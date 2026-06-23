"""
NWRE-AQG-GE-DPC
============================================
Natural-Weighted Relative-density Ellipsoid DPC with Constrained Refinement

Pipeline:
1) Sinh granular ellipsoid bằng safe split và quality gate của AQG-GE-DPC.
2) Tính khoảng cách Mahalanobis giữa các ellipsoid bằng Cholesky.
3) Xây Natural Nearest Neighborhood ở mức ellipsoid.
4) Tính mật độ ellipsoid mới từ ba nguồn: mật độ nội tại, hỗ trợ natural-neighbor,
   và mật độ tương đối trong vùng lân cận.
5) Tính delta, gamma mới và chọn tâm với ngưỡng tách thích nghi theo local scale.
6) Gán nhãn ban đầu bằng single-chain DPC ở mức ellipsoid.
7) Chỉ hiệu chỉnh các ellipsoid không chắc chắn bằng fuzzy natural-neighbor voting.
8) Chấp nhận hiệu chỉnh khi point-weighted internal quality tăng đủ, graph consistency
   không suy giảm đáng kể, và tỷ lệ điểm bị đổi nhãn nằm trong giới hạn.
9) Ánh xạ nhãn ellipsoid về điểm; ground-truth chỉ dùng ở đánh giá cuối.

Lưu ý:
- Toàn bộ neighborhood, density, center selection và refinement đều thực hiện trên
  granular ellipsoid, không chạy point-level DPC.
- Các trọng số mặc định là cấu hình khởi đầu có cơ sở từ các thành phần đã công bố,
  nhưng vẫn cần sensitivity analysis và ablation trên tập phát triển không dùng nhãn
  trong quá trình phân cụm.
"""


import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Any

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import accuracy_score, adjusted_rand_score, normalized_mutual_info_score


# ============================================================
# Ellipsoid with Cholesky + Cache ; Biểu diễn một granular ellipsoid và lưu cache các đại lượng hình học cần tính toán
# ============================================================
class Ellipsoid:
    # Khởi tạo ellipsoid từ tập dữ liệu con và lưu chỉ số gốc của các điểm dữ liệu
    def __init__(self, data: np.ndarray, indices: np.ndarray, epsilon: float = 1e-6):
        self.data = np.asarray(data, dtype=float)
        self.indices = np.asarray(indices, dtype=int)
        self.epsilon = float(epsilon)

        if self.data.ndim != 2 or self.data.shape[0] == 0:
            raise ValueError("Ellipsoid cannot be empty or non-2D.")

        self.n_samples, self.dim = self.data.shape
        self.center = np.mean(self.data, axis=0)

        self._cov_matrix = None
        self._H_matrix = None
        self._chol_factor = None
        self._rho = None
        self._lengths_rotation = None
        self._major_axis_endpoints = None
        self._density = None

    # Tính và lưu cache ma trận hiệp phương sai của ellipsoid
    @property
    def cov_matrix(self) -> np.ndarray:
        if self._cov_matrix is None:
            if self.n_samples <= 1:
                self._cov_matrix = np.zeros((self.dim, self.dim), dtype=float)
            else:
                self._cov_matrix = np.cov(self.data.T, bias=True)
        return self._cov_matrix

    # Tính ma trận hình dạng H = covariance + epsilon * I để ổn định số
    @property
    def H_matrix(self) -> np.ndarray:
        if self._H_matrix is None:
            self._H_matrix = self.cov_matrix + self.epsilon * np.eye(self.dim, dtype=float)
        return self._H_matrix

    # Phân rã Cholesky ma trận H và lưu cache để tái sử dụng
    @property
    def chol_factor(self):
        if self._chol_factor is None:
            self._chol_factor = cho_factor(self.H_matrix, lower=True, check_finite=False)
        return self._chol_factor

    # Giải hệ tuyến tính Hx = rhs bằng Cholesky, tránh nghịch đảo ma trận trực tiếp
    def solve_H(self, rhs: np.ndarray) -> np.ndarray:
        return cho_solve(self.chol_factor, rhs, check_finite=False)

    # Tính bình phương khoảng cách Mahalanobis từ các điểm đến tâm ellipsoid
    def mahal_sq_points(self, points: np.ndarray) -> np.ndarray:
        X = np.asarray(points, dtype=float)
        if X.ndim == 1:
            X = X[None, :]
        diffs = X - self.center
        solved = self.solve_H(diffs.T).T
        return np.maximum(np.einsum("ij,ij->i", diffs, solved), 0.0)

    # Tính bán kính rho để ellipsoid bao phủ các điểm dữ liệu bên trong nó
    @property
    def rho(self) -> float:
        if self._rho is None:
            self._rho = float(np.sqrt(np.max(self.mahal_sq_points(self.data))))
        return self._rho

    # Tính độ dài các bán trục và hướng xoay của ellipsoid
    @property
    def lengths_rotation(self):
        if self._lengths_rotation is None:
            eigvals_H, eigvecs_H = np.linalg.eigh(self.H_matrix)
            eigvals_H = np.maximum(eigvals_H, 1e-12)
            lengths = self.rho * np.sqrt(eigvals_H)
            self._lengths_rotation = (lengths, eigvecs_H)
        return self._lengths_rotation

    # Lấy độ dài các bán trục của ellipsoid
    @property
    def lengths(self) -> np.ndarray:
        return self.lengths_rotation[0]

    # Xác định hai điểm đầu mút gần đúng trên trục chính để phục vụ tách ellipsoid
    @property
    def major_axis_endpoints(self):
        if self._major_axis_endpoints is None:
            if self.n_samples <= 1:
                self._major_axis_endpoints = (self.center, self.center)
            else:
                center_distances = np.linalg.norm(self.data - self.center, axis=1)
                p1 = self.data[np.argmin(center_distances)]
                p2 = self.data[np.argmax(np.linalg.norm(self.data - p1, axis=1))]
                p3 = self.data[np.argmax(np.linalg.norm(self.data - p2, axis=1))]
                self._major_axis_endpoints = (p2, p3)
        return self._major_axis_endpoints

    # Tính mật độ của ellipsoid dựa trên số điểm, bán trục và khoảng cách Mahalanobis
    @property
    def density(self) -> float:
        if self._density is None:
            axes_sum = max(float(np.sum(self.lengths)), 1e-12)
            mahal = np.sqrt(self.mahal_sq_points(self.data))
            total_mahal = max(float(np.sum(mahal)), 1e-12)
            self._density = float((self.n_samples ** 2) / (axes_sum * total_mahal))
        return self._density


# ============================================================
# Data scaling; Chuẩn hóa dữ liệu đầu vào theo cấu hình của từng dataset
# ============================================================
def apply_data_scaler(data: np.ndarray, scaler_mode: str = "none") -> np.ndarray:
    mode = str(scaler_mode).lower().strip()
    if mode == "none":
        return data
    if mode == "standard":
        from sklearn.preprocessing import StandardScaler
        return StandardScaler().fit_transform(data)
    if mode == "minmax":
        from sklearn.preprocessing import MinMaxScaler
        return MinMaxScaler().fit_transform(data)
    if mode == "robust":
        from sklearn.preprocessing import RobustScaler
        return RobustScaler().fit_transform(data)
    raise ValueError("scaler_mode must be one of: none, standard, minmax, robust")


# ============================================================
# GE generation: safe split + outlier split; Duyệt danh sách ellipsoid và tách các ellipsoid có số điểm vượt ngưỡng
# ============================================================
def splits(ellipsoid_list: Sequence[Ellipsoid], num: int, epsilon: float) -> List[Ellipsoid]:
    new_ells: List[Ellipsoid] = []
    for ell in ellipsoid_list:
        if ell.n_samples < num:
            new_ells.append(ell)
        else:
            new_ells.extend(splits_ellipsoid(ell, epsilon=epsilon))
    return new_ells

# Tách một ellipsoid thành hai ellipsoid con bằng cặp điểm xa nhau và tinh chỉnh bằng Mahalanobis
def splits_ellipsoid(ellipsoid: Ellipsoid, epsilon: Optional[float] = None) -> List[Ellipsoid]:
    if ellipsoid.n_samples <= 1:
        return [ellipsoid]

    eps = ellipsoid.epsilon if epsilon is None else float(epsilon)
    data = ellipsoid.data
    indices = ellipsoid.indices
    p1, p2 = ellipsoid.major_axis_endpoints

    dist1 = np.linalg.norm(data - p1, axis=1)
    dist2 = np.linalg.norm(data - p2, axis=1)
    mask1 = dist1 < dist2
    mask2 = ~mask1

    if np.sum(mask1) == 0 or np.sum(mask2) == 0:
        return [ellipsoid]

    ell1 = Ellipsoid(data[mask1], indices[mask1], epsilon=eps)
    ell2 = Ellipsoid(data[mask2], indices[mask2], epsilon=eps)

    # Reassign points by Mahalanobis distance to child ellipsoids.
    d1 = ell1.mahal_sq_points(data)
    d2 = ell2.mahal_sq_points(data)
    mask1 = d1 < d2
    mask2 = ~mask1

    if np.sum(mask1) == 0 or np.sum(mask2) == 0:
        return [ellipsoid]

    return [
        Ellipsoid(data[mask1], indices[mask1], epsilon=eps),
        Ellipsoid(data[mask2], indices[mask2], epsilon=eps),
    ]

# Tách bổ sung các ellipsoid lớn hoặc bất ổn bằng quality gate để hạn chế tách dư thừa
def recursive_split_outlier_detection(
    initial_ellipsoids: Sequence[Ellipsoid],
    data: np.ndarray,
    t: float = 2.0,
    max_iterations: int = 10,
    epsilon: float = 1e-6,
) -> List[Ellipsoid]:
    ellipsoid_list = list(initial_ellipsoids)

    for _ in range(max_iterations):
        if not ellipsoid_list:
            break

        axes_sums = np.array([np.sum(ell.lengths) for ell in ellipsoid_list], dtype=float)
        avg_axes = float(np.mean(axes_sums))
        outliers = [ell for ell in ellipsoid_list if np.sum(ell.lengths) > 2.0 * avg_axes]

        if not outliers:
            break

        normal = [ell for ell in ellipsoid_list if ell not in outliers]
        new_ells: List[Ellipsoid] = []
        min_leaf = max(2, int(np.ceil(np.sqrt(data.shape[0]) * 0.1)))

        for ell in outliers:
            children = splits_ellipsoid(ell, epsilon=epsilon)
            if len(children) != 2 or any(child.n_samples < min_leaf for child in children):
                new_ells.append(ell)
                continue

            parent_density = ell.density
            child_density_sum = sum(child.density for child in children)
            if child_density_sum > t * parent_density:
                new_ells.extend(children)
            else:
                new_ells.append(ell)

        ellipsoid_list = normal + new_ells

    return ellipsoid_list



# ============================================================
# Ellipsoid-level Mahalanobis geometry
# ============================================================
def ellipse_mahalanobis_distance(ell_i: Ellipsoid, ell_j: Ellipsoid) -> float:
    avg_H = 0.5 * (ell_i.H_matrix + ell_j.H_matrix)
    chol_avg = cho_factor(avg_H, lower=True, check_finite=False)
    diff = ell_i.center - ell_j.center
    solved = cho_solve(chol_avg, diff, check_finite=False)
    return float(np.sqrt(max(float(diff.T @ solved), 0.0)))


def ellipse_distance(ellipsoid_list: Sequence[Ellipsoid]) -> np.ndarray:
    m = len(ellipsoid_list)
    dist_mat = np.zeros((m, m), dtype=float)
    for i in range(m):
        for j in range(i + 1, m):
            d = ellipse_mahalanobis_distance(ellipsoid_list[i], ellipsoid_list[j])
            dist_mat[i, j] = dist_mat[j, i] = d
    return dist_mat


def _validate_distance_matrix(dist_mat: np.ndarray) -> np.ndarray:
    D = np.asarray(dist_mat, dtype=float)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError("dist_mat must be square.")
    if np.any(~np.isfinite(D)) or np.any(D < -1e-12):
        raise ValueError("dist_mat contains invalid values.")
    return np.maximum(D, 0.0)


def _robust_unit_scale(values: np.ndarray) -> np.ndarray:
    """Quantile scaling to [0,1], less sensitive to extreme density values."""
    x = np.asarray(values, dtype=float)
    if x.size == 0:
        return x.copy()
    lo = float(np.quantile(x, 0.05))
    hi = float(np.quantile(x, 0.95))
    if hi <= lo + 1e-15:
        lo, hi = float(np.min(x)), float(np.max(x))
    if hi <= lo + 1e-15:
        return np.ones_like(x)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


# ============================================================
# Natural nearest-neighbor graph at ellipsoid level
# ============================================================
@dataclass(frozen=True)
class NaturalNeighborGraph:
    lambda_value: int
    knn_sets: Tuple[np.ndarray, ...]
    natural_sets: Tuple[np.ndarray, ...]
    occurrence_weight: np.ndarray
    local_scale: np.ndarray
    ranks: np.ndarray
    fallback_nodes: Tuple[int, ...]

    @property
    def mean_degree(self) -> float:
        return float(np.mean([len(x) for x in self.natural_sets])) if self.natural_sets else 0.0


@dataclass(frozen=True)
class DensityConfig:
    intrinsic_weight: float = 0.55
    neighbor_weight: float = 0.25
    relative_weight: float = 0.20
    support_power: float = 0.50
    rank_weight: float = 0.50
    mass_power: float = 0.50

    def normalized(self) -> "DensityConfig":
        s = self.intrinsic_weight + self.neighbor_weight + self.relative_weight
        if s <= 0:
            raise ValueError("Density weights must have a positive sum.")
        return DensityConfig(
            self.intrinsic_weight / s,
            self.neighbor_weight / s,
            self.relative_weight / s,
            self.support_power,
            self.rank_weight,
            self.mass_power,
        )


@dataclass(frozen=True)
class RefinementConfig:
    min_best_probability: float = 0.60
    min_probability_margin: float = 0.15
    min_uncertainty_signals: int = 2
    lower_density_factor: float = 0.30
    cluster_distance_gain: float = 0.98
    nearest_cluster_members: int = 3
    min_internal_gain: float = 0.002
    consistency_tolerance: float = 0.002
    changed_ellipsoid_ratio_limit: float = 0.20
    changed_point_ratio_limit: float = 0.10
    max_largest_point_ratio: float = 0.90


@dataclass
class RefinementDiagnostics:
    considered: int = 0
    proposed: int = 0
    changed: int = 0
    rejected_confidence: int = 0
    rejected_support: int = 0
    rejected_distance: int = 0


def build_natural_ellipsoid_graph(
    dist_mat: np.ndarray,
    max_degree: Optional[int] = None,
    use_stable_empty_stop: bool = True,
) -> NaturalNeighborGraph:
    """
    Build mutual natural-neighbor relationships between ellipsoids.

    k increases until each ellipsoid has at least one mutual neighbor or the set of
    empty neighborhoods stabilizes. A degree cap is applied only after the adaptive
    natural neighborhood is found, to retain locality in dense graphs.
    """
    D = _validate_distance_matrix(dist_mat)
    m = D.shape[0]
    if m == 0:
        return NaturalNeighborGraph(0, tuple(), tuple(), np.empty(0), np.empty(0), np.empty((0,0), int), tuple())
    if m == 1:
        empty = (np.empty(0, dtype=int),)
        return NaturalNeighborGraph(0, empty, empty, np.zeros(1), np.ones(1), np.zeros((1,1), int), tuple())

    ranked_D = D.copy()
    np.fill_diagonal(ranked_D, np.inf)
    order = np.argsort(ranked_D, axis=1)
    ranks = np.empty((m, m), dtype=int)
    for i in range(m):
        ranks[i, order[i]] = np.arange(1, m + 1)
        ranks[i, i] = 0

    chosen_mask = None
    chosen_knn = tuple()
    chosen_nat = tuple()
    chosen_k = 1
    previous_empty = None
    stable = 0

    for k in range(1, m):
        mask = np.zeros((m, m), dtype=bool)
        rows = np.repeat(np.arange(m), k)
        cols = order[:, :k].reshape(-1)
        mask[rows, cols] = True
        mutual = mask & mask.T
        knn_sets = tuple(np.flatnonzero(mask[i]).astype(int) for i in range(m))
        nat_sets = tuple(np.flatnonzero(mutual[i]).astype(int) for i in range(m))
        empty = frozenset(i for i, x in enumerate(nat_sets) if len(x) == 0)
        chosen_mask, chosen_knn, chosen_nat, chosen_k = mask, knn_sets, nat_sets, k
        if not empty:
            break
        stable = stable + 1 if empty == previous_empty else 0
        previous_empty = empty
        patience = max(1, int(np.ceil(np.log(max(m, 2)) + np.log(max(k, 1)))))
        if use_stable_empty_stop and stable >= patience:
            break

    natural = [np.asarray(x, dtype=int) for x in chosen_nat]
    fallback_nodes: List[int] = []
    for i in range(m):
        if natural[i].size == 0:
            natural[i] = np.array([int(order[i, 0])], dtype=int)
            fallback_nodes.append(i)

    if max_degree is None:
        max_degree = max(4, min(16, int(np.ceil(np.log2(max(m, 2)))) + 2))
    max_degree = max(1, int(max_degree))
    for i in range(m):
        if natural[i].size > max_degree:
            # Keep strongest mutual relations: low symmetric rank, then short distance.
            candidates = natural[i]
            score = ranks[i, candidates] + ranks[candidates, i]
            lex = np.lexsort((D[i, candidates], score))
            natural[i] = candidates[lex[:max_degree]]

    occurrence = np.sum(chosen_mask, axis=0).astype(float)
    occurrence /= max(float(m * chosen_k), 1.0)

    local_scale = np.ones(m, dtype=float)
    positive = D[D > 0]
    global_med = float(np.median(positive)) if positive.size else 1.0
    for i, neigh in enumerate(natural):
        vals = D[i, neigh]
        vals = vals[vals > 0]
        local_scale[i] = float(np.median(vals)) if vals.size else global_med
    local_scale = np.maximum(local_scale, max(global_med * 1e-9, 1e-12))

    return NaturalNeighborGraph(
        lambda_value=int(chosen_k),
        knn_sets=chosen_knn,
        natural_sets=tuple(natural),
        occurrence_weight=occurrence,
        local_scale=local_scale,
        ranks=ranks,
        fallback_nodes=tuple(fallback_nodes),
    )


def ellipsoid_similarity(i: int, j: int, D: np.ndarray, graph: NaturalNeighborGraph, rank_weight: float) -> float:
    scale = np.sqrt(graph.local_scale[i] * graph.local_scale[j]) + 1e-12
    distance_similarity = float(np.exp(-((D[i, j] / scale) ** 2)))
    symmetric_rank = float(graph.ranks[i, j] + graph.ranks[j, i])
    rank_similarity = 1.0 / (1.0 + symmetric_rank)
    rw = float(np.clip(rank_weight, 0.0, 1.0))
    return (1.0 - rw) * distance_similarity + rw * rank_similarity


# ============================================================
# Natural-weighted relative density at ellipsoid level
# ============================================================
def compute_natural_weighted_density(
    ellipsoids: Sequence[Ellipsoid],
    dist_mat: np.ndarray,
    graph: NaturalNeighborGraph,
    config: DensityConfig,
) -> Dict[str, np.ndarray]:
    cfg = config.normalized()
    D = _validate_distance_matrix(dist_mat)
    intrinsic_raw = np.array([ell.density for ell in ellipsoids], dtype=float)
    masses = np.array([ell.n_samples for ell in ellipsoids], dtype=float)
    intrinsic = _robust_unit_scale(np.log1p(np.maximum(intrinsic_raw, 0.0)))

    support = np.maximum(graph.occurrence_weight, 1e-12) ** cfg.support_power
    mass = np.log1p(masses) ** cfg.mass_power
    neighbor_raw = np.zeros(len(ellipsoids), dtype=float)
    relative_raw = np.ones(len(ellipsoids), dtype=float)

    for i, neigh in enumerate(graph.natural_sets):
        if neigh.size == 0:
            continue
        contrib = []
        for j in neigh:
            j = int(j)
            sim = ellipsoid_similarity(i, j, D, graph, cfg.rank_weight)
            contrib.append(sim * support[j] * mass[j] * (0.5 + 0.5 * intrinsic[j]))
        neighbor_raw[i] = float(np.sum(contrib))
        med = float(np.median(intrinsic_raw[neigh]))
        relative_raw[i] = intrinsic_raw[i] / (med + 1e-12)

    neighbor = _robust_unit_scale(np.log1p(np.maximum(neighbor_raw, 0.0)))
    relative = _robust_unit_scale(np.log1p(np.maximum(relative_raw, 0.0)))
    density = (
        cfg.intrinsic_weight * intrinsic
        + cfg.neighbor_weight * neighbor
        + cfg.relative_weight * relative
    )
    # A tiny deterministic tie breaker prevents ambiguous higher-density ordering.
    density = density + 1e-12 * np.arange(len(density), dtype=float)
    return {
        "density": density,
        "intrinsic_raw": intrinsic_raw,
        "intrinsic": intrinsic,
        "neighbor": neighbor,
        "relative": relative,
        "mass": masses,
        "support": support,
    }


def ellipse_min_dist(dist_mat: np.ndarray, densities: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    D = _validate_distance_matrix(dist_mat)
    rho = np.asarray(densities, dtype=float)
    m = len(rho)
    order = np.argsort(-rho, kind="mergesort")
    delta = np.zeros(m, dtype=float)
    nearest = -np.ones(m, dtype=int)
    for pos, idx in enumerate(order[1:], start=1):
        higher = order[:pos]
        j = int(higher[np.argmin(D[idx, higher])])
        nearest[idx] = j
        delta[idx] = float(D[idx, j])
    if m:
        delta[order[0]] = float(np.max(D[order[0]])) if m > 1 else 0.0
    return delta, nearest


def compute_center_score(densities: np.ndarray, deltas: np.ndarray, graph: NaturalNeighborGraph) -> np.ndarray:
    rho = _robust_unit_scale(densities)
    delta = _robust_unit_scale(deltas)
    support = _robust_unit_scale(graph.occurrence_weight)
    return rho * delta * (1.0 + 0.20 * support)


def auto_select_centers_adaptive(
    center_score: np.ndarray,
    dist_mat: np.ndarray,
    graph: NaturalNeighborGraph,
    top_k: int,
    separation_factor: float = 0.80,
) -> List[int]:
    D = _validate_distance_matrix(dist_mat)
    m = len(center_score)
    top_k = int(max(1, min(top_k, m)))
    order = np.argsort(-np.asarray(center_score), kind="mergesort")
    selected: List[int] = []
    for idx in order:
        idx = int(idx)
        if not selected:
            selected.append(idx)
        else:
            valid = True
            for c in selected:
                local_threshold = separation_factor * 0.5 * (graph.local_scale[idx] + graph.local_scale[c])
                if D[idx, c] < max(local_threshold, 1e-12):
                    valid = False
                    break
            if valid:
                selected.append(idx)
        if len(selected) == top_k:
            break
    # Fallback preserves requested cluster count, but candidates still follow center score.
    if len(selected) < top_k:
        for idx in order:
            idx = int(idx)
            if idx not in selected:
                selected.append(idx)
            if len(selected) == top_k:
                break
    return selected


# ============================================================
# Initial DPC labeling at ellipsoid level
# ============================================================
def ellipse_cluster_single_chain(
    densities: np.ndarray,
    centers: Sequence[int],
    nearest: np.ndarray,
    dist_mat: np.ndarray,
) -> np.ndarray:
    D = _validate_distance_matrix(dist_mat)
    rho = np.asarray(densities, dtype=float)
    labels = -np.ones(len(rho), dtype=int)
    centers = [int(c) for c in centers]
    for lab, c in enumerate(centers):
        labels[c] = lab
    for idx in np.argsort(-rho, kind="mergesort"):
        if labels[idx] < 0 and nearest[idx] >= 0:
            labels[idx] = labels[int(nearest[idx])]
    for idx in np.where(labels < 0)[0]:
        c = centers[int(np.argmin(D[idx, centers]))]
        labels[idx] = labels[c]
    return labels


# ============================================================
# Point-weighted internal evaluation and graph consistency
# ============================================================
def cluster_point_ratios(labels: np.ndarray, masses: np.ndarray) -> Tuple[float, float]:
    totals = np.array([np.sum(masses[labels == lab]) for lab in np.unique(labels)], dtype=float)
    s = max(float(np.sum(totals)), 1.0)
    return float(np.max(totals) / s), float(np.min(totals) / s)


def point_weighted_internal_score(
    labels: np.ndarray,
    densities: np.ndarray,
    dist_mat: np.ndarray,
    masses: np.ndarray,
) -> float:
    labels = np.asarray(labels, dtype=int)
    rho = np.asarray(densities, dtype=float)
    D = _validate_distance_matrix(dist_mat)
    masses = np.asarray(masses, dtype=float)
    unique = np.unique(labels)
    if unique.size <= 1:
        return -1e18

    reps: List[int] = []
    compact_num = 0.0
    compact_den = 0.0
    for lab in unique:
        members = np.where(labels == lab)[0]
        rep = int(members[np.argmax(rho[members])])
        reps.append(rep)
        compact_num += float(np.sum(masses[members] * D[members, rep]))
        compact_den += float(np.sum(masses[members]))
    compactness = compact_num / max(compact_den, 1e-12)

    sep_vals = [D[reps[i], reps[j]] for i in range(len(reps)) for j in range(i + 1, len(reps))]
    separation = float(np.mean(sep_vals)) if sep_vals else 0.0
    largest, smallest = cluster_point_ratios(labels, masses)
    penalty = 1.0
    if largest > 0.90:
        penalty *= 0.25
    if smallest < 0.01:
        penalty *= 0.50
    return float((separation / (compactness + 1e-12)) * penalty)


def natural_graph_consistency(labels: np.ndarray, graph: NaturalNeighborGraph, dist_mat: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=int)
    D = _validate_distance_matrix(dist_mat)
    same = total = 0.0
    visited = set()
    for i, neigh in enumerate(graph.natural_sets):
        for j in neigh:
            j = int(j)
            edge = (min(i, j), max(i, j))
            if i == j or edge in visited:
                continue
            visited.add(edge)
            w = ellipsoid_similarity(i, j, D, graph, rank_weight=0.5)
            total += w
            if labels[i] == labels[j]:
                same += w
    return same / total if total > 0 else 1.0


def _distance_to_cluster(idx: int, members: np.ndarray, D: np.ndarray, r: int) -> float:
    members = members[members != idx]
    if members.size == 0:
        return float("inf")
    values = np.sort(D[idx, members])
    return float(np.mean(values[: min(max(r, 1), values.size)]))


def _membership_probabilities(
    idx: int,
    labels: np.ndarray,
    densities: np.ndarray,
    masses: np.ndarray,
    D: np.ndarray,
    graph: NaturalNeighborGraph,
    lower_density_factor: float,
) -> Dict[int, float]:
    scores: Dict[int, float] = {}
    rho_norm = 0.2 + 0.8 * _robust_unit_scale(densities)
    for nb in graph.natural_sets[idx]:
        nb = int(nb)
        lab = int(labels[nb])
        sim = ellipsoid_similarity(idx, nb, D, graph, rank_weight=0.5)
        reliability = max(float(graph.occurrence_weight[nb]), 1e-12) ** 0.5
        density_factor = float(rho_norm[nb])
        mass_factor = float(np.sqrt(np.log1p(masses[nb])))
        hierarchy = 1.0 if densities[nb] > densities[idx] else float(lower_density_factor)
        scores[lab] = scores.get(lab, 0.0) + sim * reliability * density_factor * mass_factor * hierarchy
    total = float(sum(scores.values()))
    return {lab: val / total for lab, val in scores.items()} if total > 0 else {}


def propose_constrained_fuzzy_refinement(
    baseline_labels: np.ndarray,
    densities: np.ndarray,
    dist_mat: np.ndarray,
    masses: np.ndarray,
    centers: Sequence[int],
    nearest: np.ndarray,
    graph: NaturalNeighborGraph,
    config: RefinementConfig,
) -> Tuple[np.ndarray, RefinementDiagnostics]:
    labels = np.asarray(baseline_labels, dtype=int)
    corrected = labels.copy()
    D = _validate_distance_matrix(dist_mat)
    centers_set = {int(x) for x in centers}
    diagnostics = RefinementDiagnostics()
    radius_threshold = float(np.quantile(graph.local_scale, 0.75)) if graph.local_scale.size else 0.0

    # All proposals use baseline labels; accepted local changes do not propagate in the same pass.
    for idx in np.argsort(densities):
        idx = int(idx)
        if idx in centers_set:
            continue
        neigh = graph.natural_sets[idx]
        if neigh.size == 0:
            continue

        probs = _membership_probabilities(idx, labels, densities, masses, D, graph, config.lower_density_factor)
        if not probs:
            continue
        ranked = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        best_lab, best_prob = int(ranked[0][0]), float(ranked[0][1])
        second_prob = float(ranked[1][1]) if len(ranked) > 1 else 0.0
        current_lab = int(labels[idx])
        current_prob = float(probs.get(current_lab, 0.0))

        mixed = len(np.unique(labels[neigh])) > 1
        parent_disagreement = nearest[idx] >= 0 and labels[int(nearest[idx])] != current_lab
        low_confidence = current_prob < config.min_best_probability
        large_local_radius = graph.local_scale[idx] > radius_threshold
        signals = int(mixed) + int(parent_disagreement) + int(low_confidence) + int(large_local_radius)
        if signals < config.min_uncertainty_signals:
            continue
        diagnostics.considered += 1

        if best_lab == current_lab:
            continue
        diagnostics.proposed += 1
        if best_prob < config.min_best_probability or (best_prob - second_prob) < config.min_probability_margin:
            diagnostics.rejected_confidence += 1
            continue

        supporting = neigh[(labels[neigh] == best_lab) & (densities[neigh] > densities[idx])]
        if supporting.size == 0:
            diagnostics.rejected_support += 1
            continue

        current_members = np.where(labels == current_lab)[0]
        best_members = np.where(labels == best_lab)[0]
        d_current = _distance_to_cluster(idx, current_members, D, config.nearest_cluster_members)
        d_best = _distance_to_cluster(idx, best_members, D, config.nearest_cluster_members)
        if not np.isfinite(d_best) or d_best >= config.cluster_distance_gain * d_current:
            diagnostics.rejected_distance += 1
            continue

        corrected[idx] = best_lab
        diagnostics.changed += 1

    return corrected, diagnostics


def choose_labels_with_constrained_refinement(
    densities: np.ndarray,
    dist_mat: np.ndarray,
    masses: np.ndarray,
    centers: Sequence[int],
    nearest: np.ndarray,
    graph: NaturalNeighborGraph,
    config: RefinementConfig,
) -> Dict[str, Any]:
    single = ellipse_cluster_single_chain(densities, centers, nearest, dist_mat)
    refined, diagnostics = propose_constrained_fuzzy_refinement(
        single, densities, dist_mat, masses, centers, nearest, graph, config
    )
    single_score = point_weighted_internal_score(single, densities, dist_mat, masses)
    refined_score = point_weighted_internal_score(refined, densities, dist_mat, masses)
    single_cons = natural_graph_consistency(single, graph, dist_mat)
    refined_cons = natural_graph_consistency(refined, graph, dist_mat)

    changed_mask = single != refined
    ell_ratio = float(np.mean(changed_mask))
    point_ratio = float(np.sum(masses[changed_mask]) / max(np.sum(masses), 1.0))
    largest_ratio, _ = cluster_point_ratios(refined, masses)
    relative_gain = (refined_score - single_score) / (abs(single_score) + 1e-12)

    safe = (
        len(np.unique(refined)) == len(np.unique(single))
        and ell_ratio <= config.changed_ellipsoid_ratio_limit
        and point_ratio <= config.changed_point_ratio_limit
        and largest_ratio <= config.max_largest_point_ratio
    )
    accepted = bool(
        diagnostics.changed > 0
        and safe
        and relative_gain >= config.min_internal_gain
        and refined_cons + config.consistency_tolerance >= single_cons
    )
    return {
        "labels": refined if accepted else single,
        "single_labels": single,
        "refined_labels": refined,
        "mode": "natural_fuzzy" if accepted else "single",
        "accepted": accepted,
        "diagnostics": diagnostics,
        "single_score": single_score,
        "refined_score": refined_score,
        "relative_gain": relative_gain,
        "single_consistency": single_cons,
        "refined_consistency": refined_cons,
        "changed_ellipsoid_ratio": ell_ratio,
        "changed_point_ratio": point_ratio,
    }


def choose_best_configuration_by_internal_score(
    densities: np.ndarray,
    deltas: np.ndarray,
    center_score: np.ndarray,
    dist_mat: np.ndarray,
    masses: np.ndarray,
    nearest: np.ndarray,
    graph: NaturalNeighborGraph,
    k_candidates: Sequence[int],
    refinement_config: RefinementConfig,
    center_separation_factor: float,
) -> Dict[str, Any]:
    best = None
    for k in sorted(set(int(x) for x in k_candidates if int(x) > 0)):
        centers = auto_select_centers_adaptive(
            center_score, dist_mat, graph, k, separation_factor=center_separation_factor
        )
        choice = choose_labels_with_constrained_refinement(
            densities, dist_mat, masses, centers, nearest, graph, refinement_config
        )
        final_score = point_weighted_internal_score(choice["labels"], densities, dist_mat, masses)
        final_cons = natural_graph_consistency(choice["labels"], graph, dist_mat)
        item = {"k": k, "centers": centers, "final_score": final_score, "final_consistency": final_cons, **choice}
        if best is None or final_score > best["final_score"] + 1e-12 or (
            abs(final_score - best["final_score"]) <= 1e-12 and final_cons > best["final_consistency"]
        ):
            best = item
    if best is None:
        raise RuntimeError("No valid clustering configuration was generated.")
    return best

# ============================================================
# Evaluation; Căn chỉnh nhãn dự đoán với nhãn thật bằng thuật toán Hungarian để tính ACC
# ============================================================
def align_labels(true_labels: np.ndarray, pred_labels: np.ndarray) -> np.ndarray:
    true_classes = np.unique(true_labels)
    pred_classes = np.unique(pred_labels)
    confusion = np.zeros((len(true_classes), len(pred_classes)), dtype=float)

    for i, t in enumerate(true_classes):
        for j, p in enumerate(pred_classes):
            confusion[i, j] = np.sum((true_labels == t) & (pred_labels == p))

    row_ind, col_ind = linear_sum_assignment(-confusion)
    mapping = {pred_classes[j]: true_classes[i] for i, j in zip(row_ind, col_ind)}
    return np.array([mapping.get(label, -1) for label in pred_labels])

# In phân bố số lượng mẫu theo từng nhãn
def print_distribution(name: str, labels: np.ndarray) -> None:
    unique, counts = np.unique(labels, return_counts=True)
    print(name)
    for lab, cnt in zip(unique, counts):
        print(f"  label {int(lab):>4}: {int(cnt)}")


# ============================================================
# Plotting; Vẽ biểu đồ phân cụm 2D; nếu dữ liệu nhiều chiều thì giảm chiều bằng PCA
# ============================================================
def plot_predicted_clusters_2d(
    data: np.ndarray,
    pred_labels: np.ndarray,
    dataset_name: str = "dataset",
    show_legend: bool = False,
) -> None:
    """
    Draw clustering result as a 2D scatter chart.
    - If data has 2 dimensions: plot directly.
    - If data has more than 2 dimensions: reduce to 2D using PCA.
    This function is only for visualization and does not affect clustering results.
    """
    X = np.asarray(data, dtype=float)
    y = np.asarray(pred_labels)

    if X.ndim != 2:
        raise ValueError("Data must be a 2D array.")

    if X.shape[1] > 2:
        from sklearn.decomposition import PCA
        X_vis = PCA(n_components=2, random_state=42).fit_transform(X)
        xlabel = "PC1"
        ylabel = "PC2"
    else:
        X_vis = X
        xlabel = "Feature 1"
        ylabel = "Feature 2"

    # Plot style similar to the example image
    plt.figure(figsize=(6.4, 4.8))
    unique_labels = np.unique(y)

    for lab in unique_labels:
        mask = (y == lab)
        plt.scatter(
            X_vis[mask, 0],
            X_vis[mask, 1],
            s=16,
            alpha=0.95
        )

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"Clustering Result - {dataset_name}")
    if show_legend:
        plt.legend([f"Cluster {lab}" for lab in unique_labels], loc="best")
    plt.tight_layout()
    plt.show()



# ============================================================
# Inherited KSE dataset configurations
# ============================================================
BASE_DATASET_CONFIGS: Dict[str, Dict[str, object]] = {
    "iris": {"scaler": "none", "k": 3, "outlier_t": 1.3, "split_factor": 1.0},
    "seed": {"scaler": "none", "k": 3, "outlier_t": 1.3, "split_factor": 1.0},
    "segment_3": {"scaler": "robust", "k": 8, "outlier_t": 1.7, "split_factor": 0.8},
    "landsat_2": {"scaler": "none", "k": 5, "outlier_t": 2.0, "split_factor": 1.0},
    "msplice_2": {"scaler": "standard", "k": 3, "outlier_t": 2.0, "split_factor": 0.6},
    "rice": {"scaler": "standard", "k": 2, "outlier_t": 1.5, "split_factor": 0.6},
    "banknote": {"scaler": "standard", "k": 2, "outlier_t": 1.5, "split_factor": 0.8},
    "htru2": {"scaler": "none", "k": 2, "outlier_t": 2.0, "split_factor": 1.0},
    "breast_cancer": {"scaler": "none", "k": 2, "outlier_t": 1.5, "split_factor": 0.6},
    "hcv_data": {"scaler": "robust", "k": 2, "outlier_t": 2.0, "split_factor": 1.0},
    "dry_bean": {"scaler": "standard", "k": 7, "outlier_t": 1.3, "split_factor": 0.6},
    "rice_cammeo": {"scaler": "standard", "k": 2, "outlier_t": 1.3, "split_factor": 0.6},
}

DEFAULT_DENSITY_CONFIG = DensityConfig()
DEFAULT_REFINEMENT_CONFIG = RefinementConfig()
# ============================================================
# Danh sách dataset mặc định.
# ============================================================
def get_default_dataset_names() -> List[str]:
    return [
        "iris", "seed", "segment_3", "landsat_2",
        "msplice_2", "rice", "banknote", "htru2",
        "breast_cancer", "hcv_data", "dry_bean", "rice_cammeo",
    ]


# ============================================================
# Khai báo đường dẫn feature và label cho các bộ dữ liệu.
# ============================================================
def get_default_dataset_registry(base_dir: Path) -> Dict[str, Tuple[Path, Path]]:
    return {
        "iris": (
            base_dir / "real_dataset_and_label" / "real_datasets" / "Iris.txt",
            base_dir / "real_dataset_and_label" / "real_datasets_label" / "Iris_label.txt",
        ),
        "seed": (
            base_dir / "real_dataset_and_label" / "real_datasets" / "Seed.txt",
            base_dir / "real_dataset_and_label" / "real_datasets_label" / "Seed_label.txt",
        ),
        "segment_3": (
            base_dir / "real_dataset_and_label" / "real_datasets" / "segment_3.txt",
            base_dir / "real_dataset_and_label" / "real_datasets_label" / "segment_3_label.txt",
        ),
        "landsat_2": (
            base_dir / "real_dataset_and_label" / "real_datasets" / "landsat_2.txt",
            base_dir / "real_dataset_and_label" / "real_datasets_label" / "landsat_2_label.txt",
        ),
        "msplice_2": (
            base_dir / "real_dataset_and_label" / "real_datasets" / "msplice_2.txt",
            base_dir / "real_dataset_and_label" / "real_datasets_label" / "msplice_2_label.txt",
        ),
        "rice": (
            base_dir / "real_dataset_and_label" / "real_datasets" / "rice.txt",
            base_dir / "real_dataset_and_label" / "real_datasets_label" / "rice_label.txt",
        ),
        "banknote": (
            base_dir / "real_dataset_and_label" / "real_datasets" / "banknote.txt",
            base_dir / "real_dataset_and_label" / "real_datasets_label" / "banknote_label.txt",
        ),
        "htru2": (
            base_dir / "real_dataset_and_label" / "real_datasets" / "htru2.txt",
            base_dir / "real_dataset_and_label" / "real_datasets_label" / "htru2_label.txt",
        ),
        "breast_cancer": (
            base_dir / "dataset" / "unlabel" / "breast_cancer.txt",
            base_dir / "dataset" / "label" / "breast_cancer_label.txt",
        ),
        "hcv_data": (
            base_dir / "dataset" / "unlabel" / "hcv_data.txt",
            base_dir / "dataset" / "label" / "hcv_data_label.txt",
        ),
        "dry_bean": (
            base_dir / "dataset" / "unlabel" / "dry_bean.txt",
            base_dir / "dataset" / "label" / "dry_bean_label.txt",
        ),
        "rice_cammeo": (
            base_dir / "dataset" / "unlabel" / "rice+cammeo.txt",
            base_dir / "dataset" / "label" / "rice+cammeo_label.txt",
        ),
    }




# ============================================================
# Run NWRE-AQG-GE-DPC on one dataset
# ============================================================
def run_dataset(
    dataset_name: str,
    base_dir: Path,
    epsilon: float = 1e-6,
    show_chart: bool = False,
    verbose: bool = True,
    k_candidates: Optional[Sequence[int]] = None,
    density_config: Optional[DensityConfig] = None,
    refinement_config: Optional[RefinementConfig] = None,
    center_separation_factor: float = 0.80,
    max_natural_degree: Optional[int] = None,
) -> Dict[str, object]:
    key = dataset_name.lower()
    registry = get_default_dataset_registry(base_dir)
    if key not in registry or key not in BASE_DATASET_CONFIGS:
        raise KeyError(f"Unknown or unconfigured dataset: {dataset_name}")
    feature_file, label_file = registry[key]
    if not feature_file.exists():
        raise FileNotFoundError(feature_file)

    cfg = BASE_DATASET_CONFIGS[key]
    density_config = density_config or DEFAULT_DENSITY_CONFIG
    refinement_config = refinement_config or DEFAULT_REFINEMENT_CONFIG
    data = np.loadtxt(feature_file, dtype=float)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    data = apply_data_scaler(data, str(cfg["scaler"]))
    inherited_k = int(cfg["k"])
    num = max(5, int(np.ceil(np.sqrt(data.shape[0]) * float(cfg["split_factor"]))))
    log = print if verbose else (lambda *args, **kwargs: None)

    t0 = time.perf_counter()
    ellipsoids: List[Ellipsoid] = [Ellipsoid(data, np.arange(len(data)), epsilon)]
    while True:
        before = len(ellipsoids)
        ellipsoids = splits(ellipsoids, num, epsilon)
        if len(ellipsoids) == before:
            break
    ellipsoids = recursive_split_outlier_detection(
        ellipsoids, data, float(cfg["outlier_t"]), 10, epsilon
    )
    t_gen = time.perf_counter() - t0

    t0 = time.perf_counter()
    D = ellipse_distance(ellipsoids)
    graph = build_natural_ellipsoid_graph(D, max_degree=max_natural_degree)
    density_parts = compute_natural_weighted_density(ellipsoids, D, graph, density_config)
    densities = density_parts["density"]
    deltas, nearest = ellipse_min_dist(D, densities)
    center_score = compute_center_score(densities, deltas, graph)
    masses = density_parts["mass"]
    t_attr = time.perf_counter() - t0

    t0 = time.perf_counter()
    candidates = [inherited_k] if k_candidates is None else list(k_candidates)
    best = choose_best_configuration_by_internal_score(
        densities, deltas, center_score, D, masses, nearest, graph,
        candidates, refinement_config, center_separation_factor
    )
    t_cluster = time.perf_counter() - t0

    ell_labels = np.asarray(best["labels"], dtype=int)
    pred = np.full(len(data), -1, dtype=int)
    for i, ell in enumerate(ellipsoids):
        pred[ell.indices] = ell_labels[i]
    if np.any(pred < 0):
        raise RuntimeError("Unassigned points remain.")
    if show_chart:
        plot_predicted_clusters_2d(data, pred, key, False)

    if not label_file.exists():
        raise FileNotFoundError(label_file)
    truth = np.loadtxt(label_file, dtype=float).astype(int)
    aligned = align_labels(truth, pred)
    acc = accuracy_score(truth, aligned)
    nmi = normalized_mutual_info_score(truth, pred)
    ari = adjusted_rand_score(truth, pred)
    total = t_gen + t_attr + t_cluster
    diag: RefinementDiagnostics = best["diagnostics"]

    log("=" * 100)
    log(f"Dataset: {key} | n={len(data)}, d={data.shape[1]}, ellipsoids={len(ellipsoids)}")
    log(f"Natural lambda={graph.lambda_value}, mean degree={graph.mean_degree:.2f}")
    log(f"Centers={best['centers']} | mode={best['mode']} | changed={diag.changed}")
    log(f"Changed ellipsoid ratio={best['changed_ellipsoid_ratio']:.4f}")
    log(f"Changed point ratio={best['changed_point_ratio']:.4f}")
    log(f"Internal score single/refined={best['single_score']:.6f}/{best['refined_score']:.6f}")
    log(f"ACC={acc:.3f}, NMI={nmi:.3f}, ARI={ari:.3f}, time={total*1000:.3f} ms")

    return {
        "dataset": key, "acc": float(acc), "nmi": float(nmi), "ari": float(ari),
        "scaler": str(cfg["scaler"]), "selected_k": int(best["k"]), "mode": best["mode"],
        "n_ellipsoids": len(ellipsoids), "natural_lambda": graph.lambda_value,
        "mean_natural_degree": graph.mean_degree, "changed_ellipsoids": diag.changed,
        "changed_ellipsoid_ratio": best["changed_ellipsoid_ratio"],
        "changed_point_ratio": best["changed_point_ratio"],
        "refinement_accepted": best["accepted"], "internal_gain": best["relative_gain"],
        "total_time": total, "pred_labels": pred, "aligned_pred_labels": aligned,
        "ellipsoid_labels": ell_labels, "centers": best["centers"],
        "density_components": density_parts,
    }


def run_all_datasets(
    base_dir: Path,
    dataset_names: Optional[Sequence[str]] = None,
    epsilon: float = 1e-6,
    show_chart: bool = False,
    verbose_each_dataset: bool = False,
    density_config: Optional[DensityConfig] = None,
    refinement_config: Optional[RefinementConfig] = None,
) -> Dict[str, Dict[str, object]]:
    names = list(dataset_names) if dataset_names is not None else get_default_dataset_names()
    results: Dict[str, Dict[str, object]] = {}
    for name in names:
        try:
            results[name] = run_dataset(
                name, base_dir, epsilon, show_chart, verbose_each_dataset,
                density_config=density_config, refinement_config=refinement_config
            )
        except Exception as exc:
            results[name] = {"error": str(exc)}
            print(f"[ERROR] {name}: {exc}")

    print("\n" + "=" * 160)
    print("SUMMARY - NWRE-AQG-GE-DPC")
    print("=" * 160)
    print(f"{'Dataset':<18} {'ACC':>7} {'NMI':>7} {'ARI':>7} {'Scaler':>9} {'k':>4} {'Mode':>14} {'Ells':>6} {'Lambda':>7} {'N-Deg':>7} {'Changed':>8} {'P-Change':>9} {'Accept':>8} {'Time(ms)':>11}")
    print("-" * 160)
    for name, res in results.items():
        if "error" in res:
            print(f"{name:<18} ERROR: {res['error']}")
            continue
        print(f"{name:<18} {res['acc']:>7.3f} {res['nmi']:>7.3f} {res['ari']:>7.3f} {res['scaler']:>9} {res['selected_k']:>4} {res['mode']:>14} {res['n_ellipsoids']:>6} {res['natural_lambda']:>7} {res['mean_natural_degree']:>7.2f} {res['changed_ellipsoids']:>8} {res['changed_point_ratio']:>9.3f} {str(res['refinement_accepted']):>8} {res['total_time']*1000:>11.3f}")
    print("=" * 160)
    return results


def run_smoke_test(random_state: int = 42) -> None:
    rng = np.random.default_rng(random_state)
    a = rng.normal(size=(100, 2)) @ np.array([[1.7, 0.6], [0.0, 0.25]]) + [-2, 0]
    b = rng.normal(size=(100, 2)) @ np.array([[1.4, -0.5], [0.0, 0.30]]) + [2, 0.5]
    data = np.vstack([a, b])
    ells: List[Ellipsoid] = [Ellipsoid(data, np.arange(len(data)))]
    threshold = max(5, int(np.ceil(np.sqrt(len(data)))))
    while True:
        old = len(ells)
        ells = splits(ells, threshold, 1e-6)
        if len(ells) == old:
            break
    D = ellipse_distance(ells)
    graph = build_natural_ellipsoid_graph(D)
    parts = compute_natural_weighted_density(ells, D, graph, DEFAULT_DENSITY_CONFIG)
    delta, nearest = ellipse_min_dist(D, parts["density"])
    score = compute_center_score(parts["density"], delta, graph)
    result = choose_best_configuration_by_internal_score(
        parts["density"], delta, score, D, parts["mass"], nearest, graph,
        [2], DEFAULT_REFINEMENT_CONFIG, 0.8
    )
    assert len(result["labels"]) == len(ells)
    print(f"Smoke test passed | ellipsoids={len(ells)}, lambda={graph.lambda_value}, mode={result['mode']}")


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent
    run_all_datasets(BASE_DIR, show_chart=False, verbose_each_dataset=False)
