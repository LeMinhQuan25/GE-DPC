"""
ANW-AQG-GE-DPC
============================================
Adaptive Natural-Weighted Ellipsoid Label Refinement for AQG-GE-DPC

Mục tiêu của phiên bản mở rộng:
1) Kế thừa toàn bộ phần sinh granular ellipsoid, quality gate, Mahalanobis-Cholesky
   và ellipsoid-level cache từ AQG-GE-DPC.
2) Giữ density-delta center scoring và single-chain DPC làm cấu hình nhãn ban đầu.
3) Xây dựng Natural Nearest Neighborhood ở mức ellipsoid, không sử dụng một số
   láng giềng K cố định cho mọi ellipsoid.
4) Tính độ hỗ trợ cấu trúc của mỗi ellipsoid từ tần suất nó xuất hiện trong danh
   sách KNN của các ellipsoid khác, theo nguyên lý weighted KNN.
5) Chỉ đề xuất sửa nhãn cho ellipsoid không chắc chắn bằng fuzzy natural-neighbor
   membership; tâm cụm không bao giờ bị sửa.
6) Chỉ chấp nhận refinement khi thỏa đồng thời các safety gate toàn cục, internal
   score không giảm và natural-graph consistency không giảm.
7) Ground-truth labels chỉ được đọc và sử dụng sau khi quá trình phân cụm hoàn tất.

Lưu ý khoa học:
- Đây là một mở rộng ở mức ellipsoid, không phải bản sao point-level của FWNNN-DPC
  hoặc LW-DPC.
- Natural neighborhood sử dụng khoảng cách Mahalanobis giữa ellipsoid thay cho
  Euclidean distance giữa điểm.
- Fuzzy membership sử dụng w_ij = 1 / (1 + d_ij) và hệ số quan hệ được chuẩn hóa
  theo natural neighborhood của ellipsoid láng giềng.
- KNN-occurrence weight chỉ được dùng như độ tin cậy cấu trúc trong refinement;
  density gốc của AQG-GE-DPC được giữ nguyên để cô lập đóng góp của label refinement.
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
# DPC attributes; Tính khoảng cách Mahalanobis giữa hai ellipsoid bằng ma trận H trung bình và Cholesky
# ============================================================
def ellipse_mahalanobis_distance(ell_i: Ellipsoid, ell_j: Ellipsoid) -> float:
    avg_H = 0.5 * (ell_i.H_matrix + ell_j.H_matrix)
    chol_avg = cho_factor(avg_H, lower=True, check_finite=False)
    diff = ell_i.center - ell_j.center
    solved = cho_solve(chol_avg, diff, check_finite=False)
    return float(np.sqrt(max(float(diff.T @ solved), 0.0)))

# Tạo ma trận khoảng cách đôi một giữa tất cả các ellipsoid
def ellipse_distance(ellipsoid_list: Sequence[Ellipsoid]) -> np.ndarray:
    n = len(ellipsoid_list)
    dist_mat = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            d = ellipse_mahalanobis_distance(ellipsoid_list[i], ellipsoid_list[j])
            dist_mat[i, j] = dist_mat[j, i] = d
    return dist_mat

# Tính delta và ellipsoid có mật độ cao hơn gần nhất theo nguyên lý DPC
def ellipse_min_dist(dist_mat: np.ndarray, densities: np.ndarray):
    densities = np.asarray(densities, dtype=float)
    n = len(densities)
    order = np.argsort(-densities)
    min_dists = np.zeros(n, dtype=float)
    nearest = -np.ones(n, dtype=int)

    for idx in order[1:]:
        higher = np.where(densities > densities[idx])[0]
        if higher.size > 0:
            j = higher[np.argmin(dist_mat[idx, higher])]
            nearest[idx] = int(j)
            min_dists[idx] = dist_mat[idx, j]

    if n > 0:
        min_dists[order[0]] = np.max(min_dists)
    return min_dists, nearest


# ============================================================
# Center selection; # Tự động chọn tâm cụm dựa trên gamma và lọc các tâm quá gần nhau
# ============================================================
def auto_select_centers_quality(
    densities: np.ndarray,
    min_dists: np.ndarray,
    dist_mat: np.ndarray,
    top_k: int,
) -> List[int]:
    n = len(densities)
    if n == 0:
        return []
    if n == 1:
        return [0]

    top_k = int(max(1, min(top_k, n)))
    gamma = densities * min_dists
    order = np.argsort(-gamma)

    positive_dists = dist_mat[dist_mat > 0]
    if positive_dists.size == 0:
        return order[:top_k].tolist()

    median_dist = float(np.median(positive_dists))
    q25_dist = float(np.quantile(positive_dists, 0.25))
    min_sep = max(q25_dist, 0.20 * median_dist, 1e-12)

    selected: List[int] = []
    for idx in order:
        idx = int(idx)
        if not selected:
            selected.append(idx)
        else:
            nearest_selected_dist = float(np.min(dist_mat[idx, selected]))
            if nearest_selected_dist >= min_sep:
                selected.append(idx)
        if len(selected) >= top_k:
            break

    # Fallback if pruning removes too many centers.
    if len(selected) < top_k:
        for idx in order:
            idx = int(idx)
            if idx not in selected:
                selected.append(idx)
            if len(selected) >= top_k:
                break

    return selected[:top_k]


# ============================================================
# Label assignment; # Gán nhãn ellipsoid theo chuỗi DPC từ các tâm cụm đã chọn
# ============================================================
def ellipse_cluster_single_chain(
    densities: np.ndarray,
    centers: Sequence[int],
    nearest: np.ndarray,
    dist_mat: np.ndarray,
) -> np.ndarray:
    n = len(densities)
    labels = -np.ones(n, dtype=int)
    centers = [int(c) for c in centers if 0 <= int(c) < n]

    if len(centers) == 0:
        labels[:] = 0
        return labels

    for lab, c in enumerate(centers):
        labels[c] = lab

    order = np.argsort(-densities)
    for idx in order:
        if labels[idx] == -1 and nearest[idx] != -1:
            labels[idx] = labels[nearest[idx]]

    # Fallback for any unlabeled ellipsoid.
    for idx in np.where(labels == -1)[0]:
        c = centers[int(np.argmin(dist_mat[idx, centers]))]
        labels[idx] = labels[c]

    return labels


# ============================================================
# Adaptive natural-neighbor graph at ellipsoid level
# ============================================================
@dataclass(frozen=True)
class NaturalNeighborGraph:
    """Cấu trúc natural-neighbor được xây hoàn toàn từ ma trận khoảng cách ellipsoid."""

    lambda_value: int
    knn_sets: Tuple[np.ndarray, ...]
    natural_sets: Tuple[np.ndarray, ...]
    occurrence_weight: np.ndarray
    radii: np.ndarray
    fallback_nodes: Tuple[int, ...]

    @property
    def mean_degree(self) -> float:
        if not self.natural_sets:
            return 0.0
        return float(np.mean([len(x) for x in self.natural_sets]))


def _validate_distance_matrix(dist_mat: np.ndarray) -> np.ndarray:
    D = np.asarray(dist_mat, dtype=float)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError("dist_mat must be a square matrix.")
    if np.any(~np.isfinite(D)):
        raise ValueError("dist_mat contains NaN or infinite values.")
    if np.any(D < -1e-12):
        raise ValueError("dist_mat contains negative distances.")
    D = np.maximum(D, 0.0)
    return D


def build_natural_ellipsoid_graph(
    dist_mat: np.ndarray,
    use_stable_empty_stop: bool = True,
) -> NaturalNeighborGraph:
    """
    Tìm Natural Nearest Neighborhood (NNN) ở mức ellipsoid.

    Với bậc k:
      NN_k(i)  : k ellipsoid gần nhất của i.
      RNN_k(i) : các ellipsoid xem i là một trong k láng giềng gần nhất.
      NNN_k(i) = NN_k(i) giao RNN_k(i).

    Quá trình tăng k dần và dừng khi:
    - mọi ellipsoid đã có ít nhất một natural neighbor; hoặc
    - tập ellipsoid có NNN rỗng không đổi đủ lâu. Ngưỡng ổn định dùng
      ceil(log(M) + log(k)), tương ứng với cơ chế giảm ảnh hưởng của
      outlier/noise trong FWNNN-DPC.

    Nếu dừng sớm mà một ellipsoid vẫn có NNN rỗng, ellipsoid gần nhất được dùng
    làm fallback duy nhất, đúng với giả định mỗi đối tượng cần ít nhất một
    "true friend".
    """
    D = _validate_distance_matrix(dist_mat)
    n = D.shape[0]

    if n == 0:
        return NaturalNeighborGraph(0, tuple(), tuple(), np.empty(0), np.empty(0), tuple())
    if n == 1:
        empty = (np.empty(0, dtype=int),)
        return NaturalNeighborGraph(0, empty, empty, np.zeros(1), np.zeros(1), tuple())

    ranked = D.copy()
    np.fill_diagonal(ranked, np.inf)
    neighbor_order = np.argsort(ranked, axis=1)

    selected_knn: Tuple[np.ndarray, ...] = tuple()
    selected_natural: Tuple[np.ndarray, ...] = tuple()
    selected_mask = np.zeros((n, n), dtype=bool)
    selected_k = 1

    previous_empty: Optional[frozenset[int]] = None
    stable_rounds = 0

    for k in range(1, n):
        knn_mask = np.zeros((n, n), dtype=bool)
        rows = np.repeat(np.arange(n), k)
        cols = neighbor_order[:, :k].reshape(-1)
        knn_mask[rows, cols] = True

        mutual_mask = knn_mask & knn_mask.T
        knn_sets = tuple(np.flatnonzero(knn_mask[i]).astype(int) for i in range(n))
        natural_sets = tuple(np.flatnonzero(mutual_mask[i]).astype(int) for i in range(n))
        empty_set = frozenset(i for i, neigh in enumerate(natural_sets) if len(neigh) == 0)

        selected_knn = knn_sets
        selected_natural = natural_sets
        selected_mask = knn_mask
        selected_k = k

        if not empty_set:
            break

        if previous_empty == empty_set:
            stable_rounds += 1
        else:
            stable_rounds = 0

        previous_empty = empty_set
        patience = max(1, int(np.ceil(np.log(max(n, 2)) + np.log(max(k, 1)))))
        if use_stable_empty_stop and stable_rounds >= patience:
            break

    natural_mutable = [np.asarray(x, dtype=int).copy() for x in selected_natural]
    fallback_nodes: List[int] = []
    for i, neigh in enumerate(natural_mutable):
        if neigh.size == 0:
            nearest = int(neighbor_order[i, 0])
            natural_mutable[i] = np.array([nearest], dtype=int)
            fallback_nodes.append(i)

    occurrence_count = np.sum(selected_mask, axis=0).astype(float)
    denominator = float(max(n * selected_k, 1))
    occurrence_weight = occurrence_count / denominator

    radii = np.zeros(n, dtype=float)
    for i, neigh in enumerate(natural_mutable):
        if neigh.size > 0:
            radii[i] = float(np.max(D[i, neigh]))

    return NaturalNeighborGraph(
        lambda_value=int(selected_k),
        knn_sets=selected_knn,
        natural_sets=tuple(natural_mutable),
        occurrence_weight=occurrence_weight,
        radii=radii,
        fallback_nodes=tuple(fallback_nodes),
    )


def natural_neighbor_outlier_mask(graph: NaturalNeighborGraph) -> np.ndarray:
    """
    Đánh dấu ellipsoid có natural-neighborhood radius lớn hơn radius trung bình.
    Đây là bản chuyển trực tiếp của tiêu chuẩn outlier trong FWNNN-DPC sang mức
    ellipsoid. Mặt nạ này chỉ dùng để xác định đối tượng cần kiểm tra, không tự
    động quyết định đổi nhãn.
    """
    if graph.radii.size == 0:
        return np.empty(0, dtype=bool)
    threshold = float(np.mean(graph.radii))
    return graph.radii > threshold


def natural_graph_consistency(
    labels: np.ndarray,
    graph: NaturalNeighborGraph,
    dist_mat: np.ndarray,
) -> float:
    """Tỷ lệ trọng số natural-neighbor edges nối hai ellipsoid cùng nhãn."""
    labels = np.asarray(labels, dtype=int)
    D = _validate_distance_matrix(dist_mat)
    same_weight = 0.0
    total_weight = 0.0

    for i, neigh in enumerate(graph.natural_sets):
        for j in neigh:
            j = int(j)
            if i == j:
                continue
            weight = 1.0 / (1.0 + float(D[i, j]))
            total_weight += weight
            if labels[i] == labels[j]:
                same_weight += weight

    if total_weight <= 0.0:
        return 0.0
    return float(same_weight / total_weight)


# ============================================================
# Fuzzy natural-weighted ellipsoid label refinement
# ============================================================
@dataclass(frozen=True)
class RefinementConfig:
    """Các gate của label refinement. Mặc định dùng chung cho mọi dataset."""

    switch_margin: float = 1.50
    max_largest_ratio: float = 0.85
    changed_ratio_limit: float = 0.35
    require_higher_density_support: bool = True
    require_representative_improvement: bool = True
    internal_score_tolerance: float = 1e-12
    consistency_tolerance: float = 1e-12


@dataclass
class RefinementDiagnostics:
    considered: int = 0
    proposed: int = 0
    changed: int = 0
    outlier_candidates: int = 0
    mixed_neighbor_candidates: int = 0
    parent_disagreement_candidates: int = 0
    rejected_no_membership: int = 0
    rejected_margin: int = 0
    rejected_higher_density: int = 0
    rejected_distance: int = 0


def _cluster_representatives(labels: np.ndarray, densities: np.ndarray) -> Dict[int, int]:
    representatives: Dict[int, int] = {}
    for lab in np.unique(labels):
        members = np.where(labels == lab)[0]
        representatives[int(lab)] = int(members[np.argmax(densities[members])])
    return representatives


def _fuzzy_membership_scores(
    idx: int,
    labels: np.ndarray,
    dist_mat: np.ndarray,
    graph: NaturalNeighborGraph,
) -> Dict[int, float]:
    """
    Tính fuzzy membership của ellipsoid idx theo natural neighbors.

    Thành phần cốt lõi:
      w_ij = 1 / (1 + d_ij)
      gamma_ij = w_ij / sum_{l in NNN(j)} w_lj
      p_i(c) = sum gamma_ij * w_ij * reliability_j

    reliability_j là KNN-occurrence weight của j, được chuẩn hóa trong chính tập
    natural neighbors của i. Đây là cách sử dụng trực tiếp nguyên lý weighted KNN
    để đánh giá độ tin cậy cấu trúc của láng giềng, không thay đổi density gốc.
    """
    D = dist_mat
    neighbors = np.asarray(graph.natural_sets[idx], dtype=int)
    neighbors = neighbors[(neighbors >= 0) & (neighbors < len(labels)) & (neighbors != idx)]
    if neighbors.size == 0:
        return {}

    raw_support = np.maximum(graph.occurrence_weight[neighbors], 0.0)
    support_sum = float(np.sum(raw_support))
    if support_sum <= 1e-15:
        reliability = np.full(neighbors.size, 1.0 / neighbors.size, dtype=float)
    else:
        reliability = raw_support / support_sum

    scores: Dict[int, float] = {}
    for pos, nb in enumerate(neighbors):
        nb = int(nb)
        lab = int(labels[nb])
        if lab < 0:
            continue

        wij = 1.0 / (1.0 + float(D[idx, nb]))
        nb_neighbors = np.asarray(graph.natural_sets[nb], dtype=int)
        nb_neighbors = nb_neighbors[(nb_neighbors >= 0) & (nb_neighbors < len(labels)) & (nb_neighbors != nb)]

        if nb_neighbors.size == 0:
            denominator = wij
        else:
            denominator = float(np.sum(1.0 / (1.0 + D[nb, nb_neighbors])))
        denominator = max(denominator, 1e-15)

        gamma_ij = wij / denominator
        contribution = gamma_ij * wij * float(reliability[pos])
        scores[lab] = scores.get(lab, 0.0) + contribution

    return scores


def propose_natural_fuzzy_refinement(
    baseline_labels: np.ndarray,
    densities: np.ndarray,
    dist_mat: np.ndarray,
    centers: Sequence[int],
    nearest: np.ndarray,
    graph: NaturalNeighborGraph,
    config: RefinementConfig,
) -> Tuple[np.ndarray, RefinementDiagnostics]:
    """
    Đề xuất đổi nhãn theo một lượt duy nhất.

    Tất cả membership được tính từ baseline_labels, không dùng nhãn vừa sửa trong
    cùng lượt. Thiết kế này tránh tạo một chuỗi lan truyền mới tương tự domino
    effect. Các cluster centers được khóa cố định.
    """
    labels = np.asarray(baseline_labels, dtype=int)
    densities = np.asarray(densities, dtype=float)
    D = _validate_distance_matrix(dist_mat)
    corrected = labels.copy()
    diagnostics = RefinementDiagnostics()

    center_set = {int(c) for c in centers}
    representatives = _cluster_representatives(labels, densities)
    outlier_mask = natural_neighbor_outlier_mask(graph)

    for idx in np.argsort(densities):
        idx = int(idx)
        if idx in center_set:
            continue

        neighbors = np.asarray(graph.natural_sets[idx], dtype=int)
        neighbors = neighbors[(neighbors >= 0) & (neighbors < len(labels)) & (neighbors != idx)]
        if neighbors.size == 0:
            continue

        current_lab = int(labels[idx])
        neighbor_labs = np.unique(labels[neighbors])
        mixed_neighbors = neighbor_labs.size > 1
        parent_disagreement = bool(nearest[idx] >= 0 and labels[int(nearest[idx])] != current_lab)
        is_outlier = bool(outlier_mask[idx])

        if not (mixed_neighbors or parent_disagreement or is_outlier):
            continue

        diagnostics.considered += 1
        diagnostics.mixed_neighbor_candidates += int(mixed_neighbors)
        diagnostics.parent_disagreement_candidates += int(parent_disagreement)
        diagnostics.outlier_candidates += int(is_outlier)

        scores = _fuzzy_membership_scores(idx, labels, D, graph)
        if not scores:
            diagnostics.rejected_no_membership += 1
            continue

        best_lab, best_score = max(scores.items(), key=lambda item: item[1])
        best_lab = int(best_lab)
        best_score = float(best_score)
        current_score = float(scores.get(current_lab, 0.0))

        if best_lab == current_lab:
            continue
        diagnostics.proposed += 1

        if best_score <= float(config.switch_margin) * max(current_score, 1e-15):
            diagnostics.rejected_margin += 1
            continue

        if config.require_higher_density_support:
            valid_support = [
                int(nb) for nb in neighbors
                if labels[int(nb)] == best_lab and densities[int(nb)] > densities[idx]
            ]
            if not valid_support:
                diagnostics.rejected_higher_density += 1
                continue

        if config.require_representative_improvement:
            current_rep = representatives.get(current_lab)
            best_rep = representatives.get(best_lab)
            if current_rep is None or best_rep is None:
                diagnostics.rejected_distance += 1
                continue
            current_dist = float(D[idx, current_rep])
            best_dist = float(D[idx, best_rep])
            if best_dist >= current_dist:
                diagnostics.rejected_distance += 1
                continue

        corrected[idx] = best_lab
        diagnostics.changed += 1

    return corrected, diagnostics


# ============================================================
# Internal score and global safety gates
# ============================================================
def cluster_size_ratio(labels: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=int)
    _, counts = np.unique(labels, return_counts=True)
    if counts.size == 0:
        return 0.0
    return float(np.max(counts) / max(np.sum(counts), 1))


def internal_cluster_score(
    labels: np.ndarray,
    densities: np.ndarray,
    dist_mat: np.ndarray,
    largest_ratio_threshold: float = 0.85,
    min_ratio_threshold: float = 0.02,
    largest_penalty: float = 0.20,
    min_penalty: float = 0.50,
) -> float:
    """
    Lightweight internal score kế thừa từ AQG-GE-DPC.

    score = separation / compactness * imbalance_penalty
    Không sử dụng ground-truth labels.
    """
    labels = np.asarray(labels, dtype=int)
    densities = np.asarray(densities, dtype=float)
    D = _validate_distance_matrix(dist_mat)
    unique = sorted(np.unique(labels))
    if len(unique) <= 1:
        return -1e18

    representatives: List[int] = []
    compact_values: List[float] = []
    sizes: List[int] = []

    for lab in unique:
        members = np.where(labels == lab)[0]
        sizes.append(len(members))
        rep = int(members[np.argmax(densities[members])])
        representatives.append(rep)

        if len(members) <= 1:
            compact_values.append(0.0)
        else:
            block = D[np.ix_(members, members)]
            compact_values.append(float(np.mean(block)))

    compactness = float(np.mean(compact_values)) + 1e-12

    sep_values: List[float] = []
    for i in range(len(representatives)):
        for j in range(i + 1, len(representatives)):
            sep_values.append(float(D[representatives[i], representatives[j]]))
    separation = float(np.mean(sep_values)) if sep_values else 0.0

    sizes_arr = np.asarray(sizes, dtype=float)
    largest_ratio = float(np.max(sizes_arr) / np.sum(sizes_arr))
    min_ratio = float(np.min(sizes_arr) / np.sum(sizes_arr))

    penalty = 1.0
    if largest_ratio > float(largest_ratio_threshold):
        penalty *= float(largest_penalty)
    if min_ratio < float(min_ratio_threshold):
        penalty *= float(min_penalty)

    return float((separation / compactness) * penalty)


def _global_refinement_safety(
    baseline_labels: np.ndarray,
    refined_labels: np.ndarray,
    config: RefinementConfig,
) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    baseline_labels = np.asarray(baseline_labels, dtype=int)
    refined_labels = np.asarray(refined_labels, dtype=int)

    if len(np.unique(refined_labels)) != len(np.unique(baseline_labels)):
        reasons.append("cluster_count_changed")

    if cluster_size_ratio(refined_labels) > float(config.max_largest_ratio):
        reasons.append("largest_cluster_ratio_exceeded")

    changed_ratio = float(np.mean(baseline_labels != refined_labels))
    if changed_ratio > float(config.changed_ratio_limit):
        reasons.append("changed_ratio_exceeded")

    return len(reasons) == 0, reasons


def choose_labels_with_constrained_refinement(
    densities: np.ndarray,
    dist_mat: np.ndarray,
    centers: Sequence[int],
    nearest: np.ndarray,
    graph: NaturalNeighborGraph,
    config: RefinementConfig,
) -> Dict[str, Any]:
    """So sánh single-chain và natural-fuzzy refinement hoàn toàn bằng chỉ số nội tại."""
    single_labels = ellipse_cluster_single_chain(densities, centers, nearest, dist_mat)
    single_score = internal_cluster_score(single_labels, densities, dist_mat)
    single_consistency = natural_graph_consistency(single_labels, graph, dist_mat)

    refined_labels, diagnostics = propose_natural_fuzzy_refinement(
        baseline_labels=single_labels,
        densities=densities,
        dist_mat=dist_mat,
        centers=centers,
        nearest=nearest,
        graph=graph,
        config=config,
    )

    changed_ratio = float(np.mean(single_labels != refined_labels))
    refined_score = internal_cluster_score(refined_labels, densities, dist_mat)
    refined_consistency = natural_graph_consistency(refined_labels, graph, dist_mat)
    safe, safety_reasons = _global_refinement_safety(single_labels, refined_labels, config)

    score_ok = refined_score + config.internal_score_tolerance >= single_score
    consistency_ok = refined_consistency + config.consistency_tolerance >= single_consistency
    has_change = diagnostics.changed > 0
    accepted = bool(has_change and safe and score_ok and consistency_ok)

    if accepted:
        final_labels = refined_labels
        mode = "natural_fuzzy"
        final_score = refined_score
        final_consistency = refined_consistency
    else:
        final_labels = single_labels
        mode = "single"
        final_score = single_score
        final_consistency = single_consistency

    return {
        "mode": mode,
        "labels": final_labels,
        "single_labels": single_labels,
        "refined_labels": refined_labels,
        "single_score": float(single_score),
        "refined_score": float(refined_score),
        "final_score": float(final_score),
        "single_consistency": float(single_consistency),
        "refined_consistency": float(refined_consistency),
        "final_consistency": float(final_consistency),
        "changed_ratio": changed_ratio,
        "accepted": accepted,
        "score_ok": bool(score_ok),
        "consistency_ok": bool(consistency_ok),
        "safety_ok": bool(safe),
        "safety_reasons": safety_reasons,
        "diagnostics": diagnostics,
    }


def choose_best_configuration_by_internal_score(
    densities: np.ndarray,
    min_dists: np.ndarray,
    dist_mat: np.ndarray,
    nearest: np.ndarray,
    graph: NaturalNeighborGraph,
    k_candidates: Sequence[int],
    refinement_config: RefinementConfig,
) -> Dict[str, Any]:
    """
    Đánh giá nhẹ nhiều số tâm ứng viên và hai assignment modes.

    Ground truth không tham gia. Với mỗi k, refinement chỉ được giữ khi vượt qua
    toàn bộ acceptance gates. Cấu hình cuối được chọn theo internal score; natural
    graph consistency chỉ dùng làm tie-break.
    """
    n = len(densities)
    valid_k = sorted({int(k) for k in k_candidates if 1 <= int(k) <= n})
    if not valid_k:
        valid_k = [1]

    best: Optional[Dict[str, Any]] = None
    for k in valid_k:
        centers = auto_select_centers_quality(densities, min_dists, dist_mat, top_k=k)
        choice = choose_labels_with_constrained_refinement(
            densities=densities,
            dist_mat=dist_mat,
            centers=centers,
            nearest=nearest,
            graph=graph,
            config=refinement_config,
        )

        item: Dict[str, Any] = {
            "k": int(k),
            "centers": centers,
            **choice,
        }

        if best is None:
            best = item
            continue

        better_score = item["final_score"] > best["final_score"] + 1e-12
        tied_score = abs(item["final_score"] - best["final_score"]) <= 1e-12
        better_consistency = item["final_consistency"] > best["final_consistency"] + 1e-12
        if better_score or (tied_score and better_consistency):
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
# Các cấu hình scaler, số tâm k, outlier_t và split_factor được kế thừa nguyên
# trạng từ file AQG-GE-DPC đầu vào để cô lập tác động của label refinement mới.
# File này không dùng true labels để tune lại các giá trị bên dưới.
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

DEFAULT_REFINEMENT_CONFIG = RefinementConfig(
    switch_margin=1.50,
    max_largest_ratio=0.85,
    changed_ratio_limit=0.35,
    require_higher_density_support=True,
    require_representative_improvement=True,
)


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
# Run ANW-AQG-GE-DPC on one dataset
# ============================================================
def run_dataset(
    dataset_name: str,
    base_dir: Path,
    epsilon: float = 1e-6,
    show_chart: bool = False,
    verbose: bool = True,
    k_candidates: Optional[Sequence[int]] = None,
    refinement_config: Optional[RefinementConfig] = None,
) -> Dict[str, object]:
    key = dataset_name.lower()
    registry = get_default_dataset_registry(base_dir)

    if key not in registry:
        raise KeyError(f"Unknown dataset '{dataset_name}'. Available: {list(registry.keys())}")
    if key not in BASE_DATASET_CONFIGS:
        raise KeyError(f"Missing inherited KSE config for dataset '{dataset_name}'.")

    feature_file, label_file = registry[key]
    if not feature_file.exists():
        raise FileNotFoundError(f"Feature file not found: {feature_file}")

    cfg = BASE_DATASET_CONFIGS[key]
    refinement_config = refinement_config or DEFAULT_REFINEMENT_CONFIG

    # Chỉ đọc feature ở đầu pipeline. Ground truth được trì hoãn tới evaluation.
    data = np.loadtxt(feature_file, dtype=float)
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    scaler_mode = str(cfg["scaler"])
    outlier_t = float(cfg["outlier_t"])
    split_factor = float(cfg["split_factor"])
    inherited_k = int(cfg["k"])

    data = apply_data_scaler(data, scaler_mode=scaler_mode)
    num = max(5, int(np.ceil(np.sqrt(data.shape[0]) * split_factor)))
    log = print if verbose else (lambda *args, **kwargs: None)

    log("=" * 100)
    log("ANW-AQG-GE-DPC: constrained natural-weighted ellipsoid label refinement")
    log(f"Dataset        : {key}")
    log(f"Data shape     : n={data.shape[0]}, d={data.shape[1]}")
    log(f"Scaler         : {scaler_mode}")
    log(f"outlier_t      : {outlier_t}")
    log(f"split_factor   : {split_factor}")
    log(f"inherited k    : {inherited_k}")
    log(f"switch_margin  : {refinement_config.switch_margin}")
    log("=" * 100)

    # --------------------------------------------------------
    # 1) Kế thừa: quality-gated granular-ellipsoid generation
    # --------------------------------------------------------
    t_gen_start = time.perf_counter()
    ellipsoid_list: List[Ellipsoid] = [
        Ellipsoid(data, np.arange(data.shape[0]), epsilon=epsilon)
    ]

    while True:
        before = len(ellipsoid_list)
        ellipsoid_list = splits(ellipsoid_list, num=num, epsilon=epsilon)
        if len(ellipsoid_list) == before:
            break

    ellipsoid_list = recursive_split_outlier_detection(
        ellipsoid_list,
        data,
        t=outlier_t,
        max_iterations=10,
        epsilon=epsilon,
    )
    time_gen = time.perf_counter() - t_gen_start

    # --------------------------------------------------------
    # 2) Kế thừa: density, delta, gamma and Mahalanobis cache
    # --------------------------------------------------------
    t_attr_start = time.perf_counter()
    densities = np.array([ell.density for ell in ellipsoid_list], dtype=float)
    dist_mat = ellipse_distance(ellipsoid_list)
    min_dists, nearest = ellipse_min_dist(dist_mat, densities)
    gamma = densities * min_dists
    time_attr = time.perf_counter() - t_attr_start

    # --------------------------------------------------------
    # 3) Mới: adaptive natural-neighbor structure
    # --------------------------------------------------------
    t_graph_start = time.perf_counter()
    natural_graph = build_natural_ellipsoid_graph(dist_mat, use_stable_empty_stop=True)
    time_graph = time.perf_counter() - t_graph_start

    # --------------------------------------------------------
    # 4) Mới: constrained fuzzy label refinement + internal acceptance
    # --------------------------------------------------------
    t_cluster_start = time.perf_counter()
    if k_candidates is None:
        # Mặc định giữ đúng số tâm của bài KSE để đánh giá riêng đóng góp refinement.
        candidate_k = [inherited_k]
    else:
        candidate_k = list(k_candidates)

    best = choose_best_configuration_by_internal_score(
        densities=densities,
        min_dists=min_dists,
        dist_mat=dist_mat,
        nearest=nearest,
        graph=natural_graph,
        k_candidates=candidate_k,
        refinement_config=refinement_config,
    )
    time_cluster = time.perf_counter() - t_cluster_start

    ellipsoid_labels = np.asarray(best["labels"], dtype=int)
    centers = [int(x) for x in best["centers"]]

    # --------------------------------------------------------
    # 5) Map ellipsoid labels back to original points
    # --------------------------------------------------------
    pred_labels = np.full(len(data), -1, dtype=int)
    for ell_idx, ell in enumerate(ellipsoid_list):
        pred_labels[ell.indices] = ellipsoid_labels[ell_idx]

    if np.any(pred_labels < 0):
        raise RuntimeError("Some data points were not assigned a cluster label.")

    if show_chart:
        plot_predicted_clusters_2d(data, pred_labels, dataset_name=key, show_legend=False)

    # --------------------------------------------------------
    # 6) Final evaluation only: ground truth is first used here
    # --------------------------------------------------------
    if not label_file.exists():
        raise FileNotFoundError(f"Label file not found for final evaluation: {label_file}")
    true_labels = np.loadtxt(label_file, dtype=float).astype(int)
    if len(true_labels) != len(pred_labels):
        raise ValueError(f"Data and label length mismatch: {len(pred_labels)} vs {len(true_labels)}")

    aligned_pred = align_labels(true_labels, pred_labels)
    acc = accuracy_score(true_labels, aligned_pred)
    nmi = normalized_mutual_info_score(true_labels, pred_labels)
    ari = adjusted_rand_score(true_labels, pred_labels)

    total_time = time_gen + time_attr + time_graph + time_cluster
    diagnostics: RefinementDiagnostics = best["diagnostics"]

    log("-" * 100)
    log(f"Centers                  : {centers}")
    log(f"Selected gamma           : {[float(gamma[i]) for i in centers]}")
    log(f"Selected k               : {best['k']}")
    log(f"Label mode               : {best['mode']}")
    log(f"Ellipsoids               : {len(ellipsoid_list)}")
    log(f"Natural lambda           : {natural_graph.lambda_value}")
    log(f"Mean natural degree      : {natural_graph.mean_degree:.3f}")
    log(f"Fallback natural nodes   : {len(natural_graph.fallback_nodes)}")
    log(f"Considered/proposed      : {diagnostics.considered}/{diagnostics.proposed}")
    log(f"Accepted local changes   : {diagnostics.changed}")
    log(f"Global changed ratio     : {best['changed_ratio']:.3f}")
    log(f"Internal score S/R       : {best['single_score']:.6f}/{best['refined_score']:.6f}")
    log(f"Graph consistency S/R    : {best['single_consistency']:.6f}/{best['refined_consistency']:.6f}")
    log(f"Refinement accepted      : {best['accepted']}")
    if best["safety_reasons"]:
        log(f"Safety rejection reasons : {best['safety_reasons']}")
    log(f"ACC={acc:.3f}, NMI={nmi:.3f}, ARI={ari:.3f}, Time={total_time * 1000:.3f} ms")
    log("=" * 100)

    return {
        "dataset": key,
        "acc": float(acc),
        "nmi": float(nmi),
        "ari": float(ari),
        "scaler": scaler_mode,
        "outlier_t": outlier_t,
        "split_factor": split_factor,
        "selected_k": int(best["k"]),
        "mode": str(best["mode"]),
        "n_ellipsoids": len(ellipsoid_list),
        "natural_lambda": int(natural_graph.lambda_value),
        "mean_natural_degree": float(natural_graph.mean_degree),
        "fallback_natural_nodes": len(natural_graph.fallback_nodes),
        "changed_ellipsoids": int(diagnostics.changed),
        "changed_ratio": float(best["changed_ratio"]),
        "single_internal_score": float(best["single_score"]),
        "refined_internal_score": float(best["refined_score"]),
        "single_graph_consistency": float(best["single_consistency"]),
        "refined_graph_consistency": float(best["refined_consistency"]),
        "refinement_accepted": bool(best["accepted"]),
        "safety_reasons": list(best["safety_reasons"]),
        "time_generation": float(time_gen),
        "time_attributes": float(time_attr),
        "time_natural_graph": float(time_graph),
        "time_clustering": float(time_cluster),
        "total_time": float(total_time),
        "pred_labels": pred_labels,
        "aligned_pred_labels": aligned_pred,
        "ellipsoid_labels": ellipsoid_labels,
        "centers": centers,
    }


# ============================================================
# Run all configured datasets
# ============================================================
def run_all_datasets(
    base_dir: Path,
    dataset_names: Optional[Sequence[str]] = None,
    epsilon: float = 1e-6,
    show_chart: bool = False,
    verbose_each_dataset: bool = False,
    refinement_config: Optional[RefinementConfig] = None,
) -> Dict[str, Dict[str, object]]:
    if dataset_names is None:
        dataset_names = get_default_dataset_names()

    results: Dict[str, Dict[str, object]] = {}
    for name in dataset_names:
        try:
            results[name] = run_dataset(
                dataset_name=name,
                base_dir=base_dir,
                epsilon=epsilon,
                show_chart=show_chart,
                verbose=verbose_each_dataset,
                refinement_config=refinement_config,
            )
        except Exception as exc:
            results[name] = {"error": str(exc)}
            print(f"[ERROR] {name}: {exc}")

    print("\n" + "=" * 154)
    print("SUMMARY - ANW-AQG-GE-DPC")
    print("=" * 154)
    print(
        f"{'Dataset':<18} {'ACC':>7} {'NMI':>7} {'ARI':>7} "
        f"{'Scaler':>9} {'k':>4} {'Mode':>14} {'Ells':>6} "
        f"{'Lambda':>7} {'N-Deg':>7} {'Changed':>8} {'Accept':>8} {'Time(ms)':>11}"
    )
    print("-" * 154)

    for name, res in results.items():
        if "error" in res:
            print(f"{name:<18} ERROR: {res['error']}")
            continue
        print(
            f"{name:<18} {res['acc']:>7.3f} {res['nmi']:>7.3f} {res['ari']:>7.3f} "
            f"{res['scaler']:>9} {res['selected_k']:>4} {res['mode']:>14} "
            f"{res['n_ellipsoids']:>6} {res['natural_lambda']:>7} "
            f"{res['mean_natural_degree']:>7.2f} {res['changed_ellipsoids']:>8} "
            f"{str(res['refinement_accepted']):>8} {res['total_time'] * 1000:>11.3f}"
        )

    print("=" * 154)
    return results


# ============================================================
# Optional deterministic smoke test without external dataset files
# ============================================================
def run_smoke_test(random_state: int = 42) -> None:
    """Kiểm tra cú pháp và luồng thuật toán trên dữ liệu anisotropic tổng hợp nhỏ."""
    rng = np.random.default_rng(random_state)
    base1 = rng.normal(size=(90, 2)) @ np.array([[1.8, 0.7], [0.0, 0.25]]) + np.array([-2.0, 0.0])
    base2 = rng.normal(size=(90, 2)) @ np.array([[1.4, -0.6], [0.0, 0.30]]) + np.array([2.0, 0.5])
    data = np.vstack([base1, base2])

    ellipsoids: List[Ellipsoid] = [Ellipsoid(data, np.arange(len(data)))]
    threshold = max(5, int(np.ceil(np.sqrt(len(data)))))
    while True:
        before = len(ellipsoids)
        ellipsoids = splits(ellipsoids, num=threshold, epsilon=1e-6)
        if len(ellipsoids) == before:
            break

    densities = np.array([ell.density for ell in ellipsoids], dtype=float)
    distances = ellipse_distance(ellipsoids)
    deltas, nearest = ellipse_min_dist(distances, densities)
    graph = build_natural_ellipsoid_graph(distances)
    result = choose_best_configuration_by_internal_score(
        densities=densities,
        min_dists=deltas,
        dist_mat=distances,
        nearest=nearest,
        graph=graph,
        k_candidates=[2],
        refinement_config=DEFAULT_REFINEMENT_CONFIG,
    )
    assert len(result["labels"]) == len(ellipsoids)
    assert np.all(np.asarray(result["labels"]) >= 0)
    print(
        "Smoke test passed | "
        f"ellipsoids={len(ellipsoids)}, lambda={graph.lambda_value}, mode={result['mode']}"
    )


# ============================================================
# Program entry point
# ============================================================
if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent

    # Bật dòng dưới để kiểm tra nhanh mà không cần bộ dữ liệu ngoài.
    # run_smoke_test()

    # Chạy toàn bộ 12 dataset với cấu hình KSE được kế thừa và refinement mới.
    run_all_datasets(
        base_dir=BASE_DIR,
        epsilon=1e-6,
        show_chart=False,
        verbose_each_dataset=False,
    )
