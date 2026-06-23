
"""
CANR-AQG-GE-DPC
================
Constrained Adaptive Natural-Refinement AQG-GE-DPC

Mục tiêu:
1) Giữ nguyên toàn bộ pipeline AQG-GE-DPC gốc như một baseline candidate bắt buộc.
2) Chỉ sử dụng Natural Nearest Neighbor ở mức granular ellipsoid.
3) Không thay thế trực tiếp density gốc bằng một density mới trên toàn bộ pipeline.
4) Dùng natural-neighbor information để:
   - hỗ trợ xếp hạng center candidate;
   - phát hiện ellipsoid không chắc chắn;
   - fuzzy label refinement có kiểm soát.
5) Chỉ chấp nhận candidate mới khi chất lượng nội tại có trọng số theo số điểm
   cải thiện đủ rõ so với baseline.
6) Ground-truth labels chỉ được sử dụng ở bước đánh giá cuối cùng.

Flow:
Raw data
-> scaling
-> quality-gated granular ellipsoid generation
-> Mahalanobis distance matrix
-> AQG intrinsic density, delta, gamma
-> exact AQG baseline candidate
-> natural ellipsoid graph
-> natural-assisted center candidates
-> constrained fuzzy refinement candidates
-> point-weighted internal evaluation
-> retain baseline unless a new candidate is meaningfully better
-> map ellipsoid labels back to points
-> final ACC/NMI/ARI evaluation
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import accuracy_score, adjusted_rand_score, normalized_mutual_info_score


# ============================================================
# Configuration
# ============================================================
@dataclass(frozen=True)
class NaturalConfig:
    # Natural graph
    max_lambda: Optional[int] = None
    stable_rounds: int = 2
    max_degree: Optional[int] = None

    # Natural-assisted center scoring
    center_support_weights: Tuple[float, ...] = (0.05, 0.10, 0.15)
    center_prominence_weights: Tuple[float, ...] = (0.03, 0.05, 0.08)
    prominence_clip: float = 1.0

    # Fuzzy refinement
    min_best_probability: float = 0.62
    min_probability_margin: float = 0.15
    min_uncertainty_signals: int = 2
    lower_density_factor: float = 0.30
    nearest_cluster_members: int = 3
    cluster_distance_gain: float = 0.98
    min_neighbor_support: int = 2

    # Global safety
    changed_ellipsoid_ratio_limit: float = 0.15
    changed_point_ratio_limit: float = 0.08
    max_largest_point_ratio: float = 0.92

    # Candidate acceptance
    graph_weight: float = 0.25
    change_penalty: float = 0.75
    min_candidate_gain: float = 0.005


DEFAULT_NATURAL_CONFIG = NaturalConfig()


# ============================================================
# Ellipsoid with Cholesky + cache
# ============================================================
class Ellipsoid:
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

    @property
    def cov_matrix(self) -> np.ndarray:
        if self._cov_matrix is None:
            if self.n_samples <= 1:
                self._cov_matrix = np.zeros((self.dim, self.dim), dtype=float)
            else:
                self._cov_matrix = np.cov(self.data.T, bias=True)
        return self._cov_matrix

    @property
    def H_matrix(self) -> np.ndarray:
        if self._H_matrix is None:
            self._H_matrix = self.cov_matrix + self.epsilon * np.eye(self.dim, dtype=float)
        return self._H_matrix

    @property
    def chol_factor(self):
        if self._chol_factor is None:
            self._chol_factor = cho_factor(self.H_matrix, lower=True, check_finite=False)
        return self._chol_factor

    def solve_H(self, rhs: np.ndarray) -> np.ndarray:
        return cho_solve(self.chol_factor, rhs, check_finite=False)

    def mahal_sq_points(self, points: np.ndarray) -> np.ndarray:
        X = np.asarray(points, dtype=float)
        if X.ndim == 1:
            X = X[None, :]
        diffs = X - self.center
        solved = self.solve_H(diffs.T).T
        return np.maximum(np.einsum("ij,ij->i", diffs, solved), 0.0)

    @property
    def rho(self) -> float:
        if self._rho is None:
            self._rho = float(np.sqrt(np.max(self.mahal_sq_points(self.data))))
        return self._rho

    @property
    def lengths_rotation(self):
        if self._lengths_rotation is None:
            eigvals_H, eigvecs_H = np.linalg.eigh(self.H_matrix)
            eigvals_H = np.maximum(eigvals_H, 1e-12)
            lengths = self.rho * np.sqrt(eigvals_H)
            self._lengths_rotation = (lengths, eigvecs_H)
        return self._lengths_rotation

    @property
    def lengths(self) -> np.ndarray:
        return self.lengths_rotation[0]

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

    @property
    def density(self) -> float:
        """
        AQG intrinsic ellipsoid density.
        This density is preserved as the primary DPC density.
        """
        if self._density is None:
            axes_sum = max(float(np.sum(self.lengths)), 1e-12)
            mahal = np.sqrt(self.mahal_sq_points(self.data))
            total_mahal = max(float(np.sum(mahal)), 1e-12)
            self._density = float((self.n_samples ** 2) / (axes_sum * total_mahal))
        return self._density


# ============================================================
# Data scaling
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
# Granular ellipsoid generation
# ============================================================
def splits(ellipsoid_list: Sequence[Ellipsoid], num: int, epsilon: float) -> List[Ellipsoid]:
    new_ells: List[Ellipsoid] = []
    for ell in ellipsoid_list:
        if ell.n_samples < num:
            new_ells.append(ell)
        else:
            new_ells.extend(splits_ellipsoid(ell, epsilon=epsilon))
    return new_ells


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
# Ellipsoid distance and DPC attributes
# ============================================================
def ellipse_mahalanobis_distance(ell_i: Ellipsoid, ell_j: Ellipsoid) -> float:
    avg_H = 0.5 * (ell_i.H_matrix + ell_j.H_matrix)
    chol_avg = cho_factor(avg_H, lower=True, check_finite=False)
    diff = ell_i.center - ell_j.center
    solved = cho_solve(chol_avg, diff, check_finite=False)
    return float(np.sqrt(max(float(diff.T @ solved), 0.0)))


def ellipse_distance(ellipsoid_list: Sequence[Ellipsoid]) -> np.ndarray:
    n = len(ellipsoid_list)
    dist_mat = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            d = ellipse_mahalanobis_distance(ellipsoid_list[i], ellipsoid_list[j])
            dist_mat[i, j] = dist_mat[j, i] = d
    return dist_mat


def ellipse_min_dist(dist_mat: np.ndarray, densities: np.ndarray):
    densities = np.asarray(densities, dtype=float)
    n = len(densities)
    order = np.argsort(-densities, kind="mergesort")
    min_dists = np.zeros(n, dtype=float)
    nearest = -np.ones(n, dtype=int)

    for idx in order[1:]:
        higher = np.where(densities > densities[idx])[0]
        if higher.size == 0:
            # Stable tie fallback: earlier objects in descending order.
            earlier = order[np.where(order == idx)[0][0] - 1]
            higher = np.asarray([earlier], dtype=int)

        j = higher[np.argmin(dist_mat[idx, higher])]
        nearest[idx] = int(j)
        min_dists[idx] = float(dist_mat[idx, j])

    if n > 0:
        max_delta = float(np.max(min_dists))
        if max_delta <= 0:
            max_delta = float(np.max(dist_mat[order[0]]))
        min_dists[order[0]] = max_delta

    return min_dists, nearest


# ============================================================
# Exact AQG center selection and assignment
# ============================================================
def auto_select_centers_quality(
    densities: np.ndarray,
    min_dists: np.ndarray,
    dist_mat: np.ndarray,
    top_k: int,
    score_override: Optional[np.ndarray] = None,
) -> List[int]:
    n = len(densities)
    if n == 0:
        return []
    if n == 1:
        return [0]

    top_k = int(max(1, min(top_k, n)))
    gamma = densities * min_dists if score_override is None else np.asarray(score_override, dtype=float)
    order = np.argsort(-gamma, kind="mergesort")

    positive_dists = dist_mat[dist_mat > 0]
    if positive_dists.size == 0:
        return order[:top_k].tolist()

    median_dist = float(np.median(positive_dists))
    q25_dist = float(np.quantile(positive_dists, 0.25))
    min_sep = max(q25_dist, 0.20 * median_dist, 1e-12)

    selected: List[int] = []
    for idx in order:
        idx = int(idx)
        if not selected or float(np.min(dist_mat[idx, selected])) >= min_sep:
            selected.append(idx)
        if len(selected) >= top_k:
            break

    if len(selected) < top_k:
        for idx in order:
            idx = int(idx)
            if idx not in selected:
                selected.append(idx)
            if len(selected) >= top_k:
                break

    return selected[:top_k]


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

    order = np.argsort(-densities, kind="mergesort")
    for idx in order:
        if labels[idx] == -1 and nearest[idx] != -1:
            labels[idx] = labels[nearest[idx]]

    for idx in np.where(labels == -1)[0]:
        c = centers[int(np.argmin(dist_mat[idx, centers]))]
        labels[idx] = labels[c]

    return labels


def ellipse_cluster_conservative_graph_correction(
    densities: np.ndarray,
    dist_mat: np.ndarray,
    centers: Sequence[int],
    nearest: np.ndarray,
    switch_margin: float = 1.75,
    graph_k_factor: float = 1.0,
) -> np.ndarray:
    labels = ellipse_cluster_single_chain(densities, centers, nearest, dist_mat)
    corrected = labels.copy()
    n = len(densities)
    center_set = set(int(c) for c in centers)

    positive = dist_mat[dist_mat > 0]
    eps_dist = float(np.median(positive) * 1e-9) if positive.size else 1e-12
    k_neighbors = max(2, int(np.ceil(np.log2(max(n, 2))) * graph_k_factor))

    for idx in np.argsort(densities):
        if idx in center_set:
            continue
        higher = np.where(densities > densities[idx])[0]
        if higher.size == 0:
            continue

        higher_sorted = higher[np.argsort(dist_mat[idx, higher])]
        higher_sorted = higher_sorted[: min(k_neighbors, higher_sorted.size)]

        votes: Dict[int, float] = {}
        for nb in higher_sorted:
            lab = int(labels[nb])
            if lab < 0:
                continue
            weight = float(densities[nb] / (dist_mat[idx, nb] + eps_dist))
            votes[lab] = votes.get(lab, 0.0) + weight

        if not votes:
            continue

        current_lab = int(labels[idx])
        current_vote = votes.get(current_lab, 0.0)
        best_lab, best_vote = max(votes.items(), key=lambda kv: kv[1])

        if best_lab != current_lab and best_vote > float(switch_margin) * max(current_vote, 1e-12):
            corrected[idx] = int(best_lab)

    return corrected


def cluster_size_ratio(labels: np.ndarray) -> float:
    _, counts = np.unique(labels, return_counts=True)
    return float(np.max(counts) / max(np.sum(counts), 1))


def is_graph_safe(
    single_labels: np.ndarray,
    graph_labels: np.ndarray,
    max_largest_ratio: float = 0.85,
    changed_ratio_limit: float = 0.35,
) -> bool:
    if cluster_size_ratio(graph_labels) > float(max_largest_ratio):
        return False
    if float(np.mean(single_labels != graph_labels)) > float(changed_ratio_limit):
        return False
    return True


# ============================================================
# Natural nearest-neighbor graph on ellipsoids
# ============================================================
def build_natural_ellipsoid_graph(
    dist_mat: np.ndarray,
    config: NaturalConfig,
) -> Dict[str, object]:
    """
    Build a mutual natural-neighbor graph on ellipsoids.

    The search increases lambda until the number of ellipsoids with no
    mutual neighbor remains unchanged for `stable_rounds`. The resulting
    graph may be pruned by a degree cap to preserve locality.
    """
    n = dist_mat.shape[0]
    if n <= 1:
        return {
            "neighbors": [np.array([], dtype=int) for _ in range(n)],
            "lambda": 0,
            "mean_degree": 0.0,
            "support": np.ones(n, dtype=float),
            "ranks": np.zeros((n, n), dtype=int),
        }

    sortable = np.asarray(dist_mat, dtype=float).copy()
    np.fill_diagonal(sortable, np.inf)
    order = np.argsort(sortable, axis=1)
    ranks = np.empty((n, n), dtype=int)
    for i in range(n):
        ranks[i, order[i]] = np.arange(n)

    max_lambda = config.max_lambda
    if max_lambda is None:
        max_lambda = min(n - 1, max(4, int(np.ceil(np.sqrt(n))) + 4))
    max_lambda = int(max(1, min(max_lambda, n - 1)))

    previous_empty = None
    stable_count = 0
    chosen_lambda = max_lambda
    chosen_mask = None

    for lam in range(1, max_lambda + 1):
        knn_mask = ranks <= lam
        np.fill_diagonal(knn_mask, False)
        mutual = knn_mask & knn_mask.T
        empty_count = int(np.sum(np.sum(mutual, axis=1) == 0))

        if previous_empty is not None and empty_count == previous_empty:
            stable_count += 1
        else:
            stable_count = 0
        previous_empty = empty_count

        chosen_mask = mutual
        chosen_lambda = lam

        if empty_count == 0 and stable_count >= config.stable_rounds:
            break

    neighbors: List[np.ndarray] = []
    default_cap = min(12, max(4, int(np.ceil(np.log2(max(n, 2)))) + 2))
    degree_cap = default_cap if config.max_degree is None else int(config.max_degree)
    degree_cap = max(1, min(degree_cap, n - 1))

    for i in range(n):
        nb = np.where(chosen_mask[i])[0]
        if nb.size > degree_cap:
            # Keep strongest mutual-rank neighbors.
            mutual_rank = ranks[i, nb] + ranks[nb, i]
            keep_order = np.lexsort((dist_mat[i, nb], mutual_rank))
            nb = nb[keep_order[:degree_cap]]
        neighbors.append(np.asarray(nb, dtype=int))

    occurrence = np.zeros(n, dtype=float)
    for nb in neighbors:
        occurrence[nb] += 1.0

    positive_occ = occurrence[occurrence > 0]
    scale = float(np.median(positive_occ)) if positive_occ.size else 1.0
    support = occurrence / max(scale, 1e-12)
    support = np.clip(support, 0.0, 3.0)

    mean_degree = float(np.mean([len(nb) for nb in neighbors])) if neighbors else 0.0

    return {
        "neighbors": neighbors,
        "lambda": int(chosen_lambda),
        "mean_degree": mean_degree,
        "support": support,
        "ranks": ranks,
    }


def compute_local_prominence(
    densities: np.ndarray,
    neighbors: Sequence[np.ndarray],
    clip_value: float = 1.0,
) -> np.ndarray:
    """
    Relative density prominence, centered around zero and clipped.
    It does not replace the AQG density.
    """
    densities = np.asarray(densities, dtype=float)
    prominence = np.zeros_like(densities)

    for i, nb in enumerate(neighbors):
        if len(nb) == 0:
            prominence[i] = 0.0
            continue
        local_median = float(np.median(densities[nb]))
        ratio = (densities[i] + 1e-12) / (local_median + 1e-12)
        prominence[i] = np.clip(np.log(ratio), -clip_value, clip_value)

    return prominence


def natural_assisted_gamma(
    base_gamma: np.ndarray,
    support: np.ndarray,
    prominence: np.ndarray,
    support_weight: float,
    prominence_weight: float,
) -> np.ndarray:
    """
    Natural information only perturbs the original gamma mildly.
    Density ordering and nearest-higher-density chain are preserved.
    """
    support_centered = support - float(np.median(support))
    support_scale = float(np.median(np.abs(support_centered))) + 1e-12
    support_z = np.clip(support_centered / support_scale, -2.0, 2.0)

    adjustment = np.exp(
        float(support_weight) * support_z
        + float(prominence_weight) * prominence
    )
    return np.asarray(base_gamma, dtype=float) * adjustment


# ============================================================
# Point-weighted internal evaluation
# ============================================================
def cluster_point_ratio(labels: np.ndarray, masses: np.ndarray) -> float:
    totals = []
    for lab in np.unique(labels):
        totals.append(float(np.sum(masses[labels == lab])))
    return max(totals) / max(float(np.sum(masses)), 1e-12)


def point_weighted_internal_score(
    labels: np.ndarray,
    densities: np.ndarray,
    dist_mat: np.ndarray,
    masses: np.ndarray,
    largest_ratio_threshold: float = 0.92,
    min_ratio_threshold: float = 0.01,
) -> float:
    """
    Separation / compactness using point mass of each ellipsoid.
    """
    labels = np.asarray(labels, dtype=int)
    masses = np.asarray(masses, dtype=float)
    unique = sorted(np.unique(labels))
    if len(unique) <= 1:
        return -1e18

    reps: List[int] = []
    compact_values: List[float] = []
    cluster_masses: List[float] = []

    for lab in unique:
        members = np.where(labels == lab)[0]
        member_masses = masses[members]
        total_mass = float(np.sum(member_masses))
        cluster_masses.append(total_mass)

        # Density peak within the cluster as representative.
        rep = int(members[np.argmax(densities[members])])
        reps.append(rep)

        if len(members) <= 1:
            compact_values.append(0.0)
        else:
            compact_values.append(
                float(np.average(dist_mat[rep, members], weights=member_masses))
            )

    total_mass = max(float(np.sum(cluster_masses)), 1e-12)
    compactness = float(
        np.average(compact_values, weights=np.asarray(cluster_masses))
    ) + 1e-12

    sep_values: List[float] = []
    sep_weights: List[float] = []
    for i in range(len(reps)):
        for j in range(i + 1, len(reps)):
            sep_values.append(float(dist_mat[reps[i], reps[j]]))
            sep_weights.append(float(cluster_masses[i] * cluster_masses[j]))

    separation = (
        float(np.average(sep_values, weights=sep_weights))
        if sep_values else 0.0
    )

    ratios = np.asarray(cluster_masses, dtype=float) / total_mass
    penalty = 1.0
    if float(np.max(ratios)) > float(largest_ratio_threshold):
        penalty *= 0.20
    if float(np.min(ratios)) < float(min_ratio_threshold):
        penalty *= 0.50

    return float((separation / compactness) * penalty)


def weighted_graph_consistency(
    labels: np.ndarray,
    neighbors: Sequence[np.ndarray],
    dist_mat: np.ndarray,
    masses: np.ndarray,
) -> float:
    same_weight = 0.0
    total_weight = 0.0

    for i, nb in enumerate(neighbors):
        for j in nb:
            if j <= i:
                continue
            edge_weight = (
                np.sqrt(float(masses[i] * masses[j]))
                / (1.0 + float(dist_mat[i, j]))
            )
            total_weight += edge_weight
            if labels[i] == labels[j]:
                same_weight += edge_weight

    if total_weight <= 0:
        return 0.0
    return float(same_weight / total_weight)


def changed_point_ratio(
    baseline_labels: np.ndarray,
    candidate_labels: np.ndarray,
    masses: np.ndarray,
) -> float:
    changed = baseline_labels != candidate_labels
    return float(np.sum(masses[changed]) / max(np.sum(masses), 1e-12))


# ============================================================
# Constrained natural fuzzy refinement
# ============================================================
def _distance_to_cluster(
    idx: int,
    cluster_members: np.ndarray,
    dist_mat: np.ndarray,
    r: int,
) -> float:
    if cluster_members.size == 0:
        return np.inf
    values = np.sort(dist_mat[idx, cluster_members])
    take = min(max(1, int(r)), values.size)
    return float(np.mean(values[:take]))


def natural_fuzzy_refinement(
    baseline_labels: np.ndarray,
    centers: Sequence[int],
    densities: np.ndarray,
    nearest: np.ndarray,
    dist_mat: np.ndarray,
    natural_graph: Dict[str, object],
    masses: np.ndarray,
    config: NaturalConfig,
) -> Dict[str, object]:
    """
    One-shot refinement.
    Every proposal is computed from the same baseline labels to avoid
    correction-induced domino effects.
    """
    labels = np.asarray(baseline_labels, dtype=int)
    corrected = labels.copy()
    neighbors: Sequence[np.ndarray] = natural_graph["neighbors"]
    support = np.asarray(natural_graph["support"], dtype=float)
    ranks = np.asarray(natural_graph["ranks"], dtype=int)
    center_set = set(int(c) for c in centers)

    density_scale = np.log1p(np.maximum(densities, 0.0))
    if np.max(density_scale) > np.min(density_scale):
        density_scale = (density_scale - np.min(density_scale)) / (
            np.max(density_scale) - np.min(density_scale)
        )
    else:
        density_scale = np.ones_like(density_scale)

    proposals: Dict[int, int] = {}
    proposal_confidence: Dict[int, float] = {}

    for idx in np.argsort(densities):
        if idx in center_set:
            continue

        nb = np.asarray(neighbors[idx], dtype=int)
        if nb.size < config.min_neighbor_support:
            continue

        nb_labels = labels[nb]
        unique_nb_labels = np.unique(nb_labels)
        current_lab = int(labels[idx])
        parent_lab = int(labels[nearest[idx]]) if nearest[idx] >= 0 else current_lab

        # Uncertainty signals.
        mixed_neighbors = len(unique_nb_labels) > 1
        parent_disagreement = parent_lab != current_lab

        current_members = np.where(labels == current_lab)[0]
        current_members = current_members[current_members != idx]
        current_dist = _distance_to_cluster(
            idx, current_members, dist_mat, config.nearest_cluster_members
        )

        # Local label distribution.
        raw_scores: Dict[int, float] = {}
        same_label_count = 0

        for nb_idx in nb:
            lab = int(labels[nb_idx])
            if lab == current_lab:
                same_label_count += 1

            distance_similarity = 1.0 / (1.0 + float(dist_mat[idx, nb_idx]))
            mutual_rank = int(ranks[idx, nb_idx] + ranks[nb_idx, idx])
            rank_similarity = 1.0 / (1.0 + mutual_rank)

            higher_factor = (
                1.0 if densities[nb_idx] > densities[idx]
                else float(config.lower_density_factor)
            )
            reliability = 0.5 + 0.5 * min(float(support[nb_idx]), 2.0) / 2.0
            mass_factor = np.sqrt(np.log1p(float(masses[nb_idx])))
            density_factor = 0.5 + 0.5 * float(density_scale[nb_idx])

            contribution = (
                distance_similarity
                * (0.5 + 0.5 * rank_similarity)
                * reliability
                * mass_factor
                * density_factor
                * higher_factor
            )
            raw_scores[lab] = raw_scores.get(lab, 0.0) + contribution

        total_score = float(sum(raw_scores.values()))
        if total_score <= 0:
            continue

        probabilities = {
            lab: score / total_score for lab, score in raw_scores.items()
        }
        sorted_probs = sorted(probabilities.items(), key=lambda kv: kv[1], reverse=True)
        best_lab, best_prob = sorted_probs[0]
        second_prob = sorted_probs[1][1] if len(sorted_probs) > 1 else 0.0
        margin = float(best_prob - second_prob)

        low_current_support = (
            same_label_count / max(len(nb), 1)
            < config.min_best_probability
        )

        best_members = np.where(labels == int(best_lab))[0]
        best_dist = _distance_to_cluster(
            idx, best_members, dist_mat, config.nearest_cluster_members
        )
        geometric_ambiguity = (
            np.isfinite(current_dist)
            and best_lab != current_lab
            and best_dist < current_dist
        )

        signals = (
            int(mixed_neighbors)
            + int(parent_disagreement)
            + int(low_current_support)
            + int(geometric_ambiguity)
        )
        if signals < int(config.min_uncertainty_signals):
            continue

        if int(best_lab) == current_lab:
            continue
        if best_prob < float(config.min_best_probability):
            continue
        if margin < float(config.min_probability_margin):
            continue

        # The target cluster must be geometrically closer by a minimum margin.
        if not np.isfinite(best_dist):
            continue
        if np.isfinite(current_dist):
            if best_dist > float(config.cluster_distance_gain) * current_dist:
                continue

        proposals[int(idx)] = int(best_lab)
        proposal_confidence[int(idx)] = margin

    for idx, lab in proposals.items():
        corrected[idx] = lab

    changed_mask = corrected != labels
    changed_ell_ratio = float(np.mean(changed_mask))
    changed_pt_ratio = float(
        np.sum(masses[changed_mask]) / max(np.sum(masses), 1e-12)
    )

    safe = True
    if changed_ell_ratio > float(config.changed_ellipsoid_ratio_limit):
        safe = False
    if changed_pt_ratio > float(config.changed_point_ratio_limit):
        safe = False
    if cluster_point_ratio(corrected, masses) > float(config.max_largest_point_ratio):
        safe = False
    if len(np.unique(corrected)) != len(np.unique(labels)):
        safe = False

    return {
        "labels": corrected if safe else labels.copy(),
        "safe": bool(safe),
        "proposed_changes": int(np.sum(changed_mask)),
        "changed_ellipsoid_ratio": changed_ell_ratio,
        "changed_point_ratio": changed_pt_ratio,
        "mean_margin": (
            float(np.mean(list(proposal_confidence.values())))
            if proposal_confidence else 0.0
        ),
    }


# ============================================================
# Candidate creation and selection
# ============================================================
def make_exact_aqg_baseline(
    densities: np.ndarray,
    min_dists: np.ndarray,
    dist_mat: np.ndarray,
    nearest: np.ndarray,
    k: int,
    allow_graph: bool,
    graph_switch_margin: float,
) -> Dict[str, object]:
    centers = auto_select_centers_quality(
        densities, min_dists, dist_mat, top_k=k
    )
    single_labels = ellipse_cluster_single_chain(
        densities, centers, nearest, dist_mat
    )

    labels = single_labels
    mode = "baseline_single"

    if allow_graph:
        graph_labels = ellipse_cluster_conservative_graph_correction(
            densities,
            dist_mat,
            centers,
            nearest,
            switch_margin=graph_switch_margin,
        )
        if is_graph_safe(single_labels, graph_labels):
            labels = graph_labels
            mode = "baseline_graph"

    return {
        "mode": mode,
        "centers": centers,
        "labels": labels,
        "center_score": densities * min_dists,
    }


def evaluate_candidate(
    candidate: Dict[str, object],
    baseline_labels: np.ndarray,
    densities: np.ndarray,
    dist_mat: np.ndarray,
    masses: np.ndarray,
    natural_graph: Dict[str, object],
    baseline_internal: float,
    baseline_consistency: float,
    config: NaturalConfig,
) -> Dict[str, object]:
    labels = np.asarray(candidate["labels"], dtype=int)

    internal = point_weighted_internal_score(
        labels, densities, dist_mat, masses
    )
    consistency = weighted_graph_consistency(
        labels,
        natural_graph["neighbors"],
        dist_mat,
        masses,
    )
    p_change = changed_point_ratio(
        baseline_labels, labels, masses
    )

    # Relative quality against exact AQG baseline.
    if baseline_internal <= 0 or internal <= 0:
        geometry_gain = -1e9
    else:
        geometry_gain = float(np.log(internal / baseline_internal))

    quality_gain = (
        geometry_gain
        + float(config.graph_weight) * (consistency - baseline_consistency)
        - float(config.change_penalty) * p_change
    )

    enriched = dict(candidate)
    enriched.update({
        "internal_score": float(internal),
        "graph_consistency": float(consistency),
        "point_change_ratio": float(p_change),
        "quality_gain": float(quality_gain),
    })
    return enriched


def select_best_candidate(
    baseline: Dict[str, object],
    alternatives: Sequence[Dict[str, object]],
    densities: np.ndarray,
    dist_mat: np.ndarray,
    masses: np.ndarray,
    natural_graph: Dict[str, object],
    config: NaturalConfig,
) -> Dict[str, object]:
    baseline_labels = np.asarray(baseline["labels"], dtype=int)
    baseline_internal = point_weighted_internal_score(
        baseline_labels, densities, dist_mat, masses
    )
    baseline_consistency = weighted_graph_consistency(
        baseline_labels,
        natural_graph["neighbors"],
        dist_mat,
        masses,
    )

    baseline_eval = dict(baseline)
    baseline_eval.update({
        "internal_score": float(baseline_internal),
        "graph_consistency": float(baseline_consistency),
        "point_change_ratio": 0.0,
        "quality_gain": 0.0,
        "accepted": False,
    })

    best = baseline_eval
    for candidate in alternatives:
        evaluated = evaluate_candidate(
            candidate,
            baseline_labels,
            densities,
            dist_mat,
            masses,
            natural_graph,
            baseline_internal,
            baseline_consistency,
            config,
        )
        if (
            evaluated["quality_gain"] >= float(config.min_candidate_gain)
            and evaluated["quality_gain"] > best["quality_gain"]
        ):
            best = evaluated

    best["accepted"] = best["mode"] != baseline["mode"]
    return best


def build_and_select_candidates(
    densities: np.ndarray,
    min_dists: np.ndarray,
    dist_mat: np.ndarray,
    nearest: np.ndarray,
    masses: np.ndarray,
    k: int,
    allow_graph: bool,
    graph_switch_margin: float,
    natural_graph: Dict[str, object],
    config: NaturalConfig,
) -> Dict[str, object]:
    base_gamma = densities * min_dists
    baseline = make_exact_aqg_baseline(
        densities,
        min_dists,
        dist_mat,
        nearest,
        k,
        allow_graph,
        graph_switch_margin,
    )

    alternatives: List[Dict[str, object]] = []
    prominence = compute_local_prominence(
        densities,
        natural_graph["neighbors"],
        clip_value=config.prominence_clip,
    )
    support = np.asarray(natural_graph["support"], dtype=float)

    # Candidate 1: fuzzy refinement over exact baseline.
    baseline_refinement = natural_fuzzy_refinement(
        baseline["labels"],
        baseline["centers"],
        densities,
        nearest,
        dist_mat,
        natural_graph,
        masses,
        config,
    )
    if baseline_refinement["safe"] and baseline_refinement["proposed_changes"] > 0:
        alternatives.append({
            "mode": "baseline_natural_fuzzy",
            "centers": baseline["centers"],
            "labels": baseline_refinement["labels"],
            "center_score": baseline["center_score"],
            "proposed_changes": baseline_refinement["proposed_changes"],
            "changed_point_ratio_local": baseline_refinement["changed_point_ratio"],
        })

    # Candidates 2+: natural-assisted center ranking with original density chain.
    for support_weight in config.center_support_weights:
        for prominence_weight in config.center_prominence_weights:
            adjusted_gamma = natural_assisted_gamma(
                base_gamma,
                support,
                prominence,
                support_weight=support_weight,
                prominence_weight=prominence_weight,
            )
            centers = auto_select_centers_quality(
                densities,
                min_dists,
                dist_mat,
                top_k=k,
                score_override=adjusted_gamma,
            )
            labels = ellipse_cluster_single_chain(
                densities, centers, nearest, dist_mat
            )

            mode = (
                f"natural_center_s{support_weight:.2f}"
                f"_p{prominence_weight:.2f}"
            )
            alternatives.append({
                "mode": mode,
                "centers": centers,
                "labels": labels,
                "center_score": adjusted_gamma,
                "proposed_changes": 0,
                "changed_point_ratio_local": 0.0,
            })

            refinement = natural_fuzzy_refinement(
                labels,
                centers,
                densities,
                nearest,
                dist_mat,
                natural_graph,
                masses,
                config,
            )
            if refinement["safe"] and refinement["proposed_changes"] > 0:
                alternatives.append({
                    "mode": mode + "_fuzzy",
                    "centers": centers,
                    "labels": refinement["labels"],
                    "center_score": adjusted_gamma,
                    "proposed_changes": refinement["proposed_changes"],
                    "changed_point_ratio_local": refinement["changed_point_ratio"],
                })

    # Remove exact duplicate labelings.
    unique_alternatives: List[Dict[str, object]] = []
    seen = {np.asarray(baseline["labels"], dtype=int).tobytes()}
    for candidate in alternatives:
        key = np.asarray(candidate["labels"], dtype=int).tobytes()
        if key not in seen:
            seen.add(key)
            unique_alternatives.append(candidate)

    best = select_best_candidate(
        baseline,
        unique_alternatives,
        densities,
        dist_mat,
        masses,
        natural_graph,
        config,
    )
    best["baseline_mode"] = baseline["mode"]
    best["candidate_count"] = 1 + len(unique_alternatives)
    return best


# ============================================================
# Evaluation and plotting
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


def plot_predicted_clusters_2d(
    data: np.ndarray,
    pred_labels: np.ndarray,
    dataset_name: str = "dataset",
    show_legend: bool = False,
) -> None:
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

    plt.figure(figsize=(6.4, 4.8))
    for lab in np.unique(y):
        mask = y == lab
        plt.scatter(X_vis[mask, 0], X_vis[mask, 1], s=16, alpha=0.95)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(dataset_name)
    if show_legend:
        plt.legend([f"Cluster {lab}" for lab in np.unique(y)], loc="best")
    plt.tight_layout()
    plt.show()


# ============================================================
# Dataset configuration
# ============================================================
BEST_DATASET_CONFIGS: Dict[str, Dict[str, object]] = {
    "iris": {
        "scaler": "none", "k": 3, "allow_graph": False,
        "outlier_t": 1.3, "split_factor": 1.0, "graph_switch_margin": 1.75,
    },
    "seed": {
        "scaler": "none", "k": 3, "allow_graph": False,
        "outlier_t": 1.3, "split_factor": 1.0, "graph_switch_margin": 1.75,
    },
    "segment_3": {
        "scaler": "robust", "k": 8, "allow_graph": False,
        "outlier_t": 1.7, "split_factor": 0.8, "graph_switch_margin": 1.75,
    },
    "landsat_2": {
        "scaler": "none", "k": 5, "allow_graph": True,
        "outlier_t": 2.0, "split_factor": 1.0, "graph_switch_margin": 1.50,
    },
    "msplice_2": {
        "scaler": "standard", "k": 3, "allow_graph": True,
        "outlier_t": 2.0, "split_factor": 0.6, "graph_switch_margin": 1.30,
    },
    "rice": {
        "scaler": "standard", "k": 2, "allow_graph": True,
        "outlier_t": 1.5, "split_factor": 0.6, "graph_switch_margin": 1.30,
    },
    "banknote": {
        "scaler": "standard", "k": 2, "allow_graph": False,
        "outlier_t": 1.5, "split_factor": 0.8, "graph_switch_margin": 1.75,
    },
    "htru2": {
        "scaler": "none", "k": 2, "allow_graph": False,
        "outlier_t": 2.0, "split_factor": 1.0, "graph_switch_margin": 1.75,
    },
    "breast_cancer": {
        "scaler": "none", "k": 2, "allow_graph": False,
        "outlier_t": 1.5, "split_factor": 0.6, "graph_switch_margin": 1.75,
    },
    "hcv_data": {
        "scaler": "robust", "k": 2, "allow_graph": False,
        "outlier_t": 2.0, "split_factor": 1.0, "graph_switch_margin": 1.75,
    },
    "dry_bean": {
        "scaler": "standard", "k": 7, "allow_graph": False,
        "outlier_t": 1.3, "split_factor": 0.6, "graph_switch_margin": 1.75,
    },
    "rice_cammeo": {
        "scaler": "standard", "k": 2, "allow_graph": True,
        "outlier_t": 1.3, "split_factor": 0.6, "graph_switch_margin": 1.30,
    },
}


def get_default_dataset_names() -> List[str]:
    return [
        "iris", "seed", "segment_3", "landsat_2",
        "msplice_2", "rice", "banknote", "htru2",
        "breast_cancer", "hcv_data", "dry_bean", "rice_cammeo",
    ]


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
# Run one dataset
# ============================================================
def run_dataset(
    dataset_name: str,
    base_dir: Path,
    epsilon: float = 1e-6,
    natural_config: NaturalConfig = DEFAULT_NATURAL_CONFIG,
    show_chart: bool = False,
    verbose: bool = True,
) -> Dict[str, object]:
    key = dataset_name.lower()
    registry = get_default_dataset_registry(base_dir)

    if key not in registry:
        raise KeyError(f"Unknown dataset '{dataset_name}'.")
    if key not in BEST_DATASET_CONFIGS:
        raise KeyError(f"Missing config for '{dataset_name}'.")

    feature_file, label_file = registry[key]
    if not feature_file.exists() or not label_file.exists():
        raise FileNotFoundError(
            f"Dataset files not found for '{dataset_name}'.\n"
            f"Feature: {feature_file}\nLabel: {label_file}"
        )

    cfg = BEST_DATASET_CONFIGS[key]
    data = np.loadtxt(feature_file, dtype=float)
    true_labels = np.loadtxt(label_file, dtype=float).astype(int)

    if data.ndim == 1:
        data = data.reshape(-1, 1)
    if len(data) != len(true_labels):
        raise ValueError("Data and label length mismatch.")

    scaler_mode = str(cfg["scaler"])
    outlier_t = float(cfg["outlier_t"])
    split_factor = float(cfg["split_factor"])
    k = int(cfg["k"])
    allow_graph = bool(cfg["allow_graph"])
    graph_switch_margin = float(cfg.get("graph_switch_margin", 1.75))

    data = apply_data_scaler(data, scaler_mode)
    num = max(5, int(np.ceil(np.sqrt(data.shape[0]) * split_factor)))
    log = print if verbose else (lambda *args, **kwargs: None)

    # 1) Granular ellipsoid generation
    t_gen = time.time()
    ellipsoids: List[Ellipsoid] = [
        Ellipsoid(data, np.arange(data.shape[0]), epsilon=epsilon)
    ]
    while True:
        before = len(ellipsoids)
        ellipsoids = splits(ellipsoids, num=num, epsilon=epsilon)
        if len(ellipsoids) == before:
            break

    ellipsoids = recursive_split_outlier_detection(
        ellipsoids,
        data,
        t=outlier_t,
        max_iterations=10,
        epsilon=epsilon,
    )
    time_gen = time.time() - t_gen

    # 2) Exact AQG attributes
    t_attr = time.time()
    densities = np.asarray([ell.density for ell in ellipsoids], dtype=float)
    masses = np.asarray([ell.n_samples for ell in ellipsoids], dtype=float)
    dist_mat = ellipse_distance(ellipsoids)
    min_dists, nearest = ellipse_min_dist(dist_mat, densities)
    natural_graph = build_natural_ellipsoid_graph(
        dist_mat, natural_config
    )
    time_attr = time.time() - t_attr

    # 3) Candidate generation and unsupervised selection
    t_cluster = time.time()
    best = build_and_select_candidates(
        densities,
        min_dists,
        dist_mat,
        nearest,
        masses,
        k,
        allow_graph,
        graph_switch_margin,
        natural_graph,
        natural_config,
    )
    ellipsoid_labels = np.asarray(best["labels"], dtype=int)
    time_cluster = time.time() - t_cluster

    # 4) Map ellipsoid labels back to points
    pred_labels = np.full(len(data), -1, dtype=int)
    for i, ell in enumerate(ellipsoids):
        pred_labels[ell.indices] = ellipsoid_labels[i]

    if np.any(pred_labels < 0):
        raise RuntimeError("Some data points were not assigned.")

    if show_chart:
        plot_predicted_clusters_2d(
            data, pred_labels, dataset_name=key, show_legend=False
        )

    # 5) Final evaluation only
    aligned_pred = align_labels(true_labels, pred_labels)
    acc = accuracy_score(true_labels, aligned_pred)
    nmi = normalized_mutual_info_score(true_labels, pred_labels)
    ari = adjusted_rand_score(true_labels, pred_labels)
    total_time = time_gen + time_attr + time_cluster

    baseline_mode = str(best.get("baseline_mode", "baseline_single"))
    accepted = bool(best.get("accepted", False))
    changed_ells = int(np.sum(
        ellipsoid_labels != make_exact_aqg_baseline(
            densities, min_dists, dist_mat, nearest, k,
            allow_graph, graph_switch_margin
        )["labels"]
    ))

    log("=" * 100)
    log(f"Dataset          : {key}")
    log(f"Ellipsoids       : {len(ellipsoids)}")
    log(f"Natural lambda   : {natural_graph['lambda']}")
    log(f"Natural mean deg : {natural_graph['mean_degree']:.3f}")
    log(f"Baseline mode    : {baseline_mode}")
    log(f"Selected mode    : {best['mode']}")
    log(f"Candidate count  : {best['candidate_count']}")
    log(f"Accepted         : {accepted}")
    log(f"Quality gain     : {best['quality_gain']:.6f}")
    log(f"Changed ellips.  : {changed_ells}")
    log(f"Point change     : {best['point_change_ratio']:.6f}")
    log(f"ACC={acc:.3f}, NMI={nmi:.3f}, ARI={ari:.3f}")
    log("=" * 100)

    return {
        "dataset": key,
        "acc": float(acc),
        "nmi": float(nmi),
        "ari": float(ari),
        "scaler": scaler_mode,
        "best_k": k,
        "mode": str(best["mode"]),
        "baseline_mode": baseline_mode,
        "accepted": accepted,
        "quality_gain": float(best["quality_gain"]),
        "candidate_count": int(best["candidate_count"]),
        "n_ellipsoids": len(ellipsoids),
        "natural_lambda": int(natural_graph["lambda"]),
        "natural_mean_degree": float(natural_graph["mean_degree"]),
        "changed_ellipsoids": changed_ells,
        "changed_point_ratio": float(best["point_change_ratio"]),
        "total_time": total_time,
        "pred_labels": pred_labels,
        "aligned_pred_labels": aligned_pred,
    }


# ============================================================
# Run all datasets
# ============================================================
def run_all_datasets(
    base_dir: Path,
    dataset_names: Optional[Sequence[str]] = None,
    epsilon: float = 1e-6,
    natural_config: NaturalConfig = DEFAULT_NATURAL_CONFIG,
    show_chart: bool = False,
    verbose_each_dataset: bool = False,
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
                natural_config=natural_config,
                show_chart=show_chart,
                verbose=verbose_each_dataset,
            )
        except Exception as exc:
            results[name] = {"error": str(exc)}
            print(f"[ERROR] {name}: {exc}")

    print("\n" + "=" * 170)
    print("SUMMARY - CANR-AQG-GE-DPC")
    print("=" * 170)
    print(
        f"{'Dataset':<18} {'ACC':>7} {'NMI':>7} {'ARI':>7} "
        f"{'Scaler':>9} {'k':>3} {'Mode':>35} {'Ells':>6} "
        f"{'Lambda':>7} {'N-Deg':>7} {'Changed':>8} "
        f"{'P-Change':>9} {'Gain':>9} {'Accept':>7} {'Time(ms)':>11}"
    )
    print("-" * 170)

    for name, res in results.items():
        if "error" in res:
            print(f"{name:<18} ERROR: {res['error']}")
            continue

        print(
            f"{name:<18} {res['acc']:>7.3f} {res['nmi']:>7.3f} {res['ari']:>7.3f} "
            f"{res['scaler']:>9} {res['best_k']:>3} {res['mode']:>35} "
            f"{res['n_ellipsoids']:>6} {res['natural_lambda']:>7} "
            f"{res['natural_mean_degree']:>7.2f} {res['changed_ellipsoids']:>8} "
            f"{res['changed_point_ratio']:>9.3f} {res['quality_gain']:>9.4f} "
            f"{str(res['accepted']):>7} {res['total_time'] * 1000:>11.3f}"
        )

    print("=" * 170)
    return results


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent

    run_all_datasets(
        base_dir=BASE_DIR,
        epsilon=1e-6,
        natural_config=DEFAULT_NATURAL_CONFIG,
        show_chart=False,
        verbose_each_dataset=False,
    )
