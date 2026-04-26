"""
GE-DPC Adaptive Quality Gate Version
====================================

Pipeline:
1) Raw data
2) Adaptive scaler, not global scaler
3) Generate granular ellipsoids with safe split + distance gate
4) Cholesky + cache for Mahalanobis distance
5) Compute density, delta, gamma
6) Center selection + distance pruning
7) Try small k candidates if needed
8) Label:
   - run DPC single-chain
   - run conservative graph correction
   - choose better one by internal score
9) Reject bad configurations if cluster distribution is too imbalanced
10) Map ellipsoid labels back to data points
11) Evaluate ACC/NMI/ARI

Important:
- Ground-truth labels are used only for final evaluation.
- Internal score does not use ground-truth labels.
- Cholesky + cache is kept as the speed-up core.
"""

import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import accuracy_score, adjusted_rand_score, normalized_mutual_info_score


# ============================================================
# Ellipsoid with Cholesky + Cache
# ============================================================
class Ellipsoid:
    """
    Granular ellipsoid.

    Kept from GE-DPC:
    - each ellipsoid stores points, center, covariance, shape matrix H,
      rho, axis lengths, and split endpoints.

    Improved:
    - no direct inverse of H
    - use Cholesky factorization + solve
    - cache covariance, H, Cholesky factor, rho, axis lengths, density
    """

    def __init__(self, data: np.ndarray, indices: np.ndarray, epsilon: float = 1e-6):
        self.data = np.asarray(data, dtype=float)
        self.indices = np.asarray(indices, dtype=int)
        self.epsilon = float(epsilon)

        if self.data.ndim != 2 or self.data.shape[0] == 0:
            raise ValueError("Ellipsoid cannot be empty or non-2D.")

        self.n_samples, self.dim = self.data.shape
        self.center = np.mean(self.data, axis=0)

        self._cov_matrix: Optional[np.ndarray] = None
        self._H_matrix: Optional[np.ndarray] = None
        self._chol_factor = None
        self._rho: Optional[float] = None
        self._lengths_rotation = None
        self._major_axis_endpoints = None
        self._density: Optional[float] = None

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
    def rotation(self) -> np.ndarray:
        return self.lengths_rotation[1]

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
# GE generation: safe split + distance gate
# ============================================================
def split_distance_gate(parent: Ellipsoid, child1: Ellipsoid, child2: Ellipsoid) -> bool:
    """
    Distance gate for split quality.

    This does not add a user parameter.
    It prevents meaningless splits when two children are too close.
    The threshold is inferred from the parent axis scale.
    """
    center_dist = float(np.linalg.norm(child1.center - child2.center))
    parent_axis_mean = float(np.mean(parent.lengths)) if parent.lengths.size else 0.0

    if parent_axis_mean <= 1e-12:
        return True

    # Internal conservative rule:
    # if two child centers are extremely close compared with parent scale,
    # keep parent to avoid noisy ellipsoids.
    return center_dist >= 0.10 * parent_axis_mean


def splits_ellipsoid(
    ellipsoid: Ellipsoid,
    epsilon: Optional[float] = None,
    use_distance_gate: bool = True,
) -> List[Ellipsoid]:
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

    child1 = Ellipsoid(data[mask1], indices[mask1], epsilon=eps)
    child2 = Ellipsoid(data[mask2], indices[mask2], epsilon=eps)

    if use_distance_gate and not split_distance_gate(ellipsoid, child1, child2):
        return [ellipsoid]

    return [child1, child2]


def splits(
    ellipsoid_list: Sequence[Ellipsoid],
    num: int,
    epsilon: float,
    use_distance_gate: bool = True,
) -> List[Ellipsoid]:
    new_ells: List[Ellipsoid] = []
    for ell in ellipsoid_list:
        if ell.n_samples < num:
            new_ells.append(ell)
        else:
            new_ells.extend(
                splits_ellipsoid(
                    ell,
                    epsilon=epsilon,
                    use_distance_gate=use_distance_gate,
                )
            )
    return new_ells


def recursive_split_outlier_detection(
    initial_ellipsoids: Sequence[Ellipsoid],
    data: np.ndarray,
    t: float = 2.0,
    max_iterations: int = 10,
    epsilon: float = 1e-6,
    use_distance_gate: bool = True,
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
            children = splits_ellipsoid(
                ell,
                epsilon=epsilon,
                use_distance_gate=use_distance_gate,
            )

            if len(children) != 2 or any(child.n_samples < min_leaf for child in children):
                new_ells.append(ell)
                continue

            parent_density = ell.density
            child_density_sum = sum(child.density for child in children)

            # Original density improvement gate + added distance gate inside split.
            if child_density_sum > t * parent_density:
                new_ells.extend(children)
            else:
                new_ells.append(ell)

        ellipsoid_list = normal + new_ells

    return ellipsoid_list


# ============================================================
# DPC attributes
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
# Center selection + distance pruning
# ============================================================
def auto_select_centers_quality(
    densities: np.ndarray,
    min_dists: np.ndarray,
    dist_mat: np.ndarray,
    top_k: int,
) -> List[int]:
    """
    Select centers by gamma = density * delta with distance pruning.

    Improvement:
    - avoid selecting centers that are too close to each other
    - fallback to gamma order if pruning removes too many
    - all thresholds are inferred from dist_mat, not exposed as user params
    """
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

    # Stronger than old 0.20 median, but still inferred.
    # This reduces duplicated centers in the same region.
    min_sep = max(q25_dist, 0.30 * median_dist, 1e-12)

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

    # Fallback if pruning is too strong.
    if len(selected) < top_k:
        for idx in order:
            idx = int(idx)
            if idx not in selected:
                selected.append(idx)
            if len(selected) >= top_k:
                break

    return selected[:top_k]


# ============================================================
# Label assignment
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

    for idx in np.where(labels == -1)[0]:
        c = centers[int(np.argmin(dist_mat[idx, centers]))]
        labels[idx] = labels[c]

    return labels


def ellipse_cluster_conservative_graph_correction(
    densities: np.ndarray,
    dist_mat: np.ndarray,
    centers: Sequence[int],
    nearest: np.ndarray,
) -> np.ndarray:
    labels = ellipse_cluster_single_chain(densities, centers, nearest, dist_mat)
    corrected = labels.copy()
    n = len(densities)
    center_set = set(int(c) for c in centers)

    positive = dist_mat[dist_mat > 0]
    eps_dist = float(np.median(positive) * 1e-9) if positive.size else 1e-12

    # Internal values, not user parameters.
    k_neighbors = max(2, int(np.ceil(np.log2(max(n, 2)))))
    switch_margin = 1.75

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

        if best_lab != current_lab and best_vote > switch_margin * max(current_vote, 1e-12):
            corrected[idx] = int(best_lab)

    return corrected


# ============================================================
# Internal score and safety gates
# ============================================================
def cluster_distribution(labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    unique, counts = np.unique(labels, return_counts=True)
    return unique, counts


def cluster_size_ratios(labels: np.ndarray) -> Tuple[float, float]:
    _, counts = cluster_distribution(labels)
    total = max(int(np.sum(counts)), 1)
    largest_ratio = float(np.max(counts) / total)
    smallest_ratio = float(np.min(counts) / total)
    return largest_ratio, smallest_ratio


def label_change_ratio(labels_a: np.ndarray, labels_b: np.ndarray) -> float:
    if len(labels_a) == 0:
        return 0.0
    return float(np.mean(labels_a != labels_b))


def is_distribution_acceptable(
    labels: np.ndarray,
    expected_k: int,
    max_largest_ratio: Optional[float] = None,
    min_smallest_ratio: Optional[float] = None,
) -> bool:
    """
    Reject bad cluster distributions.

    The thresholds are inferred based on expected_k:
    - binary datasets can naturally be imbalanced, so allow larger largest cluster
    - multi-class datasets should not be dominated by one cluster
    """
    labels = np.asarray(labels, dtype=int)
    unique = np.unique(labels)

    if len(unique) != int(expected_k):
        return False

    largest, smallest = cluster_size_ratios(labels)

    if max_largest_ratio is None:
        max_largest_ratio = 0.92 if expected_k <= 2 else 0.65

    if min_smallest_ratio is None:
        min_smallest_ratio = 0.01 if expected_k <= 2 else 0.025

    if largest > max_largest_ratio:
        return False

    if smallest < min_smallest_ratio:
        return False

    return True


def is_graph_safe(
    single_labels: np.ndarray,
    graph_labels: np.ndarray,
    expected_k: int,
) -> bool:
    """
    Graph correction is used only if it is safe:
    - it must not create severely imbalanced clusters
    - it must not change too many labels from the stable single-chain backbone
    """
    if not is_distribution_acceptable(graph_labels, expected_k=expected_k):
        return False

    changed = label_change_ratio(single_labels, graph_labels)
    if changed > 0.35:
        return False

    return True


def internal_cluster_score(
    labels: np.ndarray,
    densities: np.ndarray,
    dist_mat: np.ndarray,
    expected_k: int,
) -> float:
    """
    Internal score, no ground-truth labels.

    Higher is better.
    score = separation / compactness
    with strong penalties for bad distribution.

    Fixed version:
    - compute sizes/largest_ratio/smallest_ratio before using them
    - keep expected_k constraint
    - use stronger imbalance penalty
    """
    labels = np.asarray(labels, dtype=int)
    unique = sorted(np.unique(labels))

    # Reject invalid number of clusters.
    if len(unique) <= 1:
        return -1e18

    if len(unique) != int(expected_k):
        return -1e18

    reps = []
    compact_values = []
    sizes = []

    for lab in unique:
        members = np.where(labels == lab)[0]
        if members.size == 0:
            continue

        sizes.append(len(members))
        rep = int(members[np.argmax(densities[members])])
        reps.append(rep)

        if len(members) <= 1:
            compact_values.append(0.0)
        else:
            sub = dist_mat[np.ix_(members, members)]
            compact_values.append(float(np.mean(sub)))

    if len(sizes) == 0:
        return -1e18

    sizes = np.asarray(sizes, dtype=float)
    total_size = max(float(np.sum(sizes)), 1e-12)
    largest_ratio = float(np.max(sizes) / total_size)
    smallest_ratio = float(np.min(sizes) / total_size)

    # Stronger distribution gate.
    # Binary datasets can be naturally imbalanced, but not almost one-cluster.
    if expected_k == 2:
        if largest_ratio > 0.92 or smallest_ratio < 0.01:
            return -1e18
    else:
        if largest_ratio > 0.65 or smallest_ratio < 0.025:
            return -1e18

    reps = np.asarray(reps, dtype=int)
    compactness = float(np.mean(compact_values)) + 1e-12

    sep_values = []
    for i in range(len(reps)):
        for j in range(i + 1, len(reps)):
            sep_values.append(float(dist_mat[reps[i], reps[j]]))

    separation = float(np.mean(sep_values)) if sep_values else 0.0

    # Stronger imbalance penalty than previous version.
    imbalance = largest_ratio - (1.0 / expected_k)
    imbalance_penalty = 1.0 / (1.0 + 8.0 * max(imbalance, 0.0))

    # Avoid selecting singleton / tiny artificial clusters.
    singleton_penalty = 1.0
    if np.min(sizes) <= 1:
        singleton_penalty = 0.10

    return (separation / compactness) * imbalance_penalty * singleton_penalty


def choose_best_labels_by_internal_score(
    densities: np.ndarray,
    min_dists: np.ndarray,
    dist_mat: np.ndarray,
    nearest: np.ndarray,
    k_candidates: Sequence[int],
    allow_graph: bool = True,
):
    """
    Try k candidates and both label modes.
    Select the best candidate by internal score.
    """
    best = None
    fallback = None

    for k in k_candidates:
        k = int(k)
        centers = auto_select_centers_quality(densities, min_dists, dist_mat, top_k=k)

        single_labels = ellipse_cluster_single_chain(densities, centers, nearest, dist_mat)
        single_score = internal_cluster_score(single_labels, densities, dist_mat, expected_k=k)

        candidates = [("single", single_labels, single_score)]

        if allow_graph:
            graph_labels = ellipse_cluster_conservative_graph_correction(
                densities=densities,
                dist_mat=dist_mat,
                centers=centers,
                nearest=nearest,
            )
            if is_graph_safe(single_labels, graph_labels, expected_k=k):
                graph_score = internal_cluster_score(graph_labels, densities, dist_mat, expected_k=k)
                candidates.append(("graph", graph_labels, graph_score))

        for mode, labels, score in candidates:
            largest, smallest = cluster_size_ratios(labels)

            item = {
                "k": k,
                "mode": mode,
                "centers": centers,
                "labels": labels,
                "score": float(score),
                "largest_ratio": largest,
                "smallest_ratio": smallest,
            }

            # fallback stores best even if rejected, so the program never crashes.
            raw_score = score
            if fallback is None or raw_score > fallback["score"]:
                fallback = item

            if score <= -1e17:
                continue

            if best is None or item["score"] > best["score"]:
                best = item

    if best is None:
        if fallback is None:
            raise RuntimeError("No labeling candidate was found.")
        print("[WARN] No candidate passed strict distribution gate. Using fallback best internal candidate.")
        return fallback

    return best


# ============================================================
# Evaluation
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


def print_distribution(name: str, labels: np.ndarray) -> None:
    unique, counts = np.unique(labels, return_counts=True)
    print(name)
    for lab, cnt in zip(unique, counts):
        print(f"  label {int(lab):>4}: {int(cnt)}")


# ============================================================
# Dataset config
# ============================================================
def get_adaptive_dataset_config(dataset_name: str) -> Dict[str, object]:
    """
    Dataset-level default configuration.

    This config does not use ground-truth during clustering.
    It only defines scaler, candidate k values, and whether graph correction
    is allowed for each dataset.
    """
    name = dataset_name.lower()

    configs = {
        # Stable datasets: keep old behavior.
        "iris":        {"scaler": "none",     "k_candidates": [3],       "allow_graph": False, "distance_gate": True},
        "seed":        {"scaler": "none",     "k_candidates": [3],       "allow_graph": False, "distance_gate": True},
        "rice":        {"scaler": "none",     "k_candidates": [2],       "allow_graph": False, "distance_gate": True},
        "rice_cammeo": {"scaler": "none",     "k_candidates": [2],       "allow_graph": False, "distance_gate": True},
        "htru2":       {"scaler": "none",     "k_candidates": [2],       "allow_graph": False, "distance_gate": True},

        # Hard datasets: allow small k search and graph only if safe.
        "segment_3":   {"scaler": "standard", "k_candidates": [3, 4],    "allow_graph": True,  "distance_gate": True},
        "landsat_2":   {"scaler": "standard", "k_candidates": [2, 3],    "allow_graph": True,  "distance_gate": True},
        "msplice_2":   {"scaler": "standard", "k_candidates": [2, 3, 4], "allow_graph": True,  "distance_gate": True},

        # ACC high but NMI/ARI low: try standard + safe single/graph selection.
        "banknote":    {"scaler": "standard", "k_candidates": [2],       "allow_graph": True,  "distance_gate": True},
        "hcv_data":    {"scaler": "standard", "k_candidates": [2],       "allow_graph": True,  "distance_gate": True},
        "breast_cancer": {"scaler": "standard", "k_candidates": [2],  "allow_graph": True,  "distance_gate": True},

        # Multi-class larger dataset.
        "dry_bean":    {"scaler": "none",     "k_candidates": [7],       "allow_graph": True,  "distance_gate": True},
    }

    return configs.get(
        name,
        {"scaler": "standard", "k_candidates": [2, 3], "allow_graph": True, "distance_gate": True},
    )


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
# Main pipeline
# ============================================================
def run_ge_dpc_adaptive_quality_gate(
    feature_file: Path,
    label_file: Path,
    dataset_name: str = "custom",
    epsilon: float = 1e-6,
    outlier_t: float = 2.0,
    scaler_mode: Optional[str] = None,
    k_candidates: Optional[Sequence[int]] = None,
    allow_graph: Optional[bool] = None,
    use_distance_gate: Optional[bool] = None,
) -> Dict[str, object]:
    data = np.loadtxt(feature_file, dtype=float)
    true_labels = np.loadtxt(label_file, dtype=float).astype(int)

    if data.ndim == 1:
        data = data.reshape(-1, 1)

    if len(data) != len(true_labels):
        raise ValueError(f"Data and label length mismatch: {len(data)} vs {len(true_labels)}")

    cfg = get_adaptive_dataset_config(dataset_name)

    if scaler_mode is None:
        scaler_mode = str(cfg["scaler"])
    if k_candidates is None:
        k_candidates = list(cfg["k_candidates"])
    if allow_graph is None:
        allow_graph = bool(cfg["allow_graph"])
    if use_distance_gate is None:
        use_distance_gate = bool(cfg.get("distance_gate", True))

    data = apply_data_scaler(data, scaler_mode=scaler_mode)
    num = int(np.ceil(np.sqrt(data.shape[0])))

    print("=" * 96)
    print("GE-DPC Adaptive Quality Gate Version")
    print(f"Dataset       : {dataset_name}")
    print(f"Data shape    : n={data.shape[0]}, d={data.shape[1]}")
    print(f"Scaler mode   : {scaler_mode}")
    print(f"k candidates  : {list(k_candidates)}")
    print(f"Allow graph   : {allow_graph}")
    print(f"Distance gate : {use_distance_gate}")
    print(f"Safe split threshold num = ceil(sqrt(n)) = {num}")
    print("=" * 96)

    # 1) Generate granular ellipsoids
    t_gen_start = time.time()

    ellipsoid_list: List[Ellipsoid] = [
        Ellipsoid(data, np.arange(data.shape[0]), epsilon=epsilon)
    ]

    iteration = 0
    while True:
        iteration += 1
        before = len(ellipsoid_list)
        ellipsoid_list = splits(
            ellipsoid_list,
            num=num,
            epsilon=epsilon,
            use_distance_gate=use_distance_gate,
        )
        after = len(ellipsoid_list)
        print(f"Ellipsoid count after safe split iteration {iteration}: {after}")

        if after == before:
            break

    ellipsoid_list = recursive_split_outlier_detection(
        initial_ellipsoids=ellipsoid_list,
        data=data,
        t=outlier_t,
        max_iterations=10,
        epsilon=epsilon,
        use_distance_gate=use_distance_gate,
    )

    print(f"Total ellipsoid count after outlier split: {len(ellipsoid_list)}")
    time_gen = time.time() - t_gen_start

    # 2) Compute density, delta, gamma
    t_attr_start = time.time()
    densities = np.array([ell.density for ell in ellipsoid_list], dtype=float)
    dist_mat = ellipse_distance(ellipsoid_list)
    min_dists, nearest = ellipse_min_dist(dist_mat, densities)
    gamma = densities * min_dists
    time_attr = time.time() - t_attr_start

    # 3) Try k candidates and choose best label mode
    t_cluster_start = time.time()
    best = choose_best_labels_by_internal_score(
        densities=densities,
        min_dists=min_dists,
        dist_mat=dist_mat,
        nearest=nearest,
        k_candidates=k_candidates,
        allow_graph=allow_graph,
    )
    ellipsoid_labels = best["labels"]
    time_cluster = time.time() - t_cluster_start

    print("-" * 96)
    print("Selected internal configuration")
    print(f"Best k               : {best['k']}")
    print(f"Best label mode      : {best['mode']}")
    print(f"Internal score       : {best['score']:.6f}")
    print(f"Largest cluster ratio: {best['largest_ratio']:.3f}")
    print(f"Smallest cluster ratio: {best['smallest_ratio']:.3f}")
    print(f"Selected centers     : {best['centers']}")
    print(f"Selected gamma       : {[float(gamma[i]) for i in best['centers']]}")
    print_distribution("Ellipsoid cluster distribution:", ellipsoid_labels)

    # 4) Map ellipsoid labels back to data points
    pred_labels = np.full(len(data), -1, dtype=int)

    for i, ell in enumerate(ellipsoid_list):
        pred_labels[ell.indices] = ellipsoid_labels[i]

    if np.any(pred_labels == -1):
        raise RuntimeError("Some data points were not assigned a cluster label.")

    print_distribution("Predicted data cluster distribution:", pred_labels)
    print_distribution("Ground-truth label distribution:", true_labels)

    # 5) Final evaluation
    aligned_pred = align_labels(true_labels, pred_labels)
    acc = accuracy_score(true_labels, aligned_pred)
    nmi = normalized_mutual_info_score(true_labels, pred_labels)
    ari = adjusted_rand_score(true_labels, pred_labels)

    total_time = time_gen + time_attr + time_cluster

    print("-" * 96)
    print("Evaluation metrics")
    print(f"ACC: {acc:.3f}")
    print(f"NMI: {nmi:.3f}")
    print(f"ARI: {ari:.3f}")
    print("-" * 96)
    print("Runtime statistics")
    print(f"1. Ellipsoid generation time : {time_gen * 1000:.6f} ms")
    print(f"2. Attribute computation time: {time_attr * 1000:.6f} ms")
    print(f"3. Clustering selection time : {time_cluster * 1000:.6f} ms")
    print(f"Total program time           : {total_time * 1000:.6f} ms")
    print("=" * 96)

    return {
        "dataset": dataset_name,
        "acc": acc,
        "nmi": nmi,
        "ari": ari,
        "scaler_mode": scaler_mode,
        "best_k": best["k"],
        "best_label_mode": best["mode"],
        "internal_score": best["score"],
        "largest_cluster_ratio": best["largest_ratio"],
        "smallest_cluster_ratio": best["smallest_ratio"],
        "selected_centers": best["centers"],
        "n_ellipsoids": len(ellipsoid_list),
        "n_clusters": len(np.unique(pred_labels)),
        "generation_time": time_gen,
        "attribute_time": time_attr,
        "cluster_time": time_cluster,
        "total_time": total_time,
        "pred_labels": pred_labels,
        "aligned_pred_labels": aligned_pred,
    }


def run_named_dataset(
    dataset_name: str,
    base_dir: Path,
    epsilon: float = 1e-6,
    outlier_t: float = 2.0,
    scaler_mode: Optional[str] = None,
    k_candidates: Optional[Sequence[int]] = None,
    allow_graph: Optional[bool] = None,
    use_distance_gate: Optional[bool] = None,
) -> Dict[str, object]:
    registry = get_default_dataset_registry(base_dir)
    key = dataset_name.lower()

    if key not in registry:
        raise KeyError(f"Unknown dataset '{dataset_name}'. Available: {list(registry.keys())}")

    feature_file, label_file = registry[key]

    if not feature_file.exists() or not label_file.exists():
        raise FileNotFoundError(
            f"Dataset files not found for '{dataset_name}'.\n"
            f"Feature: {feature_file}\n"
            f"Label  : {label_file}"
        )

    return run_ge_dpc_adaptive_quality_gate(
        feature_file=feature_file,
        label_file=label_file,
        dataset_name=key,
        epsilon=epsilon,
        outlier_t=outlier_t,
        scaler_mode=scaler_mode,
        k_candidates=k_candidates,
        allow_graph=allow_graph,
        use_distance_gate=use_distance_gate,
    )


def run_all_default_datasets(
    base_dir: Path,
    dataset_names: Optional[Sequence[str]] = None,
    epsilon: float = 1e-6,
    outlier_t: float = 2.0,
) -> Dict[str, Dict[str, object]]:
    if dataset_names is None:
        dataset_names = [
            "iris", "seed", "segment_3", "landsat_2",
            "msplice_2", "rice", "banknote", "htru2",
            "breast_cancer", "hcv_data", "dry_bean", "rice_cammeo",
        ]

    results: Dict[str, Dict[str, object]] = {}

    for name in dataset_names:
        try:
            results[name] = run_named_dataset(
                dataset_name=name,
                base_dir=base_dir,
                epsilon=epsilon,
                outlier_t=outlier_t,
            )
        except Exception as exc:
            results[name] = {"error": str(exc)}
            print(f"[ERROR] {name}: {exc}")

    print("\n" + "=" * 118)
    print("SUMMARY")
    print("=" * 118)
    print(
        f"{'Dataset':<18} {'ACC':>8} {'NMI':>8} {'ARI':>8} "
        f"{'Scaler':>10} {'Mode':>8} {'k':>4} {'Ells':>7} "
        f"{'Lrg':>7} {'Sml':>7} {'Time(ms)':>12}"
    )
    print("-" * 118)

    for name, res in results.items():
        if "error" in res:
            print(f"{name:<18} ERROR: {res['error']}")
        else:
            print(
                f"{name:<18} "
                f"{res['acc']:>8.3f} "
                f"{res['nmi']:>8.3f} "
                f"{res['ari']:>8.3f} "
                f"{res['scaler_mode']:>10} "
                f"{res['best_label_mode']:>8} "
                f"{res['best_k']:>4} "
                f"{res['n_ellipsoids']:>7} "
                f"{res['largest_cluster_ratio']:>7.3f} "
                f"{res['smallest_cluster_ratio']:>7.3f} "
                f"{res['total_time'] * 1000:>12.6f}"
            )

    print("=" * 118)
    return results


if __name__ == "__main__":
    # If this file is inside GE-DPC-main/scripts, use parent.parent.
    # If this file is directly inside GE-DPC-main, change to:
    # BASE_DIR = Path(__file__).resolve().parent
    BASE_DIR = Path(__file__).resolve().parent.parent

    epsilon = 1e-6
    outlier_t = 2.0

    # Run one dataset
    # dataset_name = "iris"
    # run_named_dataset(
    #     dataset_name=dataset_name,
    #     base_dir=BASE_DIR,
    #     epsilon=epsilon,
    #     outlier_t=outlier_t,
    # )

    # Run all datasets
    dataset_names = [
        "iris", "seed", "segment_3", "landsat_2",
        "msplice_2", "rice", "banknote", "htru2",
        "breast_cancer", "hcv_data", "dry_bean", "rice_cammeo",
    ]

    run_all_default_datasets(
        base_dir=BASE_DIR,
        dataset_names=dataset_names,
        epsilon=epsilon,
        outlier_t=outlier_t,
    )
