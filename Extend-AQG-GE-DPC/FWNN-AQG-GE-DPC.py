"""
FWNN-AQG-GE-DPC
================
Fuzzy-Weighted Natural-Neighbor Adaptive Quality-Gated
Granular-Ellipsoid Density Peak Clustering.

Scientific basis
----------------
This implementation extends AQG-GE-DPC at the granular-ellipsoid level by
adapting two ideas from fuzzy weighted natural-nearest-neighbor DPC:

1) Natural nearest neighbors are built on the final granular ellipsoids,
   not on all original samples. This preserves the efficiency advantage of
   granular computing because the number of ellipsoids is normally much
   smaller than the number of data points.

2) Label assignment is divided into two stages:
   - reliable ellipsoids are propagated by natural-neighbor consensus;
   - ambiguous ellipsoids are assigned by fuzzy weighted natural-neighbor
     membership, subject to conservative acceptance gates.

The method keeps the main AQG-GE-DPC components:
- adaptive data scaling;
- safe granular-ellipsoid splitting;
- density-based quality gate;
- Cholesky-based Mahalanobis computation;
- ellipsoid-level caching;
- density-delta center selection;
- internal-score-based unsupervised refinement acceptance.

Important methodological constraint
-----------------------------------
Ground-truth labels are used only in the final evaluation stage to compute
ACC, NMI, and ARI. They are not used for splitting, density estimation,
center selection, label propagation, fuzzy assignment, or model selection.

This file is a research prototype. It implements a scientifically motivated
extension rather than claiming to reproduce the original FWNNN-DPC paper
exactly, because the original paper operates at point level whereas this
implementation transfers the natural-neighbor and fuzzy-assignment principles
to granular ellipsoids.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.linalg import LinAlgError, cho_factor, cho_solve
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import accuracy_score, adjusted_rand_score, normalized_mutual_info_score


# =============================================================================
# Numerical utilities
# =============================================================================

def _safe_minmax(values: np.ndarray) -> np.ndarray:
    """Return min-max normalized values in [0, 1].

    A constant vector is mapped to zeros. This function is used only for
    unsupervised internal quantities such as density components.
    """
    x = np.asarray(values, dtype=float)
    if x.size == 0:
        return x.copy()
    lo = float(np.min(x))
    hi = float(np.max(x))
    if hi - lo <= 1e-15:
        return np.zeros_like(x, dtype=float)
    return (x - lo) / (hi - lo)


def _safe_mad(values: np.ndarray) -> float:
    """Median absolute deviation, used for robust thresholding."""
    x = np.asarray(values, dtype=float)
    if x.size == 0:
        return 0.0
    med = float(np.median(x))
    return float(np.median(np.abs(x - med)))


def _stable_soft_ratio(best: float, second: float, eps: float = 1e-12) -> float:
    """Relative membership margin in [0, 1] for non-negative scores."""
    best = max(float(best), 0.0)
    second = max(float(second), 0.0)
    return (best - second) / (best + eps)


# =============================================================================
# Granular ellipsoid with Cholesky-based cache
# =============================================================================

class Ellipsoid:
    """Granular ellipsoid representation.

    Each ellipsoid stores the original sample indices and lazily caches
    covariance-related quantities. Cholesky factorization is used to evaluate
    Mahalanobis forms without explicitly computing matrix inverses.
    """

    def __init__(self, data: np.ndarray, indices: np.ndarray, epsilon: float = 1e-6):
        self.data = np.asarray(data, dtype=float)
        self.indices = np.asarray(indices, dtype=int)
        self.epsilon = float(epsilon)

        if self.data.ndim != 2 or self.data.shape[0] == 0:
            raise ValueError("Ellipsoid data must be a non-empty 2D array.")
        if self.indices.ndim != 1 or self.indices.size != self.data.shape[0]:
            raise ValueError("indices must be a 1D array aligned with data rows.")

        self.n_samples, self.dim = self.data.shape
        self.center = np.mean(self.data, axis=0)

        self._cov_matrix: Optional[np.ndarray] = None
        self._H_matrix: Optional[np.ndarray] = None
        self._chol_factor = None
        self._rho: Optional[float] = None
        self._lengths_rotation: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self._major_axis_endpoints: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self._intrinsic_density: Optional[float] = None

    @property
    def cov_matrix(self) -> np.ndarray:
        if self._cov_matrix is None:
            if self.n_samples <= 1:
                self._cov_matrix = np.zeros((self.dim, self.dim), dtype=float)
            else:
                cov = np.cov(self.data.T, bias=True)
                if np.ndim(cov) == 0:
                    cov = np.array([[float(cov)]], dtype=float)
                self._cov_matrix = np.asarray(cov, dtype=float)
        return self._cov_matrix

    @property
    def H_matrix(self) -> np.ndarray:
        if self._H_matrix is None:
            self._H_matrix = self.cov_matrix + self.epsilon * np.eye(self.dim, dtype=float)
        return self._H_matrix

    @property
    def chol_factor(self):
        if self._chol_factor is None:
            try:
                self._chol_factor = cho_factor(self.H_matrix, lower=True, check_finite=False)
            except LinAlgError:
                # Defensive regularization for nearly singular covariance matrices.
                jitter = max(self.epsilon * 10.0, 1e-8)
                H = self.H_matrix + jitter * np.eye(self.dim, dtype=float)
                self._chol_factor = cho_factor(H, lower=True, check_finite=False)
        return self._chol_factor

    def solve_H(self, rhs: np.ndarray) -> np.ndarray:
        return cho_solve(self.chol_factor, rhs, check_finite=False)

    def mahal_sq_points(self, points: np.ndarray) -> np.ndarray:
        X = np.asarray(points, dtype=float)
        if X.ndim == 1:
            X = X[None, :]
        if X.shape[1] != self.dim:
            raise ValueError("Point dimensionality does not match ellipsoid dimensionality.")
        diffs = X - self.center
        solved = self.solve_H(diffs.T).T
        return np.maximum(np.einsum("ij,ij->i", diffs, solved), 0.0)

    @property
    def rho(self) -> float:
        """Mahalanobis radius covering the ellipsoid samples."""
        if self._rho is None:
            self._rho = float(np.sqrt(np.max(self.mahal_sq_points(self.data))))
        return self._rho

    @property
    def lengths_rotation(self) -> Tuple[np.ndarray, np.ndarray]:
        if self._lengths_rotation is None:
            eigvals, eigvecs = np.linalg.eigh(self.H_matrix)
            eigvals = np.maximum(eigvals, 1e-12)
            lengths = self.rho * np.sqrt(eigvals)
            self._lengths_rotation = (lengths, eigvecs)
        return self._lengths_rotation

    @property
    def lengths(self) -> np.ndarray:
        return self.lengths_rotation[0]

    @property
    def major_axis_endpoints(self) -> Tuple[np.ndarray, np.ndarray]:
        """Approximate farthest-point pair used to initialize binary splitting."""
        if self._major_axis_endpoints is None:
            if self.n_samples <= 1:
                self._major_axis_endpoints = (self.center.copy(), self.center.copy())
            else:
                center_dist = np.linalg.norm(self.data - self.center, axis=1)
                p0 = self.data[int(np.argmin(center_dist))]
                p1 = self.data[int(np.argmax(np.linalg.norm(self.data - p0, axis=1)))]
                p2 = self.data[int(np.argmax(np.linalg.norm(self.data - p1, axis=1)))]
                self._major_axis_endpoints = (p1, p2)
        return self._major_axis_endpoints

    @property
    def intrinsic_density(self) -> float:
        """AQG-GE-DPC-style intrinsic density.

        This component combines sample cardinality, ellipsoid axis lengths,
        and within-ellipsoid Mahalanobis dispersion.
        """
        if self._intrinsic_density is None:
            axes_sum = max(float(np.sum(self.lengths)), 1e-12)
            mahal = np.sqrt(self.mahal_sq_points(self.data))
            total_mahal = max(float(np.sum(mahal)), 1e-12)
            self._intrinsic_density = float((self.n_samples ** 2) / (axes_sum * total_mahal))
        return self._intrinsic_density


# =============================================================================
# Configuration
# =============================================================================

@dataclass(frozen=True)
class MethodConfig:
    """Dataset-independent configuration for the proposed extension."""

    epsilon: float = 1e-6
    scaler: str = "standard"

    # Granular-ellipsoid generation
    split_factor: float = 1.0
    outlier_quality_t: float = 1.5
    max_split_iterations: int = 10
    min_leaf_ratio: float = 0.10

    # Number of clusters. In controlled benchmark experiments, this can be set
    # to the known number of classes, matching common DPC evaluation practice.
    # It is not inferred from ground-truth labels inside the algorithm.
    n_clusters: int = 2

    # Natural-neighbor construction
    nnn_stable_rounds: int = 2
    nnn_max_k_factor: float = 2.0
    nnn_min_coverage: float = 0.95

    # Hybrid density
    use_hybrid_density: bool = True
    density_blend_mode: str = "adaptive"  # "adaptive" or "fixed"
    fixed_intrinsic_weight: float = 0.50

    # Reliable propagation
    consensus_threshold: float = 0.70
    min_labeled_neighbor_weight: float = 1e-12

    # Ambiguity and fuzzy assignment
    radius_mad_factor: float = 1.5
    fuzzy_membership_margin: float = 0.15
    fuzzy_max_rounds: int = 10

    # Conservative global acceptance
    max_changed_ratio: float = 0.35
    max_largest_cluster_ratio: float = 0.85
    min_cluster_ratio: float = 0.01
    internal_score_tolerance: float = 0.0


# =============================================================================
# Data scaling
# =============================================================================

def apply_data_scaler(data: np.ndarray, scaler_mode: str) -> np.ndarray:
    mode = str(scaler_mode).strip().lower()
    X = np.asarray(data, dtype=float)

    if mode == "none":
        return X.copy()
    if mode == "standard":
        from sklearn.preprocessing import StandardScaler
        return StandardScaler().fit_transform(X)
    if mode == "minmax":
        from sklearn.preprocessing import MinMaxScaler
        return MinMaxScaler().fit_transform(X)
    if mode == "robust":
        from sklearn.preprocessing import RobustScaler
        return RobustScaler().fit_transform(X)

    raise ValueError("scaler must be one of: none, standard, minmax, robust")


# =============================================================================
# Quality-gated granular-ellipsoid generation
# =============================================================================

def split_ellipsoid(ellipsoid: Ellipsoid, epsilon: Optional[float] = None) -> List[Ellipsoid]:
    """Split one ellipsoid into two children.

    Initialization uses an approximate farthest-point pair. A second assignment
    step uses child-specific Mahalanobis distance, so the split follows local
    anisotropic geometry rather than only Euclidean geometry.
    """
    if ellipsoid.n_samples <= 1:
        return [ellipsoid]

    eps = ellipsoid.epsilon if epsilon is None else float(epsilon)
    X = ellipsoid.data
    idx = ellipsoid.indices
    p1, p2 = ellipsoid.major_axis_endpoints

    d1 = np.linalg.norm(X - p1, axis=1)
    d2 = np.linalg.norm(X - p2, axis=1)
    mask_left = d1 < d2

    if np.all(mask_left) or np.all(~mask_left):
        return [ellipsoid]

    left = Ellipsoid(X[mask_left], idx[mask_left], epsilon=eps)
    right = Ellipsoid(X[~mask_left], idx[~mask_left], epsilon=eps)

    md_left = left.mahal_sq_points(X)
    md_right = right.mahal_sq_points(X)
    refined_left = md_left < md_right

    if np.all(refined_left) or np.all(~refined_left):
        return [ellipsoid]

    return [
        Ellipsoid(X[refined_left], idx[refined_left], epsilon=eps),
        Ellipsoid(X[~refined_left], idx[~refined_left], epsilon=eps),
    ]


def safe_split_pass(
    ellipsoids: Sequence[Ellipsoid],
    min_split_size: int,
    epsilon: float,
) -> List[Ellipsoid]:
    """Apply one safe split pass to ellipsoids meeting the size condition."""
    result: List[Ellipsoid] = []
    for ell in ellipsoids:
        if ell.n_samples < min_split_size:
            result.append(ell)
        else:
            result.extend(split_ellipsoid(ell, epsilon=epsilon))
    return result


def recursive_quality_refinement(
    ellipsoids: Sequence[Ellipsoid],
    full_data: np.ndarray,
    quality_t: float,
    max_iterations: int,
    epsilon: float,
    min_leaf_ratio: float,
) -> List[Ellipsoid]:
    """Reconsider unusually large ellipsoids and accept only quality-improving splits.

    A candidate parent is reconsidered when its total semi-axis length exceeds
    twice the current mean. A split is accepted only when the sum of child
    intrinsic densities exceeds quality_t times the parent intrinsic density.
    """
    current = list(ellipsoids)
    min_leaf = max(2, int(math.ceil(math.sqrt(full_data.shape[0]) * min_leaf_ratio)))

    for _ in range(int(max_iterations)):
        if not current:
            break

        axis_sums = np.array([np.sum(e.lengths) for e in current], dtype=float)
        mean_axis_sum = float(np.mean(axis_sums))
        candidate_mask = axis_sums > 2.0 * mean_axis_sum

        if not np.any(candidate_mask):
            break

        next_list: List[Ellipsoid] = []
        changed = False

        for is_candidate, parent in zip(candidate_mask, current):
            if not is_candidate:
                next_list.append(parent)
                continue

            children = split_ellipsoid(parent, epsilon=epsilon)
            if len(children) != 2 or any(c.n_samples < min_leaf for c in children):
                next_list.append(parent)
                continue

            child_density_sum = sum(c.intrinsic_density for c in children)
            if child_density_sum > float(quality_t) * parent.intrinsic_density:
                next_list.extend(children)
                changed = True
            else:
                next_list.append(parent)

        current = next_list
        if not changed:
            break

    return current


def generate_granular_ellipsoids(data: np.ndarray, cfg: MethodConfig) -> List[Ellipsoid]:
    """Complete AQG-style ellipsoid generation stage."""
    n = data.shape[0]
    min_split_size = max(5, int(math.ceil(math.sqrt(n) * cfg.split_factor)))
    ellipsoids: List[Ellipsoid] = [
        Ellipsoid(data, np.arange(n, dtype=int), epsilon=cfg.epsilon)
    ]

    while True:
        before = len(ellipsoids)
        ellipsoids = safe_split_pass(
            ellipsoids,
            min_split_size=min_split_size,
            epsilon=cfg.epsilon,
        )
        if len(ellipsoids) == before:
            break

    return recursive_quality_refinement(
        ellipsoids,
        full_data=data,
        quality_t=cfg.outlier_quality_t,
        max_iterations=cfg.max_split_iterations,
        epsilon=cfg.epsilon,
        min_leaf_ratio=cfg.min_leaf_ratio,
    )


# =============================================================================
# Ellipsoid-level Mahalanobis geometry
# =============================================================================

def ellipsoid_mahalanobis_distance(a: Ellipsoid, b: Ellipsoid) -> float:
    """Symmetric ellipsoid-center Mahalanobis distance using average shape."""
    avg_H = 0.5 * (a.H_matrix + b.H_matrix)
    try:
        chol = cho_factor(avg_H, lower=True, check_finite=False)
    except LinAlgError:
        jitter = max(a.epsilon, b.epsilon, 1e-8) * 10.0
        avg_H = avg_H + jitter * np.eye(avg_H.shape[0], dtype=float)
        chol = cho_factor(avg_H, lower=True, check_finite=False)

    diff = a.center - b.center
    solved = cho_solve(chol, diff, check_finite=False)
    return float(np.sqrt(max(float(diff.T @ solved), 0.0)))


def ellipsoid_distance_matrix(ellipsoids: Sequence[Ellipsoid]) -> np.ndarray:
    m = len(ellipsoids)
    D = np.zeros((m, m), dtype=float)
    for i in range(m):
        for j in range(i + 1, m):
            d = ellipsoid_mahalanobis_distance(ellipsoids[i], ellipsoids[j])
            D[i, j] = D[j, i] = d
    return D


# =============================================================================
# Natural nearest neighbors on granular ellipsoids
# =============================================================================

def build_ellipsoid_natural_neighbors(
    dist_mat: np.ndarray,
    stable_rounds: int = 2,
    max_k_factor: float = 2.0,
    min_coverage: float = 0.95,
) -> Tuple[List[np.ndarray], int, np.ndarray]:
    """Construct reciprocal natural-neighbor sets on ellipsoids.

    For each k, NNN_k(i) = NN_k(i) intersection RNN_k(i). The search stops
    when either sufficient coverage is reached and the empty-set pattern is
    stable, or a conservative k limit is reached.

    Returns
    -------
    nnn_sets:
        List of natural-neighbor index arrays.
    natural_eigenvalue:
        The final k used by the adaptive search.
    neighbor_order:
        Cached nearest-neighbor order for reuse.
    """
    D = np.asarray(dist_mat, dtype=float)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError("dist_mat must be a square matrix.")

    m = D.shape[0]
    if m == 0:
        return [], 0, np.empty((0, 0), dtype=int)
    if m == 1:
        return [np.empty(0, dtype=int)], 0, np.empty((1, 0), dtype=int)

    # Exclude self by placing the diagonal at +inf before sorting.
    D_work = D.copy()
    np.fill_diagonal(D_work, np.inf)
    neighbor_order = np.argsort(D_work, axis=1)

    max_k = min(
        m - 1,
        max(1, int(math.ceil(max_k_factor * math.log2(max(m, 2))))),
    )

    previous_empty: Optional[np.ndarray] = None
    unchanged_rounds = 0
    final_sets: List[np.ndarray] = [np.empty(0, dtype=int) for _ in range(m)]
    final_k = 1

    for k in range(1, max_k + 1):
        nn = neighbor_order[:, :k]
        directed = np.zeros((m, m), dtype=bool)
        rows = np.repeat(np.arange(m), k)
        directed[rows, nn.reshape(-1)] = True
        mutual = directed & directed.T

        sets_k = [np.flatnonzero(mutual[i]) for i in range(m)]
        empty = np.array([len(s) == 0 for s in sets_k], dtype=bool)
        coverage = 1.0 - float(np.mean(empty))

        if previous_empty is not None and np.array_equal(empty, previous_empty):
            unchanged_rounds += 1
        else:
            unchanged_rounds = 0

        final_sets = sets_k
        final_k = k
        previous_empty = empty

        if coverage >= float(min_coverage) and unchanged_rounds >= int(stable_rounds):
            break

    # Scientifically conservative fallback: an ellipsoid with no reciprocal
    # neighbor receives only its nearest ellipsoid, rather than forcing k to
    # grow to m-1 and connecting unrelated regions.
    for i, s in enumerate(final_sets):
        if len(s) == 0:
            final_sets[i] = np.array([int(neighbor_order[i, 0])], dtype=int)

    return final_sets, final_k, neighbor_order


def natural_neighbor_radii(
    nnn_sets: Sequence[np.ndarray],
    dist_mat: np.ndarray,
) -> np.ndarray:
    radii = np.zeros(len(nnn_sets), dtype=float)
    for i, neigh in enumerate(nnn_sets):
        if len(neigh) > 0:
            radii[i] = float(np.max(dist_mat[i, neigh]))
    return radii


# =============================================================================
# Hybrid ellipsoid density
# =============================================================================

def compute_structural_natural_neighbor_density(
    ellipsoids: Sequence[Ellipsoid],
    dist_mat: np.ndarray,
    nnn_sets: Sequence[np.ndarray],
) -> np.ndarray:
    """Natural-neighbor structural density at ellipsoid level.

    The score combines reciprocal-neighbor cardinality and locally scaled,
    size-aware affinity. Local scales use the median NNN distance, reducing
    sensitivity to globally uneven density.
    """
    m = len(ellipsoids)
    sizes = np.array([e.n_samples for e in ellipsoids], dtype=float)
    nnn_cardinality = np.array([len(s) for s in nnn_sets], dtype=float)

    local_scale = np.ones(m, dtype=float)
    positive_global = dist_mat[dist_mat > 0]
    global_median = float(np.median(positive_global)) if positive_global.size else 1.0

    for i, neigh in enumerate(nnn_sets):
        vals = dist_mat[i, neigh]
        vals = vals[vals > 0]
        local_scale[i] = float(np.median(vals)) if vals.size else global_median
        local_scale[i] = max(local_scale[i], 1e-12)

    structural = np.zeros(m, dtype=float)
    for i, neigh in enumerate(nnn_sets):
        score = float(len(neigh))
        for j in neigh:
            denom = local_scale[i] * local_scale[j] + 1e-12
            affinity = math.exp(-(dist_mat[i, j] ** 2) / denom)
            size_factor = math.sqrt(max(sizes[i] * sizes[j], 1.0))
            score += affinity * size_factor * (1.0 + nnn_cardinality[j])
        structural[i] = score

    return structural


def compute_hybrid_density(
    ellipsoids: Sequence[Ellipsoid],
    structural_density: np.ndarray,
    blend_mode: str = "adaptive",
    fixed_intrinsic_weight: float = 0.50,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Blend intrinsic and natural-neighbor structural densities.

    Adaptive blending gives larger ellipsoids more trust in intrinsic density,
    while smaller ellipsoids receive relatively more structural support from
    their natural neighborhood.
    """
    intrinsic_raw = np.array([e.intrinsic_density for e in ellipsoids], dtype=float)
    structural_raw = np.asarray(structural_density, dtype=float)

    intrinsic = _safe_minmax(intrinsic_raw)
    structural = _safe_minmax(structural_raw)

    mode = str(blend_mode).lower().strip()
    if mode == "fixed":
        alpha = np.full(len(ellipsoids), np.clip(fixed_intrinsic_weight, 0.0, 1.0))
    elif mode == "adaptive":
        sizes = np.array([e.n_samples for e in ellipsoids], dtype=float)
        median_size = max(float(np.median(sizes)), 1.0)
        alpha = sizes / (sizes + median_size)
        alpha = np.clip(alpha, 0.20, 0.80)
    else:
        raise ValueError("blend_mode must be 'adaptive' or 'fixed'.")

    hybrid = alpha * intrinsic + (1.0 - alpha) * structural
    # A tiny deterministic tie-breaker prevents equal-density ambiguity.
    hybrid += 1e-12 * np.arange(len(hybrid), dtype=float)
    return hybrid, intrinsic_raw, structural_raw


# =============================================================================
# DPC attributes and center selection
# =============================================================================

def compute_delta_and_parent(
    dist_mat: np.ndarray,
    densities: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    densities = np.asarray(densities, dtype=float)
    m = len(densities)
    order = np.argsort(-densities, kind="mergesort")
    delta = np.zeros(m, dtype=float)
    parent = -np.ones(m, dtype=int)

    for position, idx in enumerate(order):
        if position == 0:
            continue
        higher = order[:position]
        j = int(higher[np.argmin(dist_mat[idx, higher])])
        parent[idx] = j
        delta[idx] = float(dist_mat[idx, j])

    if m > 0:
        top = int(order[0])
        finite_row = dist_mat[top][dist_mat[top] > 0]
        delta[top] = float(np.max(finite_row)) if finite_row.size else 0.0

    return delta, parent


def select_centers_nnn_aware(
    densities: np.ndarray,
    delta: np.ndarray,
    dist_mat: np.ndarray,
    nnn_sets: Sequence[np.ndarray],
    n_clusters: int,
) -> List[int]:
    """Select centers by gamma with distance and NNN-basin suppression."""
    m = len(densities)
    k = max(1, min(int(n_clusters), m))
    gamma = densities * delta
    order = np.argsort(-gamma, kind="mergesort")

    positive = dist_mat[dist_mat > 0]
    if positive.size == 0:
        return order[:k].astype(int).tolist()

    min_sep = max(float(np.quantile(positive, 0.25)), 0.20 * float(np.median(positive)), 1e-12)
    nnn_sets_as_set = [set(map(int, s)) for s in nnn_sets]
    selected: List[int] = []

    for idx_raw in order:
        idx = int(idx_raw)
        if not selected:
            selected.append(idx)
        else:
            too_close = float(np.min(dist_mat[idx, selected])) < min_sep
            same_basin = any(
                (c in nnn_sets_as_set[idx]) or (idx in nnn_sets_as_set[c])
                for c in selected
            )
            # Suppress only when both geometric and neighborhood evidence
            # indicate that the candidate belongs to an already represented basin.
            if not (too_close and same_basin):
                selected.append(idx)

        if len(selected) >= k:
            break

    # Deterministic fallback guarantees the requested number of centers.
    if len(selected) < k:
        for idx_raw in order:
            idx = int(idx_raw)
            if idx not in selected:
                selected.append(idx)
            if len(selected) >= k:
                break

    return selected[:k]


# =============================================================================
# Baseline assignment and internal score
# =============================================================================

def single_chain_assignment(
    densities: np.ndarray,
    centers: Sequence[int],
    parent: np.ndarray,
    dist_mat: np.ndarray,
) -> np.ndarray:
    m = len(densities)
    labels = -np.ones(m, dtype=int)
    centers = [int(c) for c in centers]

    for lab, c in enumerate(centers):
        labels[c] = lab

    order = np.argsort(-densities, kind="mergesort")
    for idx in order:
        if labels[idx] == -1 and parent[idx] >= 0:
            labels[idx] = labels[parent[idx]]

    for idx in np.where(labels < 0)[0]:
        nearest_center = centers[int(np.argmin(dist_mat[idx, centers]))]
        labels[idx] = labels[nearest_center]

    return labels


def cluster_size_statistics(labels: np.ndarray) -> Tuple[float, float]:
    _, counts = np.unique(labels, return_counts=True)
    total = max(int(np.sum(counts)), 1)
    return float(np.max(counts) / total), float(np.min(counts) / total)


def internal_cluster_score(
    labels: np.ndarray,
    densities: np.ndarray,
    dist_mat: np.ndarray,
    max_largest_cluster_ratio: float,
    min_cluster_ratio: float,
) -> float:
    """Unsupervised separation-compactness score with imbalance penalties."""
    labels = np.asarray(labels, dtype=int)
    unique = np.unique(labels)
    if unique.size <= 1:
        return -1e18

    representatives: List[int] = []
    compactness_terms: List[float] = []
    counts: List[int] = []

    for lab in unique:
        members = np.flatnonzero(labels == lab)
        counts.append(len(members))
        representatives.append(int(members[np.argmax(densities[members])]))

        if len(members) <= 1:
            compactness_terms.append(0.0)
        else:
            block = dist_mat[np.ix_(members, members)]
            compactness_terms.append(float(np.mean(block)))

    compactness = float(np.mean(compactness_terms)) + 1e-12
    sep_terms: List[float] = []
    for i in range(len(representatives)):
        for j in range(i + 1, len(representatives)):
            sep_terms.append(float(dist_mat[representatives[i], representatives[j]]))
    separation = float(np.mean(sep_terms)) if sep_terms else 0.0

    counts_arr = np.asarray(counts, dtype=float)
    largest_ratio = float(np.max(counts_arr) / np.sum(counts_arr))
    smallest_ratio = float(np.min(counts_arr) / np.sum(counts_arr))

    penalty = 1.0
    if largest_ratio > max_largest_cluster_ratio:
        penalty *= 0.20
    if smallest_ratio < min_cluster_ratio:
        penalty *= 0.50

    return (separation / compactness) * penalty


# =============================================================================
# Reliable natural-neighbor propagation
# =============================================================================

def _neighbor_vote_scores(
    i: int,
    labels: np.ndarray,
    densities: np.ndarray,
    ellipsoid_sizes: np.ndarray,
    nnn_sets: Sequence[np.ndarray],
    dist_mat: np.ndarray,
) -> Dict[int, float]:
    votes: Dict[int, float] = {}
    neigh = nnn_sets[i]
    if len(neigh) == 0:
        return votes

    positive = dist_mat[i, neigh]
    local_scale = float(np.median(positive[positive > 0])) if np.any(positive > 0) else 1.0
    local_scale = max(local_scale, 1e-12)

    for j in neigh:
        lab = int(labels[j])
        if lab < 0:
            continue
        distance_weight = math.exp(-((dist_mat[i, j] / local_scale) ** 2))
        reliability = max(float(densities[j]), 1e-12) * math.sqrt(max(float(ellipsoid_sizes[j]), 1.0))
        votes[lab] = votes.get(lab, 0.0) + distance_weight * reliability

    return votes


def propagate_reliable_labels(
    densities: np.ndarray,
    centers: Sequence[int],
    nnn_sets: Sequence[np.ndarray],
    dist_mat: np.ndarray,
    ellipsoid_sizes: np.ndarray,
    consensus_threshold: float,
    min_labeled_neighbor_weight: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Iteratively assign ellipsoids supported by high NNN consensus."""
    m = len(densities)
    labels = -np.ones(m, dtype=int)
    confidence = np.zeros(m, dtype=float)

    for lab, c in enumerate(centers):
        labels[int(c)] = lab
        confidence[int(c)] = 1.0

    # Density-descending order follows the DPC principle while requiring
    # explicit local consensus before accepting a propagated label.
    order = np.argsort(-densities, kind="mergesort")

    progress = True
    while progress:
        progress = False
        for idx_raw in order:
            i = int(idx_raw)
            if labels[i] >= 0:
                continue

            votes = _neighbor_vote_scores(
                i, labels, densities, ellipsoid_sizes, nnn_sets, dist_mat
            )
            if not votes:
                continue

            total = float(sum(votes.values()))
            if total <= min_labeled_neighbor_weight:
                continue

            best_lab, best_vote = max(votes.items(), key=lambda kv: kv[1])
            conf = float(best_vote / (total + 1e-12))
            if conf >= consensus_threshold:
                labels[i] = int(best_lab)
                confidence[i] = conf
                progress = True

    return labels, confidence


# =============================================================================
# Ambiguous-ellipsoid detection and fuzzy assignment
# =============================================================================

def identify_ambiguous_ellipsoids(
    propagated_labels: np.ndarray,
    confidence: np.ndarray,
    nnn_sets: Sequence[np.ndarray],
    radii: np.ndarray,
    parent: np.ndarray,
    dist_mat: np.ndarray,
    consensus_threshold: float,
    radius_mad_factor: float,
) -> np.ndarray:
    """Detect unlabeled, weak-consensus, boundary, and sparse ellipsoids."""
    positive_radii = radii[radii > 0]
    if positive_radii.size:
        radius_threshold = float(np.median(positive_radii)) + radius_mad_factor * _safe_mad(positive_radii)
    else:
        radius_threshold = np.inf

    ambiguous = np.zeros(len(propagated_labels), dtype=bool)

    for i in range(len(propagated_labels)):
        if propagated_labels[i] < 0:
            ambiguous[i] = True
            continue

        if confidence[i] < consensus_threshold and confidence[i] > 0:
            ambiguous[i] = True

        neigh_labels = {
            int(propagated_labels[j])
            for j in nnn_sets[i]
            if propagated_labels[j] >= 0
        }
        if len(neigh_labels) > 1:
            ambiguous[i] = True

        if radii[i] > radius_threshold:
            ambiguous[i] = True

        p = int(parent[i])
        if p >= 0 and dist_mat[i, p] > radius_threshold:
            ambiguous[i] = True

    return np.flatnonzero(ambiguous)


def _fuzzy_memberships_for_ellipsoid(
    i: int,
    labels: np.ndarray,
    densities: np.ndarray,
    ellipsoid_sizes: np.ndarray,
    nnn_sets: Sequence[np.ndarray],
    dist_mat: np.ndarray,
    n_clusters: int,
) -> np.ndarray:
    """Compute fuzzy weighted natural-neighbor membership scores.

    The weight transfers the FWNNN-DPC principle to ellipsoids:
    - proximity contributes through 1 / (1 + distance);
    - gamma_ij normalizes the edge relative to neighbor j's own local context;
    - density and ellipsoid size provide a reliability factor.
    """
    memberships = np.zeros(n_clusters, dtype=float)

    for j in nnn_sets[i]:
        lab = int(labels[j])
        if lab < 0 or lab >= n_clusters:
            continue

        w_ij = 1.0 / (1.0 + float(dist_mat[i, j]))
        denom = 0.0
        for l in nnn_sets[j]:
            denom += 1.0 / (1.0 + float(dist_mat[l, j]))
        denom = max(denom, 1e-12)
        gamma_ij = w_ij / denom

        reliability = max(float(densities[j]), 1e-12) * math.sqrt(max(float(ellipsoid_sizes[j]), 1.0))
        memberships[lab] += gamma_ij * w_ij * reliability

    return memberships


def fuzzy_assign_ambiguous_ellipsoids(
    initial_labels: np.ndarray,
    ambiguous_indices: np.ndarray,
    densities: np.ndarray,
    ellipsoid_sizes: np.ndarray,
    nnn_sets: Sequence[np.ndarray],
    dist_mat: np.ndarray,
    centers: Sequence[int],
    n_clusters: int,
    membership_margin: float,
    max_rounds: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Assign ambiguous ellipsoids by iterative fuzzy NNN membership."""
    labels = np.asarray(initial_labels, dtype=int).copy()
    membership_confidence = np.zeros(len(labels), dtype=float)
    ambiguous_set = set(map(int, ambiguous_indices))
    center_set = set(map(int, centers))

    for _ in range(int(max_rounds)):
        changed = False
        proposals: List[Tuple[int, int, float]] = []

        for i in sorted(ambiguous_set):
            if i in center_set:
                continue

            scores = _fuzzy_memberships_for_ellipsoid(
                i,
                labels,
                densities,
                ellipsoid_sizes,
                nnn_sets,
                dist_mat,
                n_clusters,
            )
            if not np.any(scores > 0):
                continue

            order = np.argsort(-scores)
            best = int(order[0])
            second_score = float(scores[order[1]]) if len(order) > 1 else 0.0
            margin = _stable_soft_ratio(float(scores[best]), second_score)

            if margin >= membership_margin:
                proposals.append((i, best, margin))

        # Batch update prevents order-dependent propagation within one round.
        for i, lab, margin in proposals:
            if labels[i] != lab:
                labels[i] = lab
                changed = True
            membership_confidence[i] = max(membership_confidence[i], margin)

        if not changed:
            break

    # Final fallback: any still-unlabeled ellipsoid is assigned to its nearest
    # selected center. This fallback is deterministic and does not use labels.
    centers_list = [int(c) for c in centers]
    for i in np.where(labels < 0)[0]:
        c = centers_list[int(np.argmin(dist_mat[i, centers_list]))]
        labels[i] = labels[c]

    return labels, membership_confidence


# =============================================================================
# Conservative acceptance gate
# =============================================================================

def accept_refinement(
    base_labels: np.ndarray,
    refined_labels: np.ndarray,
    densities: np.ndarray,
    dist_mat: np.ndarray,
    cfg: MethodConfig,
) -> Tuple[bool, Dict[str, float]]:
    """Accept fuzzy refinement only when structural safety conditions hold."""
    base_labels = np.asarray(base_labels, dtype=int)
    refined_labels = np.asarray(refined_labels, dtype=int)

    changed_ratio = float(np.mean(base_labels != refined_labels))
    largest_ratio, smallest_ratio = cluster_size_statistics(refined_labels)

    base_score = internal_cluster_score(
        base_labels,
        densities,
        dist_mat,
        cfg.max_largest_cluster_ratio,
        cfg.min_cluster_ratio,
    )
    refined_score = internal_cluster_score(
        refined_labels,
        densities,
        dist_mat,
        cfg.max_largest_cluster_ratio,
        cfg.min_cluster_ratio,
    )

    same_cluster_count = np.unique(base_labels).size == np.unique(refined_labels).size
    safe = (
        changed_ratio <= cfg.max_changed_ratio
        and largest_ratio <= cfg.max_largest_cluster_ratio
        and smallest_ratio >= cfg.min_cluster_ratio
        and same_cluster_count
        and refined_score + cfg.internal_score_tolerance >= base_score
    )

    diagnostics = {
        "changed_ratio": changed_ratio,
        "largest_cluster_ratio": largest_ratio,
        "smallest_cluster_ratio": smallest_ratio,
        "base_internal_score": float(base_score),
        "refined_internal_score": float(refined_score),
    }
    return bool(safe), diagnostics


# =============================================================================
# Final evaluation utilities
# =============================================================================

def align_labels_hungarian(true_labels: np.ndarray, pred_labels: np.ndarray) -> np.ndarray:
    true_labels = np.asarray(true_labels)
    pred_labels = np.asarray(pred_labels)
    true_classes = np.unique(true_labels)
    pred_classes = np.unique(pred_labels)

    confusion = np.zeros((len(true_classes), len(pred_classes)), dtype=float)
    for i, t in enumerate(true_classes):
        for j, p in enumerate(pred_classes):
            confusion[i, j] = np.sum((true_labels == t) & (pred_labels == p))

    row_ind, col_ind = linear_sum_assignment(-confusion)
    mapping = {pred_classes[j]: true_classes[i] for i, j in zip(row_ind, col_ind)}
    return np.array([mapping.get(p, -1) for p in pred_labels])


# =============================================================================
# Main method
# =============================================================================

def run_fwnn_aqg_ge_dpc(
    data: np.ndarray,
    cfg: MethodConfig,
    true_labels: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    """Run the complete FWNN-AQG-GE-DPC pipeline.

    Parameters
    ----------
    data:
        Input feature matrix of shape (n_samples, n_features).
    cfg:
        Method configuration.
    true_labels:
        Optional labels used only for final external evaluation.

    Returns
    -------
    Dictionary containing predictions, unsupervised diagnostics, runtimes,
    and optional ACC/NMI/ARI values.
    """
    X_raw = np.asarray(data, dtype=float)
    if X_raw.ndim != 2 or X_raw.shape[0] == 0:
        raise ValueError("data must be a non-empty 2D array.")
    if cfg.n_clusters < 1:
        raise ValueError("n_clusters must be at least 1.")

    X = apply_data_scaler(X_raw, cfg.scaler)

    # -------------------------------------------------------------------------
    # Stage 1: quality-gated granular-ellipsoid generation
    # -------------------------------------------------------------------------
    t0 = time.perf_counter()
    ellipsoids = generate_granular_ellipsoids(X, cfg)
    generation_time = time.perf_counter() - t0

    # -------------------------------------------------------------------------
    # Stage 2: ellipsoid geometry, NNN graph, and hybrid density
    # -------------------------------------------------------------------------
    t1 = time.perf_counter()
    dist_mat = ellipsoid_distance_matrix(ellipsoids)
    nnn_sets, natural_eigenvalue, neighbor_order = build_ellipsoid_natural_neighbors(
        dist_mat,
        stable_rounds=cfg.nnn_stable_rounds,
        max_k_factor=cfg.nnn_max_k_factor,
        min_coverage=cfg.nnn_min_coverage,
    )
    radii = natural_neighbor_radii(nnn_sets, dist_mat)

    structural_density = compute_structural_natural_neighbor_density(
        ellipsoids, dist_mat, nnn_sets
    )
    if cfg.use_hybrid_density:
        densities, intrinsic_raw, structural_raw = compute_hybrid_density(
            ellipsoids,
            structural_density,
            blend_mode=cfg.density_blend_mode,
            fixed_intrinsic_weight=cfg.fixed_intrinsic_weight,
        )
    else:
        intrinsic_raw = np.array([e.intrinsic_density for e in ellipsoids], dtype=float)
        structural_raw = structural_density
        densities = _safe_minmax(intrinsic_raw) + 1e-12 * np.arange(len(ellipsoids))

    delta, parent = compute_delta_and_parent(dist_mat, densities)
    gamma = densities * delta
    attribute_time = time.perf_counter() - t1

    # -------------------------------------------------------------------------
    # Stage 3: NNN-aware centers, reliable propagation, fuzzy assignment
    # -------------------------------------------------------------------------
    t2 = time.perf_counter()
    centers = select_centers_nnn_aware(
        densities,
        delta,
        dist_mat,
        nnn_sets,
        n_clusters=cfg.n_clusters,
    )

    base_labels = single_chain_assignment(densities, centers, parent, dist_mat)
    ellipsoid_sizes = np.array([e.n_samples for e in ellipsoids], dtype=float)

    propagated_labels, propagation_confidence = propagate_reliable_labels(
        densities,
        centers,
        nnn_sets,
        dist_mat,
        ellipsoid_sizes,
        consensus_threshold=cfg.consensus_threshold,
        min_labeled_neighbor_weight=cfg.min_labeled_neighbor_weight,
    )

    ambiguous_indices = identify_ambiguous_ellipsoids(
        propagated_labels,
        propagation_confidence,
        nnn_sets,
        radii,
        parent,
        dist_mat,
        consensus_threshold=cfg.consensus_threshold,
        radius_mad_factor=cfg.radius_mad_factor,
    )

    refined_labels, fuzzy_confidence = fuzzy_assign_ambiguous_ellipsoids(
        propagated_labels,
        ambiguous_indices,
        densities,
        ellipsoid_sizes,
        nnn_sets,
        dist_mat,
        centers,
        n_clusters=cfg.n_clusters,
        membership_margin=cfg.fuzzy_membership_margin,
        max_rounds=cfg.fuzzy_max_rounds,
    )

    accepted, acceptance_diagnostics = accept_refinement(
        base_labels,
        refined_labels,
        densities,
        dist_mat,
        cfg,
    )
    final_ellipsoid_labels = refined_labels if accepted else base_labels
    label_mode = "fuzzy_nnn" if accepted else "single_chain_fallback"
    clustering_time = time.perf_counter() - t2

    # -------------------------------------------------------------------------
    # Stage 4: map ellipsoid labels back to original samples
    # -------------------------------------------------------------------------
    pred_labels = np.full(X.shape[0], -1, dtype=int)
    for i, ell in enumerate(ellipsoids):
        pred_labels[ell.indices] = final_ellipsoid_labels[i]
    if np.any(pred_labels < 0):
        raise RuntimeError("Some samples were not assigned a cluster label.")

    total_time = generation_time + attribute_time + clustering_time

    result: Dict[str, object] = {
        "pred_labels": pred_labels,
        "ellipsoid_labels": final_ellipsoid_labels,
        "base_ellipsoid_labels": base_labels,
        "centers": centers,
        "densities": densities,
        "intrinsic_density_raw": intrinsic_raw,
        "structural_density_raw": structural_raw,
        "delta": delta,
        "gamma": gamma,
        "dist_mat": dist_mat,
        "natural_neighbor_sets": nnn_sets,
        "natural_eigenvalue": natural_eigenvalue,
        "neighbor_order": neighbor_order,
        "natural_neighbor_radii": radii,
        "propagation_confidence": propagation_confidence,
        "fuzzy_confidence": fuzzy_confidence,
        "ambiguous_indices": ambiguous_indices,
        "refinement_accepted": accepted,
        "label_mode": label_mode,
        "acceptance_diagnostics": acceptance_diagnostics,
        "n_ellipsoids": len(ellipsoids),
        "generation_time_sec": generation_time,
        "attribute_time_sec": attribute_time,
        "clustering_time_sec": clustering_time,
        "total_time_sec": total_time,
        "ellipsoids": ellipsoids,
    }

    # Ground-truth labels are deliberately accessed only here.
    if true_labels is not None:
        y_true = np.asarray(true_labels)
        if y_true.shape[0] != pred_labels.shape[0]:
            raise ValueError("true_labels length does not match data length.")

        aligned = align_labels_hungarian(y_true, pred_labels)
        result.update(
            {
                "aligned_pred_labels": aligned,
                "acc": float(accuracy_score(y_true, aligned)),
                "nmi": float(normalized_mutual_info_score(y_true, pred_labels)),
                "ari": float(adjusted_rand_score(y_true, pred_labels)),
            }
        )

    return result


# =============================================================================
# File-based runner compatible with the user's existing dataset layout
# =============================================================================

def load_txt_dataset(feature_file: Path, label_file: Path) -> Tuple[np.ndarray, np.ndarray]:
    X = np.loadtxt(feature_file, dtype=float)
    y = np.loadtxt(label_file, dtype=float).astype(int)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"Feature/label length mismatch: {X.shape[0]} vs {y.shape[0]}")
    return X, y


def run_dataset_from_files(
    feature_file: Path,
    label_file: Path,
    cfg: MethodConfig,
    dataset_name: str = "dataset",
    verbose: bool = True,
) -> Dict[str, object]:
    X, y = load_txt_dataset(feature_file, label_file)
    result = run_fwnn_aqg_ge_dpc(X, cfg, true_labels=y)

    if verbose:
        diag = result["acceptance_diagnostics"]
        print("=" * 100)
        print(f"FWNN-AQG-GE-DPC | Dataset: {dataset_name}")
        print(f"Shape                 : n={X.shape[0]}, d={X.shape[1]}")
        print(f"Ellipsoids            : {result['n_ellipsoids']}")
        print(f"Natural eigenvalue    : {result['natural_eigenvalue']}")
        print(f"Centers               : {result['centers']}")
        print(f"Ambiguous ellipsoids  : {len(result['ambiguous_indices'])}")
        print(f"Refinement accepted   : {result['refinement_accepted']}")
        print(f"Final label mode      : {result['label_mode']}")
        print(
            f"ACC={result['acc']:.3f}, NMI={result['nmi']:.3f}, "
            f"ARI={result['ari']:.3f}"
        )
        print(
            "Time(ms): generation={:.3f}, attributes={:.3f}, clustering={:.3f}, total={:.3f}".format(
                result["generation_time_sec"] * 1000.0,
                result["attribute_time_sec"] * 1000.0,
                result["clustering_time_sec"] * 1000.0,
                result["total_time_sec"] * 1000.0,
            )
        )
        print(
            "Acceptance diagnostics: changed={:.3f}, largest={:.3f}, smallest={:.3f}, "
            "base_score={:.6f}, refined_score={:.6f}".format(
                diag["changed_ratio"],
                diag["largest_cluster_ratio"],
                diag["smallest_cluster_ratio"],
                diag["base_internal_score"],
                diag["refined_internal_score"],
            )
        )
        print("=" * 100)

    return result


# =============================================================================
# Dataset registry and fixed benchmark configurations
# =============================================================================

def get_default_dataset_registry(base_dir: Path) -> Dict[str, Tuple[Path, Path]]:
    """Return the same 12-dataset file layout used by the AQG-GE-DPC code.

    Parameters
    ----------
    base_dir:
        Project root containing ``real_dataset_and_label`` and ``dataset``.
    """
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


def get_default_dataset_names() -> List[str]:
    """Return the benchmark execution order used in the KSE experiment."""
    return [
        "iris", "seed", "segment_3", "landsat_2",
        "msplice_2", "rice", "banknote", "htru2",
        "breast_cancer", "hcv_data", "dry_bean", "rice_cammeo",
    ]


# Fixed benchmark settings inherited from the supplied AQG-GE-DPC runner.
# They are not searched using ground-truth labels during this execution.
BEST_DATASET_CONFIGS: Dict[str, Dict[str, object]] = {
    "iris":          {"scaler": "none",     "n_clusters": 3, "split_factor": 1.0, "outlier_quality_t": 1.3},
    "seed":          {"scaler": "none",     "n_clusters": 3, "split_factor": 1.0, "outlier_quality_t": 1.3},
    "segment_3":     {"scaler": "robust",   "n_clusters": 8, "split_factor": 0.8, "outlier_quality_t": 1.7},
    "landsat_2":     {"scaler": "none",     "n_clusters": 5, "split_factor": 1.0, "outlier_quality_t": 2.0},
    "msplice_2":     {"scaler": "standard", "n_clusters": 3, "split_factor": 0.6, "outlier_quality_t": 2.0},
    "rice":          {"scaler": "standard", "n_clusters": 2, "split_factor": 0.6, "outlier_quality_t": 1.5},
    "banknote":      {"scaler": "standard", "n_clusters": 2, "split_factor": 0.8, "outlier_quality_t": 1.5},
    "htru2":         {"scaler": "none",     "n_clusters": 2, "split_factor": 1.0, "outlier_quality_t": 2.0},
    "breast_cancer": {"scaler": "none",     "n_clusters": 2, "split_factor": 0.6, "outlier_quality_t": 1.5},
    "hcv_data":      {"scaler": "robust",   "n_clusters": 2, "split_factor": 1.0, "outlier_quality_t": 2.0},
    "dry_bean":      {"scaler": "standard", "n_clusters": 7, "split_factor": 0.6, "outlier_quality_t": 1.3},
    "rice_cammeo":   {"scaler": "standard", "n_clusters": 2, "split_factor": 0.6, "outlier_quality_t": 1.3},
}


def build_dataset_config(dataset_name: str) -> MethodConfig:
    """Create a MethodConfig for one benchmark dataset."""
    key = dataset_name.lower().strip()
    if key not in BEST_DATASET_CONFIGS:
        raise KeyError(f"Missing configuration for dataset '{dataset_name}'.")

    values = BEST_DATASET_CONFIGS[key]
    return MethodConfig(
        epsilon=1e-6,
        scaler=str(values["scaler"]),
        n_clusters=int(values["n_clusters"]),
        split_factor=float(values["split_factor"]),
        outlier_quality_t=float(values["outlier_quality_t"]),
        consensus_threshold=0.70,
        fuzzy_membership_margin=0.15,
        max_changed_ratio=0.35,
        max_largest_cluster_ratio=0.85,
    )


def find_project_base_dir(script_path: Path) -> Path:
    """Locate the project root without requiring manual path editing.

    The search checks the script directory and several parents. The first
    directory containing at least the Iris feature and label files is used.
    """
    script_dir = script_path.resolve().parent
    candidates = [script_dir] + list(script_dir.parents[:4])

    for candidate in candidates:
        registry = get_default_dataset_registry(candidate)
        iris_feature, iris_label = registry["iris"]
        if iris_feature.exists() and iris_label.exists():
            return candidate

    # Preserve the original KSE layout assumption as the diagnostic fallback.
    return script_dir.parent


def run_named_dataset(
    dataset_name: str,
    base_dir: Path,
    verbose: bool = True,
) -> Dict[str, object]:
    """Run one registered dataset using its fixed benchmark configuration."""
    key = dataset_name.lower().strip()
    registry = get_default_dataset_registry(base_dir)
    if key not in registry:
        raise KeyError(f"Unknown dataset '{dataset_name}'.")

    feature_file, label_file = registry[key]
    if not feature_file.exists() or not label_file.exists():
        raise FileNotFoundError(
            f"Dataset files not found for '{key}'.\n"
            f"Feature: {feature_file}\n"
            f"Label  : {label_file}"
        )

    cfg = build_dataset_config(key)
    result = run_dataset_from_files(
        feature_file=feature_file,
        label_file=label_file,
        cfg=cfg,
        dataset_name=key,
        verbose=verbose,
    )
    result.update({
        "dataset": key,
        "scaler": cfg.scaler,
        "n_clusters": cfg.n_clusters,
        "split_factor": cfg.split_factor,
        "outlier_quality_t": cfg.outlier_quality_t,
    })
    return result


def run_all_datasets(
    base_dir: Path,
    dataset_names: Optional[Sequence[str]] = None,
    verbose_each_dataset: bool = False,
) -> Dict[str, Dict[str, object]]:
    """Run all 12 datasets and print one consolidated result table."""
    names = list(dataset_names) if dataset_names is not None else get_default_dataset_names()
    results: Dict[str, Dict[str, object]] = {}

    for name in names:
        try:
            results[name] = run_named_dataset(
                dataset_name=name,
                base_dir=base_dir,
                verbose=verbose_each_dataset,
            )
        except Exception as exc:
            results[name] = {"error": str(exc)}
            print(f"[ERROR] {name}: {exc}")

    width = 164
    print("\n" + "=" * width)
    print("SUMMARY - FWNN-AQG-GE-DPC | 12 DATASETS")
    print(f"Project base directory: {base_dir}")
    print("=" * width)
    print(
        f"{'Dataset':<18} {'ACC':>7} {'NMI':>7} {'ARI':>7} "
        f"{'Scaler':>10} {'k':>4} {'out_t':>7} {'split':>7} "
        f"{'Ells':>7} {'Lambda':>7} {'Ambig':>7} {'Accept':>8} "
        f"{'Mode':>22} {'Time(ms)':>11}"
    )
    print("-" * width)

    valid_results: List[Dict[str, object]] = []
    for name in names:
        res = results[name]
        if "error" in res:
            print(f"{name:<18} ERROR: {res['error']}")
            continue

        valid_results.append(res)
        print(
            f"{name:<18} {res['acc']:>7.3f} {res['nmi']:>7.3f} {res['ari']:>7.3f} "
            f"{res['scaler']:>10} {res['n_clusters']:>4d} "
            f"{res['outlier_quality_t']:>7.2f} {res['split_factor']:>7.2f} "
            f"{res['n_ellipsoids']:>7d} {res['natural_eigenvalue']:>7d} "
            f"{len(res['ambiguous_indices']):>7d} {str(res['refinement_accepted']):>8} "
            f"{res['label_mode']:>22} {res['total_time_sec'] * 1000.0:>11.3f}"
        )

    print("=" * width)
    if valid_results:
        print(
            "AVERAGE".ljust(18)
            + f" {np.mean([r['acc'] for r in valid_results]):>7.3f}"
            + f" {np.mean([r['nmi'] for r in valid_results]):>7.3f}"
            + f" {np.mean([r['ari'] for r in valid_results]):>7.3f}"
            + " " * 64
            + f"{np.mean([r['total_time_sec'] * 1000.0 for r in valid_results]):>11.3f}"
        )
        print("=" * width)

    return results


# =============================================================================
# Program entry point
# =============================================================================

if __name__ == "__main__":
    BASE_DIR = find_project_base_dir(Path(__file__))

    # Run one dataset for debugging:
    # run_named_dataset("iris", base_dir=BASE_DIR, verbose=True)

    # Default behavior: run the complete 12-dataset benchmark in one command.
    run_all_datasets(
        base_dir=BASE_DIR,
        dataset_names=None,
        verbose_each_dataset=False,
    )
