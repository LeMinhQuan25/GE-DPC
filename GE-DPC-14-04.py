import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)
from sklearn.utils.extmath import randomized_svd


# ============================================================
# Pure-development GE-DPC
# ------------------------------------------------------------
# Core principles kept from original GE-DPC pipeline:
#   1) generate granular ellipsoids
#   2) compute ellipsoid attributes
#   3) compute inter-ellipsoid distances
#   4) DPC center selection on ellipsoids
#   5) propagate labels and map back to points
#
# Improvements added INSIDE the same pipeline:
#   A) adaptive approximation inside each ellipsoid
#   B) stronger minimum leaf size from safe split stage
#   C) density clipping for degenerate/small ellipsoids
#   D) pruning before exact ellipse-distance computation
#
# This file avoids dataset-level hybrid rule.
# All datasets follow one unified pipeline.
# ============================================================


class AdaptiveEllipsoid:
    def __init__(
        self,
        data: np.ndarray,
        indices: np.ndarray,
        epsilon: float = 1e-6,
        randomized_min_samples: int = 64,
        randomized_min_dim: int = 24,
        svd_oversamples: int = 8,
        svd_n_iter: int = 2,
        small_ellipsoid_size: int = 3,
        tiny_var_threshold: float = 1e-12,
        covariance_shrinkage: float = 1e-6,
    ):
        self.data = np.asarray(data, dtype=float)
        self.indices = np.asarray(indices, dtype=int)
        self.epsilon = float(epsilon)
        self.randomized_min_samples = int(randomized_min_samples)
        self.randomized_min_dim = int(randomized_min_dim)
        self.svd_oversamples = int(svd_oversamples)
        self.svd_n_iter = int(svd_n_iter)
        self.small_ellipsoid_size = int(small_ellipsoid_size)
        self.tiny_var_threshold = float(tiny_var_threshold)
        self.covariance_shrinkage = float(covariance_shrinkage)

        if self.data.ndim != 2 or len(self.data) == 0:
            raise ValueError("data must be a non-empty 2D array")

        self.n_samples, self.dim = self.data.shape
        self.center = np.mean(self.data, axis=0)
        self.Xc = self.data - self.center

        self.mode_used = "unknown"
        self.rank = 0
        self.approx_rank_used = 0
        self.is_degenerate = False
        self.is_small = self.n_samples <= self.small_ellipsoid_size

        # Shape representation:
        # H ~= U diag(lambda) U^T + lambda_perp * (I - UU^T)
        self.U = np.zeros((self.dim, 0), dtype=float)
        self.shape_eigs = np.full(self.dim, self.epsilon, dtype=float)
        self.parallel_eigs = np.zeros((0,), dtype=float)
        self.perp_eig = self.epsilon
        self.inv_parallel = np.zeros((0,), dtype=float)

        self._fit_shape_adaptive()
        self.rho = self._compute_rho()
        self.lengths = self._compute_lengths()
        self.major_axis_endpoints = self._compute_major_axis_endpoints()

    def _auto_rank(self) -> int:
        rank_target = max(6, int(np.ceil(2.0 * np.sqrt(self.dim))))
        return int(min(self.dim, self.n_samples, rank_target))

    def _fit_shape_adaptive(self) -> None:
        # Case 1: degenerate / too small ellipsoid
        if self.n_samples <= 1 or np.allclose(self.Xc, 0.0):
            self._set_degenerate("degenerate")
            return

        if self.n_samples <= self.small_ellipsoid_size:
            self._fit_small_covariance()
            return

        # Case 2: medium ellipsoid -> cheap covariance eig
        # Case 3: larger ellipsoid -> randomized SVD when truly worthwhile
        if self.n_samples >= self.randomized_min_samples and self.dim >= self.randomized_min_dim:
            self._fit_randomized_low_rank()
        else:
            self._fit_medium_covariance()

    def _set_degenerate(self, mode: str) -> None:
        self.mode_used = mode
        self.rank = 0
        self.approx_rank_used = 0
        self.is_degenerate = True
        self.U = np.zeros((self.dim, 0), dtype=float)
        self.parallel_eigs = np.zeros((0,), dtype=float)
        self.inv_parallel = np.zeros((0,), dtype=float)
        self.perp_eig = self.epsilon
        self.shape_eigs = np.full(self.dim, self.epsilon, dtype=float)

    def _fit_small_covariance(self) -> None:
        # Cheap and stable path for small ellipsoids.
        # We keep full covariance in small dimension space.
        cov = np.cov(self.Xc, rowvar=False, bias=True)
        if np.ndim(cov) == 0:
            cov = np.array([[float(cov)]], dtype=float)
        cov = np.atleast_2d(cov)
        if cov.shape != (self.dim, self.dim):
            cov = np.diag(np.var(self.Xc, axis=0))

        tr = float(np.trace(cov)) / max(self.dim, 1)
        cov = cov + (self.covariance_shrinkage * max(tr, 1.0)) * np.eye(self.dim)

        evals, evecs = np.linalg.eigh(cov)
        evals = np.maximum(evals, self.epsilon)
        order = np.argsort(evals)[::-1]
        evals = evals[order]
        evecs = evecs[:, order]

        keep = evals > self.tiny_var_threshold
        kept_evals = evals[keep]
        kept_evecs = evecs[:, keep]

        if len(kept_evals) == 0:
            self._set_degenerate("small_degenerate")
            return

        self.mode_used = "small_cov_eig"
        self.rank = len(kept_evals)
        self.approx_rank_used = self.rank
        self.U = kept_evecs
        self.parallel_eigs = kept_evals
        self.inv_parallel = 1.0 / np.maximum(self.parallel_eigs, self.epsilon)
        self.perp_eig = max(float(np.min(self.parallel_eigs)), self.epsilon)
        self.shape_eigs = np.maximum(evals, self.epsilon)
        self.is_degenerate = self.rank == 0

    def _fit_medium_covariance(self) -> None:
        cov = np.cov(self.Xc, rowvar=False, bias=True)
        if np.ndim(cov) == 0:
            cov = np.array([[float(cov)]], dtype=float)
        cov = np.atleast_2d(cov)
        if cov.shape != (self.dim, self.dim):
            cov = np.diag(np.var(self.Xc, axis=0))

        tr = float(np.trace(cov)) / max(self.dim, 1)
        cov = cov + (self.covariance_shrinkage * max(tr, 1.0)) * np.eye(self.dim)

        evals, evecs = np.linalg.eigh(cov)
        evals = np.maximum(evals, self.epsilon)
        order = np.argsort(evals)[::-1]
        evals = evals[order]
        evecs = evecs[:, order]

        target_rank = self._auto_rank()
        keep = evals > self.tiny_var_threshold
        keep_idx = np.where(keep)[0][:target_rank]

        if len(keep_idx) == 0:
            self._set_degenerate("medium_degenerate")
            return

        self.mode_used = "medium_cov_eig"
        self.rank = len(keep_idx)
        self.approx_rank_used = self.rank
        self.U = evecs[:, keep_idx]
        self.parallel_eigs = evals[keep_idx]
        self.inv_parallel = 1.0 / np.maximum(self.parallel_eigs, self.epsilon)

        dropped = evals[self.rank:]
        if len(dropped) > 0:
            self.perp_eig = max(float(np.mean(dropped)), self.epsilon)
        else:
            self.perp_eig = max(float(np.min(self.parallel_eigs)), self.epsilon)

        self.shape_eigs = evals
        self.is_degenerate = self.rank == 0

    def _fit_randomized_low_rank(self) -> None:
        target_rank = self._auto_rank()
        target_rank = min(target_rank, min(self.Xc.shape))
        if target_rank <= 0:
            self._set_degenerate("randomized_degenerate")
            return

        try:
            _, s, vt = randomized_svd(
                self.Xc,
                n_components=target_rank,
                n_oversamples=min(self.svd_oversamples, max(2, self.dim)),
                n_iter=self.svd_n_iter,
                random_state=0,
            )
            cov_eigs = (s ** 2) / max(self.n_samples, 1)
            cov_eigs = np.maximum(cov_eigs, self.epsilon)
            keep = cov_eigs > self.tiny_var_threshold

            if not np.any(keep):
                self._set_degenerate("randomized_degenerate")
                return

            self.mode_used = "randomized_svd"
            self.U = vt[keep].T
            self.parallel_eigs = cov_eigs[keep]
            self.rank = len(self.parallel_eigs)
            self.approx_rank_used = int(target_rank)
            self.inv_parallel = 1.0 / np.maximum(self.parallel_eigs, self.epsilon)

            residual_energy = np.var(self.Xc, axis=0).sum() - self.parallel_eigs.sum()
            residual_dims = max(self.dim - self.rank, 1)
            self.perp_eig = max(float(residual_energy / residual_dims), self.epsilon)

            tail = np.full(max(0, self.dim - self.rank), self.perp_eig, dtype=float)
            self.shape_eigs = np.concatenate([self.parallel_eigs, tail])
            self.is_degenerate = self.rank == 0
        except Exception:
            self._fit_medium_covariance()
            self.mode_used = "fallback_medium_cov_eig"

    def mahal_sq_points(self, points: np.ndarray) -> np.ndarray:
        X = np.asarray(points, dtype=float) - self.center
        if X.ndim == 1:
            X = X[None, :]

        if self.rank == 0:
            return np.sum(X * X, axis=1) / max(self.perp_eig, self.epsilon)

        proj = X @ self.U
        parallel_sq = np.sum((proj ** 2) * self.inv_parallel[None, :], axis=1)

        total_sq = np.sum(X * X, axis=1)
        proj_sq = np.sum(proj * proj, axis=1)
        perp_energy = np.maximum(total_sq - proj_sq, 0.0)
        perp_sq = perp_energy / max(self.perp_eig, self.epsilon)
        return parallel_sq + perp_sq

    def _compute_rho(self) -> float:
        mahal_sq = self.mahal_sq_points(self.data)
        mahal_sq = np.maximum(mahal_sq, 0.0)
        return float(np.sqrt(np.max(mahal_sq)))

    def _compute_lengths(self) -> np.ndarray:
        return self.rho * np.sqrt(np.maximum(self.shape_eigs, self.epsilon))

    def _compute_major_axis_endpoints(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.n_samples <= 1:
            return self.center, self.center

        if self.rank > 0:
            axis = self.U[:, 0]
            scores = self.Xc @ axis
            i_min = int(np.argmin(scores))
            i_max = int(np.argmax(scores))
            return self.data[i_min], self.data[i_max]

        center_distances = np.linalg.norm(self.data - self.center, axis=1)
        p1 = self.data[np.argmin(center_distances)]
        p2 = self.data[np.argmax(np.linalg.norm(self.data - p1, axis=1))]
        p3 = self.data[np.argmax(np.linalg.norm(self.data - p2, axis=1))]
        return p2, p3

    @property
    def outer_radius(self) -> float:
        return float(np.max(self.lengths)) if len(self.lengths) else 0.0


# ------------------------------
# Metrics and helper functions
# ------------------------------

def get_num(ellipsoid: AdaptiveEllipsoid) -> int:
    return ellipsoid.n_samples


def calculate_ellipsoid_density(
    ellipsoid: AdaptiveEllipsoid,
    density_small_penalty: float = 0.2,
    density_degenerate_penalty: float = 0.05,
    min_center_size: int = 4,
) -> float:
    n_samples = ellipsoid.n_samples
    axes_sum = float(np.sum(ellipsoid.lengths))
    axes_sum = max(axes_sum, 1e-12)

    mahal_sq = ellipsoid.mahal_sq_points(ellipsoid.data)
    total_mahal = float(np.sum(np.sqrt(np.maximum(mahal_sq, 0.0))))
    total_mahal = max(total_mahal, 1e-12)

    base_density = float((n_samples ** 2) / (axes_sum * total_mahal))

    # Density clipping / penalty for unreliable ellipsoids
    penalty = 1.0
    if ellipsoid.is_degenerate:
        penalty *= density_degenerate_penalty
    if ellipsoid.n_samples < min_center_size:
        penalty *= density_small_penalty

    # Additional stability: forbid density explosion from tiny ellipsoids
    effective_cap = max(1.0, float(n_samples))
    clipped_density = min(base_density, base_density if n_samples >= min_center_size else effective_cap)
    return float(clipped_density * penalty)


# ------------------------------
# Splitting
# ------------------------------

def build_ellipsoid(data: np.ndarray, indices: np.ndarray, ellipsoid_kwargs: Dict) -> AdaptiveEllipsoid:
    return AdaptiveEllipsoid(data, indices, **ellipsoid_kwargs)


def split_one_ellipsoid(
    ellipsoid: AdaptiveEllipsoid,
    ellipsoid_kwargs: Dict,
    min_leaf_size: int,
) -> List[AdaptiveEllipsoid]:
    if ellipsoid.n_samples <= max(1, min_leaf_size):
        return [ellipsoid]

    data = ellipsoid.data
    indices = ellipsoid.indices
    p1, p2 = ellipsoid.major_axis_endpoints

    d1 = np.linalg.norm(data - p1, axis=1)
    d2 = np.linalg.norm(data - p2, axis=1)
    mask1 = d1 < d2
    mask2 = ~mask1

    c1, c2 = data[mask1], data[mask2]
    i1, i2 = indices[mask1], indices[mask2]

    if len(c1) < min_leaf_size or len(c2) < min_leaf_size:
        return [ellipsoid]

    ell1 = build_ellipsoid(c1, i1, ellipsoid_kwargs)
    ell2 = build_ellipsoid(c2, i2, ellipsoid_kwargs)

    dist1_sq = ell1.mahal_sq_points(data)
    dist2_sq = ell2.mahal_sq_points(data)
    mask1 = dist1_sq < dist2_sq
    mask2 = ~mask1

    c1, c2 = data[mask1], data[mask2]
    i1, i2 = indices[mask1], indices[mask2]

    if len(c1) < min_leaf_size or len(c2) < min_leaf_size:
        return [ellipsoid]

    ell1 = build_ellipsoid(c1, i1, ellipsoid_kwargs)
    ell2 = build_ellipsoid(c2, i2, ellipsoid_kwargs)
    return [ell1, ell2]


def safe_split_stage(
    root: AdaptiveEllipsoid,
    split_threshold: int,
    ellipsoid_kwargs: Dict,
    min_leaf_size: int,
) -> List[AdaptiveEllipsoid]:
    ellipsoid_list = [root]
    split_iter = 0

    while True:
        split_iter += 1
        before = len(ellipsoid_list)
        new_list: List[AdaptiveEllipsoid] = []

        for ell in ellipsoid_list:
            if get_num(ell) < split_threshold:
                new_list.append(ell)
            else:
                new_list.extend(split_one_ellipsoid(ell, ellipsoid_kwargs, min_leaf_size=min_leaf_size))

        ellipsoid_list = new_list
        after = len(ellipsoid_list)

        print(
            f"Ellipsoid count after safe split iteration {split_iter} "
            f"(Số lượng ellipsoid sau lần phân tách an toàn thứ {split_iter}): {after}"
        )

        if after == before:
            break

    return ellipsoid_list


def recursive_split_outlier_detection(
    initial_ellipsoids: List[AdaptiveEllipsoid],
    data: np.ndarray,
    t: float = 2.0,
    max_iterations: int = 10,
    ellipsoid_kwargs: Optional[Dict] = None,
    min_leaf_size: int = 4,
) -> List[AdaptiveEllipsoid]:
    ellipsoid_list = initial_ellipsoids.copy()
    ellipsoid_kwargs = ellipsoid_kwargs or {}

    for _ in range(max_iterations):
        if not ellipsoid_list:
            break

        axes_sums = np.array([np.sum(ell.lengths) for ell in ellipsoid_list], dtype=float)
        axes_sum_avg = float(np.mean(axes_sums)) if len(axes_sums) > 0 else 0.0

        outliers = [ell for ell in ellipsoid_list if np.sum(ell.lengths) > 2.0 * axes_sum_avg]
        if not outliers:
            break

        kept = [ell for ell in ellipsoid_list if ell not in outliers]
        replaced: List[AdaptiveEllipsoid] = []

        for ell in outliers:
            children = split_one_ellipsoid(ell, ellipsoid_kwargs, min_leaf_size=min_leaf_size)
            if len(children) != 2:
                replaced.append(ell)
                continue

            if any(ch.n_samples < min_leaf_size for ch in children):
                replaced.append(ell)
                continue

            parent_density = calculate_ellipsoid_density(ell)
            child_density_sum = sum(calculate_ellipsoid_density(ch) for ch in children)

            # controlled outlier split
            if child_density_sum > t * parent_density and not any(ch.is_degenerate for ch in children):
                replaced.extend(children)
            else:
                replaced.append(ell)

        ellipsoid_list = kept + replaced

    return ellipsoid_list


# ------------------------------
# Distance with pruning
# ------------------------------

def _pair_distance_fast(ell_i: AdaptiveEllipsoid, ell_j: AdaptiveEllipsoid) -> float:
    diff = ell_i.center - ell_j.center
    eps = max(ell_i.epsilon, ell_j.epsilon)

    mats = []
    if ell_i.rank > 0:
        mats.append(ell_i.U * np.sqrt(0.5 * ell_i.parallel_eigs)[None, :])
    if ell_j.rank > 0:
        mats.append(ell_j.U * np.sqrt(0.5 * ell_j.parallel_eigs)[None, :])

    if not mats:
        return float(np.linalg.norm(diff) / np.sqrt(eps))

    A = np.concatenate(mats, axis=1)
    At_diff = A.T @ diff
    M = np.eye(A.shape[1]) + (A.T @ A) / eps

    try:
        z = np.linalg.solve(M, At_diff)
    except np.linalg.LinAlgError:
        z = np.linalg.pinv(M) @ At_diff

    quad = (diff @ diff) / eps - (At_diff @ z) / (eps ** 2)
    return float(np.sqrt(max(quad, 0.0)))


def ellipse_distance_pruned(
    ellipsoid_list: List[AdaptiveEllipsoid],
    prune_margin: float = 3.0,
) -> Tuple[np.ndarray, Dict[str, int]]:
    n = len(ellipsoid_list)
    dist_mat = np.full((n, n), np.inf, dtype=float)
    np.fill_diagonal(dist_mat, 0.0)

    exact_pairs = 0
    pruned_pairs = 0

    centers = np.array([ell.center for ell in ellipsoid_list], dtype=float)
    outer_radii = np.array([ell.outer_radius for ell in ellipsoid_list], dtype=float)

    for i in range(n):
        for j in range(i + 1, n):
            center_dist = float(np.linalg.norm(centers[i] - centers[j]))
            bound = outer_radii[i] + outer_radii[j]

            # Fast pruning: if clearly far apart, use cheap center-based bound instead
            if center_dist > prune_margin * max(bound, 1e-12):
                dist_mat[i, j] = dist_mat[j, i] = center_dist
                pruned_pairs += 1
                continue

            d = _pair_distance_fast(ellipsoid_list[i], ellipsoid_list[j])
            dist_mat[i, j] = dist_mat[j, i] = d
            exact_pairs += 1

    stats = {
        "total_pairs": n * (n - 1) // 2,
        "exact_pairs": exact_pairs,
        "pruned_pairs": pruned_pairs,
    }
    return dist_mat, stats


# ------------------------------
# DPC utilities
# ------------------------------

def ellipse_min_dist(dist_mat: np.ndarray, densities: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    densities = np.asarray(densities, dtype=float)
    order = np.argsort(-densities)

    min_dists = np.zeros(len(densities), dtype=float)
    nearest = -np.ones(len(densities), dtype=int)

    for i in order[1:]:
        mask = densities > densities[i]
        if np.any(mask):
            cand = np.where(mask)[0]
            idx_local = np.argmin(dist_mat[i, cand])
            nearest[i] = cand[idx_local]
            min_dists[i] = dist_mat[i, nearest[i]]
        else:
            min_dists[i] = np.max(dist_mat[i][np.isfinite(dist_mat[i])])

    if len(order) > 0:
        finite_vals = min_dists[np.isfinite(min_dists)]
        min_dists[order[0]] = np.max(finite_vals) if len(finite_vals) > 0 else 0.0

    return min_dists, nearest


def auto_select_centers_robust(
    ellipsoid_list: List[AdaptiveEllipsoid],
    densities: np.ndarray,
    min_dists: np.ndarray,
    mode: str = "knee",
    top_k: Optional[int] = None,
    min_centers: int = 1,
    max_centers: Optional[int] = None,
    min_center_size: int = 4,
) -> List[int]:
    densities = np.asarray(densities, dtype=float)
    min_dists = np.asarray(min_dists, dtype=float)
    n = len(densities)
    if n == 0:
        return []

    valid = np.array(
        [
            (ell.n_samples >= min_center_size) and (not ell.is_degenerate)
            for ell in ellipsoid_list
        ],
        dtype=bool,
    )

    gamma = densities * min_dists
    safe_gamma = gamma.copy()
    safe_gamma[~valid] = -np.inf

    order = np.argsort(-safe_gamma)
    valid_order = [idx for idx in order if np.isfinite(safe_gamma[idx])]

    if not valid_order:
        order = np.argsort(-gamma)
        k = 1 if top_k is None else max(1, min(top_k, n))
        return order[:k].tolist()

    if top_k is not None:
        k = int(max(1, min(top_k, len(valid_order))))
        return valid_order[:k]

    g = np.array([safe_gamma[i] for i in valid_order], dtype=float)
    m = len(g)

    if m == 1 or np.allclose(g, g[0]):
        k = min_centers if max_centers is None else min(min_centers, max_centers)
        return valid_order[:k]

    if mode == "threshold":
        thr = np.mean(g) + np.std(g)
        centers = [valid_order[i] for i in range(m) if g[i] > thr]
        if len(centers) < min_centers:
            centers = valid_order[:min_centers]
        if max_centers is not None:
            centers = centers[:max_centers]
        return centers

    x = np.arange(m, dtype=float)
    x_norm = x / max(m - 1, 1)
    y_norm = (g - g.min()) / max(g.max() - g.min(), 1e-12)

    p1 = np.array([x_norm[0], y_norm[0]], dtype=float)
    p2 = np.array([x_norm[-1], y_norm[-1]], dtype=float)
    line_vec = p2 - p1
    line_norm = np.linalg.norm(line_vec)

    if line_norm < 1e-12:
        knee_idx = min_centers - 1
    else:
        dists = [
            abs(np.cross(line_vec, np.array([xi, yi]) - p1)) / line_norm
            for xi, yi in zip(x_norm, y_norm)
        ]
        knee_idx = int(np.argmax(dists))

    k = max(min_centers, knee_idx + 1)
    if max_centers is not None:
        k = min(k, max_centers)

    return valid_order[:min(k, m)]


def ellipse_cluster(densities: np.ndarray, centers: List[int], nearest: np.ndarray) -> np.ndarray:
    labels = -np.ones(len(densities), dtype=int)

    for i, c in enumerate(centers):
        labels[c] = i

    order = np.argsort(-np.asarray(densities))
    for idx in order:
        if labels[idx] == -1 and nearest[idx] != -1:
            labels[idx] = labels[nearest[idx]]

    if np.any(labels == -1):
        if len(centers) == 0:
            labels[labels == -1] = 0
        else:
            labels[labels == -1] = len(centers)

    return labels


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


# ------------------------------
# Reporting helpers
# ------------------------------

def summarize_modes(ellipsoid_list: List[AdaptiveEllipsoid]) -> None:
    if not ellipsoid_list:
        return

    ranks = [ell.approx_rank_used for ell in ellipsoid_list]
    modes: Dict[str, int] = {}
    for ell in ellipsoid_list:
        modes[ell.mode_used] = modes.get(ell.mode_used, 0) + 1

    print("Approximation summary (Tóm tắt xấp xỉ):")
    print("- Adaptive mode inside each ellipsoid: degenerate / cheap covariance eig / randomized SVD")
    print(f"- Rank min / avg / max: {np.min(ranks)} / {np.mean(ranks):.2f} / {np.max(ranks)}")
    print(f"- Modes used: {modes}")


def describe_center_strategy(auto_center_mode, auto_center_k, min_centers, max_centers):
    if auto_center_k is not None:
        return f"robust top-k mode: top_k={auto_center_k} with degenerate/small-center filtering"
    return f"robust auto mode: mode={auto_center_mode}, min_centers={min_centers}, max_centers={max_centers}"


# ------------------------------
# Main runner
# ------------------------------

def run_ge_dpc_pure_development(
    feature_file,
    label_file,
    epsilon: float = 1e-6,
    outlier_t: float = 2.0,
    auto_center_mode: str = "knee",
    auto_center_k: Optional[int] = None,
    min_centers: int = 1,
    max_centers: Optional[int] = None,
    min_leaf_size: Optional[int] = None,
    prune_margin: float = 3.0,
    min_center_size: int = 4,
):
    data = np.loadtxt(feature_file, dtype=float)
    true_labels = np.loadtxt(label_file, dtype=float).astype(int)

    n, d = data.shape
    if min_leaf_size is None:
        min_leaf_size = max(4, int(np.ceil(np.sqrt(n) * 0.08)))

    split_threshold = int(np.ceil(np.sqrt(n)))

    ellipsoid_kwargs = dict(
        epsilon=epsilon,
        randomized_min_samples=64,
        randomized_min_dim=24,
        svd_oversamples=8,
        svd_n_iter=2,
        small_ellipsoid_size=3,
        tiny_var_threshold=1e-12,
        covariance_shrinkage=1e-6,
    )

    t_gen_start = time.time()
    root = build_ellipsoid(data, np.arange(n, dtype=int), ellipsoid_kwargs)

    print("Initial ellipsoid count (Số lượng ellipsoid ban đầu): 1")
    print(f"Input data shape (Kích thước dữ liệu đầu vào): n={n}, d={d}")
    print("Approximation mode: pure-development adaptive ellipsoid fitting")
    print("Adaptive rule: degenerate/small -> cheap covariance eig -> randomized SVD for sufficiently complex ellipsoids")
    print(f"Safe split threshold: ceil(sqrt(n)) = {split_threshold}")
    print(f"Minimum leaf size: {min_leaf_size}")
    print(f"Distance pruning margin: {prune_margin}")

    ellipsoid_list = safe_split_stage(
        root,
        split_threshold=split_threshold,
        ellipsoid_kwargs=ellipsoid_kwargs,
        min_leaf_size=min_leaf_size,
    )
    print(f"Total ellipsoid count after safe splitting (Tổng số ellipsoid sau phân tách an toàn): {len(ellipsoid_list)}")

    ellipsoid_list = recursive_split_outlier_detection(
        ellipsoid_list,
        data,
        t=outlier_t,
        max_iterations=10,
        ellipsoid_kwargs=ellipsoid_kwargs,
        min_leaf_size=min_leaf_size,
    )

    print(
        f"Total ellipsoid count after outlier-detection splitting "
        f"(Tổng số ellipsoid sau phân tách bằng phát hiện ngoại lệ): {len(ellipsoid_list)}"
    )
    print(f"A total of {len(ellipsoid_list)} ellipsoids were generated (Tổng cộng đã tạo {len(ellipsoid_list)} ellipsoid)")
    summarize_modes(ellipsoid_list)

    t_gen_end = time.time()
    time_gen = t_gen_end - t_gen_start

    t_attr_start = time.time()
    densities = np.array(
        [
            calculate_ellipsoid_density(
                ell,
                density_small_penalty=0.2,
                density_degenerate_penalty=0.05,
                min_center_size=min_center_size,
            )
            for ell in ellipsoid_list
        ],
        dtype=float,
    )

    dist_matrix, prune_stats = ellipse_distance_pruned(ellipsoid_list, prune_margin=prune_margin)
    min_dists, nearest = ellipse_min_dist(dist_matrix, densities)

    axes_sums = [np.sum(ell.lengths) for ell in ellipsoid_list]
    axes_sum_avg = np.mean(axes_sums) if axes_sums else 0.0
    outlier_ellipsoids = [ell for ell in ellipsoid_list if np.sum(ell.lengths) > 2 * axes_sum_avg]

    print(f"Number of outlier ellipsoids (Số lượng ellipsoid ngoại lệ): {len(outlier_ellipsoids)}")
    print(
        f"Distance pruning stats: total_pairs={prune_stats['total_pairs']}, "
        f"exact_pairs={prune_stats['exact_pairs']}, pruned_pairs={prune_stats['pruned_pairs']}"
    )

    t_attr_end = time.time()
    time_attr = t_attr_end - t_attr_start

    print("Auto-selecting cluster centers from decision values (Tự động chọn tâm cụm từ các giá trị decision)...")
    print("Center strategy:", describe_center_strategy(auto_center_mode, auto_center_k, min_centers, max_centers))

    selected = auto_select_centers_robust(
        ellipsoid_list,
        densities,
        min_dists,
        mode=auto_center_mode,
        top_k=auto_center_k,
        min_centers=min_centers,
        max_centers=max_centers,
        min_center_size=min_center_size,
    )

    gamma = densities * min_dists
    print(f"Selected cluster centers (Các tâm cụm đã chọn): {selected}")
    print(f"Selected gamma values (Giá trị gamma của các tâm cụm): {[float(gamma[i]) for i in selected]}")

    t_cluster_start = time.time()
    ge_labels = ellipse_cluster(densities, selected, nearest)
    t_cluster_end = time.time()
    time_cluster = t_cluster_end - t_cluster_start

    print("Mapping data points in progress (Đang ánh xạ điểm dữ liệu, không tính vào thời gian phân cụm)...")
    pred_labels = np.full(len(data), -1, dtype=int)
    for i, ell in enumerate(ellipsoid_list):
        pred_labels[ell.indices] = ge_labels[i]

    if np.any(pred_labels == -1):
        raise RuntimeError("Some data points were not assigned a label.")

    print("Calculating evaluation metrics (Đang tính các chỉ số đánh giá)...")
    aligned_pred = align_labels(true_labels, pred_labels)

    acc = accuracy_score(true_labels, aligned_pred)
    nmi = normalized_mutual_info_score(true_labels, pred_labels)
    ari = adjusted_rand_score(true_labels, pred_labels)

    print(f"ACC: {acc:.3f}")
    print(f"NMI: {nmi:.3f}")
    print(f"ARI: {ari:.3f}")
    print("-" * 30)
    print("Runtime statistics details (Chi tiết thống kê thời gian chạy):")
    print(f"1. Ellipsoid generation time (Thời gian tạo ellipsoid): {time_gen:.14f} seconds (giây)")
    print(f"2. Attribute computation time (Thời gian tính thuộc tính): {time_attr:.14f} seconds (giây)")
    print(f"3. Clustering computation time (Thời gian tính phân cụm): {time_cluster:.14f} seconds (giây) (mapping time excluded / không tính thời gian ánh xạ)")
    print("-" * 30)
    print(f"Total effective runtime (attributes + clustering) (Tổng thời gian hiệu dụng: thuộc tính + phân cụm): {(time_attr + time_cluster):.14f} seconds (giây)")
    print(f"Total effective runtime of the program (Tổng thời gian chạy hiệu dụng của chương trình): {(time_gen + time_attr + time_cluster):.14f} seconds (giây)")

    return {
        "ACC": acc,
        "NMI": nmi,
        "ARI": ari,
        "time_gen": time_gen,
        "time_attr": time_attr,
        "time_cluster": time_cluster,
        "n_ellipsoids": len(ellipsoid_list),
        "selected_centers": selected,
        "pred_labels": pred_labels,
        "ge_labels": ge_labels,
        "densities": densities,
        "min_dists": min_dists,
        "gamma": gamma,
        "distance_prune_stats": prune_stats,
    }


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent
    # feature_file = BASE_DIR / 'data' / 'miniboone' / 'miniboone.txt'
    # label_file = BASE_DIR / 'data' / 'miniboone' / 'miniboone_label.txt'

    feature_file = BASE_DIR / 'real_dataset_and_label' / 'real_datasets' / 'Iris.txt'
    label_file = BASE_DIR / 'real_dataset_and_label' / 'real_datasets_label' / 'Iris_label.txt'
    # feature_file = BASE_DIR / 'real_dataset_and_label' / 'real_datasets' / 'Seed.txt'
    # label_file = BASE_DIR / 'real_dataset_and_label' / 'real_datasets_label' / 'Seed_label.txt'
    # feature_file = BASE_DIR / 'real_dataset_and_label' / 'real_datasets' / 'segment_3.txt'
    # label_file = BASE_DIR / 'real_dataset_and_label' / 'real_datasets_label' / 'segment_3_label.txt'
    # feature_file = BASE_DIR / 'real_dataset_and_label' / 'real_datasets' / 'landsat_2.txt'
    # label_file = BASE_DIR / 'real_dataset_and_label' / 'real_datasets_label' / 'landsat_2_label.txt'
    # feature_file = BASE_DIR / 'real_dataset_and_label' / 'real_datasets' / 'msplice_2.txt'
    # label_file = BASE_DIR / 'real_dataset_and_label' / 'real_datasets_label' / 'msplice_2_label.txt'
    # feature_file = BASE_DIR / 'real_dataset_and_label' / 'real_datasets' / 'rice.txt'
    # label_file = BASE_DIR / 'real_dataset_and_label' / 'real_datasets_label' / 'rice_label.txt'
    # feature_file = BASE_DIR / 'real_dataset_and_label' / 'real_datasets' / 'banknote.txt'
    # label_file = BASE_DIR / 'real_dataset_and_label' / 'real_datasets_label' / 'banknote_label.txt'
    
    # feature_file = BASE_DIR / 'dataset' / 'unlabel' / 'breast_cancer.txt'
    # label_file = BASE_DIR / 'dataset' / 'label' / 'breast_cancer_label.txt'
    # feature_file = BASE_DIR / 'dataset' / 'unlabel' / 'hcv_data.txt'
    # label_file = BASE_DIR / 'dataset' / 'label' / 'hcv_data_label.txt'
    # feature_file = BASE_DIR / 'dataset' / 'unlabel' / 'dry_bean.txt'
    # label_file = BASE_DIR / 'dataset' / 'label' / 'dry_bean_label.txt'
    # feature_file = BASE_DIR / 'dataset' / 'unlabel' / 'rice+cammeo.txt'
    # label_file = BASE_DIR / 'dataset' / 'label' / 'rice+cammeo_label.txt'

    run_ge_dpc_pure_development(
        feature_file=feature_file,
        label_file=label_file,
        epsilon=1e-6,
        outlier_t=2.0,
        auto_center_mode="knee",
        auto_center_k=None,
        min_centers=1,
        max_centers=None,
        min_leaf_size=None,
        prune_margin=3.0,
        min_center_size=4,
    )
