"""
GE-DPC Cholesky + Conservative Graph Quality Version
====================================================

Muc tieu cua file:
1) Giu nguyen phan sinh granular-ellipsoid cua GE-DPC.
2) Giu Cholesky + cache de tang toc Mahalanobis distance.
3) Cai tien chon center bang gamma + redundancy pruning, ke ca khi dung top_k.
4) KHONG dung graph propagation manh nua.
5) Dung DPC single-chain lam backbone, sau do chi sua nhe bang conservative graph correction.
6) Tat merge mac dinh de tranh gop sai cum.
7) Them log chan doan phan phoi cum du doan.

Ghi chu:
- Khong dung ground-truth label trong qua trinh phan cum.
- Label that chi dung de tinh ACC/NMI/ARI sau cung.
- File nay co the copy truc tiep vao project va chay.
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
    Granular Ellipsoid cho GE-DPC.

    Ke thua tu GE-DPC goc:
    - Moi ellipsoid giu tap diem con, tam, covariance, shape matrix H,
      rho, truc chinh va endpoints de phuc vu split.

    Huong phat trien:
    - Khong tinh inv(H) truc tiep.
    - Dung Cholesky decomposition + solve de tinh Mahalanobis distance.
    - Cache cac dai luong lap lai: covariance, H, Cholesky factor,
      rho, axes, endpoints, density.
    """

    def __init__(self, data: np.ndarray, indices: np.ndarray, epsilon: float = 1e-6):
        self.data = np.asarray(data, dtype=float)
        self.indices = np.asarray(indices, dtype=int)
        self.epsilon = float(epsilon)

        if self.data.ndim != 2 or self.data.shape[0] == 0:
            raise ValueError("Ellipsoid cannot be created with empty or non-2D data.")

        self.n_samples, self.dim = self.data.shape
        self.center = np.mean(self.data, axis=0)

        self._cov_matrix: Optional[np.ndarray] = None
        self._H_matrix: Optional[np.ndarray] = None
        self._chol_factor: Optional[Tuple[np.ndarray, bool]] = None
        self._rho: Optional[float] = None
        self._lengths_rotation: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self._major_axis_endpoints: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self._density: Optional[float] = None

    @property
    def cov_matrix(self) -> np.ndarray:
        """Covariance matrix. bias=True de bam sat code GE-DPC goc."""
        if self._cov_matrix is None:
            if self.n_samples == 1:
                self._cov_matrix = np.zeros((self.dim, self.dim), dtype=float)
            else:
                self._cov_matrix = np.cov(self.data.T, bias=True)
        return self._cov_matrix

    @property
    def H_matrix(self) -> np.ndarray:
        """Shape matrix H = covariance + epsilon * I."""
        if self._H_matrix is None:
            self._H_matrix = self.cov_matrix + self.epsilon * np.eye(self.dim, dtype=float)
        return self._H_matrix

    @property
    def chol_factor(self) -> Tuple[np.ndarray, bool]:
        """Cholesky factor cua H, tinh mot lan roi cache."""
        if self._chol_factor is None:
            self._chol_factor = cho_factor(self.H_matrix, lower=True, check_finite=False)
        return self._chol_factor

    def solve_H(self, rhs: np.ndarray) -> np.ndarray:
        """Giai Hx = rhs bang Cholesky solve."""
        return cho_solve(self.chol_factor, rhs, check_finite=False)

    def mahal_sq_points(self, points: np.ndarray) -> np.ndarray:
        """
        Binh phuong Mahalanobis distance tu nhieu diem den ellipsoid.
        Goc: diff^T inv(H) diff.
        Moi: diff^T solve(H, diff).
        """
        X = np.asarray(points, dtype=float)
        if X.ndim == 1:
            X = X[None, :]

        diffs = X - self.center
        solved = self.solve_H(diffs.T).T
        mahal_sq = np.einsum("ij,ij->i", diffs, solved)
        return np.maximum(mahal_sq, 0.0)

    @property
    def rho(self) -> float:
        """Ban kinh Mahalanobis lon nhat cua cac diem trong ellipsoid."""
        if self._rho is None:
            self._rho = float(np.sqrt(np.max(self.mahal_sq_points(self.data))))
        return self._rho

    @property
    def lengths_rotation(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Tra ve lengths va rotation.
        Neu lambda la tri rieng cua H thi length = rho * sqrt(lambda).
        Cach nay tranh eig(inv(H)).
        """
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
    def major_axis_endpoints(self) -> Tuple[np.ndarray, np.ndarray]:
        """Lay hai dau mut truc chinh de split ellipsoid."""
        if self._major_axis_endpoints is None:
            if self.n_samples <= 1:
                self._major_axis_endpoints = (self.center, self.center)
            else:
                center_distances = np.linalg.norm(self.data - self.center, axis=1)
                point1 = self.data[np.argmin(center_distances)]
                point2 = self.data[np.argmax(np.linalg.norm(self.data - point1, axis=1))]
                point3 = self.data[np.argmax(np.linalg.norm(self.data - point2, axis=1))]
                self._major_axis_endpoints = (point2, point3)
        return self._major_axis_endpoints

    @property
    def density(self) -> float:
        """
        Mat do ellipsoid theo cong thuc GE-DPC goc,
        nhung Mahalanobis distance duoc tinh bang Cholesky solve.
        """
        if self._density is None:
            axes_sum = max(float(np.sum(self.lengths)), 1e-12)
            mahal_sq = self.mahal_sq_points(self.data)
            total_mahal = max(float(np.sum(np.sqrt(mahal_sq))), 1e-12)
            self._density = float((self.n_samples ** 2) / (axes_sum * total_mahal))
        return self._density


# ============================================================
# Helper functions
# ============================================================
def get_num(ellipsoid: Ellipsoid) -> int:
    return ellipsoid.n_samples


def calculate_ellipsoid_density(ellipsoid: Ellipsoid) -> float:
    return ellipsoid.density


def robust_scale(values: np.ndarray) -> np.ndarray:
    """Scale ve [0, 1] bang min-max an toan."""
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values
    v_min = float(np.min(values))
    v_max = float(np.max(values))
    return (values - v_min) / max(v_max - v_min, 1e-12)


# ============================================================
# Split stage: kept from GE-DPC logic
# ============================================================
def splits(ellipsoid_list: Sequence[Ellipsoid], num: int, epsilon: float) -> List[Ellipsoid]:
    """Safe split theo kich thuoc ellipsoid."""
    new_ells: List[Ellipsoid] = []
    for ell in ellipsoid_list:
        if get_num(ell) < num:
            new_ells.append(ell)
        else:
            new_ells.extend(splits_ellipsoid(ell, epsilon=epsilon))
    return new_ells


def splits_ellipsoid(ellipsoid: Ellipsoid, epsilon: Optional[float] = None) -> List[Ellipsoid]:
    """
    Tach mot ellipsoid thanh hai ellipsoid con.
    Phase 1: chia theo hai dau mut truc chinh.
    Phase 2: gan lai diem bang Mahalanobis distance.
    """
    if ellipsoid.n_samples <= 1:
        return [ellipsoid]

    eps = ellipsoid.epsilon if epsilon is None else float(epsilon)
    data = ellipsoid.data
    indices = ellipsoid.indices
    point1, point2 = ellipsoid.major_axis_endpoints

    dist_to_point1 = np.linalg.norm(data - point1, axis=1)
    dist_to_point2 = np.linalg.norm(data - point2, axis=1)

    cluster1_mask = dist_to_point1 < dist_to_point2
    cluster2_mask = ~cluster1_mask

    cluster1 = data[cluster1_mask]
    cluster2 = data[cluster2_mask]
    cluster1_idx = indices[cluster1_mask]
    cluster2_idx = indices[cluster2_mask]

    if len(cluster1) == 0 or len(cluster2) == 0:
        return [ellipsoid]

    ell1 = Ellipsoid(cluster1, cluster1_idx, epsilon=eps)
    ell2 = Ellipsoid(cluster2, cluster2_idx, epsilon=eps)

    dist1_sq = ell1.mahal_sq_points(data)
    dist2_sq = ell2.mahal_sq_points(data)
    new_cluster1_mask = dist1_sq < dist2_sq
    new_cluster2_mask = ~new_cluster1_mask

    cluster1 = data[new_cluster1_mask]
    cluster2 = data[new_cluster2_mask]
    cluster1_idx = indices[new_cluster1_mask]
    cluster2_idx = indices[new_cluster2_mask]

    if len(cluster1) == 0 or len(cluster2) == 0:
        return [ellipsoid]

    ell1 = Ellipsoid(cluster1, cluster1_idx, epsilon=eps)
    ell2 = Ellipsoid(cluster2, cluster2_idx, epsilon=eps)
    return [ell1, ell2]


def recursive_split_outlier_detection(
    initial_ellipsoids: Sequence[Ellipsoid],
    data: np.ndarray,
    t: float = 2.0,
    max_iterations: int = 10,
    epsilon: float = 1e-6,
) -> List[Ellipsoid]:
    """
    Outlier-detection split.
    Giu logic goc: chi split ellipsoid bat thuong neu tong density con du tot.
    """
    ellipsoid_list = list(initial_ellipsoids)

    for _ in range(max_iterations):
        if not ellipsoid_list:
            break

        axes_sums = np.array([np.sum(ell.lengths) for ell in ellipsoid_list], dtype=float)
        axes_sum_avg = float(np.mean(axes_sums)) if len(axes_sums) > 0 else 0.0

        outlier_ellipsoids = [ell for ell in ellipsoid_list if np.sum(ell.lengths) > 2.0 * axes_sum_avg]
        if not outlier_ellipsoids:
            break

        normal_ellipsoids = [ell for ell in ellipsoid_list if ell not in outlier_ellipsoids]
        new_ellipsoids: List[Ellipsoid] = []
        min_leaf = max(2, int(np.ceil(np.sqrt(data.shape[0]) * 0.1)))

        for outlier_ell in outlier_ellipsoids:
            children = splits_ellipsoid(outlier_ell, epsilon=epsilon)

            if len(children) != 2:
                new_ellipsoids.append(outlier_ell)
                continue

            if any(child.n_samples < min_leaf for child in children):
                new_ellipsoids.append(outlier_ell)
                continue

            parent_density = calculate_ellipsoid_density(outlier_ell)
            child_density_sum = sum(calculate_ellipsoid_density(child) for child in children)

            if child_density_sum > t * parent_density:
                new_ellipsoids.extend(children)
            else:
                new_ellipsoids.append(outlier_ell)

        ellipsoid_list = normal_ellipsoids + new_ellipsoids

    return ellipsoid_list


# ============================================================
# Ellipsoid-to-ellipsoid distance
# ============================================================
def ellipse_mahalanobis_distance(ellipsoid_i: Ellipsoid, ellipsoid_j: Ellipsoid) -> Tuple[float, float, float]:
    """
    Distance giua hai ellipsoid dua tren tam va average H.
    Dung Cholesky solve thay cho inverse.
    """
    center_i = ellipsoid_i.center
    center_j = ellipsoid_j.center
    avg_H = 0.5 * (ellipsoid_i.H_matrix + ellipsoid_j.H_matrix)

    chol_avg = cho_factor(avg_H, lower=True, check_finite=False)
    diff = center_i - center_j
    solved = cho_solve(chol_avg, diff, check_finite=False)
    dist_sq = float(diff.T @ solved)
    dist = float(np.sqrt(max(dist_sq, 0.0)))
    return dist, dist, dist


def ellipse_distance(ellipsoid_list: Sequence[Ellipsoid]) -> np.ndarray:
    """Tao ma tran khoang cach giua cac ellipsoid."""
    n = len(ellipsoid_list)
    dist_mat = np.zeros((n, n), dtype=float)

    for i in range(n):
        for j in range(i + 1, n):
            _, _, rel_dist = ellipse_mahalanobis_distance(ellipsoid_list[i], ellipsoid_list[j])
            dist_mat[i, j] = dist_mat[j, i] = rel_dist

    return dist_mat


# ============================================================
# DPC attributes
# ============================================================
def ellipse_min_dist(dist_mat: np.ndarray, densities: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Tinh delta/min-distance toi ellipsoid co density cao hon gan nhat."""
    densities = np.asarray(densities, dtype=float)
    order = np.argsort(-densities)
    min_dists = np.zeros(len(densities), dtype=float)
    nearest = -np.ones(len(densities), dtype=int)

    for i in order[1:]:
        mask = densities > densities[i]
        if np.any(mask):
            candidates = np.where(mask)[0]
            idx_local = np.argmin(dist_mat[i, candidates])
            nearest[i] = candidates[idx_local]
            min_dists[i] = dist_mat[i, nearest[i]]
        else:
            min_dists[i] = np.max(dist_mat[i])

    if len(order) > 0:
        min_dists[order[0]] = np.max(min_dists)

    return min_dists, nearest


# ============================================================
# Improved center selection
# ============================================================
def auto_select_centers_quality(
    densities: np.ndarray,
    min_dists: np.ndarray,
    dist_mat: np.ndarray,
    top_k: Optional[int] = None,
    min_centers: int = 2,
    max_centers: Optional[int] = None,
) -> List[int]:
    """
    Cai tien chon center.

    Diem quan trong cua ban nay:
    - Van dung gamma = density * delta.
    - Neu top_k != None thi VAN dung redundancy pruning, khong return thang top-k.
    - Neu top_k == None thi tu dong chon ung vien bang gamma-drop.
    - Redundancy pruning giup tranh nhieu center nam qua gan nhau.
    """
    densities = np.asarray(densities, dtype=float)
    min_dists = np.asarray(min_dists, dtype=float)
    n = len(densities)

    if n == 0:
        return []
    if n == 1:
        return [0]

    gamma = densities * min_dists
    order = np.argsort(-gamma)

    positive_dists = dist_mat[dist_mat > 0]
    if positive_dists.size == 0:
        target = min_centers if top_k is None else int(top_k)
        return order[: max(1, min(target, n))].tolist()

    median_dist = float(np.median(positive_dists))
    q25_dist = float(np.quantile(positive_dists, 0.25))

    # min_sep duoc suy ra tu dist_matrix, khong phai tham so nguoi dung.
    # Gia tri 0.20 * median bao thu hon 0.25 de tranh pruning qua manh.
    min_sep = max(q25_dist, 0.20 * median_dist, 1e-12)

    # --------------------------------------------------------
    # Case 1: biet truoc so center top_k.
    # Van pruning de tranh center trung vung.
    # --------------------------------------------------------
    if top_k is not None:
        target_k = int(max(1, min(top_k, n)))
        if max_centers is not None:
            target_k = min(target_k, max_centers)
        target_k = max(min_centers, target_k)
        target_k = min(target_k, n)

        selected: List[int] = []
        for idx in order:
            idx = int(idx)
            if not selected:
                selected.append(idx)
            else:
                nearest_selected_dist = float(np.min(dist_mat[idx, selected]))
                if nearest_selected_dist >= min_sep:
                    selected.append(idx)

            if len(selected) >= target_k:
                break

        # Neu pruning qua manh thi bo sung theo gamma cao nhat.
        if len(selected) < target_k:
            for idx in order:
                idx = int(idx)
                if idx not in selected:
                    selected.append(idx)
                if len(selected) >= target_k:
                    break

        return selected[:target_k]

    # --------------------------------------------------------
    # Case 2: tu dong uoc luong so center bang gamma-drop.
    # --------------------------------------------------------
    g = gamma[order]
    g_norm = robust_scale(g)

    if np.allclose(g_norm, g_norm[0]):
        candidate_k = min(max(min_centers, 1), n)
    else:
        drops = (g_norm[:-1] - g_norm[1:]) / np.maximum(g_norm[:-1], 1e-12)
        if drops.size == 0:
            candidate_k = min_centers
        else:
            head_limit = max(min(n - 1, int(np.ceil(np.sqrt(n))) + min_centers), min_centers)
            head_drops = drops[:head_limit]
            candidate_k = int(np.argmax(head_drops)) + 1
            candidate_k = max(candidate_k, min_centers)

    pool_size = max(candidate_k * 3, min_centers * 3, int(np.ceil(np.sqrt(n))))
    if max_centers is not None:
        pool_size = max(pool_size, max_centers)
    pool_size = min(pool_size, n)
    pool = order[:pool_size]

    selected: List[int] = []
    for idx in pool:
        idx = int(idx)
        if not selected:
            selected.append(idx)
        else:
            nearest_selected_dist = float(np.min(dist_mat[idx, selected]))
            if nearest_selected_dist >= min_sep:
                selected.append(idx)

        if max_centers is not None and len(selected) >= max_centers:
            break

    if len(selected) < min_centers:
        for idx in order:
            idx = int(idx)
            if idx not in selected:
                selected.append(idx)
            if len(selected) >= min_centers:
                break

    if max_centers is not None:
        selected = selected[:max_centers]

    return selected


# ============================================================
# Conservative graph correction
# ============================================================
def ellipse_cluster_single_chain(
    densities: np.ndarray,
    centers: Sequence[int],
    nearest: np.ndarray,
    dist_mat: np.ndarray,
) -> np.ndarray:
    """
    DPC labeling goc: moi ellipsoid nhan nhan tu nearest higher-density.
    Ham nay dung lam backbone on dinh truoc khi correction.
    """
    densities = np.asarray(densities, dtype=float)
    n = len(densities)
    labels = -np.ones(n, dtype=int)

    centers = [int(c) for c in centers if 0 <= int(c) < n]
    if len(centers) == 0:
        labels[:] = 0
        return labels

    for label_id, c in enumerate(centers):
        labels[c] = label_id

    order = np.argsort(-densities)
    for idx in order:
        if labels[idx] == -1 and nearest[idx] != -1:
            labels[idx] = labels[nearest[idx]]

    # Fallback: neu con ellipsoid chua nhan nhan, gan ve center gan nhat.
    for idx in np.where(labels == -1)[0]:
        nearest_center = centers[int(np.argmin(dist_mat[idx, centers]))]
        labels[idx] = labels[nearest_center]

    return labels


def ellipse_cluster_conservative_graph_correction(
    densities: np.ndarray,
    dist_mat: np.ndarray,
    centers: Sequence[int],
    nearest: np.ndarray,
) -> np.ndarray:
    """
    Conservative graph correction.

    Khac voi graph propagation manh:
    1) Gan nhan bang DPC single-chain truoc.
    2) Chi doi nhan neu graph voting cho bang chung manh hon ro rang.
    3) Giu tinh on dinh cua DPC goc, tranh lam muot nhan qua muc.
    """
    densities = np.asarray(densities, dtype=float)
    n = len(densities)

    labels = ellipse_cluster_single_chain(
        densities=densities,
        centers=centers,
        nearest=nearest,
        dist_mat=dist_mat,
    )
    corrected = labels.copy()

    centers = [int(c) for c in centers if 0 <= int(c) < n]
    center_set = set(centers)

    positive_dists = dist_mat[dist_mat > 0]
    eps_dist = float(np.median(positive_dists) * 1e-9) if positive_dists.size else 1e-12

    # Noi bo, khong dua ra thanh tham so nguoi dung.
    k_neighbors = max(2, int(np.ceil(np.log2(max(n, 2)))))
    switch_margin = 1.75

    # Duyet tu density thap den cao, vi nhung diem thap/mep cum moi can sua.
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

        # Chi doi nhan neu vote moi thang ro rang.
        if best_lab != current_lab and best_vote > switch_margin * max(current_vote, 1e-12):
            corrected[idx] = int(best_lab)

    return corrected


# ============================================================
# Optional very conservative merge. Default OFF.
# ============================================================
def merge_close_clusters_light(
    ellipsoid_labels: np.ndarray,
    densities: np.ndarray,
    dist_mat: np.ndarray,
) -> np.ndarray:
    """
    Merge cum nhe neu center bi chon du.
    Mac dinh nen tat. Chi bat khi selected centers qua nhieu.
    """
    labels = np.asarray(ellipsoid_labels, dtype=int).copy()
    unique_labels = sorted([int(x) for x in np.unique(labels) if x >= 0])
    if len(unique_labels) <= 1:
        return labels

    representatives: Dict[int, int] = {}
    for lab in unique_labels:
        members = np.where(labels == lab)[0]
        if members.size == 0:
            continue
        representatives[lab] = int(members[np.argmax(densities[members])])

    reps = list(representatives.values())
    if len(reps) <= 1:
        return labels

    center_dists = []
    for a in range(len(reps)):
        for b in range(a + 1, len(reps)):
            d = float(dist_mat[reps[a], reps[b]])
            if d > 0:
                center_dists.append(d)

    if len(center_dists) == 0:
        return labels

    center_dists = np.asarray(center_dists, dtype=float)
    # Bao thu hon ban cu: chi merge neu rat gan, quantile 10%.
    merge_threshold = float(np.quantile(center_dists, 0.10))
    if merge_threshold <= 0:
        return labels

    parent = {lab: lab for lab in unique_labels}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            rep_a = representatives[ra]
            rep_b = representatives[rb]
            if densities[rep_a] >= densities[rep_b]:
                parent[rb] = ra
            else:
                parent[ra] = rb

    for i, lab_i in enumerate(unique_labels):
        for lab_j in unique_labels[i + 1:]:
            rep_i = representatives[lab_i]
            rep_j = representatives[lab_j]
            d = float(dist_mat[rep_i, rep_j])
            if 0 < d <= merge_threshold:
                union(lab_i, lab_j)

    merged = labels.copy()
    for lab in unique_labels:
        merged[labels == lab] = find(lab)

    final_unique = sorted(np.unique(merged))
    remap = {old: new for new, old in enumerate(final_unique)}
    merged = np.array([remap[x] for x in merged], dtype=int)
    return merged


# ============================================================
# Evaluation and diagnostics
# ============================================================
def align_labels(true_labels: np.ndarray, pred_labels: np.ndarray) -> np.ndarray:
    """Hungarian matching de tinh ACC cho clustering."""
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
    """In phan phoi nhan de chan doan cum bi lech hay khong."""
    unique, counts = np.unique(labels, return_counts=True)
    print(name)
    for lab, cnt in zip(unique, counts):
        print(f"  label {int(lab):>4}: {int(cnt)}")


# ============================================================
# Main improved GE-DPC pipeline
# ============================================================
def run_ge_dpc_cholesky_conservative_quality(
    feature_file: Path,
    label_file: Path,
    epsilon: float = 1e-6,
    outlier_t: float = 2.0,
    auto_center_k: Optional[int] = None,
    min_centers: int = 2,
    max_centers: Optional[int] = None,
    enable_light_merge: bool = False,
) -> Dict[str, object]:
    """
    Pipeline GE-DPC cai tien on dinh chat luong.

    Buoc 1. Doc du lieu va nhan that.
    Buoc 2. Sinh granular-ellipsoid.
    Buoc 3. Toi uu tinh toan hinh hoc bang Cholesky/cache.
    Buoc 4. Tinh DPC attributes o muc ellipsoid.
    Buoc 5. Cai tien chon center bang gamma + redundancy pruning.
    Buoc 6. Gan nhan DPC goc + conservative graph correction.
    Buoc 7. Merge cum nhe neu bat enable_light_merge.
    Buoc 8. Anh xa nhan ellipsoid ve diem du lieu.
    Buoc 9. Danh gia ACC/NMI/ARI va thoi gian chay.
    """
    data = np.loadtxt(feature_file, dtype=float)
    true_labels = np.loadtxt(label_file, dtype=float).astype(int)

    if data.ndim == 1:
        data = data.reshape(-1, 1)

    if len(data) != len(true_labels):
        raise ValueError(f"Data and label length mismatch: {len(data)} vs {len(true_labels)}")

    num = int(np.ceil(np.sqrt(data.shape[0])))

    # ------------------------------
    # 1-2) Sinh ellipsoid
    # ------------------------------
    t_gen_start = time.time()
    initial_indices = np.arange(data.shape[0], dtype=int)
    initial_ellipsoid = Ellipsoid(data, initial_indices, epsilon=epsilon)
    ellipsoid_list: List[Ellipsoid] = [initial_ellipsoid]

    print("=" * 80)
    print("GE-DPC Cholesky + Conservative Graph Quality Improvement")
    print(f"Data shape: n={data.shape[0]}, d={data.shape[1]}")
    print(f"Safe split threshold num = ceil(sqrt(n)) = {num}")
    print("Initial ellipsoid count: 1")

    iteration = 0
    while True:
        iteration += 1
        before = len(ellipsoid_list)
        ellipsoid_list = splits(ellipsoid_list, num=num, epsilon=epsilon)
        after = len(ellipsoid_list)
        print(f"Ellipsoid count after safe split iteration {iteration}: {after}")
        if after == before:
            break

    print(f"Total ellipsoid count after safe splitting: {len(ellipsoid_list)}")

    ellipsoid_list = recursive_split_outlier_detection(
        ellipsoid_list,
        data,
        t=outlier_t,
        max_iterations=10,
        epsilon=epsilon,
    )
    print(f"Total ellipsoid count after outlier-detection splitting: {len(ellipsoid_list)}")
    t_gen_end = time.time()
    time_gen = t_gen_end - t_gen_start

    # ------------------------------
    # 3-4) DPC attributes
    # ------------------------------
    t_attr_start = time.time()
    densities = np.array([calculate_ellipsoid_density(ell) for ell in ellipsoid_list], dtype=float)
    dist_matrix = ellipse_distance(ellipsoid_list)
    min_dists, nearest = ellipse_min_dist(dist_matrix, densities)
    gamma = densities * min_dists
    t_attr_end = time.time()
    time_attr = t_attr_end - t_attr_start

    # ------------------------------
    # 5) Center selection
    # ------------------------------
    print("Auto-selecting cluster centers with gamma + redundancy pruning...")
    selected = auto_select_centers_quality(
        densities=densities,
        min_dists=min_dists,
        dist_mat=dist_matrix,
        top_k=auto_center_k,
        min_centers=min_centers,
        max_centers=max_centers,
    )
    print(f"Number of selected centers: {len(selected)}")
    print(f"Selected centers: {selected}")
    print(f"Selected gamma values: {[float(gamma[i]) for i in selected]}")

    # ------------------------------
    # 6-7) Conservative graph correction + optional merge
    # ------------------------------
    t_cluster_start = time.time()
    ellipsoid_labels_before_merge = ellipse_cluster_conservative_graph_correction(
        densities=densities,
        dist_mat=dist_matrix,
        centers=selected,
        nearest=nearest,
    )

    if enable_light_merge:
        ellipsoid_labels = merge_close_clusters_light(
            ellipsoid_labels=ellipsoid_labels_before_merge,
            densities=densities,
            dist_mat=dist_matrix,
        )
    else:
        ellipsoid_labels = ellipsoid_labels_before_merge

    t_cluster_end = time.time()
    time_cluster = t_cluster_end - t_cluster_start

    n_clusters_before = len(np.unique(ellipsoid_labels_before_merge))
    n_clusters_after = len(np.unique(ellipsoid_labels))
    print(f"Clusters before light merge: {n_clusters_before}")
    print(f"Clusters after light merge : {n_clusters_after}")
    print_distribution("Ellipsoid cluster distribution:", ellipsoid_labels)

    # ------------------------------
    # 8) Mapping ellipsoid label -> data point label
    # ------------------------------
    print("Mapping ellipsoid labels to data points...")
    pred_labels = np.full(len(data), -1, dtype=int)
    for i, ell in enumerate(ellipsoid_list):
        pred_labels[ell.indices] = ellipsoid_labels[i]

    if np.any(pred_labels == -1):
        raise RuntimeError("Some data points were not assigned a cluster label.")

    print_distribution("Predicted data cluster distribution:", pred_labels)
    print_distribution("Ground-truth label distribution:", true_labels)

    # ------------------------------
    # 9) Evaluation
    # ------------------------------
    aligned_pred_labels = align_labels(true_labels, pred_labels)
    acc = accuracy_score(true_labels, aligned_pred_labels)
    nmi = normalized_mutual_info_score(true_labels, pred_labels)
    ari = adjusted_rand_score(true_labels, pred_labels)

    total_valid_time1 = time_attr + time_cluster
    total_valid_time2 = time_gen + time_attr + time_cluster

    print("-" * 80)
    print("Evaluation metrics:")
    print(f"ACC: {acc:.3f}")
    print(f"NMI: {nmi:.3f}")
    print(f"ARI: {ari:.3f}")
    print("-" * 80)
    print("Runtime statistics:")
    # print(f"1. Ellipsoid generation time: {time_gen:.14f} seconds")
    # print(f"2. Attribute computation time: {time_attr:.14f} seconds")
    # print(f"3. Clustering computation time: {time_cluster:.14f} seconds")
    # print(f"Total effective runtime (attributes + clustering): {total_valid_time1:.14f} seconds")
    # print(f"Total effective runtime of the program: {total_valid_time2:.14f} seconds")
    print(f"1. Ellipsoid generation time: {time_gen * 1000:.6f} ms")
    print(f"2. Attribute computation time: {time_attr * 1000:.6f} ms")
    print(f"3. Clustering computation time: {time_cluster * 1000:.6f} ms")
    print(f"Total effective runtime (attributes + clustering): {total_valid_time1 * 1000:.6f} ms")
    print(f"Total effective runtime of the program: {total_valid_time2 * 1000:.6f} ms")
    print("-" * 80)

    return {
        "acc": acc,
        "nmi": nmi,
        "ari": ari,
        "selected_centers": selected,
        "n_ellipsoids": len(ellipsoid_list),
        "n_clusters_before_merge": n_clusters_before,
        "n_clusters_after_merge": n_clusters_after,
        "pred_labels": pred_labels,
        "aligned_pred_labels": aligned_pred_labels,
        "generation_time": time_gen,
        "attribute_time": time_attr,
        "cluster_time": time_cluster,
        "total_valid_time_attr_cluster": total_valid_time1,
        "total_valid_time_program": total_valid_time2,
    }


# ============================================================
# Dataset registry
# ============================================================
def get_default_dataset_registry(base_dir: Path) -> Dict[str, Tuple[Path, Path]]:
    """Registry giu nguyen theo cau truc project cua ban."""
    return {
        "miniboone": (
            base_dir / "data" / "miniboone" / "miniboone.txt",
            base_dir / "data" / "miniboone" / "miniboone_label.txt",
        ),
        "covertype": (
            base_dir / "data" / "covertype" / "covertype.txt",
            base_dir / "data" / "covertype" / "covertype_label.txt",
        ),
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


def get_dataset_default_k(dataset_name: str) -> int:
    """
    Default k de test on dinh theo so lop thuong dung cua cac dataset.
    Neu ban muon auto hoan toan, dat auto_center_k=None o main.
    """
    defaults = {
        "iris": 3,
        "seed": 3,
        "segment_3": 3,
        "landsat_2": 2,
        "msplice_2": 2,
        "rice": 2,
        "banknote": 2,
        "htru2": 2,
        "breast_cancer": 2,
        "hcv_data": 2,
        "dry_bean": 7,
        "rice_cammeo": 2,
    }
    return defaults.get(dataset_name.lower(), 2)


def run_named_dataset(
    dataset_name: str,
    base_dir: Path,
    epsilon: float = 1e-6,
    outlier_t: float = 2.0,
    auto_center_k: Optional[int] = None,
    min_centers: int = 2,
    max_centers: Optional[int] = None,
    enable_light_merge: bool = False,
) -> Dict[str, object]:
    """Chay mot dataset theo ten."""
    registry = get_default_dataset_registry(base_dir)
    key = dataset_name.lower()
    if key not in registry:
        raise KeyError(f"Unknown dataset_name: {dataset_name}. Available: {list(registry.keys())}")

    feature_file, label_file = registry[key]
    if not feature_file.exists() or not label_file.exists():
        raise FileNotFoundError(
            f"Dataset files not found for '{dataset_name}'.\n"
            f"Feature: {feature_file}\nLabel: {label_file}"
        )

    print("=" * 80)
    print(f"Running dataset: {dataset_name}")
    print(f"Feature file: {feature_file}")
    print(f"Label file  : {label_file}")
    print("=" * 80)

    return run_ge_dpc_cholesky_conservative_quality(
        feature_file=feature_file,
        label_file=label_file,
        epsilon=epsilon,
        outlier_t=outlier_t,
        auto_center_k=auto_center_k,
        min_centers=min_centers,
        max_centers=max_centers,
        enable_light_merge=enable_light_merge,
    )


def run_all_default_datasets(
    base_dir: Path,
    dataset_names: Optional[Sequence[str]] = None,
    epsilon: float = 1e-6,
    outlier_t: float = 2.0,
    use_dataset_default_k: bool = True,
    auto_center_k: Optional[int] = None,
    min_centers: int = 2,
    max_centers: Optional[int] = None,
    enable_light_merge: bool = False,
) -> Dict[str, Dict[str, object]]:
    """Chay nhieu dataset va in bang tong hop."""
    registry = get_default_dataset_registry(base_dir)
    if dataset_names is None:
        dataset_names = list(registry.keys())

    results: Dict[str, Dict[str, object]] = {}

    for dataset_name in dataset_names:
        try:
            if use_dataset_default_k:
                k = get_dataset_default_k(dataset_name)
                local_min = k
                local_max = k
            else:
                k = auto_center_k
                local_min = min_centers
                local_max = max_centers

            results[dataset_name] = run_named_dataset(
                dataset_name=dataset_name,
                base_dir=base_dir,
                epsilon=epsilon,
                outlier_t=outlier_t,
                auto_center_k=k,
                min_centers=local_min,
                max_centers=local_max,
                enable_light_merge=enable_light_merge,
            )
        except Exception as exc:
            results[dataset_name] = {"error": str(exc)}
            print(f"[ERROR] Dataset '{dataset_name}' failed: {exc}")

    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"{'Dataset':<18} {'ACC':>8} {'NMI':>8} {'ARI':>8} {'Ells':>8} {'C_before':>10} {'C_after':>9} {'Time(s)':>12}")
    print("-" * 100)
    for name, res in results.items():
        if "error" in res:
            print(f"{name:<18} ERROR: {res['error']}")
        else:
            print(
                f"{name:<18} "
                f"{res['acc']:>8.3f} "
                f"{res['nmi']:>8.3f} "
                f"{res['ari']:>8.3f} "
                f"{res['n_ellipsoids']:>8} "
                f"{res['n_clusters_before_merge']:>10} "
                f"{res['n_clusters_after_merge']:>9} "
                f"{res['total_valid_time_program']:>12.6f}"
            )
    print("=" * 100)

    return results


if __name__ == "__main__":
    # Neu file nam trong GE-DPC-main/scripts thi dung parent.parent.
    # Neu file nam truc tiep trong GE-DPC-main thi doi thanh Path(__file__).resolve().parent.
    BASE_DIR = Path(__file__).resolve().parent.parent

    epsilon = 1e-6
    outlier_t = 2.0

    # ========================================================
    # Khuyen nghi de test segment_3:
    # - segment_3 thuong nen dung k=3.
    # - Tat merge de tranh gop sai cum.
    # ========================================================
    # dataset_name = "segment_3"

    # auto_center_k = get_dataset_default_k(dataset_name)  # segment_3 -> 3
    # min_centers = auto_center_k
    # max_centers = auto_center_k
    # enable_light_merge = False
    dataset_name = "hcv_data"

    auto_center_k = 2
    min_centers = 2
    max_centers = 2
    enable_light_merge = False

    run_named_dataset(
        dataset_name=dataset_name,
        base_dir=BASE_DIR,
        epsilon=epsilon,
        outlier_t=outlier_t,
        auto_center_k=auto_center_k,
        min_centers=min_centers,
        max_centers=max_centers,
        enable_light_merge=enable_light_merge,
    )

    # ========================================================
    # Neu muon chay 8 dataset hay dung, bo comment doan duoi.
    # use_dataset_default_k=True se dung k theo tung dataset.
    # ========================================================
    # dataset_names = [
    #     "iris", "seed", "segment_3", "landsat_2",
    #     "msplice_2", "rice", "banknote", "htru2",
    # ]
    # all_results = run_all_default_datasets(
    #     base_dir=BASE_DIR,
    #     dataset_names=dataset_names,
    #     epsilon=epsilon,
    #     outlier_t=outlier_t,
    #     use_dataset_default_k=True,
    #     enable_light_merge=False,
    # )
