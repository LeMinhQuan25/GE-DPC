"""
GE-DPC Quality-Improved Version
================================

Huong phat trien:
1) Giu nguyen sinh granular-ellipsoid cua GE-DPC.
2) Giu Cholesky + cache de tang toc Mahalanobis distance.
3) Cai tien chon center bang gamma + redundancy pruning.
4) Thay gan nhan don bang graph-based multi-neighbor propagation.
5) Them merge cum nhe neu chon du center.

Ghi chu quan trong:
- File nay khong dung ground-truth label trong qua trinh phan cum.
- Label that chi dung de danh gia ACC/NMI/ARI sau cung.
- Cac nguong moi deu duoc suy ra tu dist_matrix/gamma, han che them tham so thu cong.
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

    Phan moi / huong phat trien:
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
        """Covariance matrix. Bias=True de bam sat code GE-DPC goc."""
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


def robust_scale(values: np.ndarray) -> np.ndarray:
    """Scale ve [0, 1] bang min-max an toan."""
    values = np.asarray(values, dtype=float)
    v_min = float(np.min(values)) if values.size else 0.0
    v_max = float(np.max(values)) if values.size else 1.0
    return (values - v_min) / max(v_max - v_min, 1e-12)


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

    Khac voi ban goc:
    - Van dung gamma = density * delta.
    - Neu top_k != None thi cho phep chay theo k biet truoc.
    - Neu top_k == None thi tu dong chon ung vien bang gamma-gap/knee.
    - Them redundancy pruning: loai center qua gan center da chon.

    Muc tieu:
    - Giam truong hop chon thieu center.
    - Giam truong hop nhieu center nam cung mot vung mat do.
    - Khong dung label that.
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

    if top_k is not None:
        target_k = int(max(1, min(top_k, n)))
        if max_centers is not None:
            target_k = min(target_k, max_centers)
        target_k = max(min_centers, target_k)
        return order[: min(target_k, n)].tolist()

    # ------------------------------
    # 1) Tu dong uoc luong so center ung vien bang gamma drop.
    # ------------------------------
    g = gamma[order]
    g_norm = robust_scale(g)

    if np.allclose(g_norm, g_norm[0]):
        candidate_k = min(max(min_centers, 1), n)
    else:
        # Ratio drop giua cac gamma lien tiep.
        drops = (g_norm[:-1] - g_norm[1:]) / np.maximum(g_norm[:-1], 1e-12)
        if drops.size == 0:
            candidate_k = min_centers
        else:
            # Vi tri drop lon nhat trong phan dau danh sach.
            head_limit = max(min(n - 1, int(np.ceil(np.sqrt(n))) + min_centers), min_centers)
            head_drops = drops[:head_limit]
            candidate_k = int(np.argmax(head_drops)) + 1
            candidate_k = max(candidate_k, min_centers)

    # Lay du ung vien de redundancy pruning co co hoi chon center tot.
    pool_size = max(candidate_k * 3, min_centers * 3, int(np.ceil(np.sqrt(n))))
    if max_centers is not None:
        pool_size = max(pool_size, max_centers)
    pool_size = min(pool_size, n)
    pool = order[:pool_size]

    # ------------------------------
    # 2) Redundancy pruning tu dong bang median positive distance.
    # ------------------------------
    positive_dists = dist_mat[dist_mat > 0]
    if positive_dists.size == 0:
        return order[:max(min_centers, 1)].tolist()

    median_dist = float(np.median(positive_dists))
    q25_dist = float(np.quantile(positive_dists, 0.25))
    # min_sep khong phai tham so thu cong: suy ra tu phan bo distance.
    min_sep = max(q25_dist, 0.25 * median_dist, 1e-12)

    selected: List[int] = []
    for idx in pool:
        if not selected:
            selected.append(int(idx))
        else:
            nearest_selected_dist = float(np.min(dist_mat[idx, selected]))
            if nearest_selected_dist >= min_sep:
                selected.append(int(idx))

        if max_centers is not None and len(selected) >= max_centers:
            break

    # Neu pruning qua manh thi bo sung theo gamma cao nhat.
    if len(selected) < min_centers:
        for idx in order:
            if int(idx) not in selected:
                selected.append(int(idx))
            if len(selected) >= min_centers:
                break

    if max_centers is not None:
        selected = selected[:max_centers]

    return selected


# ============================================================
# Graph-based multi-neighbor propagation
# ============================================================
def ellipse_cluster_graph_propagation(
    densities: np.ndarray,
    dist_mat: np.ndarray,
    centers: Sequence[int],
) -> np.ndarray:
    """
    Graph-based multi-neighbor label propagation o muc ellipsoid.

    Khac voi ellipse_cluster() goc:
    - Goc: moi ellipsoid nhan nhan tu 1 nearest higher-density neighbor.
    - Moi: moi ellipsoid nhan vote tu nhieu higher-density neighbors da co nhan.

    Cach tinh vote:
        vote(label_j) += density_j / distance(i, j)

    Uu diem:
    - Giam loi lan truyen theo chuoi don.
    - Tang on dinh khi center hoac nearest higher-density bi lech.
    - Chi xu ly tren so ellipsoid nen chi phi nho.
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
    positive_dists = dist_mat[dist_mat > 0]
    eps_dist = float(np.median(positive_dists) * 1e-9) if positive_dists.size else 1e-12

    # So neighbor noi bo, suy ra tu so ellipsoid, khong yeu cau nguoi dung tune.
    k_neighbors = max(2, int(np.ceil(np.log2(max(n, 2)))))

    for idx in order:
        if labels[idx] != -1:
            continue

        higher = np.where(densities > densities[idx])[0]
        if higher.size == 0:
            # Fallback: gan ve center gan nhat.
            nearest_center = centers[int(np.argmin(dist_mat[idx, centers]))]
            labels[idx] = labels[nearest_center]
            continue

        # Chi lay higher-density neighbors gan nhat.
        higher_sorted = higher[np.argsort(dist_mat[idx, higher])]
        higher_sorted = higher_sorted[: min(k_neighbors, higher_sorted.size)]

        votes: Dict[int, float] = {}
        for nb in higher_sorted:
            nb_label = int(labels[nb])
            if nb_label == -1:
                continue
            w = float(densities[nb] / (dist_mat[idx, nb] + eps_dist))
            votes[nb_label] = votes.get(nb_label, 0.0) + w

        if votes:
            labels[idx] = max(votes.items(), key=lambda kv: kv[1])[0]
        else:
            # Neu chua co higher neighbor nao da co nhan, fallback ve center gan nhat.
            nearest_center = centers[int(np.argmin(dist_mat[idx, centers]))]
            labels[idx] = labels[nearest_center]

    # Fallback cuoi cung neu con ellipsoid chua co nhan.
    unlabeled = np.where(labels == -1)[0]
    for idx in unlabeled:
        nearest_center = centers[int(np.argmin(dist_mat[idx, centers]))]
        labels[idx] = labels[nearest_center]

    return labels


# ============================================================
# Light cluster merge
# ============================================================
def merge_close_clusters_light(
    ellipsoid_labels: np.ndarray,
    centers: Sequence[int],
    densities: np.ndarray,
    dist_mat: np.ndarray,
) -> np.ndarray:
    """
    Merge cum nhe neu center bi chon du.

    Nguyen tac bao thu:
    - Chi merge khi hai cluster-center rat gan nhau theo phan bo distance center-center.
    - Khong dung label that.
    - Dung DSU/Union-Find de hop nhat label.

    Muc tieu:
    - Neu center selection chon du center trong cung mot vung mat do,
      buoc nay gom lai de giam over-segmentation.
    - Neu cac center cach xa nhau, buoc nay gan nhu khong anh huong.
    """
    labels = np.asarray(ellipsoid_labels, dtype=int).copy()
    unique_labels = sorted([int(x) for x in np.unique(labels) if x >= 0])
    if len(unique_labels) <= 1:
        return labels

    # Dai dien moi cum: ellipsoid co density cao nhat trong cum.
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
    merge_threshold = float(np.quantile(center_dists, 0.20))
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
            # Giu root la cluster co representative density cao hon.
            rep_a = representatives[ra]
            rep_b = representatives[rb]
            if densities[rep_a] >= densities[rep_b]:
                parent[rb] = ra
            else:
                parent[ra] = rb

    # Merge bao thu: chi merge cac representative rat gan nhau.
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

    # Nen lai label ve 0..k-1.
    final_unique = sorted(np.unique(merged))
    remap = {old: new for new, old in enumerate(final_unique)}
    merged = np.array([remap[x] for x in merged], dtype=int)
    return merged


# ============================================================
# Evaluation
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


# ============================================================
# Main improved GE-DPC pipeline
# ============================================================
def run_ge_dpc_cholesky_graph_quality(
    feature_file: Path,
    label_file: Path,
    epsilon: float = 1e-6,
    outlier_t: float = 2.0,
    auto_center_k: Optional[int] = None,
    min_centers: int = 2,
    max_centers: Optional[int] = None,
    enable_light_merge: bool = True,
) -> Dict[str, object]:
    """
    Pipeline GE-DPC cai tien chat luong.

    Buoc 1. Doc du lieu va nhan that.
    Buoc 2. Sinh granular-ellipsoid.
    Buoc 3. Toi uu tinh toan hinh hoc bang Cholesky/cache.
    Buoc 4. Tinh DPC attributes o muc ellipsoid.
    Buoc 5. Cai tien chon center.
    Buoc 6. Gan nhan bang graph-based multi-neighbor propagation.
    Buoc 7. Merge cum nhe neu chon du center.
    Buoc 8. Anh xa nhan ellipsoid ve diem du lieu.
    Buoc 9. Danh gia ACC/NMI/ARI va thoi gian chay.
    """
    data = np.loadtxt(feature_file, dtype=float)
    true_labels = np.loadtxt(label_file, dtype=float).astype(int)

    if data.ndim == 1:
        data = data.reshape(-1, 1)

    num = int(np.ceil(np.sqrt(data.shape[0])))

    # ------------------------------
    # 1-2) Sinh ellipsoid
    # ------------------------------
    t_gen_start = time.time()
    initial_indices = np.arange(data.shape[0], dtype=int)
    initial_ellipsoid = Ellipsoid(data, initial_indices, epsilon=epsilon)
    ellipsoid_list: List[Ellipsoid] = [initial_ellipsoid]

    print("=" * 80)
    print("GE-DPC Cholesky + Graph Quality Improvement")
    print(f"Data shape: n={data.shape[0]}, d={data.shape[1]}")
    print(f"Safe split threshold num = ceil(sqrt(n)) = {num}")
    print("Initial ellipsoid count (So luong ellipsoid ban dau): 1")

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
    # 5) Improved center selection
    # ------------------------------
    print("Auto-selecting cluster centers with quality-aware redundancy pruning...")
    selected = auto_select_centers_quality(
        densities=densities,
        min_dists=min_dists,
        dist_mat=dist_matrix,
        top_k=auto_center_k,
        min_centers=min_centers,
        max_centers=max_centers,
    )
    print(f"Selected centers: {selected}")
    print(f"Selected gamma values: {[float(gamma[i]) for i in selected]}")

    # ------------------------------
    # 6-7) Graph propagation + light merge
    # ------------------------------
    t_cluster_start = time.time()
    ellipsoid_labels_before_merge = ellipse_cluster_graph_propagation(
        densities=densities,
        dist_mat=dist_matrix,
        centers=selected,
    )

    if enable_light_merge:
        ellipsoid_labels = merge_close_clusters_light(
            ellipsoid_labels=ellipsoid_labels_before_merge,
            centers=selected,
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

    # ------------------------------
    # 8) Mapping ellipsoid label -> data point label
    # ------------------------------
    print("Mapping ellipsoid labels to data points...")
    pred_labels = np.full(len(data), -1, dtype=int)
    for i, ell in enumerate(ellipsoid_list):
        pred_labels[ell.indices] = ellipsoid_labels[i]

    if np.any(pred_labels == -1):
        raise RuntimeError("Some data points were not assigned a cluster label.")

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
    print(f"1. Ellipsoid generation time: {time_gen:.14f} seconds")
    print(f"2. Attribute computation time: {time_attr:.14f} seconds")
    print(f"3. Clustering computation time: {time_cluster:.14f} seconds")
    print(f"Total effective runtime (attributes + clustering): {total_valid_time1:.14f} seconds")
    print(f"Total effective runtime of the program: {total_valid_time2:.14f} seconds")
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
    """
    Registry giu nguyen tu source cua ban.
    Neu dataset nao khong ton tai, run_all_default_datasets() se bo qua va bao loi.
    """
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


def run_named_dataset(
    dataset_name: str,
    base_dir: Path,
    epsilon: float = 1e-6,
    outlier_t: float = 2.0,
    auto_center_k: Optional[int] = None,
    min_centers: int = 2,
    max_centers: Optional[int] = None,
    enable_light_merge: bool = True,
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

    return run_ge_dpc_cholesky_graph_quality(
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
    auto_center_k: Optional[int] = None,
    min_centers: int = 2,
    max_centers: Optional[int] = None,
    enable_light_merge: bool = True,
) -> Dict[str, Dict[str, object]]:
    """Chay nhieu dataset va in bang tong hop."""
    registry = get_default_dataset_registry(base_dir)
    if dataset_names is None:
        dataset_names = list(registry.keys())

    results: Dict[str, Dict[str, object]] = {}

    for dataset_name in dataset_names:
        try:
            results[dataset_name] = run_named_dataset(
                dataset_name=dataset_name,
                base_dir=base_dir,
                epsilon=epsilon,
                outlier_t=outlier_t,
                auto_center_k=auto_center_k,
                min_centers=min_centers,
                max_centers=max_centers,
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

    # Khuyen nghi:
    # - auto_center_k=None de tu dong chon center.
    # - min_centers co the dat theo kien thuc dataset neu biet so lop toi thieu.
    # - max_centers=None de khong ep cung; light merge se gom neu chon du.
    auto_center_k = None
    min_centers = 2
    max_centers = None
    enable_light_merge = False

    # Chay mot dataset.
    dataset_name = "segment_3"
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

    # Chay 8 dataset ban hay dung thi sua danh sach nay cho dung bo can chay.
    # dataset_names = [
    #     "iris", "seed", "segment_3", "landsat_2",
    #     "msplice_2", "rice", "banknote", "htru2",
    # ]
    # all_results = run_all_default_datasets(
    #     base_dir=BASE_DIR,
    #     dataset_names=dataset_names,
    #     epsilon=epsilon,
    #     outlier_t=outlier_t,
    #     auto_center_k=auto_center_k,
    #     min_centers=min_centers,
    #     max_centers=max_centers,
    #     enable_light_merge=enable_light_merge,
    # )
