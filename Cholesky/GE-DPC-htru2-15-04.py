import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import accuracy_score, adjusted_rand_score, normalized_mutual_info_score

# GE-DPC with Cholesky + Cache - htru2 dedicated version
# - Keep the new development logic: Cholesky + cache
# - Do NOT use approximate matrix computation / randomized SVD
# - Customize ONLY center selection for htru2
# - Keep ellipsoid generation / split / density / distance / DPC assignment logic


class Ellipsoid:
    def __init__(self, data: np.ndarray, indices: np.ndarray, epsilon: float = 1e-6):
        self.data = np.asarray(data, dtype=float)
        self.indices = np.asarray(indices, dtype=int)
        self.epsilon = float(epsilon)

        if self.data.ndim != 2 or self.data.shape[0] == 0:
            raise ValueError("Ellipsoid cannot be created with empty or non-2D data.")

        self.n_samples, self.dim = self.data.shape
        self.center = np.mean(self.data, axis=0)

        # cache
        self._cov_matrix: Optional[np.ndarray] = None
        self._H_matrix: Optional[np.ndarray] = None
        self._chol_factor: Optional[Tuple[np.ndarray, bool]] = None
        self._rho: Optional[float] = None
        self._lengths_rotation: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self._major_axis_endpoints: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self._density: Optional[float] = None

    @property
    def cov_matrix(self) -> np.ndarray:
        if self._cov_matrix is None:
            if self.n_samples == 1:
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
    def chol_factor(self) -> Tuple[np.ndarray, bool]:
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
        mahal_sq = np.einsum("ij,ij->i", diffs, solved)
        return np.maximum(mahal_sq, 0.0)

    @property
    def rho(self) -> float:
        if self._rho is None:
            self._rho = float(np.sqrt(np.max(self.mahal_sq_points(self.data))))
        return self._rho

    @property
    def lengths_rotation(self) -> Tuple[np.ndarray, np.ndarray]:
        if self._lengths_rotation is None:
            eigvals_H, eigvecs_H = np.linalg.eigh(self.H_matrix)
            eigvals_H = np.maximum(eigvals_H, 1e-12)
            self._lengths_rotation = (self.rho * np.sqrt(eigvals_H), eigvecs_H)
        return self._lengths_rotation

    @property
    def lengths(self) -> np.ndarray:
        return self.lengths_rotation[0]

    @property
    def rotation(self) -> np.ndarray:
        return self.lengths_rotation[1]

    @property
    def major_axis_endpoints(self) -> Tuple[np.ndarray, np.ndarray]:
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
        if self._density is None:
            axes_sum = max(float(np.sum(self.lengths)), 1e-12)
            mahal_sq = self.mahal_sq_points(self.data)
            total_mahal = max(float(np.sum(np.sqrt(mahal_sq))), 1e-12)
            self._density = float((self.n_samples ** 2) / (axes_sum * total_mahal))
        return self._density


def get_num(ellipsoid: Ellipsoid) -> int:
    return ellipsoid.n_samples


def calculate_ellipsoid_density(ellipsoid: Ellipsoid) -> float:
    return ellipsoid.density


def splits(ellipsoid_list: Sequence[Ellipsoid], num: int, epsilon: float) -> List[Ellipsoid]:
    new_ells: List[Ellipsoid] = []
    for ell in ellipsoid_list:
        if get_num(ell) < num:
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

    return [Ellipsoid(cluster1, cluster1_idx, epsilon=eps), Ellipsoid(cluster2, cluster2_idx, epsilon=eps)]


def recursive_split_outlier_detection(initial_ellipsoids, data, t=2.0, max_iterations=10, epsilon=1e-6):
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
        new_ellipsoids = []
        min_leaf = max(2, int(np.ceil(np.sqrt(data.shape[0]) * 0.1)))

        for outlier_ell in outlier_ellipsoids:
            children = splits_ellipsoid(outlier_ell, epsilon=epsilon)
            if len(children) != 2 or any(child.n_samples < min_leaf for child in children):
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


def ellipse_mahalanobis_distance(ellipsoid_i: Ellipsoid, ellipsoid_j: Ellipsoid):
    avg_H = 0.5 * (ellipsoid_i.H_matrix + ellipsoid_j.H_matrix)
    chol_avg = cho_factor(avg_H, lower=True, check_finite=False)
    diff = ellipsoid_i.center - ellipsoid_j.center
    solved = cho_solve(chol_avg, diff, check_finite=False)
    dist = float(np.sqrt(max(float(diff.T @ solved), 0.0)))
    return dist, dist, dist


def ellipse_distance(ellipsoid_list: Sequence[Ellipsoid]) -> np.ndarray:
    n = len(ellipsoid_list)
    dist_mat = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            _, _, rel_dist = ellipse_mahalanobis_distance(ellipsoid_list[i], ellipsoid_list[j])
            dist_mat[i, j] = dist_mat[j, i] = rel_dist
    return dist_mat


def ellipse_min_dist(dist_mat: np.ndarray, densities: np.ndarray):
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


def auto_select_centers(densities, min_dists, mode="knee", top_k=None, min_centers=1, max_centers=None):
    densities = np.asarray(densities, dtype=float)
    min_dists = np.asarray(min_dists, dtype=float)
    n = len(densities)
    if n == 0:
        return []

    gamma = densities * min_dists
    order = np.argsort(-gamma)

    if top_k is not None:
        return order[:int(max(1, min(top_k, n)))].tolist()

    if mode == "threshold":
        g = gamma[order]
        thr = np.mean(g) + np.std(g)
        centers = order[g > thr].tolist()
        if len(centers) < min_centers:
            centers = order[:min_centers].tolist()
        if max_centers is not None:
            centers = centers[:max_centers]
        return centers

    g = gamma[order]
    if n == 1 or np.allclose(g, g[0]):
        k = min_centers if max_centers is None else min(min_centers, max_centers)
        return order[:k].tolist()

    x = np.arange(n, dtype=float)
    x_norm = x / max(n - 1, 1)
    y_norm = (g - g.min()) / max(g.max() - g.min(), 1e-12)
    p1 = np.array([x_norm[0], y_norm[0]], dtype=float)
    p2 = np.array([x_norm[-1], y_norm[-1]], dtype=float)
    line_vec = p2 - p1
    line_norm = np.linalg.norm(line_vec)

    if line_norm < 1e-12:
        knee_idx = min_centers - 1
    else:
        dists = [abs(np.cross(line_vec, np.array([xi, yi]) - p1)) / line_norm for xi, yi in zip(x_norm, y_norm)]
        knee_idx = int(np.argmax(dists))

    k = max(min_centers, knee_idx + 1)
    if max_centers is not None:
        k = min(k, max_centers)
    return order[:min(k, n)].tolist()


# htru2 special center selection
# - keep new GE-DPC logic
# - only replace center selection rule for htru2
def auto_select_two_peaks_left_right(densities, min_dists):
    densities = np.asarray(densities, dtype=float)
    min_dists = np.asarray(min_dists, dtype=float)
    n = len(densities)
    if n == 0:
        return []
    if n == 1:
        return [0]

    gamma = densities * min_dists
    density_median = float(np.median(densities))
    left_candidates = np.where(densities <= density_median)[0]
    right_candidates = np.where(densities > density_median)[0]

    if len(left_candidates) == 0 or len(right_candidates) == 0:
        return np.argsort(-gamma)[:2].tolist()

    # ưu tiên delta cao; density chỉ hỗ trợ nhẹ để tránh peak nhiễu yếu
    left_score = min_dists[left_candidates] * (1.0 + 0.05 * densities[left_candidates])
    right_score = min_dists[right_candidates] * (1.0 + 0.05 * densities[right_candidates])

    left_center = int(left_candidates[np.argmax(left_score)])
    right_center = int(right_candidates[np.argmax(right_score)])

    if left_center == right_center:
        return np.argsort(-gamma)[:2].tolist()

    selected = [left_center, right_center]
    selected = sorted(selected, key=lambda i: gamma[i], reverse=True)
    return selected


def ellipse_cluster(densities, centers, nearest):
    labels = -np.ones(len(densities), dtype=int)
    for i, c in enumerate(centers):
        labels[c] = i
    order = np.argsort(-np.asarray(densities))
    for idx in order:
        if labels[idx] == -1 and nearest[idx] != -1:
            labels[idx] = labels[nearest[idx]]
    if np.any(labels == -1):
        labels[labels == -1] = 0 if len(centers) == 0 else len(centers)
    return labels


def align_labels(true_labels, pred_labels):
    true_classes = np.unique(true_labels)
    pred_classes = np.unique(pred_labels)
    confusion = np.zeros((len(true_classes), len(pred_classes)), dtype=float)
    for i, t in enumerate(true_classes):
        for j, p in enumerate(pred_classes):
            confusion[i, j] = np.sum((true_labels == t) & (pred_labels == p))
    row_ind, col_ind = linear_sum_assignment(-confusion)
    mapping = {pred_classes[j]: true_classes[i] for i, j in zip(row_ind, col_ind)}
    return np.array([mapping.get(label, -1) for label in pred_labels])


def run_ge_dpc_htru2_cholesky_cache(feature_file: Path, label_file: Path, epsilon: float = 1e-6, outlier_t: float = 2.0, auto_center_mode: str = "knee", auto_center_k: Optional[int] = None, min_centers: int = 1, max_centers: Optional[int] = None):
    data = np.loadtxt(feature_file, dtype=float)
    true_labels = np.loadtxt(label_file, dtype=float).astype(int)
    dataset_name = feature_file.stem.lower()
    num = int(np.ceil(np.sqrt(data.shape[0])))

    t_gen_start = time.time()
    ellipsoid_list = [Ellipsoid(data, np.arange(data.shape[0], dtype=int), epsilon=epsilon)]
    print("Initial ellipsoid count (Số lượng ellipsoid ban đầu): 1")

    iteration = 0
    while True:
        iteration += 1
        before = len(ellipsoid_list)
        ellipsoid_list = splits(ellipsoid_list, num=num, epsilon=epsilon)
        after = len(ellipsoid_list)
        print(f"Ellipsoid count after safe split iteration {iteration} (Số lượng ellipsoid sau lần phân tách an toàn thứ {iteration}): {after}")
        if after == before:
            break

    print(f"Total ellipsoid count after safe splitting (Tổng số ellipsoid sau phân tách an toàn): {len(ellipsoid_list)}")
    ellipsoid_list = recursive_split_outlier_detection(ellipsoid_list, data, t=outlier_t, max_iterations=10, epsilon=epsilon)
    print(f"Total ellipsoid count after outlier-detection splitting (Tổng số ellipsoid sau phân tách bằng phát hiện ngoại lệ): {len(ellipsoid_list)}")
    print(f"A total of {len(ellipsoid_list)} ellipsoids were generated (Tổng cộng đã tạo {len(ellipsoid_list)} ellipsoid)")
    time_gen = time.time() - t_gen_start

    t_attr_start = time.time()
    densities = np.array([calculate_ellipsoid_density(ell) for ell in ellipsoid_list], dtype=float)
    dist_matrix = ellipse_distance(ellipsoid_list)
    min_dists, nearest = ellipse_min_dist(dist_matrix, densities)
    axes_sums = [np.sum(ell.lengths) for ell in ellipsoid_list]
    axes_sum_avg = np.mean(axes_sums) if axes_sums else 0.0
    outlier_ellipsoids = [ell for ell in ellipsoid_list if np.sum(ell.lengths) > 2 * axes_sum_avg]
    print(f"Number of outlier ellipsoids (Số lượng ellipsoid ngoại lệ): {len(outlier_ellipsoids)}")
    time_attr = time.time() - t_attr_start

    print("Auto-selecting cluster centers from decision values (Tự động chọn tâm cụm từ các giá trị decision)...")
    if dataset_name == "htru2":
        selected = auto_select_two_peaks_left_right(densities, min_dists)
        print("Selection rule: htru2 special mode -> two peaks left/right (Chế độ riêng htru2 -> 2 peak trái/phải)")
    else:
        selected = auto_select_centers(densities, min_dists, mode=auto_center_mode, top_k=auto_center_k, min_centers=min_centers, max_centers=max_centers)
        print(f"Selection rule: default auto mode = {auto_center_mode}")

    gamma = densities * min_dists
    print(f"Selected cluster centers (Các tâm cụm đã chọn): {selected}")
    print(f"Selected gamma values (Giá trị gamma của các tâm cụm): {[float(gamma[i]) for i in selected]}")
    print(f"Selected (density, min_dist): {[(float(densities[i]), float(min_dists[i])) for i in selected]}")

    t_cluster_start = time.time()
    ellipsoid_labels = ellipse_cluster(densities, selected, nearest)
    time_cluster = time.time() - t_cluster_start

    print("Mapping data points in progress (Đang ánh xạ điểm dữ liệu, không tính vào thời gian phân cụm)...")
    pred_labels = np.full(len(data), -1, dtype=int)
    for i, ell in enumerate(ellipsoid_list):
        pred_labels[ell.indices] = ellipsoid_labels[i]

    if np.any(pred_labels == -1):
        raise RuntimeError("Some data points were not assigned a cluster label.")

    print("Calculating evaluation metrics (Đang tính các chỉ số đánh giá)...")
    aligned_pred_labels = align_labels(true_labels, pred_labels)
    acc = accuracy_score(true_labels, aligned_pred_labels)
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
    total_valid_time1 = time_attr + time_cluster
    total_valid_time2 = time_gen + time_attr + time_cluster
    print("-" * 30)
    print(f"Total effective runtime (attributes + clustering) (Tổng thời gian hiệu dụng: thuộc tính + phân cụm): {total_valid_time1:.14f} seconds (giây)")
    print(f"Total effective runtime of the program (Tổng thời gian chạy hiệu dụng của chương trình): {total_valid_time2:.14f} seconds (giây)")
    print("-" * 30)

    return {
        "acc": acc,
        "nmi": nmi,
        "ari": ari,
        "selected_centers": selected,
        "n_ellipsoids": len(ellipsoid_list),
        "pred_labels": pred_labels,
        "aligned_pred_labels": aligned_pred_labels,
        "generation_time": time_gen,
        "attribute_time": time_attr,
        "cluster_time": time_cluster,
        "total_valid_time_attr_cluster": total_valid_time1,
        "total_valid_time_program": total_valid_time2,
    }


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent
    feature_file = BASE_DIR / "real_dataset_and_label" / "real_datasets" / "htru2.txt"
    label_file = BASE_DIR / "real_dataset_and_label" / "real_datasets_label" / "htru2_label.txt"

    epsilon = 1e-6
    outlier_t = 2.0
    auto_center_mode = "knee"
    auto_center_k = 2
    min_centers = 2
    max_centers = 2

    run_ge_dpc_htru2_cholesky_cache(
        feature_file=feature_file,
        label_file=label_file,
        epsilon=epsilon,
        outlier_t=outlier_t,
        auto_center_mode=auto_center_mode,
        auto_center_k=auto_center_k,
        min_centers=min_centers,
        max_centers=max_centers,
    )
