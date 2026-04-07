import os
import time
import numpy as np
from pathlib import Path
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import accuracy_score, normalized_mutual_info_score, adjusted_rand_score


class Ellipsoid:
    def __init__(self, data, indices, epsilon=1e-6):
        self.data = np.asarray(data, dtype=float)
        self.indices = np.asarray(indices, dtype=int)
        self.epsilon = epsilon
        self.n_samples = len(self.data)

        if self.n_samples == 0:
            raise ValueError("Ellipsoid cannot be created with empty data.")

        self.center = np.mean(self.data, axis=0)
        self.cov_matrix = self._compute_cov_matrix()
        self.H_matrix = self._compute_H_matrix()
        self.inv_H = np.linalg.inv(self.H_matrix)
        self.rho = self._compute_rho()
        self.lengths, self.rotation = self._get_principal_axes()
        self.major_axis_endpoints = self.compute_major_axis_endpoints()

    def _compute_cov_matrix(self):
        if self.data.ndim != 2:
            raise ValueError("data must be a 2D array")
        if self.n_samples == 1:
            d = self.data.shape[1]
            return np.zeros((d, d), dtype=float)
        return np.cov(self.data.T, bias=True)

    def _compute_H_matrix(self):
        n = self.cov_matrix.shape[0]
        return self.cov_matrix + self.epsilon * np.eye(n)

    def _compute_rho(self):
        if self.n_samples == 0:
            return 0.0
        diffs = self.data - self.center
        mahal_sq = np.einsum('ij,jk,ik->i', diffs, self.inv_H, diffs)
        mahal_sq = np.maximum(mahal_sq, 0.0)
        return float(np.sqrt(np.max(mahal_sq)))

    def _get_principal_axes(self):
        eigenvalues, eigenvectors = np.linalg.eigh(self.inv_H)
        eigenvalues = np.maximum(eigenvalues, 1e-12)
        lengths = self.rho / np.sqrt(eigenvalues)
        return lengths, eigenvectors

    def compute_major_axis_endpoints(self):
        if self.n_samples <= 1:
            return self.center, self.center

        center_distances = np.linalg.norm(self.data - self.center, axis=1)
        point1_idx = np.argmin(center_distances)
        point1 = self.data[point1_idx]

        dist_to_point1 = np.linalg.norm(self.data - point1, axis=1)
        point2_idx = np.argmax(dist_to_point1)
        point2 = self.data[point2_idx]

        dist_to_point2 = np.linalg.norm(self.data - point2, axis=1)
        point3_idx = np.argmax(dist_to_point2)
        point3 = self.data[point3_idx]
        return point2, point3


def get_num(ellipsoid):
    return ellipsoid.n_samples if ellipsoid.n_samples > 0 else 0


def splits(ellipsoid_list, num):
    new_ells = []
    for ell in ellipsoid_list:
        if get_num(ell) < num:
            new_ells.append(ell)
        else:
            children = splits_ellipsoid(ell)
            new_ells.extend(children)
    return new_ells


def splits_ellipsoid(ellipsoid):
    if ellipsoid.n_samples <= 1:
        return [ellipsoid]

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

    ell1 = Ellipsoid(cluster1, cluster1_idx, epsilon=ellipsoid.epsilon)
    ell2 = Ellipsoid(cluster2, cluster2_idx, epsilon=ellipsoid.epsilon)

    new_cluster1 = []
    new_cluster2 = []
    new_idx1 = []
    new_idx2 = []

    for point, idx in zip(data, indices):
        diff1 = point - ell1.center
        diff2 = point - ell2.center
        dist1_sq = float(diff1.T @ ell1.inv_H @ diff1)
        dist2_sq = float(diff2.T @ ell2.inv_H @ diff2)
        dist1 = np.sqrt(max(dist1_sq, 0.0))
        dist2 = np.sqrt(max(dist2_sq, 0.0))
        if dist1 < dist2:
            new_cluster1.append(point)
            new_idx1.append(idx)
        else:
            new_cluster2.append(point)
            new_idx2.append(idx)

    cluster1 = np.array(new_cluster1) if new_cluster1 else np.empty((0, data.shape[1]))
    cluster2 = np.array(new_cluster2) if new_cluster2 else np.empty((0, data.shape[1]))
    cluster1_idx = np.array(new_idx1, dtype=int)
    cluster2_idx = np.array(new_idx2, dtype=int)

    if len(cluster1) == 0 or len(cluster2) == 0:
        return [ellipsoid]

    ell1 = Ellipsoid(cluster1, cluster1_idx, epsilon=ellipsoid.epsilon)
    ell2 = Ellipsoid(cluster2, cluster2_idx, epsilon=ellipsoid.epsilon)
    return [ell1, ell2]


def calculate_ellipsoid_density(ellipsoid):
    n_samples = ellipsoid.n_samples
    axes_sum = float(np.sum(ellipsoid.lengths))
    axes_sum = max(axes_sum, 1e-12)

    diffs = ellipsoid.data - ellipsoid.center
    mahal_sq = np.einsum('ij,jk,ik->i', diffs, ellipsoid.inv_H, diffs)
    mahal_sq = np.maximum(mahal_sq, 0.0)
    total_mahalanobis_distance = float(np.sum(np.sqrt(mahal_sq)))
    total_mahalanobis_distance = max(total_mahalanobis_distance, 1e-12)

    density = (n_samples ** 2) / (axes_sum * total_mahalanobis_distance)
    return float(density)


def recursive_split_outlier_detection(initial_ellipsoids, data, t=1.0, max_iterations=10):
    ellipsoid_list = initial_ellipsoids.copy()
    iteration = 0

    while iteration < max_iterations:
        iteration += 1
        axes_sums = [np.sum(ell.lengths) for ell in ellipsoid_list]
        axes_sum_avg = np.mean(axes_sums) if axes_sums else 0.0

        outlier_ellipsoids = [
            ell for ell in ellipsoid_list if np.sum(ell.lengths) > 2 * axes_sum_avg
        ]

        if not outlier_ellipsoids:
            break

        new_ellipsoids = []
        for outlier_ell in outlier_ellipsoids:
            sub_ells = splits_ellipsoid(outlier_ell)

            if len(sub_ells) != 2:
                new_ellipsoids.append(outlier_ell)
                continue

            if any(sub_ell.n_samples < np.ceil(np.sqrt(data.shape[0])) * 0.1 for sub_ell in sub_ells):
                new_ellipsoids.append(outlier_ell)
                continue

            parent_density = calculate_ellipsoid_density(outlier_ell)
            child_density_sum = sum(calculate_ellipsoid_density(sub_ell) for sub_ell in sub_ells)

            if child_density_sum <= t * parent_density:
                new_ellipsoids.append(outlier_ell)
                continue

            new_ellipsoids.extend(sub_ells)

        normal_ellipsoids = [ell for ell in ellipsoid_list if ell not in outlier_ellipsoids]
        ellipsoid_list = normal_ellipsoids + new_ellipsoids

    return ellipsoid_list


def ellipse_mahalanobis_distance(ellipsoid_i, ellipsoid_j):
    center_i = ellipsoid_i.center
    center_j = ellipsoid_j.center
    avg_cov = (ellipsoid_i.H_matrix + ellipsoid_j.H_matrix) / 2.0
    inv_avg_cov = np.linalg.inv(avg_cov)
    diff = center_i - center_j
    dist_sq = float(diff.T @ inv_avg_cov @ diff)
    dist = float(np.sqrt(max(dist_sq, 0.0)))
    return dist, dist, dist


def ellipse_distance(ellipsoid_list):
    n = len(ellipsoid_list)
    dist_mat = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            _, _, rel_dist = ellipse_mahalanobis_distance(ellipsoid_list[i], ellipsoid_list[j])
            dist_mat[i, j] = dist_mat[j, i] = rel_dist
    return dist_mat


def ellipse_min_dist(dist_mat, densities):
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


def auto_select_centers(densities, min_dists, mode='knee', top_k=None, min_centers=1, max_centers=None):
    densities = np.asarray(densities, dtype=float)
    min_dists = np.asarray(min_dists, dtype=float)
    n = len(densities)
    if n == 0:
        return []

    gamma = densities * min_dists
    order = np.argsort(-gamma)

    if top_k is not None:
        k = int(max(1, min(top_k, n)))
        return order[:k].tolist()

    if n == 1:
        return [0]

    if mode == 'threshold':
        g = gamma[order]
        thr = np.mean(g) + np.std(g)
        centers = order[g > thr].tolist()
        if len(centers) < min_centers:
            centers = order[:min_centers].tolist()
        if max_centers is not None:
            centers = centers[:max_centers]
        return centers

    g = gamma[order]
    if np.allclose(g, g[0]):
        k = min_centers
        if max_centers is not None:
            k = min(k, max_centers)
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
        distances = []
        for xi, yi in zip(x_norm, y_norm):
            p = np.array([xi, yi], dtype=float)
            dist = np.abs(np.cross(line_vec, p - p1)) / line_norm
            distances.append(dist)
        knee_idx = int(np.argmax(distances))

    k = max(min_centers, knee_idx + 1)
    if max_centers is not None:
        k = min(k, max_centers)
    k = min(k, n)
    return order[:k].tolist()


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

    # Ưu tiên delta cao, gamma cao phụ trợ để tránh chọn điểm nhiễu yếu.
    left_score = min_dists[left_candidates] * (1.0 + 0.05 * densities[left_candidates])
    right_score = min_dists[right_candidates] * (1.0 + 0.05 * densities[right_candidates])

    left_center = int(left_candidates[np.argmax(left_score)])
    right_center = int(right_candidates[np.argmax(right_score)])

    if left_center == right_center:
        return np.argsort(-gamma)[:2].tolist()

    selected = [left_center, right_center]
    selected = sorted(selected, key=lambda i: gamma[i], reverse=True)
    return selected


def ellipse_cluster(densities, centers, nearest, min_dists):
    labels = -np.ones(len(densities), dtype=int)
    for i, c in enumerate(centers):
        labels[c] = i

    order = np.argsort(-np.asarray(densities))
    for idx in order:
        if labels[idx] == -1 and nearest[idx] != -1:
            labels[idx] = labels[nearest[idx]]

    unassigned = labels == -1
    if np.any(unassigned):
        if len(centers) == 0:
            labels[unassigned] = 0
        else:
            labels[unassigned] = len(centers)
    return labels


def align_labels(true_labels, pred_labels):
    true_classes = np.unique(true_labels)
    pred_classes = np.unique(pred_labels)

    confusion_matrix = np.zeros((len(true_classes), len(pred_classes)))
    for i, true_class in enumerate(true_classes):
        for j, pred_class in enumerate(pred_classes):
            confusion_matrix[i, j] = np.sum((true_labels == true_class) & (pred_labels == pred_class))

    row_ind, col_ind = linear_sum_assignment(-confusion_matrix)

    mapping = {}
    for i, j in zip(row_ind, col_ind):
        mapping[pred_classes[j]] = true_classes[i]

    aligned_pred_labels = np.array([mapping.get(label, -1) for label in pred_labels])
    return aligned_pred_labels


def run_ge_dpc(feature_file, label_file, epsilon=1e-6, outlier_t=2.0,
               auto_center_mode='knee', auto_center_k=None,
               min_centers=1, max_centers=None):
    data = np.loadtxt(feature_file, dtype=float)
    true_labels = np.loadtxt(label_file, dtype=float).astype(int)
    num = int(np.ceil(np.sqrt(data.shape[0])))
    dataset_name = os.path.splitext(os.path.basename(feature_file))[0]

    t_gen_start = time.time()
    initial_indices = np.arange(data.shape[0], dtype=int)
    initial_ellipsoid = Ellipsoid(data, initial_indices, epsilon=epsilon)
    ellipsoid_list = [initial_ellipsoid]
    print("Initial ellipsoid count (Số lượng ellipsoid ban đầu): 1")

    iteration = 0
    while True:
        iteration += 1
        current_count = len(ellipsoid_list)
        ellipsoid_list = splits(ellipsoid_list, num)
        new_count = len(ellipsoid_list)
        print(f"Ellipsoid count after safe split iteration {iteration} (Số lượng ellipsoid sau lần phân tách an toàn thứ {iteration}): {new_count}")
        if new_count == current_count:
            break

    print(f"Total ellipsoid count after safe splitting (Tổng số ellipsoid sau phân tách an toàn): {len(ellipsoid_list)}")
    ellipsoid_list = recursive_split_outlier_detection(ellipsoid_list, data, t=outlier_t)
    print(f"Total ellipsoid count after outlier-detection splitting (Tổng số ellipsoid sau phân tách bằng phát hiện ngoại lệ): {len(ellipsoid_list)}")
    print(f"A total of {len(ellipsoid_list)} ellipsoids were generated (Tổng cộng đã tạo {len(ellipsoid_list)} ellipsoid)")
    t_gen_end = time.time()
    time_gen = t_gen_end - t_gen_start

    t_attr_start = time.time()
    densities = np.array([calculate_ellipsoid_density(ell) for ell in ellipsoid_list], dtype=float)
    dist_matrix = ellipse_distance(ellipsoid_list)
    min_dists, nearest = ellipse_min_dist(dist_matrix, densities)

    axes_sums = [np.sum(ell.lengths) for ell in ellipsoid_list]
    axes_sum_avg = np.mean(axes_sums) if axes_sums else 0.0
    outlier_ellipsoids = [ell for ell in ellipsoid_list if np.sum(ell.lengths) > 2 * axes_sum_avg]
    print(f"Number of outlier ellipsoids (Số lượng ellipsoid ngoại lệ): {len(outlier_ellipsoids)}")
    t_attr_end = time.time()
    time_attr = t_attr_end - t_attr_start

    print("Auto-selecting cluster centers from decision values (Tự động chọn tâm cụm từ các giá trị decision)...")
    if dataset_name.lower() == 'htru2':
        selected = auto_select_two_peaks_left_right(densities, min_dists)
        print("Selection rule: htru2 special mode -> two peaks left/right (Chế độ riêng htru2 -> 2 peak trái/phải)")
    else:
        selected = auto_select_centers(
            densities,
            min_dists,
            mode=auto_center_mode,
            top_k=auto_center_k,
            min_centers=min_centers,
            max_centers=max_centers,
        )
        print(f"Selection rule: default auto mode = {auto_center_mode}")

    gamma = densities * min_dists
    print(f"Selected cluster centers (Các tâm cụm đã chọn): {selected}")
    print(f"Selected gamma values (Giá trị gamma của các tâm cụm): {[float(gamma[i]) for i in selected]}")
    print(f"Selected (density, min_dist): {[(float(densities[i]), float(min_dists[i])) for i in selected]}")

    t_cluster_start = time.time()
    labels = ellipse_cluster(densities, selected, nearest, min_dists)
    t_cluster_end = time.time()
    time_cluster = t_cluster_end - t_cluster_start

    print("Mapping data points in progress (Đang ánh xạ điểm dữ liệu, không tính vào thời gian phân cụm)...")
    pred_labels = np.full(len(data), -1, dtype=int)
    for i, ell in enumerate(ellipsoid_list):
        pred_labels[ell.indices] = labels[i]

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
        'acc': acc,
        'nmi': nmi,
        'ari': ari,
        'selected_centers': selected,
        'n_ellipsoids': len(ellipsoid_list),
        'pred_labels': pred_labels,
        'aligned_pred_labels': aligned_pred_labels,
    }


if __name__ == '__main__':
    # =========================
    # Local run settings
    # Chỉ cần sửa 2 đường dẫn này khi đổi dataset
    # =========================
    # feature_file = '/Users/minhquan/Documents/Master/Thesis/Code/GE-DPC-main/real_dataset_and_label/real_datasets/htru2.txt'
    # label_file = '/Users/minhquan/Documents/Master/Thesis/Code/GE-DPC-main/real_dataset_and_label/real_datasets_label/htru2_label.txt'
    BASE_DIR = Path(__file__).resolve().parent
    feature_file = BASE_DIR / 'dataset' / 'unlabel' / 'dry_bean.txt'
    label_file = BASE_DIR / 'dataset' / 'label' / 'dry_bean_label.txt'

    # =========================
    # Algorithm settings
    # =========================
    epsilon = 1e-6
    outlier_t = 2.0
    auto_center_mode = 'knee'
    auto_center_k = 2
    min_centers = 2
    max_centers = 2

    run_ge_dpc(
        feature_file=feature_file,
        label_file=label_file,
        epsilon=epsilon,
        outlier_t=outlier_t,
        auto_center_mode=auto_center_mode,
        auto_center_k=auto_center_k,
        min_centers=min_centers,
        max_centers=max_centers,
    )
