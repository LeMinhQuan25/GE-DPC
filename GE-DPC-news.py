import time
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import accuracy_score, normalized_mutual_info_score, adjusted_rand_score
from sklearn.utils.extmath import randomized_svd


# ============================================================
# GE-DPC for high-dimensional data
# Main idea:
# 1) Replace exact covariance inverse with low-rank approximate shape model
# 2) Use randomized SVD on centered data inside each ellipsoid
# 3) Use Woodbury identity for fast Mahalanobis distance and pair distance
# This keeps the GE-DPC spirit: covariance-aware ellipsoids + Mahalanobis-based splitting/DPC
# while reducing the main O(d^3) bottleneck in high dimensions.
# ============================================================


class FastEllipsoid:
    def __init__(self, data, indices, epsilon=1e-6, approx_rank=16, oversamples=8, n_iter=2):
        self.data = np.asarray(data, dtype=float)
        self.indices = np.asarray(indices, dtype=int)
        self.epsilon = float(epsilon)
        self.approx_rank = int(max(1, approx_rank))
        self.oversamples = int(max(2, oversamples))
        self.n_iter = int(max(0, n_iter))

        if self.data.ndim != 2 or len(self.data) == 0:
            raise ValueError("data must be a non-empty 2D array")

        self.n_samples, self.dim = self.data.shape
        self.center = np.mean(self.data, axis=0)
        self._fit_low_rank_shape()
        self.rho = self._compute_rho()
        self.lengths = self._compute_lengths()
        self.major_axis_endpoints = self.compute_major_axis_endpoints()

    def _fit_low_rank_shape(self):
        Xc = self.data - self.center
        d = self.dim

        if self.n_samples <= 1 or np.allclose(Xc, 0.0):
            self.rank = 0
            self.U = np.zeros((d, 0), dtype=float)
            self.cov_eigs = np.zeros((0,), dtype=float)
            self._inv_diag = np.zeros((0,), dtype=float)
            self._H_eigs = np.full(d, self.epsilon, dtype=float)
            return

        max_rank = min(self.approx_rank, d, self.n_samples)
        if max_rank <= 0:
            max_rank = 1

        # randomized SVD on centered samples matrix Xc (n x d)
        # singular values s -> covariance eigenvalues = s^2 / n
        try:
            _, s, Vt = randomized_svd(
                Xc,
                n_components=max_rank,
                n_oversamples=min(self.oversamples, max(2, d)),
                n_iter=self.n_iter,
                random_state=0,
            )
            U_feat = Vt.T
        except Exception:
            # fallback to exact SVD on small blocks
            _, s, Vt = np.linalg.svd(Xc, full_matrices=False)
            s = s[:max_rank]
            U_feat = Vt[:max_rank].T

        cov_eigs = (s ** 2) / max(self.n_samples, 1)
        keep = cov_eigs > 1e-12
        self.U = U_feat[:, keep]
        self.cov_eigs = cov_eigs[keep]
        self.rank = len(self.cov_eigs)
        self._inv_diag = 1.0 / (self.cov_eigs + self.epsilon)

        # approximate spectrum of H = S + eps I
        self._H_eigs = np.concatenate([
            self.cov_eigs + self.epsilon,
            np.full(max(0, d - self.rank), self.epsilon, dtype=float)
        ])

    def mahal_sq_points(self, points):
        X = np.asarray(points, dtype=float) - self.center
        if X.ndim == 1:
            X = X[None, :]

        if self.rank == 0:
            return np.sum(X * X, axis=1) / self.epsilon

        proj = X @ self.U                                # (m, r)
        parallel_sq = np.sum((proj ** 2) * self._inv_diag[None, :], axis=1)
        total_sq = np.sum(X * X, axis=1)
        proj_sq = np.sum(proj * proj, axis=1)
        perp_sq = np.maximum(total_sq - proj_sq, 0.0) / self.epsilon
        return parallel_sq + perp_sq

    def _compute_rho(self):
        mahal_sq = self.mahal_sq_points(self.data)
        mahal_sq = np.maximum(mahal_sq, 0.0)
        return float(np.sqrt(np.max(mahal_sq)))

    def _compute_lengths(self):
        # semi-axis lengths = rho * sqrt(eig(H))
        return self.rho * np.sqrt(np.maximum(self._H_eigs, 1e-18))

    def compute_major_axis_endpoints(self):
        if self.n_samples <= 1:
            return self.center, self.center

        # same spirit as original code: approximate farthest-point pair
        center_distances = np.linalg.norm(self.data - self.center, axis=1)
        point1 = self.data[np.argmin(center_distances)]
        point2 = self.data[np.argmax(np.linalg.norm(self.data - point1, axis=1))]
        point3 = self.data[np.argmax(np.linalg.norm(self.data - point2, axis=1))]
        return point2, point3


def get_num(ellipsoid):
    return ellipsoid.n_samples


def calculate_ellipsoid_density(ellipsoid):
    n_samples = ellipsoid.n_samples
    axes_sum = float(np.sum(ellipsoid.lengths))
    axes_sum = max(axes_sum, 1e-12)
    mahal_sq = ellipsoid.mahal_sq_points(ellipsoid.data)
    total_mahal = float(np.sum(np.sqrt(np.maximum(mahal_sq, 0.0))))
    total_mahal = max(total_mahal, 1e-12)
    return float((n_samples ** 2) / (axes_sum * total_mahal))


def splits(ellipsoid_list, num, ellipsoid_kwargs):
    new_ells = []
    for ell in ellipsoid_list:
        if get_num(ell) < num:
            new_ells.append(ell)
        else:
            new_ells.extend(splits_ellipsoid(ell, ellipsoid_kwargs))
    return new_ells


def splits_ellipsoid(ellipsoid, ellipsoid_kwargs):
    if ellipsoid.n_samples <= 1:
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
    if len(c1) == 0 or len(c2) == 0:
        return [ellipsoid]

    ell1 = FastEllipsoid(c1, i1, **ellipsoid_kwargs)
    ell2 = FastEllipsoid(c2, i2, **ellipsoid_kwargs)

    # Mahalanobis refinement step from the paper, but accelerated
    dist1_sq = ell1.mahal_sq_points(data)
    dist2_sq = ell2.mahal_sq_points(data)
    mask1 = dist1_sq < dist2_sq
    mask2 = ~mask1

    c1, c2 = data[mask1], data[mask2]
    i1, i2 = indices[mask1], indices[mask2]
    if len(c1) == 0 or len(c2) == 0:
        return [ellipsoid]

    ell1 = FastEllipsoid(c1, i1, **ellipsoid_kwargs)
    ell2 = FastEllipsoid(c2, i2, **ellipsoid_kwargs)
    return [ell1, ell2]


def recursive_split_outlier_detection(initial_ellipsoids, data, t=1.7, max_iterations=10, ellipsoid_kwargs=None):
    ellipsoid_list = initial_ellipsoids.copy()
    for _ in range(max_iterations):
        if not ellipsoid_list:
            break

        axes_sums = np.array([np.sum(ell.lengths) for ell in ellipsoid_list], dtype=float)
        axes_sum_avg = float(np.mean(axes_sums)) if len(axes_sums) > 0 else 0.0
        outliers = [ell for ell in ellipsoid_list if np.sum(ell.lengths) > 2.0 * axes_sum_avg]
        if not outliers:
            break

        kept = [ell for ell in ellipsoid_list if ell not in outliers]
        replaced = []
        min_leaf = max(2, int(np.ceil(np.sqrt(data.shape[0]) * 0.1)))

        for ell in outliers:
            children = splits_ellipsoid(ell, ellipsoid_kwargs)
            if len(children) != 2:
                replaced.append(ell)
                continue
            if any(ch.n_samples < min_leaf for ch in children):
                replaced.append(ell)
                continue

            parent_density = calculate_ellipsoid_density(ell)
            child_density_sum = sum(calculate_ellipsoid_density(ch) for ch in children)
            if child_density_sum > t * parent_density:
                replaced.extend(children)
            else:
                replaced.append(ell)

        ellipsoid_list = kept + replaced
    return ellipsoid_list


def _pair_distance_fast(ell_i, ell_j):
    # H_i ≈ eps I + U_i diag(s_i) U_i^T
    # H_j ≈ eps I + U_j diag(s_j) U_j^T
    # H_avg ≈ eps I + A A^T, where A concatenates both factors with sqrt(1/2) scaling.
    diff = ell_i.center - ell_j.center
    eps = max(ell_i.epsilon, ell_j.epsilon)

    mats = []
    if ell_i.rank > 0:
        mats.append(ell_i.U * np.sqrt(0.5 * ell_i.cov_eigs)[None, :])
    if ell_j.rank > 0:
        mats.append(ell_j.U * np.sqrt(0.5 * ell_j.cov_eigs)[None, :])

    if not mats:
        return float(np.linalg.norm(diff) / np.sqrt(eps))

    A = np.concatenate(mats, axis=1)  # (d, k)
    At_diff = A.T @ diff               # (k,)
    M = np.eye(A.shape[1]) + (A.T @ A) / eps
    try:
        z = np.linalg.solve(M, At_diff)
    except np.linalg.LinAlgError:
        z = np.linalg.pinv(M) @ At_diff

    quad = (diff @ diff) / eps - (At_diff @ z) / (eps ** 2)
    return float(np.sqrt(max(quad, 0.0)))


def ellipse_distance(ellipsoid_list):
    n = len(ellipsoid_list)
    dist_mat = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            d = _pair_distance_fast(ellipsoid_list[i], ellipsoid_list[j])
            dist_mat[i, j] = dist_mat[j, i] = d
    return dist_mat


def ellipse_min_dist(dist_mat, densities):
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


def ellipse_cluster(densities, centers, nearest):
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

def run_ge_dpc_highdim_fast(
    feature_file,
    label_file,
    epsilon=1e-6,
    outlier_t=1.7,
    auto_center_mode='knee',
    auto_center_k=None,
    min_centers=1,
    max_centers=None,
    approx_rank=16,
    svd_oversamples=8,
    svd_n_iter=2,
):
    data = np.loadtxt(feature_file, dtype=float)
    true_labels = np.loadtxt(label_file, dtype=float).astype(int)

    ellipsoid_kwargs = dict(
        epsilon=epsilon,
        approx_rank=approx_rank,
        oversamples=svd_oversamples,
        n_iter=svd_n_iter,
    )

    num = int(np.ceil(np.sqrt(data.shape[0])))

    t_gen_start = time.time()
    root = FastEllipsoid(data, np.arange(data.shape[0], dtype=int), **ellipsoid_kwargs)
    ellipsoid_list = [root]
    print("Initial ellipsoid count (Số lượng ellipsoid ban đầu): 1")

    split_iter = 0
    while True:
        split_iter += 1
        before = len(ellipsoid_list)
        ellipsoid_list = splits(ellipsoid_list, num, ellipsoid_kwargs)
        after = len(ellipsoid_list)
        print(f"Ellipsoid count after safe split iteration {split_iter} (Số lượng ellipsoid sau lần phân tách an toàn thứ {split_iter}): {after}")
        if after == before:
            break

    print(f"Total ellipsoid count after safe splitting (Tổng số ellipsoid sau phân tách an toàn): {len(ellipsoid_list)}")

    ellipsoid_list = recursive_split_outlier_detection(
        ellipsoid_list,
        data,
        t=outlier_t,
        max_iterations=10,
        ellipsoid_kwargs=ellipsoid_kwargs,
    )
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
    selected = auto_select_centers(
        densities,
        min_dists,
        mode=auto_center_mode,
        top_k=auto_center_k,
        min_centers=min_centers,
        max_centers=max_centers,
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
        'aligned_pred_labels': aligned_pred,
        'generation_time': time_gen,
        'attribute_time': time_attr,
        'cluster_time': time_cluster,
        'total_valid_time_attr_cluster': total_valid_time1,
        'total_valid_time_program': total_valid_time2,
    }

if __name__ == '__main__':
    # -------------------------
    # Change only these 2 paths
    # -------------------------
    BASE_DIR = Path(__file__).resolve().parent
    feature_file = BASE_DIR / 'dataset' / 'unlabel' / 'dry_bean.txt'
    label_file = BASE_DIR / 'dataset' / 'label' / 'dry_bean_label.txt'
    # feature_file = '/Users/quanle/Documents/Master/Thesis/Code/GE-DPC-main/dataset/unlabel/rice+cammeo.txt'
    # label_file = '/Users/quanle/Documents/Master/Thesis/Code/GE-DPC-main/dataset/label/rice+cammeo_label.txt'

    # -------------------------
    # Core parameters
    # -------------------------
    epsilon = 1e-6
    outlier_t = 1.7          # paper default / recommended threshold

    # Center selection
    auto_center_mode = 'knee'
    auto_center_k = 7
    min_centers = 7
    max_centers = 7

    # High-dimensional acceleration knobs
    approx_rank = 16         # try 8, 16, 24, 32
    svd_oversamples = 8
    svd_n_iter = 2

    run_ge_dpc_highdim_fast(
        feature_file=feature_file,
        label_file=label_file,
        epsilon=epsilon,
        outlier_t=outlier_t,
        auto_center_mode=auto_center_mode,
        auto_center_k=auto_center_k,
        min_centers=min_centers,
        max_centers=max_centers,
        approx_rank=approx_rank,
        svd_oversamples=svd_oversamples,
        svd_n_iter=svd_n_iter,
    )
