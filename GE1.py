import time
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import accuracy_score, normalized_mutual_info_score, adjusted_rand_score
from sklearn.utils.extmath import randomized_svd


# ============================================================
# GE-DPC hướng phát triển cho dữ liệu nhiều chiều
# ------------------------------------------------------------
# Mục tiêu:
# - Giữ tinh thần code gốc của GE-DPC
# - Chỉ thay phần biểu diễn ellipsoid và tính khoảng cách Mahalanobis
#   để phù hợp hơn với dữ liệu nhiều chiều
#
# Quy ước comment:
# - [GỐC]: logic giữ gần code gốc
# - [MỚI]: phần thêm cho hướng phát triển
# ============================================================


# ============================================================
# [MỚI] Cấu hình nội bộ cố định cho hướng phát triển
# ------------------------------------------------------------
# 3 tham số này là tham số kỹ thuật nội bộ.
# Không đưa ra ngoài để tune, không truyền ở main.
# ============================================================
_INTERNAL_APPROX_RANK = 16
_INTERNAL_SVD_OVERSAMPLES = 8
_INTERNAL_SVD_N_ITER = 2


# ============================================================
# [MỚI] Ellipsoid tăng tốc cho dữ liệu nhiều chiều
# ------------------------------------------------------------
# So với code gốc:
# - code gốc dùng full covariance + inv(H)
# - bản này dùng low-rank approximation + randomized SVD
# ============================================================
class FastEllipsoid:
    def __init__(self, data, indices, epsilon):
        """
        [MỚI]
        Khởi tạo một ellipsoid tăng tốc.

        Tham số:
        - data: dữ liệu thuộc ellipsoid, shape (n_samples, d)
        - indices: chỉ số các điểm trong tập dữ liệu gốc
        - epsilon: tham số gốc của thuật toán
        """
        self.data = np.asarray(data, dtype=float)
        self.indices = np.asarray(indices, dtype=int)
        self.epsilon = float(epsilon)

        if self.data.ndim != 2 or len(self.data) == 0:
            raise ValueError("data phải là mảng 2 chiều và không được rỗng.")

        self.n_samples, self.dim = self.data.shape
        self.center = np.mean(self.data, axis=0)

        self._fit_low_rank_shape()
        self.rho = self._compute_rho()
        self.lengths = self._compute_lengths()
        self.major_axis_endpoints = self.compute_major_axis_endpoints()

    def _fit_low_rank_shape(self):
        """
        [MỚI]
        Xấp xỉ shape matrix bằng low-rank SVD.

        Ý nghĩa:
        - thay full covariance inverse của code gốc
        - giảm chi phí tính toán ở dữ liệu nhiều chiều
        """
        Xc = self.data - self.center
        d = self.dim

        if self.n_samples <= 1 or np.allclose(Xc, 0.0):
            self.rank = 0
            self.U = np.zeros((d, 0), dtype=float)
            self.cov_eigs = np.zeros((0,), dtype=float)
            self._inv_diag = np.zeros((0,), dtype=float)
            self._H_eigs = np.full(d, self.epsilon, dtype=float)
            return

        max_rank = min(_INTERNAL_APPROX_RANK, d, self.n_samples)
        if max_rank <= 0:
            max_rank = 1

        try:
            _, s, Vt = randomized_svd(
                Xc,
                n_components=max_rank,
                n_oversamples=min(_INTERNAL_SVD_OVERSAMPLES, max(2, d)),
                n_iter=_INTERNAL_SVD_N_ITER,
                random_state=0,
            )
            U_feat = Vt.T
        except Exception:
            _, s, Vt = np.linalg.svd(Xc, full_matrices=False)
            s = s[:max_rank]
            U_feat = Vt[:max_rank].T

        cov_eigs = (s ** 2) / max(self.n_samples, 1)
        keep = cov_eigs > 1e-12

        self.U = U_feat[:, keep]
        self.cov_eigs = cov_eigs[keep]
        self.rank = len(self.cov_eigs)
        self._inv_diag = 1.0 / (self.cov_eigs + self.epsilon)

        self._H_eigs = np.concatenate([
            self.cov_eigs + self.epsilon,
            np.full(max(0, d - self.rank), self.epsilon, dtype=float)
        ])

    def mahal_sq_points(self, points):
        """
        [MỚI]
        Tính Mahalanobis distance bình phương từ nhiều điểm tới ellipsoid.
        """
        X = np.asarray(points, dtype=float) - self.center
        if X.ndim == 1:
            X = X[None, :]

        if self.rank == 0:
            return np.sum(X * X, axis=1) / self.epsilon

        proj = X @ self.U
        parallel_sq = np.sum((proj ** 2) * self._inv_diag[None, :], axis=1)

        total_sq = np.sum(X * X, axis=1)
        proj_sq = np.sum(proj * proj, axis=1)
        perp_sq = np.maximum(total_sq - proj_sq, 0.0) / self.epsilon

        return parallel_sq + perp_sq

    def _compute_rho(self):
        """
        [GẦN GỐC]
        Tính rho = khoảng cách Mahalanobis lớn nhất từ tâm tới các điểm.
        """
        mahal_sq = self.mahal_sq_points(self.data)
        mahal_sq = np.maximum(mahal_sq, 0.0)
        return float(np.sqrt(np.max(mahal_sq)))

    def _compute_lengths(self):
        """
        [GẦN GỐC]
        Tính độ dài các bán trục của ellipsoid.
        """
        return self.rho * np.sqrt(np.maximum(self._H_eigs, 1e-18))

    def compute_major_axis_endpoints(self):
        """
        [GỐC]
        Tìm 2 đầu mút gần đúng của trục chính dài nhất.
        """
        if self.n_samples <= 1:
            return self.center, self.center

        center_distances = np.linalg.norm(self.data - self.center, axis=1)
        point1 = self.data[np.argmin(center_distances)]
        point2 = self.data[np.argmax(np.linalg.norm(self.data - point1, axis=1))]
        point3 = self.data[np.argmax(np.linalg.norm(self.data - point2, axis=1))]
        return point2, point3


def get_num(ellipsoid):
    """
    [GỐC]
    Lấy số lượng điểm trong ellipsoid.
    """
    return ellipsoid.n_samples


def calculate_ellipsoid_density(ellipsoid):
    """
    [GỐC]
    Tính mật độ ellipsoid theo tinh thần code gốc.
    """
    n_samples = ellipsoid.n_samples

    axes_sum = float(np.sum(ellipsoid.lengths))
    axes_sum = max(axes_sum, 1e-12)

    mahal_sq = ellipsoid.mahal_sq_points(ellipsoid.data)
    total_mahal = float(np.sum(np.sqrt(np.maximum(mahal_sq, 0.0))))
    total_mahal = max(total_mahal, 1e-12)

    return float((n_samples ** 2) / (axes_sum * total_mahal))


def splits(ellipsoid_list, num):
    """
    [GỐC]
    Safe split toàn bộ danh sách ellipsoid.
    """
    new_ells = []
    for ell in ellipsoid_list:
        if get_num(ell) < num:
            new_ells.append(ell)
        else:
            new_ells.extend(splits_ellipsoid(ell))
    return new_ells


def splits_ellipsoid(ellipsoid):
    """
    [GỐC]
    Tách 1 ellipsoid thành 2 ellipsoid con.

    Quy trình:
    1. Chia sơ bộ theo khoảng cách tới 2 đầu mút trục chính
    2. Tạo 2 ellipsoid tạm
    3. Refine bằng Mahalanobis
    4. Tạo lại 2 ellipsoid cuối
    """
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

    ell1 = FastEllipsoid(c1, i1, epsilon=ellipsoid.epsilon)
    ell2 = FastEllipsoid(c2, i2, epsilon=ellipsoid.epsilon)

    dist1_sq = ell1.mahal_sq_points(data)
    dist2_sq = ell2.mahal_sq_points(data)

    mask1 = dist1_sq < dist2_sq
    mask2 = ~mask1

    c1, c2 = data[mask1], data[mask2]
    i1, i2 = indices[mask1], indices[mask2]

    if len(c1) == 0 or len(c2) == 0:
        return [ellipsoid]

    ell1 = FastEllipsoid(c1, i1, epsilon=ellipsoid.epsilon)
    ell2 = FastEllipsoid(c2, i2, epsilon=ellipsoid.epsilon)

    return [ell1, ell2]


def recursive_split_outlier_detection(initial_ellipsoids, data, t):
    """
    [GỐC]
    Tách thêm ellipsoid ngoại lệ.

    Khác với bản trước:
    - bỏ default parameter để tránh lặp tham số
    """
    ellipsoid_list = initial_ellipsoids.copy()
    max_iterations = 10

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
            children = splits_ellipsoid(ell)

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
    """
    [MỚI]
    Tính khoảng cách giữa 2 ellipsoid theo công thức tăng tốc.
    """
    diff = ell_i.center - ell_j.center
    eps = max(ell_i.epsilon, ell_j.epsilon)

    mats = []
    if ell_i.rank > 0:
        mats.append(ell_i.U * np.sqrt(0.5 * ell_i.cov_eigs)[None, :])
    if ell_j.rank > 0:
        mats.append(ell_j.U * np.sqrt(0.5 * ell_j.cov_eigs)[None, :])

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


def ellipse_distance(ellipsoid_list):
    """
    [GẦN GỐC]
    Tạo ma trận khoảng cách giữa các ellipsoid.
    """
    n = len(ellipsoid_list)
    dist_mat = np.zeros((n, n), dtype=float)

    for i in range(n):
        for j in range(i + 1, n):
            d = _pair_distance_fast(ellipsoid_list[i], ellipsoid_list[j])
            dist_mat[i, j] = dist_mat[j, i] = d

    return dist_mat


def ellipse_min_dist(dist_mat, densities):
    """
    [GỐC]
    Tính khoảng cách nhỏ nhất tới ellipsoid có density lớn hơn.
    """
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


def auto_select_centers(densities, min_dists, mode, top_k, min_centers, max_centers):
    """
    [PHẦN BẠN ĐÃ THÊM TỪ TRƯỚC]
    Tự động chọn center thay cho chọn thủ công trên decision graph.

    Không để default parameter để tránh lặp 100%.
    """
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
        dists = [
            abs(np.cross(line_vec, np.array([xi, yi]) - p1)) / line_norm
            for xi, yi in zip(x_norm, y_norm)
        ]
        knee_idx = int(np.argmax(dists))

    k = max(min_centers, knee_idx + 1)
    if max_centers is not None:
        k = min(k, max_centers)

    return order[:min(k, n)].tolist()


def ellipse_cluster(densities, centers, nearest):
    """
    [GỐC]
    Gán nhãn cụm cho ellipsoid theo logic DPC.
    """
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
    """
    Hàm phụ trợ đánh giá:
    map nhãn clustering sang nhãn thật để tính ACC.
    """
    true_classes = np.unique(true_labels)
    pred_classes = np.unique(pred_labels)

    confusion = np.zeros((len(true_classes), len(pred_classes)), dtype=float)
    for i, t in enumerate(true_classes):
        for j, p in enumerate(pred_classes):
            confusion[i, j] = np.sum((true_labels == t) & (pred_labels == p))

    row_ind, col_ind = linear_sum_assignment(-confusion)
    mapping = {pred_classes[j]: true_classes[i] for i, j in zip(row_ind, col_ind)}

    return np.array([mapping.get(label, -1) for label in pred_labels])


def run_ge_dpc_highdim(
    feature_file,
    label_file,
    epsilon,
    outlier_t,
    auto_center_mode,
    auto_center_k,
    min_centers,
    max_centers,
):
    """
    Hàm chạy chính của hướng phát triển.

    Chỉ còn đúng 6 tham số chính:
    - epsilon
    - outlier_t
    - auto_center_mode
    - auto_center_k
    - min_centers
    - max_centers
    """
    data = np.loadtxt(feature_file, dtype=float)
    true_labels = np.loadtxt(label_file, dtype=float).astype(int)

    num = int(np.ceil(np.sqrt(data.shape[0])))

    t_gen_start = time.time()

    root = FastEllipsoid(data, np.arange(data.shape[0], dtype=int), epsilon=epsilon)
    ellipsoid_list = [root]
    print("Initial ellipsoid count (Số lượng ellipsoid ban đầu): 1")

    split_iter = 0
    while True:
        split_iter += 1
        before = len(ellipsoid_list)
        ellipsoid_list = splits(ellipsoid_list, num)
        after = len(ellipsoid_list)

        print(
            f"Ellipsoid count after safe split iteration {split_iter} "
            f"(Số lượng ellipsoid sau lần phân tách an toàn thứ {split_iter}): {after}"
        )

        if after == before:
            break

    print(f"Total ellipsoid count after safe splitting (Tổng số ellipsoid sau phân tách an toàn): {len(ellipsoid_list)}")

    ellipsoid_list = recursive_split_outlier_detection(
        initial_ellipsoids=ellipsoid_list,
        data=data,
        t=outlier_t,
    )

    print(
        f"Total ellipsoid count after outlier-detection splitting "
        f"(Tổng số ellipsoid sau phân tách bằng phát hiện ngoại lệ): {len(ellipsoid_list)}"
    )
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
        densities=densities,
        min_dists=min_dists,
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
        raise RuntimeError("Một số điểm dữ liệu chưa được gán nhãn.")

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
    print(f"3. Clustering computation time (Thời gian tính phân cụm): {time_cluster:.14f} seconds (giây)")
    print("-" * 30)

    total_valid_time1 = time_attr + time_cluster
    total_valid_time2 = time_gen + time_attr + time_cluster

    print(
        f"Total effective runtime (attributes + clustering) "
        f"(Tổng thời gian hiệu dụng: thuộc tính + phân cụm): {total_valid_time1:.14f} seconds (giây)"
    )
    print(
        f"Total effective runtime of the program "
        f"(Tổng thời gian chạy hiệu dụng của chương trình): {total_valid_time2:.14f} seconds (giây)"
    )
    print("-" * 30)

    return {
        "acc": acc,
        "nmi": nmi,
        "ari": ari,
        "selected_centers": selected,
        "n_ellipsoids": len(ellipsoid_list),
        "pred_labels": pred_labels,
        "aligned_pred_labels": aligned_pred,
        "generation_time": time_gen,
        "attribute_time": time_attr,
        "cluster_time": time_cluster,
        "total_valid_time_attr_cluster": total_valid_time1,
        "total_valid_time_program": total_valid_time2,
    }


if __name__ == "__main__":
    # ========================================================
    # Chỉ sửa 2 đường dẫn này khi đổi dataset
    # ========================================================
    BASE_DIR = Path(__file__).resolve().parent
    feature_file = BASE_DIR / "dataset" / "unlabel" / "dry_bean.txt"
    label_file = BASE_DIR / "dataset" / "label" / "dry_bean_label.txt"

    # ========================================================
    # 6 tham số chính duy nhất
    # Không lặp trong chữ ký hàm
    # ========================================================
    epsilon = 1e-6
    outlier_t = 2.0

    auto_center_mode = "knee"
    auto_center_k = 7
    min_centers = 7
    max_centers = 7

    run_ge_dpc_highdim(
        feature_file=feature_file,
        label_file=label_file,
        epsilon=epsilon,
        outlier_t=outlier_t,
        auto_center_mode=auto_center_mode,
        auto_center_k=auto_center_k,
        min_centers=min_centers,
        max_centers=max_centers,
    )