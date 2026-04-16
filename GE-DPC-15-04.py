import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import accuracy_score, adjusted_rand_score, normalized_mutual_info_score


# ============================================================
# GE-DPC with Cholesky + Cache
# ------------------------------------------------------------
# Mục tiêu của bản này:
# 1) Giữ nguyên logic chính của code gốc GE-DPC.
# 2) KHÔNG dùng approximate matrix computation / randomized SVD.
# 3) Thay các phép nghịch đảo ma trận tường minh bằng Cholesky + solve.
# 4) Thêm cache cho các đại lượng bị dùng lặp lại nhiều lần.
#
# Ghi chú kế thừa / phần mới:
# - Kế thừa từ code gốc:
#   + Quy trình tạo ellipsoid
#   + Safe split
#   + Outlier-detection split
#   + Density, ellipse distance, min-dist, center selection, label mapping
#   + Cách đánh giá ACC / NMI / ARI
#
# - Phần mới trong bản này:
#   + Không còn np.linalg.inv(H)
#   + Dùng cho_factor / cho_solve để tính Mahalanobis distance
#   + Cache: covariance, H, Cholesky factor, rho, principal axes,
#            major-axis endpoints, density
#   + Vector hóa một số phép tính điểm-đến-ellipsoid
# ============================================================


class Ellipsoid:
    """
    Granular Ellipsoid cho GE-DPC.

    Kế thừa ý tưởng từ code gốc:
    - Mỗi ellipsoid giữ tập điểm con, tâm, covariance, shape matrix H,
      rho, chiều dài các trục và hai đầu mút trục chính để phục vụ split.

    Điểm mới của bản này:
    - Không tính inv(H) trực tiếp.
    - Dùng Cholesky decomposition của H để tính Mahalanobis distance.
    - Dùng cache để tránh tính lại các đại lượng trung gian.
    """

    def __init__(self, data: np.ndarray, indices: np.ndarray, epsilon: float = 1e-6):
        self.data = np.asarray(data, dtype=float)
        self.indices = np.asarray(indices, dtype=int)
        self.epsilon = float(epsilon)

        if self.data.ndim != 2 or self.data.shape[0] == 0:
            raise ValueError("Ellipsoid cannot be created with empty or non-2D data.")

        self.n_samples, self.dim = self.data.shape
        self.center = np.mean(self.data, axis=0)

        # ------------------------------
        # Cache nội bộ
        # ------------------------------
        self._cov_matrix: Optional[np.ndarray] = None
        self._H_matrix: Optional[np.ndarray] = None
        self._chol_factor: Optional[Tuple[np.ndarray, bool]] = None
        self._rho: Optional[float] = None
        self._lengths_rotation: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self._major_axis_endpoints: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self._density: Optional[float] = None

    # ========================================================
    # Cached geometric properties
    # ========================================================
    @property
    def cov_matrix(self) -> np.ndarray:
        """
        Ma trận hiệp phương sai.

        Kế thừa logic gốc:
        - n_samples == 1 -> covariance = 0
        - bias=True để khớp code gốc
        """
        if self._cov_matrix is None:
            if self.n_samples == 1:
                self._cov_matrix = np.zeros((self.dim, self.dim), dtype=float)
            else:
                self._cov_matrix = np.cov(self.data.T, bias=True)
        return self._cov_matrix

    @property
    def H_matrix(self) -> np.ndarray:
        """
        Shape matrix H = covariance + epsilon * I.

        Kế thừa logic gốc.
        Đây là ma trận được dùng trong Mahalanobis distance.
        """
        if self._H_matrix is None:
            self._H_matrix = self.cov_matrix + self.epsilon * np.eye(self.dim, dtype=float)
        return self._H_matrix

    @property
    def chol_factor(self) -> Tuple[np.ndarray, bool]:
        """
        Phân rã Cholesky của H.

        Phần mới:
        - Thay cho inv(H) trong code gốc.
        - Chỉ tính một lần rồi cache.
        """
        if self._chol_factor is None:
            self._chol_factor = cho_factor(self.H_matrix, lower=True, check_finite=False)
        return self._chol_factor

    def solve_H(self, rhs: np.ndarray) -> np.ndarray:
        """
        Giải hệ H x = rhs bằng Cholesky.

        Phần mới, dùng để thay cho nhân với inv(H).
        Hỗ trợ cả rhs là vector hoặc ma trận.
        """
        return cho_solve(self.chol_factor, rhs, check_finite=False)

    def mahal_sq_points(self, points: np.ndarray) -> np.ndarray:
        """
        Tính bình phương khoảng cách Mahalanobis từ nhiều điểm đến ellipsoid.

        Kế thừa mục đích từ code gốc, nhưng phần tính toán là mới:
        - Gốc: diff^T inv(H) diff
        - Mới: diff^T solve(H, diff)

        Đầu ra luôn là vector 1D có độ dài = số điểm.
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
        """
        Bán kính Mahalanobis lớn nhất từ tâm đến các điểm trong ellipsoid.

        Kế thừa logic gốc, chỉ đổi cách tính distance sang Cholesky solve.
        """
        if self._rho is None:
            self._rho = float(np.sqrt(np.max(self.mahal_sq_points(self.data))))
        return self._rho

    @property
    def lengths_rotation(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Trả về:
        - lengths: độ dài các trục chính của ellipsoid
        - rotation: ma trận vector riêng

        Kế thừa logic gốc nhưng tránh eig(inv(H)).
        Vì nếu lambda là trị riêng của H, thì trị riêng của inv(H) là 1/lambda.
        Trong code gốc:
            lengths = rho / sqrt(eig(inv(H)))
        Tương đương:
            lengths = rho * sqrt(eig(H))

        Cách này giữ đúng hình học nhưng ổn định số hơn.
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
        """
        Lấy hai điểm đầu mút trục chính để chia ellipsoid.

        Kế thừa hoàn toàn logic gốc:
        - tìm điểm gần tâm nhất
        - tìm điểm xa nhất khỏi điểm đó
        - rồi tìm điểm xa nhất khỏi điểm vừa tìm
        """
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
        Mật độ ellipsoid.

        Kế thừa công thức gốc.
        Phần mới: sử dụng Mahalanobis bằng Cholesky solve và cache kết quả.
        """
        if self._density is None:
            axes_sum = max(float(np.sum(self.lengths)), 1e-12)
            mahal_sq = self.mahal_sq_points(self.data)
            total_mahal = max(float(np.sum(np.sqrt(mahal_sq))), 1e-12)
            self._density = float((self.n_samples ** 2) / (axes_sum * total_mahal))
        return self._density


# ============================================================
# Helper functions (kế thừa logic gốc)
# ============================================================
def get_num(ellipsoid: Ellipsoid) -> int:
    """Kế thừa code gốc: trả về số điểm của ellipsoid."""
    return ellipsoid.n_samples


def calculate_ellipsoid_density(ellipsoid: Ellipsoid) -> float:
    """Kế thừa ý nghĩa gốc, nhưng tận dụng cache density."""
    return ellipsoid.density


# ============================================================
# Split stage
# ============================================================
def splits(ellipsoid_list: Sequence[Ellipsoid], num: int, epsilon: float) -> List[Ellipsoid]:
    """
    Safe split cho toàn bộ danh sách ellipsoid.

    Kế thừa logic gốc:
    - Nếu ellipsoid có ít hơn num điểm thì giữ nguyên.
    - Ngược lại thì thử tách bằng splits_ellipsoid.
    """
    new_ells: List[Ellipsoid] = []
    for ell in ellipsoid_list:
        if get_num(ell) < num:
            new_ells.append(ell)
        else:
            new_ells.extend(splits_ellipsoid(ell, epsilon=epsilon))
    return new_ells


def splits_ellipsoid(ellipsoid: Ellipsoid, epsilon: Optional[float] = None) -> List[Ellipsoid]:
    """
    Tách một ellipsoid thành hai ellipsoid con.

    Kế thừa logic gốc gồm 2 pha:
    1) Chia sơ bộ theo 2 đầu mút trục chính.
    2) Gán lại mỗi điểm cho ellipsoid gần hơn theo Mahalanobis distance.

    Phần mới:
    - Pha 2 được vector hóa bằng mahal_sq_points(), không còn loop điểm + inv(H).
    """
    if ellipsoid.n_samples <= 1:
        return [ellipsoid]

    eps = ellipsoid.epsilon if epsilon is None else float(epsilon)
    data = ellipsoid.data
    indices = ellipsoid.indices
    point1, point2 = ellipsoid.major_axis_endpoints

    # ------------------------------
    # Phase 1: split sơ bộ theo khoảng cách Euclidean đến 2 đầu mút
    # ------------------------------
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

    # ------------------------------
    # Phase 2: gán lại bằng Mahalanobis distance
    # ------------------------------
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


# ============================================================
# Outlier-detection split stage
# ============================================================
def recursive_split_outlier_detection(
    initial_ellipsoids: Sequence[Ellipsoid],
    data: np.ndarray,
    t: float = 2.0,
    max_iterations: int = 10,
    epsilon: float = 1e-6,
) -> List[Ellipsoid]:
    """
    Tách tiếp các ellipsoid ngoại lệ.

    Kế thừa logic gốc:
    - Một ellipsoid bị xem là ngoại lệ nếu tổng độ dài trục > 2 * trung bình.
    - Chỉ chấp nhận split nếu tổng mật độ con đủ lớn hơn mật độ cha theo ngưỡng t.
    """
    ellipsoid_list = list(initial_ellipsoids)

    for _ in range(max_iterations):
        if not ellipsoid_list:
            break

        axes_sums = np.array([np.sum(ell.lengths) for ell in ellipsoid_list], dtype=float)
        axes_sum_avg = float(np.mean(axes_sums)) if len(axes_sums) > 0 else 0.0

        outlier_ellipsoids = [
            ell for ell in ellipsoid_list if np.sum(ell.lengths) > 2.0 * axes_sum_avg
        ]
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
    Khoảng cách giữa hai ellipsoid dựa trên tâm và average covariance.

    Kế thừa logic gốc:
    - avg_cov = (H_i + H_j) / 2
    - đo Mahalanobis giữa hai tâm theo avg_cov

    Phần mới:
    - Không nghịch đảo avg_cov trực tiếp.
    - Dùng Cholesky solve trên avg_cov.

    Trả về 3 giá trị để giữ đúng giao diện cũ.
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
    """
    Tạo ma trận khoảng cách giữa các ellipsoid.

    Kế thừa logic gốc:
    - tính nửa trên ma trận rồi đối xứng xuống dưới

    Phần mới:
    - Có pair-cache cục bộ để tránh tính lặp trong cùng một lần gọi.
    """
    n = len(ellipsoid_list)
    dist_mat = np.zeros((n, n), dtype=float)
    pair_cache: Dict[Tuple[int, int], float] = {}

    for i in range(n):
        for j in range(i + 1, n):
            key = (i, j)
            if key not in pair_cache:
                _, _, rel_dist = ellipse_mahalanobis_distance(ellipsoid_list[i], ellipsoid_list[j])
                pair_cache[key] = rel_dist
            dist_mat[i, j] = dist_mat[j, i] = pair_cache[key]

    return dist_mat


# ============================================================
# DPC attribute computation and clustering
# ============================================================
def ellipse_min_dist(dist_mat: np.ndarray, densities: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Tính delta/min-distance tới ellipsoid có density cao hơn.

    Kế thừa code gốc.
    """
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


def auto_select_centers(
    densities: np.ndarray,
    min_dists: np.ndarray,
    mode: str = "knee",
    top_k: Optional[int] = None,
    min_centers: int = 1,
    max_centers: Optional[int] = None,
) -> List[int]:
    """
    Tự động chọn center theo gamma = density * delta.

    Kế thừa logic gốc đầy đủ:
    - top_k cố định
    - threshold
    - knee
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


def ellipse_cluster(densities: np.ndarray, centers: Sequence[int], nearest: np.ndarray) -> np.ndarray:
    """
    Gán nhãn ellipsoid theo nguyên lý Density Peak Clustering.

    Kế thừa logic gốc.
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


# ============================================================
# Evaluation
# ============================================================
def align_labels(true_labels: np.ndarray, pred_labels: np.ndarray) -> np.ndarray:
    """
    Hungarian matching để căn chỉnh nhãn dự đoán với nhãn thật.

    Kế thừa code gốc.
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


# ============================================================
# Main GE-DPC pipeline
# ============================================================
def run_ge_dpc_cholesky_cache(
    feature_file: Path,
    label_file: Path,
    epsilon: float = 1e-6,
    outlier_t: float = 2.0,
    auto_center_mode: str = "knee",
    auto_center_k: Optional[int] = None,
    min_centers: int = 1,
    max_centers: Optional[int] = None,
) -> Dict[str, object]:
    """
    Pipeline hoàn chỉnh GE-DPC dùng Cholesky + cache.

    Kế thừa cấu trúc code gốc gần như toàn bộ.
    Phần mới chỉ nằm ở cách xử lý ma trận và cache.
    """
    data = np.loadtxt(feature_file, dtype=float)
    true_labels = np.loadtxt(label_file, dtype=float).astype(int)
    num = int(np.ceil(np.sqrt(data.shape[0])))

    # ------------------------------
    # 1) Sinh ellipsoid
    # ------------------------------
    t_gen_start = time.time()
    initial_indices = np.arange(data.shape[0], dtype=int)
    initial_ellipsoid = Ellipsoid(data, initial_indices, epsilon=epsilon)
    ellipsoid_list: List[Ellipsoid] = [initial_ellipsoid]
    print("Initial ellipsoid count (Số lượng ellipsoid ban đầu): 1")

    iteration = 0
    while True:
        iteration += 1
        before = len(ellipsoid_list)
        ellipsoid_list = splits(ellipsoid_list, num=num, epsilon=epsilon)
        after = len(ellipsoid_list)
        print(
            f"Ellipsoid count after safe split iteration {iteration} "
            f"(Số lượng ellipsoid sau lần phân tách an toàn thứ {iteration}): {after}"
        )
        if after == before:
            break

    print(
        f"Total ellipsoid count after safe splitting "
        f"(Tổng số ellipsoid sau phân tách an toàn): {len(ellipsoid_list)}"
    )

    ellipsoid_list = recursive_split_outlier_detection(
        ellipsoid_list,
        data,
        t=outlier_t,
        max_iterations=10,
        epsilon=epsilon,
    )
    print(
        f"Total ellipsoid count after outlier-detection splitting "
        f"(Tổng số ellipsoid sau phân tách bằng phát hiện ngoại lệ): {len(ellipsoid_list)}"
    )
    print(f"A total of {len(ellipsoid_list)} ellipsoids were generated (Tổng cộng đã tạo {len(ellipsoid_list)} ellipsoid)")
    t_gen_end = time.time()
    time_gen = t_gen_end - t_gen_start

    # ------------------------------
    # 2) Tính attributes cho DPC
    # ------------------------------
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

    # ------------------------------
    # 3) Chọn center
    # ------------------------------
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

    # ------------------------------
    # 4) Phân cụm ellipsoid
    # ------------------------------
    t_cluster_start = time.time()
    ellipsoid_labels = ellipse_cluster(densities, selected, nearest)
    t_cluster_end = time.time()
    time_cluster = t_cluster_end - t_cluster_start

    # ------------------------------
    # 5) Ánh xạ nhãn ellipsoid -> điểm dữ liệu
    # ------------------------------
    print("Mapping data points in progress (Đang ánh xạ điểm dữ liệu, không tính vào thời gian phân cụm)...")
    pred_labels = np.full(len(data), -1, dtype=int)
    for i, ell in enumerate(ellipsoid_list):
        pred_labels[ell.indices] = ellipsoid_labels[i]

    if np.any(pred_labels == -1):
        raise RuntimeError("Some data points were not assigned a cluster label.")

    # ------------------------------
    # 6) Đánh giá
    # ------------------------------
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
    print(
        f"3. Clustering computation time (Thời gian tính phân cụm): {time_cluster:.14f} seconds (giây) "
        f"(mapping time excluded / không tính thời gian ánh xạ)"
    )
    total_valid_time1 = time_attr + time_cluster
    total_valid_time2 = time_gen + time_attr + time_cluster
    print("-" * 30)
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
        "aligned_pred_labels": aligned_pred_labels,
        "generation_time": time_gen,
        "attribute_time": time_attr,
        "cluster_time": time_cluster,
        "total_valid_time_attr_cluster": total_valid_time1,
        "total_valid_time_program": total_valid_time2,
    }


# ============================================================
# Dataset registry for the 8 datasets referenced by the original code
# ------------------------------------------------------------
# Ghi chú:
# - 7 bộ nằm trong real_dataset_and_label
# - 1 bộ miniboone nằm trong data/miniboone
# - Hàm này giúp bạn chạy lần lượt 8 bộ theo cùng một source code
# ============================================================
def get_default_dataset_registry(base_dir: Path) -> Dict[str, Tuple[Path, Path]]:
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
    auto_center_mode: str = "knee",
    auto_center_k: Optional[int] = None,
    min_centers: int = 1,
    max_centers: Optional[int] = None,
) -> Dict[str, object]:
    """
    Chạy một dataset theo tên trong registry mặc định.
    """
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

    return run_ge_dpc_cholesky_cache(
        feature_file=feature_file,
        label_file=label_file,
        epsilon=epsilon,
        outlier_t=outlier_t,
        auto_center_mode=auto_center_mode,
        auto_center_k=auto_center_k,
        min_centers=min_centers,
        max_centers=max_centers,
    )


def run_all_default_datasets(
    base_dir: Path,
    epsilon: float = 1e-6,
    outlier_t: float = 2.0,
    auto_center_mode: str = "knee",
    auto_center_k: Optional[int] = None,
    min_centers: int = 1,
    max_centers: Optional[int] = None,
) -> Dict[str, Dict[str, object]]:
    """
    Chạy toàn bộ 8 dataset trong registry mặc định.

    Ghi chú:
    - Hàm này không ép bạn phải dùng cùng một top-k cho mọi dataset.
    - Nhưng để bám đúng source gốc hiện tại của bạn, nó giữ đúng một giao diện thống nhất.
    """
    results: Dict[str, Dict[str, object]] = {}
    registry = get_default_dataset_registry(base_dir)

    for dataset_name in registry.keys():
        try:
            results[dataset_name] = run_named_dataset(
                dataset_name=dataset_name,
                base_dir=base_dir,
                epsilon=epsilon,
                outlier_t=outlier_t,
                auto_center_mode=auto_center_mode,
                auto_center_k=auto_center_k,
                min_centers=min_centers,
                max_centers=max_centers,
            )
        except Exception as exc:
            results[dataset_name] = {"error": str(exc)}
            print(f"[ERROR] Dataset '{dataset_name}' failed: {exc}")

    return results


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent

    # ========================================================
    # Algorithm settings
    # --------------------------------------------------------
    # Kế thừa từ code gốc:
    # - epsilon và outlier_t giữ nguyên tinh thần paper/source gốc
    # - center selection vẫn giữ nguyên giao diện cũ
    # ========================================================
    epsilon = 1e-6
    outlier_t = 2.0

    auto_center_mode = "knee" #Tham số sẽ thay đổi theo từng tập data để chọn center
    auto_center_k = 2
    min_centers = 2
    max_centers = 2

    # --------------------------------------------------------
    # Chọn 1 dataset để chạy
    # --------------------------------------------------------
    dataset_name = "miniboone"

    run_named_dataset(
        dataset_name=dataset_name,
        base_dir=BASE_DIR,
        epsilon=epsilon,
        outlier_t=outlier_t,
        auto_center_mode=auto_center_mode,
        auto_center_k=auto_center_k,
        min_centers=min_centers,
        max_centers=max_centers,
    )

    # --------------------------------------------------------
    # Nếu muốn chạy cả 8 bộ mặc định thì bỏ comment đoạn dưới
    # --------------------------------------------------------
    # all_results = run_all_default_datasets(
    #     base_dir=BASE_DIR,
    #     epsilon=epsilon,
    #     outlier_t=outlier_t,
    #     auto_center_mode=auto_center_mode,
    #     auto_center_k=auto_center_k,
    #     min_centers=min_centers,
    #     max_centers=max_centers,
    # )
    # print(all_results)
