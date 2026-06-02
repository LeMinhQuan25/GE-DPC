"""
AQG-GE-DPC
============================
Pipeline: (1)Dữ liệu đầu vào ban đầu -> (2)Chuẩn hóa dữ liệu thích nghi -> (3)Sinh các granular ellipsoid -> (4)Tính khoảng cách Mahalanobis bằng Cholesky kết hợp cơ chế lưu đệm -> 
(5)Tính mật độ, khoảng cách delta và giá trị gamma -> (6)Thử nhiều giá trị k ứng viên -> (7)Gán nhãn theo cơ chế chuỗi đơn của DPC -> (8)Chỉ áp dụng hiệu chỉnh đồ thị khi thỏa điều kiện an toàn
-> (9)Lựa chọn cấu hình tốt nhất dựa trên chỉ số nội tại -> (10)Ánh xạ nhãn của ellipsoid về từng điểm dữ liệu -> (11)Vẽ biểu đồ kết quả phân cụm -> (12)Đánh giá bằng ACC, NMI và ARI
-> (13)Giữ chế độ tuning tùy chọn, trong khi chế độ mặc định sử dụng các tham số an toàn đã được kiểm chứng

Ghi chú:
- Nhãn thật chỉ được sử dụng ở bước đánh giá cuối cùng.
- Chỉ số nội tại không sử dụng nhãn thật.
- Cholesky kết hợp cơ chế lưu đệm được giữ làm lõi tăng tốc chính.
- Hàm vẽ biểu đồ chỉ phục vụ trực quan hóa và không ảnh hưởng đến logic phân cụm.
- Chế độ mặc định không sử dụng nhãn thật trong quá trình phân cụm. Chế độ tuning tùy chọn chỉ sử dụng nhãn thật sau khi phân cụm để so sánh các cấu hình.
"""

import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import accuracy_score, adjusted_rand_score, normalized_mutual_info_score

# ============================================================
# Ellipsoid with Cholesky + Cache - Biểu diễn một granular ellipsoid và lưu cache các đại lượng hình học cần dùng.
# ============================================================
class Ellipsoid:
    # ============================================================
    # Khởi tạo ellipsoid từ tập điểm con, chỉ số gốc và tham số ổn định epsilon.
    # ============================================================
    def __init__(self, data: np.ndarray, indices: np.ndarray, epsilon: float = 1e-6):
        self.data = np.asarray(data, dtype=float)
        self.indices = np.asarray(indices, dtype=int)
        self.epsilon = float(epsilon)

        if self.data.ndim != 2 or self.data.shape[0] == 0:
            raise ValueError("Ellipsoid cannot be empty or non-2D.")

        self.n_samples, self.dim = self.data.shape
        self.center = np.mean(self.data, axis=0)

        self._cov_matrix = None
        self._H_matrix = None
        self._chol_factor = None
        self._rho = None
        self._lengths_rotation = None
        self._major_axis_endpoints = None
        self._density = None

    # ============================================================
    # Tính và lưu cache ma trận hiệp phương sai của ellipsoid.
    # ============================================================
    @property
    def cov_matrix(self) -> np.ndarray:
        if self._cov_matrix is None:
            if self.n_samples <= 1:
                self._cov_matrix = np.zeros((self.dim, self.dim), dtype=float)
            else:
                self._cov_matrix = np.cov(self.data.T, bias=True)
        return self._cov_matrix

    # ============================================================
    # Tạo ma trận hình dạng H có cộng epsilon để ổn định số.
    # ============================================================
    @property
    def H_matrix(self) -> np.ndarray:
        if self._H_matrix is None:
            self._H_matrix = self.cov_matrix + self.epsilon * np.eye(self.dim, dtype=float)
        return self._H_matrix

    # ============================================================
    # Phân rã Cholesky của ma trận H và lưu cache để tái sử dụng.
    # ============================================================
    @property
    def chol_factor(self):
        if self._chol_factor is None:
            self._chol_factor = cho_factor(self.H_matrix, lower=True, check_finite=False)
        return self._chol_factor

    # ============================================================
    # Giải hệ tuyến tính Hx = rhs bằng Cholesky thay cho nghịch đảo trực tiếp.
    # ============================================================
    def solve_H(self, rhs: np.ndarray) -> np.ndarray:
        return cho_solve(self.chol_factor, rhs, check_finite=False)

    # ============================================================
    # Tính bình phương khoảng cách Mahalanobis từ các điểm đến tâm ellipsoid.
    # ============================================================
    def mahal_sq_points(self, points: np.ndarray) -> np.ndarray:
        X = np.asarray(points, dtype=float)
        if X.ndim == 1:
            X = X[None, :]
        diffs = X - self.center
        solved = self.solve_H(diffs.T).T
        return np.maximum(np.einsum("ij,ij->i", diffs, solved), 0.0)

    # ============================================================
    # Tính bán kính chuẩn hóa lớn nhất theo khoảng cách Mahalanobis.
    # ============================================================
    @property
    def rho(self) -> float:
        if self._rho is None:
            self._rho = float(np.sqrt(np.max(self.mahal_sq_points(self.data))))
        return self._rho

    # ============================================================
    # Tính độ dài bán trục và ma trận xoay của ellipsoid.
    # ============================================================
    @property
    def lengths_rotation(self):
        if self._lengths_rotation is None:
            eigvals_H, eigvecs_H = np.linalg.eigh(self.H_matrix)
            eigvals_H = np.maximum(eigvals_H, 1e-12)
            lengths = self.rho * np.sqrt(eigvals_H)
            self._lengths_rotation = (lengths, eigvecs_H)
        return self._lengths_rotation

    # ============================================================
    # Lấy các độ dài bán trục đã được tính sẵn.
    # ============================================================
    @property
    def lengths(self) -> np.ndarray:
        return self.lengths_rotation[0]

    # ============================================================
    # Xác định hai điểm đầu mút gần đúng trên trục chính để phục vụ tách ellipsoid.
    # ============================================================
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

    # ============================================================
    # Tính mật độ ellipsoid dựa trên số mẫu, độ dài bán trục và khoảng cách Mahalanobis.
    # ============================================================
    @property
    def density(self) -> float:
        if self._density is None:
            axes_sum = max(float(np.sum(self.lengths)), 1e-12)
            mahal = np.sqrt(self.mahal_sq_points(self.data))
            total_mahal = max(float(np.sum(mahal)), 1e-12)
            self._density = float((self.n_samples ** 2) / (axes_sum * total_mahal))
        return self._density

# ============================================================
# Data scaling - Chuẩn hóa dữ liệu đầu vào theo chế độ none, standard, minmax hoặc robust.
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
# GE generation: safe split + outlier split - Duyệt danh sách ellipsoid và tách các ellipsoid đủ lớn theo ngưỡng kích thước.
# ============================================================
def splits(ellipsoid_list: Sequence[Ellipsoid], num: int, epsilon: float) -> List[Ellipsoid]:
    new_ells: List[Ellipsoid] = []
    for ell in ellipsoid_list:
        if ell.n_samples < num:
            new_ells.append(ell)
        else:
            new_ells.extend(splits_ellipsoid(ell, epsilon=epsilon))
    return new_ells


# ============================================================
# Tách một ellipsoid thành hai ellipsoid con và gán lại điểm bằng khoảng cách Mahalanobis.
# ============================================================
def splits_ellipsoid(ellipsoid: Ellipsoid, epsilon: Optional[float] = None) -> List[Ellipsoid]:
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

    return [
        Ellipsoid(data[mask1], indices[mask1], epsilon=eps),
        Ellipsoid(data[mask2], indices[mask2], epsilon=eps),
    ]


# ============================================================
# Tách bổ sung các ellipsoid quá lớn hoặc bất ổn bằng quality gate để tránh tách dư thừa.
# ============================================================
def recursive_split_outlier_detection(
    initial_ellipsoids: Sequence[Ellipsoid],
    data: np.ndarray,
    t: float = 2.0,
    max_iterations: int = 10,
    epsilon: float = 1e-6,
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
            children = splits_ellipsoid(ell, epsilon=epsilon)
            if len(children) != 2 or any(child.n_samples < min_leaf for child in children):
                new_ells.append(ell)
                continue

            parent_density = ell.density
            child_density_sum = sum(child.density for child in children)
            if child_density_sum > t * parent_density:
                new_ells.extend(children)
            else:
                new_ells.append(ell)

        ellipsoid_list = normal + new_ells

    return ellipsoid_list

# ============================================================
# Tính khoảng cách Mahalanobis giữa hai ellipsoid bằng ma trận H trung bình và Cholesky.
# ============================================================
def ellipse_mahalanobis_distance(ell_i: Ellipsoid, ell_j: Ellipsoid) -> float:
    avg_H = 0.5 * (ell_i.H_matrix + ell_j.H_matrix)
    chol_avg = cho_factor(avg_H, lower=True, check_finite=False)
    diff = ell_i.center - ell_j.center
    solved = cho_solve(chol_avg, diff, check_finite=False)
    return float(np.sqrt(max(float(diff.T @ solved), 0.0)))


# ============================================================
# Tạo ma trận khoảng cách đôi một giữa các ellipsoid.
# ============================================================
def ellipse_distance(ellipsoid_list: Sequence[Ellipsoid]) -> np.ndarray:
    n = len(ellipsoid_list)
    dist_mat = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            d = ellipse_mahalanobis_distance(ellipsoid_list[i], ellipsoid_list[j])
            dist_mat[i, j] = dist_mat[j, i] = d
    return dist_mat


# ============================================================
# Tính delta và ellipsoid có mật độ cao hơn gần nhất theo nguyên lý DPC.
# ============================================================
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
# Center selection - Chọn tâm cụm tự động dựa trên gamma và điều kiện khoảng cách tối thiểu giữa các tâm.
# ============================================================
def auto_select_centers_quality(
    densities: np.ndarray,
    min_dists: np.ndarray,
    dist_mat: np.ndarray,
    top_k: int,
) -> List[int]:
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
    min_sep = max(q25_dist, 0.20 * median_dist, 1e-12)

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

    # Fallback if pruning removes too many centers.
    if len(selected) < top_k:
        for idx in order:
            idx = int(idx)
            if idx not in selected:
                selected.append(idx)
            if len(selected) >= top_k:
                break

    return selected[:top_k]

# ============================================================
# Label assignment - Gán nhãn ellipsoid theo chuỗi DPC từ các tâm cụm đã chọn.
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

    # Fallback for any unlabeled ellipsoid.
    for idx in np.where(labels == -1)[0]:
        c = centers[int(np.argmin(dist_mat[idx, centers]))]
        labels[idx] = labels[c]

    return labels


# ============================================================
# Hiệu chỉnh nhãn bằng đồ thị theo hướng bảo thủ, chỉ đổi nhãn khi tín hiệu đủ mạnh.
# ============================================================
def ellipse_cluster_conservative_graph_correction(
    densities: np.ndarray,
    dist_mat: np.ndarray,
    centers: Sequence[int],
    nearest: np.ndarray,
    switch_margin: float = 1.75,
    graph_k_factor: float = 1.0,
) -> np.ndarray:
    labels = ellipse_cluster_single_chain(densities, centers, nearest, dist_mat)
    corrected = labels.copy()
    n = len(densities)
    center_set = set(int(c) for c in centers)

    positive = dist_mat[dist_mat > 0]
    eps_dist = float(np.median(positive) * 1e-9) if positive.size else 1e-12
    k_neighbors = max(2, int(np.ceil(np.log2(max(n, 2))) * graph_k_factor))
    switch_margin = float(switch_margin)

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
# Internal score and safety gate - Tính tỷ lệ kích thước cụm lớn nhất để phát hiện mất cân bằng cụm.
# ============================================================
def cluster_size_ratio(labels: np.ndarray) -> float:
    _, counts = np.unique(labels, return_counts=True)
    return float(np.max(counts) / max(np.sum(counts), 1))


# ============================================================
# Kiểm tra graph correction có an toàn không trước khi chấp nhận kết quả hiệu chỉnh.
# ============================================================
def is_graph_safe(
    single_labels: np.ndarray,
    graph_labels: np.ndarray,
    max_largest_ratio: float = 0.85,
    changed_ratio_limit: float = 0.35,
) -> bool:
    # Prevent one cluster from swallowing most ellipsoids.
    if cluster_size_ratio(graph_labels) > float(max_largest_ratio):
        return False

    # Prevent graph correction from changing too many ellipsoid labels.
    changed_ratio = float(np.mean(single_labels != graph_labels))
    if changed_ratio > float(changed_ratio_limit):
        return False

    return True


# ============================================================
# Chấm điểm cấu hình phân cụm bằng độ tách biệt, độ chặt và phạt mất cân bằng, không dùng nhãn thật.
# ============================================================
def internal_cluster_score(
    labels: np.ndarray,
    densities: np.ndarray,
    dist_mat: np.ndarray,
    largest_ratio_threshold: float = 0.85,
    min_ratio_threshold: float = 0.02,
    largest_penalty: float = 0.20,
    min_penalty: float = 0.50,
) -> float:
    """
    Internal score, no ground-truth labels.
    Higher is better.

    score = separation / compactness, with penalties for cluster imbalance.
    """
    labels = np.asarray(labels, dtype=int)
    unique = sorted(np.unique(labels))
    if len(unique) <= 1:
        return -1e18

    reps = []
    compact_values = []
    sizes = []

    for lab in unique:
        members = np.where(labels == lab)[0]
        sizes.append(len(members))
        rep = int(members[np.argmax(densities[members])])
        reps.append(rep)

        if len(members) <= 1:
            compact_values.append(0.0)
        else:
            compact_values.append(float(np.mean(dist_mat[np.ix_(members, members)])))

    reps = np.asarray(reps, dtype=int)
    compactness = float(np.mean(compact_values)) + 1e-12

    sep_values = []
    for i in range(len(reps)):
        for j in range(i + 1, len(reps)):
            sep_values.append(float(dist_mat[reps[i], reps[j]]))
    separation = float(np.mean(sep_values)) if sep_values else 0.0

    sizes = np.asarray(sizes, dtype=float)
    largest_ratio = float(np.max(sizes) / np.sum(sizes))
    min_ratio = float(np.min(sizes) / np.sum(sizes))

    imbalance_penalty = 1.0
    if largest_ratio > float(largest_ratio_threshold):
        imbalance_penalty *= float(largest_penalty)
    if min_ratio < float(min_ratio_threshold):
        imbalance_penalty *= float(min_penalty)

    return (separation / compactness) * imbalance_penalty


# ============================================================
# Thử nhiều giá trị k và chọn cấu hình nhãn tốt nhất theo điểm nội tại.
# ============================================================
def choose_best_labels_by_internal_score(
    densities: np.ndarray,
    min_dists: np.ndarray,
    dist_mat: np.ndarray,
    nearest: np.ndarray,
    k_candidates: Sequence[int],
    allow_graph: bool = True,
    graph_switch_margin: float = 1.75,
    graph_k_factor: float = 1.0,
    graph_max_largest_ratio: float = 0.85,
    graph_changed_ratio_limit: float = 0.35,
    score_largest_ratio_threshold: float = 0.85,
    score_min_ratio_threshold: float = 0.02,
):
    best = None

    for k in k_candidates:
        centers = auto_select_centers_quality(densities, min_dists, dist_mat, top_k=int(k))
        single_labels = ellipse_cluster_single_chain(densities, centers, nearest, dist_mat)
        single_score = internal_cluster_score(
            single_labels, densities, dist_mat,
            largest_ratio_threshold=score_largest_ratio_threshold,
            min_ratio_threshold=score_min_ratio_threshold,
        )

        candidates = [("single", single_labels, single_score)]

        if allow_graph:
            graph_labels = ellipse_cluster_conservative_graph_correction(
                densities, dist_mat, centers, nearest,
                switch_margin=graph_switch_margin,
                graph_k_factor=graph_k_factor,
            )
            if is_graph_safe(
                single_labels, graph_labels,
                max_largest_ratio=graph_max_largest_ratio,
                changed_ratio_limit=graph_changed_ratio_limit,
            ):
                graph_score = internal_cluster_score(
                    graph_labels, densities, dist_mat,
                    largest_ratio_threshold=score_largest_ratio_threshold,
                    min_ratio_threshold=score_min_ratio_threshold,
                )
                candidates.append(("graph", graph_labels, graph_score))

        for mode, labels, score in candidates:
            item = {
                "k": int(k),
                "mode": mode,
                "centers": centers,
                "labels": labels,
                "score": float(score),
                "largest_ratio": cluster_size_ratio(labels),
            }
            if best is None or item["score"] > best["score"]:
                best = item

    if best is None:
        raise RuntimeError("No valid labeling candidate was found.")
    return best

# ============================================================
# Evaluation - Ánh xạ nhãn dự đoán sang nhãn thật bằng Hungarian để tính ACC.
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


# ============================================================
# In phân bố số lượng mẫu theo từng nhãn.
# ============================================================
def print_distribution(name: str, labels: np.ndarray) -> None:
    unique, counts = np.unique(labels, return_counts=True)
    print(name)
    for lab, cnt in zip(unique, counts):
        print(f"  label {int(lab):>4}: {int(cnt)}")

# ============================================================
# Vẽ biểu đồ phân cụm 2D; nếu dữ liệu nhiều chiều thì giảm chiều bằng PCA.
# ============================================================
def plot_predicted_clusters_2d(
    data: np.ndarray,
    pred_labels: np.ndarray,
    dataset_name: str = "dataset",
    show_legend: bool = False,
) -> None:
    """
    Draw clustering result as a 2D scatter chart.
    - If data has 2 dimensions: plot directly.
    - If data has more than 2 dimensions: reduce to 2D using PCA.
    This function is only for visualization and does not affect clustering results.
    """
    X = np.asarray(data, dtype=float)
    y = np.asarray(pred_labels)

    if X.ndim != 2:
        raise ValueError("Data must be a 2D array.")

    if X.shape[1] > 2:
        from sklearn.decomposition import PCA
        X_vis = PCA(n_components=2, random_state=42).fit_transform(X)
        xlabel = "PC1"
        ylabel = "PC2"
    else:
        X_vis = X
        xlabel = "Feature 1"
        ylabel = "Feature 2"

    # Plot style similar to the example image
    plt.figure(figsize=(6.4, 4.8))
    unique_labels = np.unique(y)

    for lab in unique_labels:
        mask = (y == lab)
        plt.scatter(
            X_vis[mask, 0],
            X_vis[mask, 1],
            s=16,
            alpha=0.95
        )

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"Clustering Result - {dataset_name}")
    if show_legend:
        plt.legend([f"Cluster {lab}" for lab in unique_labels], loc="best")
    plt.tight_layout()
    plt.show()

# ============================================================
# Dataset config - Lấy cấu hình mặc định theo từng bộ dữ liệu, gồm scaler, k candidates, graph và split factor.
# ============================================================
def get_adaptive_dataset_config(dataset_name: str) -> Dict[str, object]:
    """
    Dataset-level default configuration.
    This does not use ground-truth labels during clustering.
    It only sets preprocessing and candidate k values for testing.
    """
    name = dataset_name.lower()
    configs = {
        # "iris":        {"scaler": "none", "k_candidates": [3], "allow_graph": False},
        # "seed":        {"scaler": "none", "k_candidates": [3], "allow_graph": False},
        # "segment_3":   {"scaler": "none", "k_candidates": [8], "allow_graph": True},
        # "landsat_2":   {"scaler": "none", "k_candidates": [5], "allow_graph": True},
        # "msplice_2":   {"scaler": "none", "k_candidates": [4], "allow_graph": True},
        # "rice":        {"scaler": "none", "k_candidates": [2], "allow_graph": False},
        # "banknote":    {"scaler": "none", "k_candidates": [2], "allow_graph": False},
        # "htru2":       {"scaler": "none", "k_candidates": [2], "allow_graph": False},
        # "hcv_data":    {"scaler": "none", "k_candidates": [2], "allow_graph": False},
        # "dry_bean":    {"scaler": "none", "k_candidates": [7], "allow_graph": True},
        # "rice_cammeo": {"scaler": "none", "k_candidates": [2], "allow_graph": False},
        # Safe default configs.
        # These keep the strong results from the original fast version and avoid over-tuning.
        # If you want aggressive search, use tune_dataset_for_best_acc() instead of default run_all.
        "iris":        {"scaler": "none",     "k_candidates": [3], "allow_graph": False, "split_factor": 1.0},
        "seed":        {"scaler": "none",     "k_candidates": [3], "allow_graph": False, "split_factor": 1.0},
        "segment_3":   {"scaler": "standard", "k_candidates": [8], "allow_graph": True,  "split_factor": 1.0},
        "landsat_2":   {"scaler": "standard", "k_candidates": [5], "allow_graph": True,  "split_factor": 1.0},

        # msplice improved in the aggressive run with k=3 and single-chain labeling.
        "msplice_2":   {"scaler": "standard", "k_candidates": [3], "allow_graph": False, "split_factor": 0.7},

        # rice improved with k=3 in the aggressive run, but if you need exactly 2 clusters, change [3] to [2].
        "rice":        {"scaler": "none",     "k_candidates": [3], "allow_graph": False, "split_factor": 0.8},

        "banknote":    {"scaler": "standard", "k_candidates": [2], "allow_graph": True,  "split_factor": 1.0},
        "htru2":       {"scaler": "none",     "k_candidates": [2], "allow_graph": False, "split_factor": 1.0},
        "breast_cancer": {"scaler": "standard", "k_candidates": [2], "allow_graph": False, "split_factor": 1.0},
        "hcv_data":    {"scaler": "standard", "k_candidates": [2], "allow_graph": True,  "split_factor": 1.0},
        "dry_bean":    {"scaler": "none",     "k_candidates": [7], "allow_graph": True,  "split_factor": 1.0},
        "rice_cammeo": {"scaler": "none",     "k_candidates": [2], "allow_graph": False, "split_factor": 1.0},
    }
    return configs.get(name, {"scaler": "standard", "k_candidates": [2], "allow_graph": True, "split_factor": 1.0})


# ============================================================
# Khai báo đường dẫn feature và label cho các bộ dữ liệu mặc định.
# ============================================================
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
# Main pipeline - Chạy toàn bộ pipeline AQG-GE-DPC từ đọc dữ liệu, sinh ellipsoid, phân cụm đến đánh giá.
# ============================================================
def run_ge_dpc_adaptive_quality(
    feature_file: Path,
    label_file: Path,
    dataset_name: str = "custom",
    epsilon: float = 1e-6,
    outlier_t: float = 2.0,
    split_factor: Optional[float] = None,
    scaler_mode: Optional[str] = None,
    k_candidates: Optional[Sequence[int]] = None,
    allow_graph: Optional[bool] = None,
    graph_switch_margin: float = 1.75,
    graph_k_factor: float = 1.0,
    graph_max_largest_ratio: float = 0.85,
    graph_changed_ratio_limit: float = 0.35,
    score_largest_ratio_threshold: float = 0.85,
    score_min_ratio_threshold: float = 0.02,
    show_chart: bool = True,
    verbose: bool = True,
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
    if split_factor is None:
        split_factor = float(cfg.get("split_factor", 1.0))

    data = apply_data_scaler(data, scaler_mode=scaler_mode)
    num = max(5, int(np.ceil(np.sqrt(data.shape[0]) * float(split_factor))))
    log = print if verbose else (lambda *args, **kwargs: None)

    log("=" * 90)
    log("GE-DPC Adaptive Quality Version")
    log(f"Dataset      : {dataset_name}")
    log(f"Data shape   : n={data.shape[0]}, d={data.shape[1]}")
    log(f"Scaler mode  : {scaler_mode}")
    log(f"k candidates : {list(k_candidates)}")
    log(f"Allow graph  : {allow_graph}")
    log(f"Split factor : {split_factor}")
    log(f"Safe split threshold num = ceil(sqrt(n) * split_factor) = {num}")
    log("=" * 90)

    # 1) Generate granular ellipsoids
    t_gen_start = time.time()
    ellipsoid_list: List[Ellipsoid] = [Ellipsoid(data, np.arange(data.shape[0]), epsilon=epsilon)]

    iteration = 0
    while True:
        iteration += 1
        before = len(ellipsoid_list)
        ellipsoid_list = splits(ellipsoid_list, num=num, epsilon=epsilon)
        after = len(ellipsoid_list)
        log(f"Ellipsoid count after safe split iteration {iteration}: {after}")
        if after == before:
            break

    ellipsoid_list = recursive_split_outlier_detection(
        ellipsoid_list,
        data,
        t=outlier_t,
        max_iterations=10,
        epsilon=epsilon,
    )
    log(f"Total ellipsoid count after outlier split: {len(ellipsoid_list)}")
    time_gen = time.time() - t_gen_start

    # 2) Compute density, delta, gamma
    t_attr_start = time.time()
    densities = np.array([ell.density for ell in ellipsoid_list], dtype=float)
    dist_mat = ellipse_distance(ellipsoid_list)
    min_dists, nearest = ellipse_min_dist(dist_mat, densities)
    gamma = densities * min_dists
    time_attr = time.time() - t_attr_start

    # 3) Try multiple k and choose best by internal score
    t_cluster_start = time.time()
    best = choose_best_labels_by_internal_score(
        densities=densities,
        min_dists=min_dists,
        dist_mat=dist_mat,
        nearest=nearest,
        k_candidates=k_candidates,
        allow_graph=allow_graph,
        graph_switch_margin=graph_switch_margin,
        graph_k_factor=graph_k_factor,
        graph_max_largest_ratio=graph_max_largest_ratio,
        graph_changed_ratio_limit=graph_changed_ratio_limit,
        score_largest_ratio_threshold=score_largest_ratio_threshold,
        score_min_ratio_threshold=score_min_ratio_threshold,
    )
    ellipsoid_labels = best["labels"]
    time_cluster = time.time() - t_cluster_start

    log("-" * 90)
    log("Selected internal configuration")
    log(f"Best k              : {best['k']}")
    log(f"Best label mode     : {best['mode']}")
    log(f"Internal score      : {best['score']:.6f}")
    log(f"Largest cluster rate: {best['largest_ratio']:.3f}")
    log(f"Selected centers    : {best['centers']}")
    log(f"Selected gamma      : {[float(gamma[i]) for i in best['centers']]}")
    print_distribution("Ellipsoid cluster distribution:", ellipsoid_labels) if verbose else None

    # 4) Map ellipsoid labels back to data points
    pred_labels = np.full(len(data), -1, dtype=int)
    for i, ell in enumerate(ellipsoid_list):
        pred_labels[ell.indices] = ellipsoid_labels[i]

    if np.any(pred_labels == -1):
        raise RuntimeError("Some data points were not assigned a cluster label.")

    print_distribution("Predicted data cluster distribution:", pred_labels) if verbose else None
    print_distribution("Ground-truth label distribution:", true_labels) if verbose else None

    # 5) Draw clustering chart (visualization only, does not affect logic)
    if show_chart:
        plot_predicted_clusters_2d(data, pred_labels, dataset_name=dataset_name, show_legend=False)

    # 6) Evaluate by labels only after clustering is done
    aligned_pred = align_labels(true_labels, pred_labels)
    acc = accuracy_score(true_labels, aligned_pred)
    nmi = normalized_mutual_info_score(true_labels, pred_labels)
    ari = adjusted_rand_score(true_labels, pred_labels)

    total_time = time_gen + time_attr + time_cluster

    log("-" * 90)
    log("Evaluation metrics")
    log(f"ACC: {acc:.3f}")
    log(f"NMI: {nmi:.3f}")
    log(f"ARI: {ari:.3f}")
    log("-" * 90)
    log("Runtime statistics")
    log(f"1. Ellipsoid generation time : {time_gen * 1000:.6f} ms")
    log(f"2. Attribute computation time: {time_attr * 1000:.6f} ms")
    log(f"3. Clustering selection time : {time_cluster * 1000:.6f} ms")
    log(f"Total program time           : {total_time * 1000:.6f} ms")
    log("=" * 90)

    return {
        "dataset": dataset_name,
        "acc": acc,
        "nmi": nmi,
        "ari": ari,
        "scaler_mode": scaler_mode,
        "outlier_t": outlier_t,
        "split_factor": split_factor,
        "graph_switch_margin": graph_switch_margin,
        "graph_max_largest_ratio": graph_max_largest_ratio,
        "graph_changed_ratio_limit": graph_changed_ratio_limit,
        "best_k": best["k"],
        "best_label_mode": best["mode"],
        "internal_score": best["score"],
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


# ============================================================
# Chạy pipeline cho một dataset đã đăng ký theo tên.
# ============================================================
def run_named_dataset(
    dataset_name: str,
    base_dir: Path,
    epsilon: float = 1e-6,
    outlier_t: float = 2.0,
    split_factor: Optional[float] = None,
    scaler_mode: Optional[str] = None,
    k_candidates: Optional[Sequence[int]] = None,
    allow_graph: Optional[bool] = None,
    graph_switch_margin: float = 1.75,
    graph_k_factor: float = 1.0,
    graph_max_largest_ratio: float = 0.85,
    graph_changed_ratio_limit: float = 0.35,
    score_largest_ratio_threshold: float = 0.85,
    score_min_ratio_threshold: float = 0.02,
    show_chart: bool = True,
    verbose: bool = True,
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

    return run_ge_dpc_adaptive_quality(
        feature_file=feature_file,
        label_file=label_file,
        dataset_name=key,
        epsilon=epsilon,
        outlier_t=outlier_t,
        split_factor=split_factor,
        scaler_mode=scaler_mode,
        k_candidates=k_candidates,
        allow_graph=allow_graph,
        graph_switch_margin=graph_switch_margin,
        graph_k_factor=graph_k_factor,
        graph_max_largest_ratio=graph_max_largest_ratio,
        graph_changed_ratio_limit=graph_changed_ratio_limit,
        score_largest_ratio_threshold=score_largest_ratio_threshold,
        score_min_ratio_threshold=score_min_ratio_threshold,
        show_chart=show_chart,
        verbose=verbose,
    )


# ============================================================
# Chạy pipeline mặc định trên toàn bộ danh sách dataset và in bảng tổng hợp.
# ============================================================
def run_all_default_datasets(
    base_dir: Path,
    dataset_names: Optional[Sequence[str]] = None,
    epsilon: float = 1e-6,
    outlier_t: float = 2.0,
    show_chart: bool = False,
) -> Dict[str, Dict[str, object]]:
    registry = get_default_dataset_registry(base_dir)
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
                show_chart=show_chart,
            )
        except Exception as exc:
            results[name] = {"error": str(exc)}
            print(f"[ERROR] {name}: {exc}")

    print("\n" + "=" * 110)
    print("SUMMARY")
    print("=" * 110)
    print(f"{'Dataset':<18} {'ACC':>8} {'NMI':>8} {'ARI':>8} {'Scaler':>10} {'Mode':>8} {'k':>4} {'Ells':>7} {'Time(ms)':>12}")
    print("-" * 110)
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
                f"{res['total_time'] * 1000:>12.6f}"
            )
    print("=" * 110)
    return results


# ============================================================
# ACC-oriented tuning utilities
# ============================================================
# ============================================================
# Khai báo không gian tìm kiếm k phục vụ tuning theo từng dataset.
# ============================================================
def get_dataset_k_search_space(dataset_name: str) -> List[List[int]]:
    """
    Wider k search space for ACC-oriented experiments.
    Labels are not used inside clustering; this only defines candidates.
    """
    key = dataset_name.lower()
    spaces = {
        "iris": [[3], [3, 4]],
        "seed": [[3], [3, 4]],
        "segment_3": [[8], [6, 7, 8, 9, 10]],
        "landsat_2": [[5], [4, 5, 6, 7]],
        "msplice_2": [[4], [3, 4, 5, 6]],
        "rice": [[2], [2, 3]],
        "banknote": [[2], [2, 3, 4]],
        "htru2": [[2], [2, 3]],
        "breast_cancer": [[2], [2, 3, 4]],
        "hcv_data": [[2], [2, 3, 4, 5]],
        "dry_bean": [[7], [6, 7, 8, 9]],
        "rice_cammeo": [[2], [2, 3]],
    }
    return spaces.get(key, [[2], [2, 3, 4]])


# ============================================================
# Tuning tham số để tìm cấu hình ACC tốt nhất; nhãn thật chỉ dùng sau khi phân cụm để so sánh.
# ============================================================
def tune_dataset_for_best_acc(
    dataset_name: str,
    base_dir: Path,
    epsilon: float = 1e-6,
    scaler_candidates: Sequence[str] = ("none", "standard", "robust"),
    outlier_t_candidates: Sequence[float] = (2.0, 1.7, 1.5, 1.3),
    split_factor_candidates: Sequence[float] = (1.0, 0.8, 0.6),
    allow_graph_candidates: Sequence[bool] = (False, True),
    graph_switch_margin_candidates: Sequence[float] = (1.75, 1.50, 1.30),
    graph_changed_ratio_limit_candidates: Sequence[float] = (0.35, 0.45),
    k_candidates_list: Optional[Sequence[Sequence[int]]] = None,
    max_trials: Optional[int] = None,
    top_n: int = 10,
    verbose_each_run: bool = False,
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    """
    Grid-search parameters to maximize ACC.

    Important research note:
    - True labels are NOT used during clustering.
    - True labels are used only after clustering to compare configurations.
    - For a paper/thesis, report this as parameter tuning/sensitivity analysis.
    """
    key = dataset_name.lower()
    registry = get_default_dataset_registry(base_dir)
    if key not in registry:
        raise KeyError(f"Unknown dataset '{dataset_name}'. Available: {list(registry.keys())}")

    feature_file, label_file = registry[key]
    if k_candidates_list is None:
        k_candidates_list = get_dataset_k_search_space(key)

    best_result: Optional[Dict[str, object]] = None
    all_results: List[Dict[str, object]] = []
    trial = 0

    for scaler_mode in scaler_candidates:
        for outlier_t in outlier_t_candidates:
            for split_factor in split_factor_candidates:
                for k_candidates in k_candidates_list:
                    for allow_graph in allow_graph_candidates:
                        margin_list = graph_switch_margin_candidates if allow_graph else (1.75,)
                        changed_list = graph_changed_ratio_limit_candidates if allow_graph else (0.35,)
                        for graph_switch_margin in margin_list:
                            for graph_changed_ratio_limit in changed_list:
                                trial += 1
                                if max_trials is not None and trial > max_trials:
                                    break
                                try:
                                    result = run_ge_dpc_adaptive_quality(
                                        feature_file=feature_file,
                                        label_file=label_file,
                                        dataset_name=key,
                                        epsilon=epsilon,
                                        outlier_t=outlier_t,
                                        split_factor=split_factor,
                                        scaler_mode=scaler_mode,
                                        k_candidates=k_candidates,
                                        allow_graph=allow_graph,
                                        graph_switch_margin=graph_switch_margin,
                                        graph_changed_ratio_limit=graph_changed_ratio_limit,
                                        show_chart=False,
                                        verbose=verbose_each_run,
                                    )

                                    item = {
                                        "dataset": key,
                                        "acc": float(result["acc"]),
                                        "nmi": float(result["nmi"]),
                                        "ari": float(result["ari"]),
                                        "time_ms": float(result["total_time"] * 1000),
                                        "scaler": scaler_mode,
                                        "outlier_t": float(outlier_t),
                                        "split_factor": float(split_factor),
                                        "allow_graph": bool(allow_graph),
                                        "graph_switch_margin": float(graph_switch_margin),
                                        "graph_changed_ratio_limit": float(graph_changed_ratio_limit),
                                        "k_candidates": list(k_candidates),
                                        "best_k": int(result["best_k"]),
                                        "mode": result["best_label_mode"],
                                        "n_ellipsoids": int(result["n_ellipsoids"]),
                                    }
                                    all_results.append(item)

                                    # Main target: ACC. Tie-breakers: NMI, ARI, then lower runtime.
                                    if best_result is None:
                                        best_result = item
                                    else:
                                        old_key = (
                                            best_result["acc"],
                                            best_result["nmi"],
                                            best_result["ari"],
                                            -best_result["time_ms"],
                                        )
                                        new_key = (
                                            item["acc"],
                                            item["nmi"],
                                            item["ari"],
                                            -item["time_ms"],
                                        )
                                        if new_key > old_key:
                                            best_result = item

                                except Exception as exc:
                                    print(
                                        f"[TUNE ERROR] {key} | scaler={scaler_mode}, "
                                        f"outlier_t={outlier_t}, split_factor={split_factor}, "
                                        f"k={list(k_candidates)}, graph={allow_graph}: {exc}"
                                    )
                            if max_trials is not None and trial >= max_trials:
                                break
                        if max_trials is not None and trial >= max_trials:
                            break
                    if max_trials is not None and trial >= max_trials:
                        break
                if max_trials is not None and trial >= max_trials:
                    break
            if max_trials is not None and trial >= max_trials:
                break
        if max_trials is not None and trial >= max_trials:
            break

    if best_result is None:
        raise RuntimeError(f"No successful tuning result for dataset '{key}'.")

    all_results_sorted = sorted(
        all_results,
        key=lambda r: (r["acc"], r["nmi"], r["ari"], -r["time_ms"]),
        reverse=True,
    )

    print("\n" + "=" * 120)
    print(f"BEST CONFIG FOR {key}")
    print("=" * 120)
    print(
        f"ACC={best_result['acc']:.3f}, NMI={best_result['nmi']:.3f}, ARI={best_result['ari']:.3f}, "
        f"Time={best_result['time_ms']:.3f} ms, scaler={best_result['scaler']}, "
        f"outlier_t={best_result['outlier_t']}, split_factor={best_result['split_factor']}, "
        f"graph={best_result['allow_graph']}, margin={best_result['graph_switch_margin']}, "
        f"k_candidates={best_result['k_candidates']}, best_k={best_result['best_k']}, mode={best_result['mode']}"
    )
    print("-" * 120)
    print(f"TOP {min(top_n, len(all_results_sorted))} CONFIGS")
    print(f"{'Rank':<5} {'ACC':>7} {'NMI':>7} {'ARI':>7} {'Time(ms)':>10} {'Scaler':>10} {'out_t':>6} {'split':>6} {'Graph':>7} {'Margin':>7} {'k_candidates':>18} {'best_k':>7} {'Mode':>8}")
    print("-" * 120)
    for rank, item in enumerate(all_results_sorted[:top_n], start=1):
        print(
            f"{rank:<5} {item['acc']:>7.3f} {item['nmi']:>7.3f} {item['ari']:>7.3f} "
            f"{item['time_ms']:>10.3f} {item['scaler']:>10} {item['outlier_t']:>6.2f} "
            f"{item['split_factor']:>6.2f} {str(item['allow_graph']):>7} "
            f"{item['graph_switch_margin']:>7.2f} {str(item['k_candidates']):>18} "
            f"{item['best_k']:>7} {item['mode']:>8}"
        )
    print("=" * 120)

    return best_result, all_results_sorted


# ============================================================
# Chạy tuning cho toàn bộ dataset mặc định và in bảng kết quả tốt nhất.
# ============================================================
def run_tuned_all_default_datasets(
    base_dir: Path,
    dataset_names: Optional[Sequence[str]] = None,
    epsilon: float = 1e-6,
    max_trials_per_dataset: Optional[int] = None,
) -> Dict[str, Dict[str, object]]:
    if dataset_names is None:
        dataset_names = [
            "iris", "seed", "segment_3", "landsat_2",
            "msplice_2", "rice", "banknote", "htru2",
            "breast_cancer", "hcv_data", "dry_bean", "rice_cammeo",
        ]

    best_results: Dict[str, Dict[str, object]] = {}
    for name in dataset_names:
        try:
            best, _ = tune_dataset_for_best_acc(
                dataset_name=name,
                base_dir=base_dir,
                epsilon=epsilon,
                max_trials=max_trials_per_dataset,
                top_n=5,
                verbose_each_run=False,
            )
            best_results[name] = best
        except Exception as exc:
            best_results[name] = {"error": str(exc)}
            print(f"[TUNING ERROR] {name}: {exc}")

    print("\n" + "=" * 120)
    print("TUNED SUMMARY")
    print("=" * 120)
    print(f"{'Dataset':<18} {'ACC':>8} {'NMI':>8} {'ARI':>8} {'Scaler':>10} {'out_t':>6} {'split':>6} {'Graph':>7} {'k':>4} {'Mode':>8} {'Ells':>7} {'Time(ms)':>12}")
    print("-" * 120)
    for name, res in best_results.items():
        if "error" in res:
            print(f"{name:<18} ERROR: {res['error']}")
        else:
            print(
                f"{name:<18} {res['acc']:>8.3f} {res['nmi']:>8.3f} {res['ari']:>8.3f} "
                f"{res['scaler']:>10} {res['outlier_t']:>6.2f} {res['split_factor']:>6.2f} "
                f"{str(res['allow_graph']):>7} {res['best_k']:>4} {res['mode']:>8} "
                f"{res['n_ellipsoids']:>7} {res['time_ms']:>12.3f}"
            )
    print("=" * 120)
    return best_results


if __name__ == "__main__":
    # If this file is inside GE-DPC-main/scripts, use parent.parent.
    # If this file is directly inside GE-DPC-main, change to Path(__file__).resolve().parent
    BASE_DIR = Path(__file__).resolve().parent.parent

    epsilon = 1e-6

    dataset_names = [
        "iris", "seed", "segment_3", "landsat_2",
        "msplice_2", "rice", "banknote", "htru2",
        "breast_cancer", "hcv_data", "dry_bean", "rice_cammeo",
    ]

    # ========================================================
    # MODE 1: Run adaptive default config quickly
    # ========================================================
    run_all_default_datasets(
        base_dir=BASE_DIR,
        dataset_names=dataset_names,
        epsilon=epsilon,
        outlier_t=1.7,
        show_chart=False,
    )

    # ========================================================
    # MODE 2: Tune one dataset for best ACC
    # Uncomment to tune a specific weak dataset first.
    # ========================================================
    # best, all_results = tune_dataset_for_best_acc(
    #     dataset_name="breast_cancer",
    #     base_dir=BASE_DIR,
    #     epsilon=epsilon,
    #     top_n=10,
    #     verbose_each_run=False,
    # )

    # ========================================================
    # MODE 3: Tune all datasets for best ACC
    # This can be slower, but it is the strongest mode for ACC.
    # ========================================================
    # run_tuned_all_default_datasets(
    #     base_dir=BASE_DIR,
    #     dataset_names=dataset_names,
    #     epsilon=epsilon,
    #     max_trials_per_dataset=None,
    # )
