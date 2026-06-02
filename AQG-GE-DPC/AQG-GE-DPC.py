"""
AQG-GE-DPC
============================
Pipeline:
1) Đọc dữ liệu đầu vào
2) Chuẩn hóa dữ liệu theo cấu hình tốt nhất của từng dataset
3) Sinh granular ellipsoids bằng safe split và quality gate
4) Tính khoảng cách Mahalanobis bằng Cholesky kết hợp cache
5) Tính density, delta và gamma
6) Chọn tâm cụm và gán nhãn theo DPC
7) Chỉ áp dụng graph correction nếu dataset được cấu hình sử dụng graph
8) Ánh xạ nhãn ellipsoid về từng điểm dữ liệu
9) Đánh giá ACC, NMI, ARI và thời gian chạy

Ghi chú:
- Nhãn thật chỉ được sử dụng ở bước đánh giá cuối cùng.
- Không còn phần tuning/grid-search trong file này.
- Các tham số mặc định được lấy từ best configuration đã tune trước đó cho 12 dataset.
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
# Ellipsoid with Cholesky + Cache ; Biểu diễn một granular ellipsoid và lưu cache các đại lượng hình học cần tính toán
# ============================================================
class Ellipsoid:
    # Khởi tạo ellipsoid từ tập dữ liệu con và lưu chỉ số gốc của các điểm dữ liệu
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

    # Tính và lưu cache ma trận hiệp phương sai của ellipsoid
    @property
    def cov_matrix(self) -> np.ndarray:
        if self._cov_matrix is None:
            if self.n_samples <= 1:
                self._cov_matrix = np.zeros((self.dim, self.dim), dtype=float)
            else:
                self._cov_matrix = np.cov(self.data.T, bias=True)
        return self._cov_matrix

    # Tính ma trận hình dạng H = covariance + epsilon * I để ổn định số
    @property
    def H_matrix(self) -> np.ndarray:
        if self._H_matrix is None:
            self._H_matrix = self.cov_matrix + self.epsilon * np.eye(self.dim, dtype=float)
        return self._H_matrix

    # Phân rã Cholesky ma trận H và lưu cache để tái sử dụng
    @property
    def chol_factor(self):
        if self._chol_factor is None:
            self._chol_factor = cho_factor(self.H_matrix, lower=True, check_finite=False)
        return self._chol_factor

    # Giải hệ tuyến tính Hx = rhs bằng Cholesky, tránh nghịch đảo ma trận trực tiếp
    def solve_H(self, rhs: np.ndarray) -> np.ndarray:
        return cho_solve(self.chol_factor, rhs, check_finite=False)

    # Tính bình phương khoảng cách Mahalanobis từ các điểm đến tâm ellipsoid
    def mahal_sq_points(self, points: np.ndarray) -> np.ndarray:
        X = np.asarray(points, dtype=float)
        if X.ndim == 1:
            X = X[None, :]
        diffs = X - self.center
        solved = self.solve_H(diffs.T).T
        return np.maximum(np.einsum("ij,ij->i", diffs, solved), 0.0)

    # Tính bán kính rho để ellipsoid bao phủ các điểm dữ liệu bên trong nó
    @property
    def rho(self) -> float:
        if self._rho is None:
            self._rho = float(np.sqrt(np.max(self.mahal_sq_points(self.data))))
        return self._rho

    # Tính độ dài các bán trục và hướng xoay của ellipsoid
    @property
    def lengths_rotation(self):
        if self._lengths_rotation is None:
            eigvals_H, eigvecs_H = np.linalg.eigh(self.H_matrix)
            eigvals_H = np.maximum(eigvals_H, 1e-12)
            lengths = self.rho * np.sqrt(eigvals_H)
            self._lengths_rotation = (lengths, eigvecs_H)
        return self._lengths_rotation

    # Lấy độ dài các bán trục của ellipsoid
    @property
    def lengths(self) -> np.ndarray:
        return self.lengths_rotation[0]

    # Xác định hai điểm đầu mút gần đúng trên trục chính để phục vụ tách ellipsoid
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

    # Tính mật độ của ellipsoid dựa trên số điểm, bán trục và khoảng cách Mahalanobis
    @property
    def density(self) -> float:
        if self._density is None:
            axes_sum = max(float(np.sum(self.lengths)), 1e-12)
            mahal = np.sqrt(self.mahal_sq_points(self.data))
            total_mahal = max(float(np.sum(mahal)), 1e-12)
            self._density = float((self.n_samples ** 2) / (axes_sum * total_mahal))
        return self._density


# ============================================================
# Data scaling; Chuẩn hóa dữ liệu đầu vào theo cấu hình của từng dataset
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
# GE generation: safe split + outlier split; Duyệt danh sách ellipsoid và tách các ellipsoid có số điểm vượt ngưỡng
# ============================================================
def splits(ellipsoid_list: Sequence[Ellipsoid], num: int, epsilon: float) -> List[Ellipsoid]:
    new_ells: List[Ellipsoid] = []
    for ell in ellipsoid_list:
        if ell.n_samples < num:
            new_ells.append(ell)
        else:
            new_ells.extend(splits_ellipsoid(ell, epsilon=epsilon))
    return new_ells

# Tách một ellipsoid thành hai ellipsoid con bằng cặp điểm xa nhau và tinh chỉnh bằng Mahalanobis
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

# Tách bổ sung các ellipsoid lớn hoặc bất ổn bằng quality gate để hạn chế tách dư thừa
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
# DPC attributes; Tính khoảng cách Mahalanobis giữa hai ellipsoid bằng ma trận H trung bình và Cholesky
# ============================================================
def ellipse_mahalanobis_distance(ell_i: Ellipsoid, ell_j: Ellipsoid) -> float:
    avg_H = 0.5 * (ell_i.H_matrix + ell_j.H_matrix)
    chol_avg = cho_factor(avg_H, lower=True, check_finite=False)
    diff = ell_i.center - ell_j.center
    solved = cho_solve(chol_avg, diff, check_finite=False)
    return float(np.sqrt(max(float(diff.T @ solved), 0.0)))

# Tạo ma trận khoảng cách đôi một giữa tất cả các ellipsoid
def ellipse_distance(ellipsoid_list: Sequence[Ellipsoid]) -> np.ndarray:
    n = len(ellipsoid_list)
    dist_mat = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            d = ellipse_mahalanobis_distance(ellipsoid_list[i], ellipsoid_list[j])
            dist_mat[i, j] = dist_mat[j, i] = d
    return dist_mat

# Tính delta và ellipsoid có mật độ cao hơn gần nhất theo nguyên lý DPC
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
# Center selection; # Tự động chọn tâm cụm dựa trên gamma và lọc các tâm quá gần nhau
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
# Label assignment; # Gán nhãn ellipsoid theo chuỗi DPC từ các tâm cụm đã chọn
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

# Hiệu chỉnh nhãn bằng đồ thị theo hướng bảo thủ, chỉ đổi nhãn khi tín hiệu đủ mạnh
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
# Internal score and safety gate; Tính tỷ lệ kích thước của cụm lớn nhất để kiểm tra mất cân bằng cụm
# ============================================================
def cluster_size_ratio(labels: np.ndarray) -> float:
    _, counts = np.unique(labels, return_counts=True)
    return float(np.max(counts) / max(np.sum(counts), 1))

# Kiểm tra graph correction có an toàn không trước khi chấp nhận nhãn hiệu chỉnh
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

# Chấm điểm cấu hình phân cụm bằng độ tách biệt, độ chặt và mức cân bằng cụm
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

# Chọn cấu hình nhãn tốt nhất theo chỉ số nội tại, không sử dụng nhãn thật
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
# Evaluation; Căn chỉnh nhãn dự đoán với nhãn thật bằng thuật toán Hungarian để tính ACC
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

# In phân bố số lượng mẫu theo từng nhãn
def print_distribution(name: str, labels: np.ndarray) -> None:
    unique, counts = np.unique(labels, return_counts=True)
    print(name)
    for lab, cnt in zip(unique, counts):
        print(f"  label {int(lab):>4}: {int(cnt)}")


# ============================================================
# Plotting; Vẽ biểu đồ phân cụm 2D; nếu dữ liệu nhiều chiều thì giảm chiều bằng PCA
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
# Best dataset configs - Cấu hình tốt nhất đã tune cho từng dataset.
# ============================================================
BEST_DATASET_CONFIGS: Dict[str, Dict[str, object]] = {
    "iris": {
        "scaler": "none", "k": 3, "allow_graph": False,
        "outlier_t": 1.3, "split_factor": 1.0, "graph_switch_margin": 1.75,
    },
    "seed": {
        "scaler": "none", "k": 3, "allow_graph": False,
        "outlier_t": 1.3, "split_factor": 1.0, "graph_switch_margin": 1.75,
    },
    "segment_3": {
        "scaler": "robust", "k": 8, "allow_graph": False,
        "outlier_t": 1.7, "split_factor": 0.8, "graph_switch_margin": 1.75,
    },
    "landsat_2": {
        "scaler": "none", "k": 5, "allow_graph": True,
        "outlier_t": 2.0, "split_factor": 1.0, "graph_switch_margin": 1.50,
    },
    "msplice_2": {
        "scaler": "standard", "k": 3, "allow_graph": True,
        "outlier_t": 2.0, "split_factor": 0.6, "graph_switch_margin": 1.30,
    },
    "rice": {
        "scaler": "standard", "k": 2, "allow_graph": True,
        "outlier_t": 1.5, "split_factor": 0.6, "graph_switch_margin": 1.30,
    },
    "banknote": {
        "scaler": "standard", "k": 2, "allow_graph": False,
        "outlier_t": 1.5, "split_factor": 0.8, "graph_switch_margin": 1.75,
    },
    "htru2": {
        "scaler": "none", "k": 2, "allow_graph": False,
        "outlier_t": 2.0, "split_factor": 1.0, "graph_switch_margin": 1.75,
    },
    "breast_cancer": {
        "scaler": "none", "k": 2, "allow_graph": False,
        "outlier_t": 1.5, "split_factor": 0.6, "graph_switch_margin": 1.75,
    },
    "hcv_data": {
        "scaler": "robust", "k": 2, "allow_graph": False,
        "outlier_t": 2.0, "split_factor": 1.0, "graph_switch_margin": 1.75,
    },
    "dry_bean": {
        "scaler": "standard", "k": 7, "allow_graph": False,
        "outlier_t": 1.3, "split_factor": 0.6, "graph_switch_margin": 1.75,
    },
    "rice_cammeo": {
        "scaler": "standard", "k": 2, "allow_graph": True,
        "outlier_t": 1.3, "split_factor": 0.6, "graph_switch_margin": 1.30,
    },
}


# ============================================================
# Danh sách dataset mặc định.
# ============================================================
def get_default_dataset_names() -> List[str]:
    return [
        "iris", "seed", "segment_3", "landsat_2",
        "msplice_2", "rice", "banknote", "htru2",
        "breast_cancer", "hcv_data", "dry_bean", "rice_cammeo",
    ]


# ============================================================
# Khai báo đường dẫn feature và label cho các bộ dữ liệu.
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
# Chạy AQG-GE-DPC cho một dataset với best params đã cố định.
# ============================================================
def run_dataset(
    dataset_name: str,
    base_dir: Path,
    epsilon: float = 1e-6,
    show_chart: bool = False,
    verbose: bool = True,
) -> Dict[str, object]:
    key = dataset_name.lower()
    registry = get_default_dataset_registry(base_dir)

    if key not in registry:
        raise KeyError(f"Unknown dataset '{dataset_name}'. Available: {list(registry.keys())}")
    if key not in BEST_DATASET_CONFIGS:
        raise KeyError(f"Missing best config for dataset '{dataset_name}'.")

    feature_file, label_file = registry[key]
    if not feature_file.exists() or not label_file.exists():
        raise FileNotFoundError(
            f"Dataset files not found for '{dataset_name}'.\n"
            f"Feature: {feature_file}\n"
            f"Label  : {label_file}"
        )

    cfg = BEST_DATASET_CONFIGS[key]
    data = np.loadtxt(feature_file, dtype=float)
    true_labels = np.loadtxt(label_file, dtype=float).astype(int)

    if data.ndim == 1:
        data = data.reshape(-1, 1)
    if len(data) != len(true_labels):
        raise ValueError(f"Data and label length mismatch: {len(data)} vs {len(true_labels)}")

    scaler_mode = str(cfg["scaler"])
    outlier_t = float(cfg["outlier_t"])
    split_factor = float(cfg["split_factor"])
    k = int(cfg["k"])
    allow_graph = bool(cfg["allow_graph"])
    graph_switch_margin = float(cfg.get("graph_switch_margin", 1.75))

    data = apply_data_scaler(data, scaler_mode=scaler_mode)
    num = max(5, int(np.ceil(np.sqrt(data.shape[0]) * split_factor)))
    log = print if verbose else (lambda *args, **kwargs: None)

    log("=" * 90)
    log("AQG-GE-DPC with fixed best parameters")
    log(f"Dataset      : {key}")
    log(f"Data shape   : n={data.shape[0]}, d={data.shape[1]}")
    log(f"Scaler       : {scaler_mode}")
    log(f"outlier_t    : {outlier_t}")
    log(f"split_factor : {split_factor}")
    log(f"k            : {k}")
    log(f"allow_graph  : {allow_graph}")
    log(f"graph_margin : {graph_switch_margin}")
    log("=" * 90)

    # 1) Generate granular ellipsoids
    t_gen_start = time.time()
    ellipsoid_list: List[Ellipsoid] = [Ellipsoid(data, np.arange(data.shape[0]), epsilon=epsilon)]

    while True:
        before = len(ellipsoid_list)
        ellipsoid_list = splits(ellipsoid_list, num=num, epsilon=epsilon)
        after = len(ellipsoid_list)
        if after == before:
            break

    ellipsoid_list = recursive_split_outlier_detection(
        ellipsoid_list,
        data,
        t=outlier_t,
        max_iterations=10,
        epsilon=epsilon,
    )
    time_gen = time.time() - t_gen_start

    # 2) Compute density, delta, gamma
    t_attr_start = time.time()
    densities = np.array([ell.density for ell in ellipsoid_list], dtype=float)
    dist_mat = ellipse_distance(ellipsoid_list)
    min_dists, nearest = ellipse_min_dist(dist_mat, densities)
    gamma = densities * min_dists
    time_attr = time.time() - t_attr_start

    # 3) Select centers and assign labels using fixed k
    t_cluster_start = time.time()
    centers = auto_select_centers_quality(densities, min_dists, dist_mat, top_k=k)
    single_labels = ellipse_cluster_single_chain(densities, centers, nearest, dist_mat)
    label_mode = "single"
    ellipsoid_labels = single_labels

    if allow_graph:
        graph_labels = ellipse_cluster_conservative_graph_correction(
            densities,
            dist_mat,
            centers,
            nearest,
            switch_margin=graph_switch_margin,
        )
        if is_graph_safe(single_labels, graph_labels):
            ellipsoid_labels = graph_labels
            label_mode = "graph"

    time_cluster = time.time() - t_cluster_start

    # 4) Map ellipsoid labels back to data points
    pred_labels = np.full(len(data), -1, dtype=int)
    for i, ell in enumerate(ellipsoid_list):
        pred_labels[ell.indices] = ellipsoid_labels[i]

    if np.any(pred_labels == -1):
        raise RuntimeError("Some data points were not assigned a cluster label.")

    # 5) Draw clustering chart
    if show_chart:
        plot_predicted_clusters_2d(data, pred_labels, dataset_name=key, show_legend=False)

    # 6) Final evaluation only
    aligned_pred = align_labels(true_labels, pred_labels)
    acc = accuracy_score(true_labels, aligned_pred)
    nmi = normalized_mutual_info_score(true_labels, pred_labels)
    ari = adjusted_rand_score(true_labels, pred_labels)
    total_time = time_gen + time_attr + time_cluster

    log("-" * 90)
    log(f"Centers       : {centers}")
    log(f"Selected gamma: {[float(gamma[i]) for i in centers]}")
    log(f"Label mode    : {label_mode}")
    log(f"Ellipsoids    : {len(ellipsoid_list)}")
    log(f"ACC={acc:.3f}, NMI={nmi:.3f}, ARI={ari:.3f}, Time={total_time * 1000:.3f} ms")
    log("=" * 90)

    return {
        "dataset": key,
        "acc": float(acc),
        "nmi": float(nmi),
        "ari": float(ari),
        "scaler": scaler_mode,
        "outlier_t": outlier_t,
        "split_factor": split_factor,
        "allow_graph": allow_graph,
        "graph_switch_margin": graph_switch_margin,
        "best_k": k,
        "mode": label_mode,
        "n_ellipsoids": len(ellipsoid_list),
        "total_time": total_time,
        "pred_labels": pred_labels,
        "aligned_pred_labels": aligned_pred,
    }


# ============================================================
# Chạy thẳng 12 dataset bằng best params và in bảng tổng hợp.
# ============================================================
def run_all_datasets(
    base_dir: Path,
    dataset_names: Optional[Sequence[str]] = None,
    epsilon: float = 1e-6,
    show_chart: bool = False,
    verbose_each_dataset: bool = False,
) -> Dict[str, Dict[str, object]]:
    if dataset_names is None:
        dataset_names = get_default_dataset_names()

    results: Dict[str, Dict[str, object]] = {}
    for name in dataset_names:
        try:
            results[name] = run_dataset(
                dataset_name=name,
                base_dir=base_dir,
                epsilon=epsilon,
                show_chart=show_chart,
                verbose=verbose_each_dataset,
            )
        except Exception as exc:
            results[name] = {"error": str(exc)}
            print(f"[ERROR] {name}: {exc}")

    print("\n" + "=" * 125)
    print("SUMMARY - AQG-GE-DPC fixed best parameters")
    print("=" * 125)
    print(
        f"{'Dataset':<18} {'ACC':>8} {'NMI':>8} {'ARI':>8} "
        f"{'Scaler':>10} {'out_t':>6} {'split':>6} {'Graph':>7} "
        f"{'k':>4} {'Mode':>8} {'Ells':>7} {'Time(ms)':>12}"
    )
    print("-" * 125)

    for name, res in results.items():
        if "error" in res:
            print(f"{name:<18} ERROR: {res['error']}")
        else:
            print(
                f"{name:<18} {res['acc']:>8.3f} {res['nmi']:>8.3f} {res['ari']:>8.3f} "
                f"{res['scaler']:>10} {res['outlier_t']:>6.2f} {res['split_factor']:>6.2f} "
                f"{str(res['allow_graph']):>7} {res['best_k']:>4} {res['mode']:>8} "
                f"{res['n_ellipsoids']:>7} {res['total_time'] * 1000:>12.3f}"
            )

    print("=" * 125)
    return results

# ============================================================
# Điểm bắt đầu chương trình: thiết lập thư mục gốc và chạy toàn bộ dataset
# ============================================================
if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent
    # Run 1 tập dataset để kiểm tra nhanh
    # run_dataset(
    #     dataset_name="iris",
    #     base_dir=BASE_DIR,
    #     epsilon=1e-6,
    #     show_chart=True,
    #     verbose=True,
    # )

    # Run toàn bộ dataset với best params đã cố định và in bảng tổng hợp.
    run_all_datasets(
        base_dir=BASE_DIR,
        epsilon=1e-6,
        show_chart=False,
        verbose_each_dataset=False,
    )
