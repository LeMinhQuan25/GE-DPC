import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, normalized_mutual_info_score, adjusted_rand_score
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from scipy.optimize import linear_sum_assignment  
import time  

class Ellipsoid:
    def __init__(self, data, epsilon=1e-6):
        self.data = data  # Feature matrix (Ma trận đặc trưng)
        self.center = np.mean(self.data, axis=0)  # Ellipsoid center (Tâm ellipsoid)
        self.epsilon = epsilon  # Regularization parameter (Tham số chuẩn hóa)
        self.cov_matrix = self._compute_cov_matrix()  # Covariance matrix (Ma trận hiệp phương sai)
        self.H_matrix = self._compute_H_matrix()  # H matrix (Σ + εI) (Ma trận H: Σ + εI)
        self.inv_H = np.linalg.inv(self.H_matrix)  # Inverse of H matrix (Ma trận nghịch đảo của H)
        self.n_samples = len(data)  # Number of samples (Số lượng mẫu)
        self.rho = self._compute_rho()# Compute rho to ensure the ellipsoid covers all points (Tính rho để đảm bảo ellipsoid bao phủ tất cả điểm)
        self.lengths, self.rotation = self._get_principal_axes()# Compute eigenvalues and eigenvectors for visualization (Tính trị riêng và vector riêng để trực quan hóa)
        self.major_axis_endpoints  = self.compute_major_axis_endpoints()  # Compute the two endpoints of the major axis (Tính hai đầu mút của trục dài nhất)
    def _compute_cov_matrix(self):
        return np.cov(self.data.T, bias=True)
    def _compute_H_matrix(self):
        n = self.cov_matrix.shape[0]
        return self.cov_matrix + self.epsilon * np.eye(n)
    def _compute_rho(self):
        if self.n_samples == 0:
            return 0
        # Compute the Mahalanobis distance from all points to the center (Tính khoảng cách Mahalanobis từ mọi điểm đến tâm)
        mahalanobis_distances = []
        for point in self.data:
            diff = point - self.center
            mahalanobis_distances.append(np.sqrt(diff.T @ self.inv_H @ diff))
        # Take the maximum distance (Lấy khoảng cách lớn nhất)
        max_distance = np.max(mahalanobis_distances)
        return max_distance
    def _get_principal_axes(self):
        eigenvalues, eigenvectors = np.linalg.eigh(self.inv_H)
        lengths = self.rho / np.sqrt(eigenvalues)  # Principal axis lengths (Độ dài các trục chính)
        return lengths, eigenvectors
    def compute_major_axis_endpoints(self):
        if self.n_samples == 0:
            return self.center, self.center
        # 1. Select the point closest to the ellipsoid center as the starting point (Chọn điểm gần tâm ellipsoid nhất làm điểm bắt đầu)
        center_distances = np.linalg.norm(self.data - self.center, axis=1)
        point1_idx = np.argmin(center_distances)
        point1 = self.data[point1_idx]
        # 2. Find point2, the farthest point from point1 (Tìm point2 là điểm xa point1 nhất)
        dist_to_point1 = np.linalg.norm(self.data - point1, axis=1)
        point2_idx = np.argmax(dist_to_point1)
        point2 = self.data[point2_idx]
        # 3. Find point3, the farthest point from point2 (Tìm point3 là điểm xa point2 nhất)
        dist_to_point2 = np.linalg.norm(self.data - point2, axis=1)
        point3_idx = np.argmax(dist_to_point2)
        point3 = self.data[point3_idx]
        return point2, point3
# Get the number of data points inside the ellipsoid (Lấy số lượng điểm dữ liệu trong ellipsoid) 
def get_num(ellipsoid):
    return ellipsoid.n_samples if ellipsoid.n_samples > 0 else 0
# Split ellipsoids (Phân tách ellipsoid) 
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
    # Get the farthest point pair (Lấy cặp điểm xa nhất)
    point1, point2 = ellipsoid.major_axis_endpoints
    # Use Euclidean distance for initial assignment (Dùng khoảng cách Euclid để gán ban đầu)
    dist_to_point1 = np.linalg.norm(data - point1, axis=1)
    dist_to_point2 = np.linalg.norm(data - point2, axis=1)
    # Assign points to two clusters (Gán điểm vào hai cụm)
    cluster1_mask = dist_to_point1 < dist_to_point2
    cluster1 = data[cluster1_mask]
    cluster2 = data[~cluster1_mask]
    # If either cluster is empty, return the original ellipsoid (Nếu một cụm rỗng thì trả về ellipsoid gốc)
    if len(cluster1) == 0 or len(cluster2) == 0:
        return [ellipsoid]
    # Create initial child ellipsoids (Tạo các ellipsoid con ban đầu)
    ell1 = Ellipsoid(cluster1)
    ell2 = Ellipsoid(cluster2)
    # Perform one Mahalanobis-distance refinement using each child ellipsoid’s own covariance matrix (Tinh chỉnh một lần bằng khoảng cách Mahalanobis với ma trận hiệp phương sai riêng của từng ellipsoid con)
    new_cluster1 = []
    new_cluster2 = []
    for point in data:
        # Compute the Mahalanobis distance to the centers of the two child ellipsoids using their own covariance matrices (Tính khoảng cách Mahalanobis tới tâm của hai ellipsoid con bằng ma trận hiệp phương sai riêng của chúng)
        diff1 = point - ell1.center
        dist1 = np.sqrt(diff1.T @ ell1.inv_H @ diff1) if len(cluster1) > 0 else float('inf')
        diff2 = point - ell2.center
        dist2 = np.sqrt(diff2.T @ ell2.inv_H @ diff2) if len(cluster2) > 0 else float('inf')
        # Assign to the closer child ellipsoid (Gán vào ellipsoid con gần hơn)
        if dist1 < dist2:
            new_cluster1.append(point)
        else:
            new_cluster2.append(point)
    # Update clusters (Cập nhật các cụm)
    cluster1 = np.array(new_cluster1) if new_cluster1 else np.empty((0, data.shape[1]))
    cluster2 = np.array(new_cluster2) if new_cluster2 else np.empty((0, data.shape[1]))
    # If either cluster is empty, return the original ellipsoid (Nếu một cụm rỗng thì trả về ellipsoid gốc)
    if len(cluster1) == 0 or len(cluster2) == 0:
        return [ellipsoid]
    # Create the final child ellipsoids (Tạo các ellipsoid con cuối cùng)
    ell1 = Ellipsoid(cluster1)
    ell2 = Ellipsoid(cluster2)
    return [ell1, ell2]
def recursive_split_outlier_detection(initial_ellipsoids, data, t=1.0, max_iterations=10):
    ellipsoid_list = initial_ellipsoids.copy()
    iteration = 0
    while iteration < max_iterations:
        iteration += 1
        # Compute the sum of semi-axis lengths for all ellipsoids (Tính tổng độ dài các bán trục của tất cả ellipsoid)
        axes_sums = [np.sum(ell.lengths) for ell in ellipsoid_list]
        axes_sum_avg = np.mean(axes_sums) if axes_sums else 0
        # Filter outlier ellipsoids: sum of semi-axis lengths > 2 × average (Lọc ellipsoid ngoại lai: tổng bán trục > 2 lần trung bình) 
        outlier_ellipsoids = [ell for ell in ellipsoid_list 
                             if np.sum(ell.lengths) > 2 * axes_sum_avg]
        # If there are no outlier ellipsoids, return directly (Nếu không có ellipsoid ngoại lai thì trả về ngay)
        if not outlier_ellipsoids:
            break
        # Split all outlier ellipsoids (Phân tách tất cả ellipsoid ngoại lai)
        new_ellipsoids = []
        for outlier_ell in outlier_ellipsoids:
            # Split the ellipsoid (Phân tách ellipsoid)
            sub_ells = splits_ellipsoid(outlier_ell)
            # Handle split failure (Xử lý khi phân tách thất bại)
            if len(sub_ells) != 2:
                new_ellipsoids.append(outlier_ell)
                continue
            # If any child ellipsoid has fewer points than the threshold (Nếu một ellipsoid con có số điểm nhỏ hơn ngưỡng)
            if any(sub_ell.n_samples < np.ceil(np.sqrt(data.shape[0]))  * 0.1 for sub_ell in sub_ells):
                new_ellipsoids.append(outlier_ell)
                continue
            # Compute the density condition (Tính điều kiện mật độ)
            parent_density = calculate_ellipsoid_density(outlier_ell)
            child_density_sum = sum(calculate_ellipsoid_density(sub_ell) for sub_ell in sub_ells)
            # Density condition not satisfied (Điều kiện mật độ không thỏa)
            if child_density_sum <= t * parent_density:
                new_ellipsoids.append(outlier_ell)
                continue
            # Density condition satisfied, keep the split child ellipsoids (Điều kiện mật độ thỏa, giữ lại các ellipsoid con sau phân tách)
            new_ellipsoids.extend(sub_ells)
        # Update the ellipsoid list: keep normal ellipsoids + newly split ellipsoids (Cập nhật danh sách ellipsoid: giữ ellipsoid bình thường + ellipsoid mới sau phân tách)
        normal_ellipsoids = [ell for ell in ellipsoid_list if ell not in outlier_ellipsoids]
        ellipsoid_list = normal_ellipsoids + new_ellipsoids
    return ellipsoid_list
# Compute ellipsoid density (Tính mật độ của ellipsoid) 
def calculate_ellipsoid_density(ellipsoid):
    n_samples = ellipsoid.n_samples
    # Compute the sum of semi-axis lengths (Tính tổng độ dài các bán trục)
    axes_sum = np.sum(ellipsoid.lengths)
    # Compute the sum of Mahalanobis distances from all points to the center (Tính tổng khoảng cách Mahalanobis từ mọi điểm đến tâm)
    total_mahalanobis_distance = 0
    for point in ellipsoid.data:
        diff = point - ellipsoid.center
        mahalanobis_dist = np.sqrt(diff.T @ ellipsoid.inv_H @ diff)
        total_mahalanobis_distance += mahalanobis_dist
    # (number of points squared) / (sum of semi-axis lengths × sum of Mahalanobis distances) ((bình phương số điểm) / (tổng bán trục × tổng khoảng cách Mahalanobis))
    density = (n_samples ** 2) / (axes_sum * total_mahalanobis_distance)
    return density
# Compute the average Mahalanobis distance between ellipsoids using the average covariance matrix (Tính khoảng cách Mahalanobis trung bình giữa các ellipsoid bằng ma trận hiệp phương sai trung bình)
def ellipse_mahalanobis_distance(ellipsoid_i, ellipsoid_j):
    center_i = ellipsoid_i.center
    center_j = ellipsoid_j.center
    # Compute the average of the two ellipsoids’ covariance matrices (Tính trung bình của hai ma trận hiệp phương sai ellipsoid)
    avg_cov = (ellipsoid_i.H_matrix + ellipsoid_j.H_matrix) / 2
    # Compute the inverse matrix (Tính ma trận nghịch đảo)
    inv_avg_cov = np.linalg.inv(avg_cov)
    # Compute the difference between the centers (Tính độ chênh giữa hai tâm)
    diff = center_i - center_j
    # Compute the Mahalanobis distance using the average covariance matrix (Tính khoảng cách Mahalanobis bằng ma trận hiệp phương sai trung bình)
    dist = np.sqrt(diff.T @ inv_avg_cov @ diff)
    return dist, dist, dist  # Return three identical values to keep the interface consistent (Trả về ba giá trị giống nhau để giữ giao diện hàm nhất quán)
# Compute the relative distance matrix for all ellipsoids (Tính ma trận khoảng cách tương đối giữa tất cả ellipsoid)
def ellipse_distance(ellipsoid_list):
    n = len(ellipsoid_list)
    dist_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            _, _, rel_dist = ellipse_mahalanobis_distance(ellipsoid_list[i], ellipsoid_list[j])
            dist_mat[i, j] = dist_mat[j, i] = rel_dist
    return dist_mat
# Compute the shortest distance from each ellipsoid to a higher-density ellipsoid (Tính khoảng cách ngắn nhất từ mỗi ellipsoid đến ellipsoid có mật độ cao hơn) 
def ellipse_min_dist(dist_mat, densities):
    densities = np.asarray(densities)       
    order = np.argsort(-densities)       
    min_dists = np.zeros(len(densities))       
    nearest = -np.ones(len(densities), dtype=int)
    for i in order[1:]:
        mask = densities > densities[i]
        if np.any(mask):       
            idx = np.argmin(dist_mat[i, mask])
            nearest[i] = np.where(mask)[0][idx]       
            min_dists[i] = dist_mat[i, nearest[i]]
        else:
            min_dists[i] = np.max(dist_mat[i])       
    min_dists[order[0]] = np.max(min_dists)       
    return min_dists, nearest 
# Draw the decision graph (Vẽ biểu đồ quyết định) 
def ellipse_draw_decision(densities, min_dists):
    densities = np.asarray(densities)       
    min_dists = np.asarray(min_dists)     
    fig, ax = plt.subplots()
    scatter = ax.scatter(densities, min_dists, 
                        c='black',  
                        s=40, 
                        edgecolors='w',
                        linewidths=0.5,
                        alpha=1)
    ax.set_xlabel('Density')
    ax.set_ylabel('Minimum Distance')
    selected = []
    def on_select(eclick, erelease):
        xmin, xmax = sorted([eclick.xdata, erelease.xdata])       
        ymin, ymax = sorted([eclick.ydata, erelease.ydata])       
        mask = (densities >= xmin) & (densities <= xmax) & (min_dists >= ymin) & (min_dists <= ymax)
        selected.extend(np.where(mask)[0].tolist())       
        plt.close()       
    # Create a rectangle selector (Tạo bộ chọn hình chữ nhật) 
    rs = RectangleSelector(ax, on_select,
                          useblit=True,
                          button=[1],
                          minspanx=5, minspany=5,
                          spancoords='pixels',
                          interactive=True,
                          )
    plt.tight_layout()       
    plt.show(block=True)       
    return selected 
# Cluster according to the user-selected cluster centers (Phân cụm theo các tâm cụm do người dùng chọn) 
def ellipse_cluster(densities, centers, nearest, min_dists):
    labels = -np.ones(len(densities), dtype=int)
    for i, c in enumerate(centers):
        labels[c] = i 
    order = np.argsort(-densities)       
    for idx in order:
        if labels[idx] == -1 and nearest[idx] != -1:
            labels[idx] = labels[nearest[idx]]
    # Handle unassigned points # Prevent errors when the user misses a selection; in practice, manually selected centers should not produce an independent noise cluster (Xử lý các điểm chưa được gán # Tránh lỗi khi người dùng chọn thiếu; thực tế việc chọn tâm thủ công sẽ không tạo ra một cụm nhiễu độc lập)
    unassigned = labels == -1 
    if np.any(unassigned):       
        labels[unassigned] = len(centers)
    return labels 
# Align labels using the Hungarian algorithm (Căn chỉnh nhãn bằng thuật toán Hungarian)
def align_labels(true_labels, pred_labels):
    # Get the unique values of true labels and predicted labels (Lấy các giá trị duy nhất của nhãn thật và nhãn dự đoán)
    true_classes = np.unique(true_labels)
    pred_classes = np.unique(pred_labels)
    # Create the confusion matrix (Tạo ma trận nhầm lẫn)
    confusion_matrix = np.zeros((len(true_classes), len(pred_classes)))
    for i, true_class in enumerate(true_classes):
        for j, pred_class in enumerate(pred_classes):
            confusion_matrix[i, j] = np.sum((true_labels == true_class) & (pred_labels == pred_class))
    # Use the Hungarian algorithm to find the best matching (Dùng thuật toán Hungarian để tìm ánh xạ tốt nhất)
    row_ind, col_ind = linear_sum_assignment(-confusion_matrix)
    # Create the mapping dictionary (Tạo từ điển ánh xạ)
    mapping = {}
    for i, j in zip(row_ind, col_ind):
        mapping[pred_classes[j]] = true_classes[i]
    # Apply the mapping (Áp dụng ánh xạ)
    aligned_pred_labels = np.array([mapping.get(label, -1) for label in pred_labels])
    return aligned_pred_labels
# Main function (Hàm chính) 
if __name__ == "__main__":
    # Please change this to your local path (Hãy đổi thành đường dẫn local của bạn)
    # feature_file = '/Users/quanle/Documents/Master/Thesis/Code/GE-DPC-main/real_dataset_and_label/real_datasets/htru2.txt' 
    # label_file = '/Users/quanle/Documents/Master/Thesis/Code/GE-DPC-main/real_dataset_and_label/real_datasets_label/htru2_label.txt' 
    # feature_file = '/Users/quanle/Documents/Master/Thesis/Code/GE-DPC-main/dataset/unlabel/hcv_data.txt'
    # label_file = '/Users/quanle/Documents/Master/Thesis/Code/GE-DPC-main/dataset/label/hcv_data_label.txt'
    BASE_DIR = Path(__file__).resolve().parent
    feature_file = BASE_DIR / 'dataset' / 'unlabel' / 'dry_bean.txt'
    label_file = BASE_DIR / 'dataset' / 'label' / 'dry_bean_label.txt'
    data = np.loadtxt(feature_file)
    # true_labels = np.loadtxt(label_file, dtype=int)
    true_labels = np.loadtxt(label_file, dtype=float).astype(int)
    num = np.ceil(np.sqrt(data.shape[0]))
    
    # Timing 1: ellipsoid generation stage (Đo thời gian 1: giai đoạn sinh ellipsoid)
    t_gen_start = time.time()
    # Initialize the first ellipsoid and compute its attributes (Khởi tạo ellipsoid đầu tiên và tính các thuộc tính) 
    initial_ellipsoid = Ellipsoid(data)
    ellipsoid_list = [initial_ellipsoid]
    print(f"Initial ellipsoid count (Số lượng ellipsoid ban đầu): 1")
    # Safe splitting process (Quy trình phân tách an toàn) 
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
    # Use the outlier-detection-based splitting method (note that data is passed in) (Dùng phương pháp phân tách dựa trên phát hiện ngoại lai, lưu ý có truyền data vào)
    ellipsoid_list = recursive_split_outlier_detection(ellipsoid_list, data, t=2.0)
    print(f"Total ellipsoid count after outlier-detection splitting (Tổng số ellipsoid sau phân tách bằng phát hiện ngoại lệ): {len(ellipsoid_list)}")
    print(f"A total of {len(ellipsoid_list)} ellipsoids were generated (Tổng cộng đã tạo {len(ellipsoid_list)} ellipsoid)")
    t_gen_end = time.time()
    time_gen = t_gen_end - t_gen_start
    
    # Timing 2: attribute computation stage (Đo thời gian 2: giai đoạn tính thuộc tính)
    t_attr_start = time.time()
    # Compute ellipsoid attributes (Tính thuộc tính ellipsoid)
    densities = []
    for ell in ellipsoid_list:
        quality = calculate_ellipsoid_density(ell)
        densities.append(quality)       
    # Convert to a NumPy array (Chuyển sang mảng NumPy) 
    densities = np.array(densities)    
    dist_matrix = ellipse_distance(ellipsoid_list)
    min_dists, nearest = ellipse_min_dist(dist_matrix, densities)
    # Compute the number of outlier ellipsoids (also counted as part of the computation cost) (Tính số lượng ellipsoid ngoại lai, cũng được tính vào chi phí tính toán)
    axes_sums = [np.sum(ell.lengths) for ell in ellipsoid_list]
    axes_sum_avg = np.mean(axes_sums) if axes_sums else 0
    outlier_ellipsoids = [ell for ell in ellipsoid_list 
                          if np.sum(ell.lengths) > 2 * axes_sum_avg]
    print(f"Number of outlier ellipsoids (Số lượng ellipsoid ngoại lệ): {len(outlier_ellipsoids)}")
    t_attr_end = time.time()
    time_attr = t_attr_end - t_attr_start
    # Manual interaction stage (not included in the total time) (Giai đoạn tương tác thủ công, không tính vào tổng thời gian)
    print("Please select cluster centers on the decision graph using mouse drag (Vui lòng chọn tâm cụm trên biểu đồ quyết định bằng cách kéo chuột để tạo vùng chọn hình chữ nhật)")
    # This function includes plt.show(block=True), which is the main source of manual waiting time (Hàm này có plt.show(block=True), là phần chờ thao tác thủ công chính)
    selected = ellipse_draw_decision(densities, min_dists)
    print(f"Selected cluster centers (Các tâm cụm đã chọn): {selected}")
    
    # Timing 3: clustering computation stage (Đo thời gian 3: giai đoạn tính toán phân cụm)
    t_cluster_start = time.time()
    labels = ellipse_cluster(densities, selected, nearest, min_dists)
    t_cluster_end = time.time()
    
    time_cluster = t_cluster_end - t_cluster_start
    # Map ellipsoid labels to data points (not included in timing) (Ánh xạ nhãn ellipsoid về các điểm dữ liệu, không tính vào thời gian)
    print("Mapping data points in progress (Đang ánh xạ điểm dữ liệu, không tính vào thời gian phân cụm)...")
    pred_labels = np.zeros(len(data), dtype=int)
    for i, ell in enumerate(ellipsoid_list):
        # Find the index of the data point in the original dataset (Tìm chỉ số của điểm dữ liệu trong dữ liệu gốc)
        for point in ell.data:
            # Use approximate matching to find the corresponding point (Dùng so khớp gần đúng để tìm điểm tương ứng)
            distances = np.linalg.norm(data - point, axis=1)
            idx = np.argmin(distances)
            pred_labels[idx] = labels[i]
    # Evaluation (not included in timing) (Đánh giá, không tính vào thời gian)
    print("Calculating evaluation metrics (Đang tính các chỉ số đánh giá)...")
    # Align labels using the Hungarian algorithm (Căn chỉnh nhãn bằng thuật toán Hungarian)
    aligned_pred_labels = align_labels(true_labels, pred_labels)
    # Compute evaluation metrics (Tính các chỉ số đánh giá)
    acc = accuracy_score(true_labels, aligned_pred_labels)
    nmi = normalized_mutual_info_score(true_labels, pred_labels)  
    ari = adjusted_rand_score(true_labels, pred_labels)  
    # Output metrics (In các chỉ số)
    print(f"ACC: {acc:.3f}")
    print(f"NMI: {nmi:.3f}")
    print(f"ARI: {ari:.3f}")
    # Print timing statistics (In thống kê thời gian)
    print("-" * 30)
    print("Runtime statistics details (Chi tiết thống kê thời gian chạy):")
    print(f"1. Ellipsoid generation time (Thời gian tạo ellipsoid): {time_gen:.14f} seconds (giây)")
    print(f"2. Attribute computation time (Thời gian tính thuộc tính): {time_attr:.14f} seconds (giây)")
    print(f"3. Clustering computation time (Thời gian tính phân cụm): {time_cluster:.14f} seconds (giây) (mapping time excluded / không tính thời gian ánh xạ)")
    # Compute the total effective time (generation + attributes + clustering), strictly excluding manual interaction time (Tính tổng thời gian hiệu dụng: sinh + thuộc tính + phân cụm, loại trừ hoàn toàn thời gian tương tác thủ công)
    total_valid_time1 = time_attr + time_cluster
    total_valid_time2 = time_gen + time_attr + time_cluster
    print("-" * 30)
    print(f"Total effective runtime (attributes + clustering) (Tổng thời gian hiệu dụng: thuộc tính + phân cụm): {total_valid_time1:.14f} seconds (giây)")
    print(f"Total effective runtime of the program (Tổng thời gian chạy hiệu dụng của chương trình, đã loại trừ tương tác thủ công): {total_valid_time2:.14f} seconds (giây)")
    print("-" * 30)

