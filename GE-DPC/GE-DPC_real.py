import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, normalized_mutual_info_score, adjusted_rand_score
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from scipy.optimize import linear_sum_assignment  
import time  

class Ellipsoid:
    def __init__(self, data, epsilon=1e-6):
        self.data = data  # 特征矩阵
        self.center = np.mean(self.data, axis=0)  # 椭球中心
        self.epsilon = epsilon  # 正则化参数
        self.cov_matrix = self._compute_cov_matrix()  # 协方差矩阵
        self.H_matrix = self._compute_H_matrix()  # H矩阵（Σ + εI）
        self.inv_H = np.linalg.inv(self.H_matrix)  # H的逆矩阵
        self.n_samples = len(data)  # 样本数量
        self.rho = self._compute_rho()# 计算rho值，确保椭球能覆盖所有点
        self.lengths, self.rotation = self._get_principal_axes()# 计算特征值和特征向量用于可视化
        self.major_axis_endpoints  = self.compute_major_axis_endpoints()  # 最长轴两个端点计算
    def _compute_cov_matrix(self):
        return np.cov(self.data.T, bias=True)
    def _compute_H_matrix(self):
        n = self.cov_matrix.shape[0]
        return self.cov_matrix + self.epsilon * np.eye(n)
    def _compute_rho(self):
        if self.n_samples == 0:
            return 0
        # 计算所有点到中心的马氏距离
        mahalanobis_distances = []
        for point in self.data:
            diff = point - self.center
            mahalanobis_distances.append(np.sqrt(diff.T @ self.inv_H @ diff))
        # 取最大距离
        max_distance = np.max(mahalanobis_distances)
        return max_distance
    def _get_principal_axes(self):
        eigenvalues, eigenvectors = np.linalg.eigh(self.inv_H)
        lengths = self.rho / np.sqrt(eigenvalues)  # 主轴长度
        return lengths, eigenvectors
    def compute_major_axis_endpoints(self):
        if self.n_samples == 0:
            return self.center, self.center
        # 1. 选择离椭球中心最近的点作为起始点
        center_distances = np.linalg.norm(self.data - self.center, axis=1)
        point1_idx = np.argmin(center_distances)
        point1 = self.data[point1_idx]
        # 2. 找到离point1最远的点point2
        dist_to_point1 = np.linalg.norm(self.data - point1, axis=1)
        point2_idx = np.argmax(dist_to_point1)
        point2 = self.data[point2_idx]
        # 3. 找到离point2最远的点point3
        dist_to_point2 = np.linalg.norm(self.data - point2, axis=1)
        point3_idx = np.argmax(dist_to_point2)
        point3 = self.data[point3_idx]
        return point2, point3
# 获取椭球内数据点的数量 
def get_num(ellipsoid):
    return ellipsoid.n_samples if ellipsoid.n_samples > 0 else 0
# 划分椭球 
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
    # 获取最远点对
    point1, point2 = ellipsoid.major_axis_endpoints
    # 使用欧氏距离进行初始分配
    dist_to_point1 = np.linalg.norm(data - point1, axis=1)
    dist_to_point2 = np.linalg.norm(data - point2, axis=1)
    # 分配点到两个簇
    cluster1_mask = dist_to_point1 < dist_to_point2
    cluster1 = data[cluster1_mask]
    cluster2 = data[~cluster1_mask]
    # 如果任一簇为空，返回原椭球
    if len(cluster1) == 0 or len(cluster2) == 0:
        return [ellipsoid]
    # 创建初始子椭球
    ell1 = Ellipsoid(cluster1)
    ell2 = Ellipsoid(cluster2)
    # 进行一次马氏距离优化（使用子椭球自己的协方差矩阵）
    new_cluster1 = []
    new_cluster2 = []
    for point in data:
        # 计算到两个子椭球中心的马氏距离（使用子椭球自己的协方差矩阵）
        diff1 = point - ell1.center
        dist1 = np.sqrt(diff1.T @ ell1.inv_H @ diff1) if len(cluster1) > 0 else float('inf')
        diff2 = point - ell2.center
        dist2 = np.sqrt(diff2.T @ ell2.inv_H @ diff2) if len(cluster2) > 0 else float('inf')
        # 分配到距离更近的子椭球
        if dist1 < dist2:
            new_cluster1.append(point)
        else:
            new_cluster2.append(point)
    # 更新簇
    cluster1 = np.array(new_cluster1) if new_cluster1 else np.empty((0, data.shape[1]))
    cluster2 = np.array(new_cluster2) if new_cluster2 else np.empty((0, data.shape[1]))
    # 如果任一簇为空，返回原椭球
    if len(cluster1) == 0 or len(cluster2) == 0:
        return [ellipsoid]
    # 创建最终的子椭球
    ell1 = Ellipsoid(cluster1)
    ell2 = Ellipsoid(cluster2)
    return [ell1, ell2]
def recursive_split_outlier_detection(initial_ellipsoids, data, t=1.0, max_iterations=10):
    ellipsoid_list = initial_ellipsoids.copy()
    iteration = 0
    while iteration < max_iterations:
        iteration += 1
        # 计算所有椭球的半轴长之和
        axes_sums = [np.sum(ell.lengths) for ell in ellipsoid_list]
        axes_sum_avg = np.mean(axes_sums) if axes_sums else 0
        # 筛选离群椭球：半轴长之和 > 平均值*2 
        outlier_ellipsoids = [ell for ell in ellipsoid_list 
                             if np.sum(ell.lengths) > 2 * axes_sum_avg]
        # 如果没有离群椭球，直接返回
        if not outlier_ellipsoids:
            break
        # 分割所有离群椭球
        new_ellipsoids = []
        for outlier_ell in outlier_ellipsoids:
            # 分割椭球
            sub_ells = splits_ellipsoid(outlier_ell)
            # 分割失败处理
            if len(sub_ells) != 2:
                new_ellipsoids.append(outlier_ell)
                continue
            # 任一子椭球点数小于阈值
            if any(sub_ell.n_samples < np.ceil(np.sqrt(data.shape[0]))  * 0.1 for sub_ell in sub_ells):
                new_ellipsoids.append(outlier_ell)
                continue
            # 计算密度条件
            parent_density = calculate_ellipsoid_density(outlier_ell)
            child_density_sum = sum(calculate_ellipsoid_density(sub_ell) for sub_ell in sub_ells)
            # 密度条件不满足
            if child_density_sum <= t * parent_density:
                new_ellipsoids.append(outlier_ell)
                continue
            # 密度条件满足，保留分割后的子椭球
            new_ellipsoids.extend(sub_ells)
        # 更新椭球列表：保留正常椭球 + 分割后的新椭球
        normal_ellipsoids = [ell for ell in ellipsoid_list if ell not in outlier_ellipsoids]
        ellipsoid_list = normal_ellipsoids + new_ellipsoids
    return ellipsoid_list
# 计算椭球的密度 
def calculate_ellipsoid_density(ellipsoid):
    n_samples = ellipsoid.n_samples
    # 计算半轴长度之和
    axes_sum = np.sum(ellipsoid.lengths)
    # 计算所有点到球心的马氏距离之和
    total_mahalanobis_distance = 0
    for point in ellipsoid.data:
        diff = point - ellipsoid.center
        mahalanobis_dist = np.sqrt(diff.T @ ellipsoid.inv_H @ diff)
        total_mahalanobis_distance += mahalanobis_dist
    # (点数平方) / (半轴长之和*马氏距离之和)
    density = (n_samples ** 2) / (axes_sum * total_mahalanobis_distance)
    return density
# 计算椭球间的平均马氏距离（使用平均协方差矩阵）
def ellipse_mahalanobis_distance(ellipsoid_i, ellipsoid_j):
    center_i = ellipsoid_i.center
    center_j = ellipsoid_j.center
    # 计算两个椭球协方差矩阵的平均值
    avg_cov = (ellipsoid_i.H_matrix + ellipsoid_j.H_matrix) / 2
    # 计算逆矩阵
    inv_avg_cov = np.linalg.inv(avg_cov)
    # 计算中心点之间的差异
    diff = center_i - center_j
    # 使用平均协方差矩阵计算马氏距离
    dist = np.sqrt(diff.T @ inv_avg_cov @ diff)
    return dist, dist, dist  # 为了保持接口一致，返回三个相同的值
# 计算所有椭球间的相对距离矩阵
def ellipse_distance(ellipsoid_list):
    n = len(ellipsoid_list)
    dist_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            _, _, rel_dist = ellipse_mahalanobis_distance(ellipsoid_list[i], ellipsoid_list[j])
            dist_mat[i, j] = dist_mat[j, i] = rel_dist
    return dist_mat
# 计算每个椭球到密度更高椭球的最短距离 
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
# 绘制决策图 
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
    # 创建矩形选择器 
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
# 根据用户选定的聚类中心划分簇 
def ellipse_cluster(densities, centers, nearest, min_dists):
    labels = -np.ones(len(densities), dtype=int)
    for i, c in enumerate(centers):
        labels[c] = i 
    order = np.argsort(-densities)       
    for idx in order:
        if labels[idx] == -1 and nearest[idx] != -1:
            labels[idx] = labels[nearest[idx]]
    # 处理未分配点 # 防止用户“漏选”报错，实际上手动选择中心是不会出现独立的噪声簇
    unassigned = labels == -1 
    if np.any(unassigned):       
        labels[unassigned] = len(centers)
    return labels 
# 使用匈牙利算法对齐标签
def align_labels(true_labels, pred_labels):
    # 获取真实标签和预测标签的唯一值
    true_classes = np.unique(true_labels)
    pred_classes = np.unique(pred_labels)
    # 创建混淆矩阵
    confusion_matrix = np.zeros((len(true_classes), len(pred_classes)))
    for i, true_class in enumerate(true_classes):
        for j, pred_class in enumerate(pred_classes):
            confusion_matrix[i, j] = np.sum((true_labels == true_class) & (pred_labels == pred_class))
    # 使用匈牙利算法找到最佳匹配
    row_ind, col_ind = linear_sum_assignment(-confusion_matrix)
    # 创建映射字典
    mapping = {}
    for i, j in zip(row_ind, col_ind):
        mapping[pred_classes[j]] = true_classes[i]
    # 应用映射
    aligned_pred_labels = np.array([mapping.get(label, -1) for label in pred_labels])
    return aligned_pred_labels
# 主函数 
if __name__ == "__main__":
    # 请修改为您本地的路径
    # feature_file = '/Users/quanle/Documents/Master/Thesis/Code/GE-DPC-main/real_dataset_and_label/real_datasets/Iris.txt' 
    # label_file = '/Users/quanle/Documents/Master/Thesis/Code/GE-DPC-main/real_dataset_and_label/real_datasets_label/Iris_label.txt' 
    BASE_DIR = Path(__file__).resolve().parent
    feature_file = BASE_DIR / 'dataset' / 'unlabel' / 'dry_bean.txt'
    label_file = BASE_DIR / 'dataset' / 'label' / 'dry_bean_label.txt'
    data = np.loadtxt(feature_file)
    # true_labels = np.loadtxt(label_file, dtype=int)
    true_labels = np.loadtxt(label_file, dtype=float).astype(int)
    num = np.ceil(np.sqrt(data.shape[0]))
    
    # 计时 1：椭球生成阶段
    t_gen_start = time.time()
    # 初始化第一个椭球并计算属性 
    initial_ellipsoid = Ellipsoid(data)
    ellipsoid_list = [initial_ellipsoid]
    print(f"初始椭球个数: 1")
    # 安全分割流程 
    iteration = 0
    while True:
        iteration += 1
        current_count = len(ellipsoid_list)
        ellipsoid_list = splits(ellipsoid_list, num)
        new_count = len(ellipsoid_list)
        print(f"第{iteration}次安全分割后椭球个数: {new_count}")
        if new_count == current_count:
            break  
    print(f"安全分割后总椭球个数: {len(ellipsoid_list)}")
    # 使用基于离群值检测的分割方法 (注意传入了data)
    ellipsoid_list = recursive_split_outlier_detection(ellipsoid_list, data, t=2.0)
    print(f"离群值检测分割后总椭球个数: {len(ellipsoid_list)}")
    print(f"一共生成了 {len(ellipsoid_list)} 个椭球")
    t_gen_end = time.time()
    time_gen = t_gen_end - t_gen_start
    
    # 计时 2：属性计算阶段
    t_attr_start = time.time()
    # 计算椭球属性
    densities = []
    for ell in ellipsoid_list:
        quality = calculate_ellipsoid_density(ell)
        densities.append(quality)       
    # 转换为NumPy数组 
    densities = np.array(densities)    
    dist_matrix = ellipse_distance(ellipsoid_list)
    min_dists, nearest = ellipse_min_dist(dist_matrix, densities)
    # 计算离群椭球的个数 (也算作计算开销的一部分)
    axes_sums = [np.sum(ell.lengths) for ell in ellipsoid_list]
    axes_sum_avg = np.mean(axes_sums) if axes_sums else 0
    outlier_ellipsoids = [ell for ell in ellipsoid_list 
                          if np.sum(ell.lengths) > 2 * axes_sum_avg]
    print(f"离群椭球的个数: {len(outlier_ellipsoids)}")
    t_attr_end = time.time()
    time_attr = t_attr_end - t_attr_start
    # 人工交互阶段 (不计入总时间)
    print("请在决策图上选择聚类中心（用鼠标拖拽矩形选择区域）")
    # 这个函数包含 plt.show(block=True)，是主要的人工耗时点
    selected = ellipse_draw_decision(densities, min_dists)
    print(f"选择的聚类中心: {selected}")
    
    # 计时 3：聚类计算阶段
    t_cluster_start = time.time()
    labels = ellipse_cluster(densities, selected, nearest, min_dists)
    t_cluster_end = time.time()
    
    time_cluster = t_cluster_end - t_cluster_start
    # 将椭球标签映射到数据点 (不计入耗时)
    print("正在进行数据映射（不计入Clustering Process Time）...")
    pred_labels = np.zeros(len(data), dtype=int)
    for i, ell in enumerate(ellipsoid_list):
        # 找到数据点在原始数据中的索引
        for point in ell.data:
            # 使用近似匹配来找到对应的点
            distances = np.linalg.norm(data - point, axis=1)
            idx = np.argmin(distances)
            pred_labels[idx] = labels[i]
    # 评估 (不计入耗时)
    print("正在计算评估指标...")
    # 使用匈牙利算法对齐标签
    aligned_pred_labels = align_labels(true_labels, pred_labels)
    # 计算评估指标
    acc = accuracy_score(true_labels, aligned_pred_labels)
    nmi = normalized_mutual_info_score(true_labels, pred_labels)  
    ari = adjusted_rand_score(true_labels, pred_labels)  
    # 输出指标
    print(f"ACC: {acc:.3f}")
    print(f"NMI: {nmi:.3f}")
    print(f"ARI: {ari:.3f}")
    # 打印时间统计
    print("-" * 30)
    print("运行时间统计详情：")
    print(f"1. 椭球生成时间 (Generation)  : {time_gen:.14f} 秒")
    print(f"2. 属性计算时间 (Attributes)  : {time_attr:.14f} 秒")
    print(f"3. 聚类计算时间 (Clustering)  : {time_cluster:.14f} 秒 (已剔除映射时间)")
    # 计算总有效时间 (生成 + 属性 + 聚类)，严格排除人工交互时间
    total_valid_time1 = time_attr + time_cluster
    total_valid_time2 = time_gen + time_attr + time_cluster
    print("-" * 30)
    print(f": {total_valid_time1:.14f} 秒")
    print(f"程序总有效运行时间 (已扣除人工交互): {total_valid_time2:.14f} 秒")
    print("-" * 30)

