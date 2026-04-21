import time
import numpy as np 
from pathlib import Path
import matplotlib.pyplot as plt 
from matplotlib.patches import Ellipse 
from matplotlib.widgets import RectangleSelector 

class Ellipsoid:
    def __init__(self, data, epsilon=1e-6):#, rho_factor=1):
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
        if self.data.shape[0] < 2:
            return np.zeros((self.data.shape[1], self.data.shape[1]))
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
        max_distance = np.max(mahalanobis_distances)
        return max_distance
    def _get_principal_axes(self):
        eigenvalues, eigenvectors = np.linalg.eigh(self.inv_H)
        with np.errstate(divide='ignore', invalid='ignore'):
            lengths = self.rho / np.sqrt(eigenvalues)  # 主轴长度
        lengths = np.nan_to_num(lengths)
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
    def get_ellipse_params(self):
        # 使用从_get_principal_axes计算得到的主轴长度和旋转矩阵
        angle_rad = np.arctan2(self.rotation[1, 0], self.rotation[0, 0])
        angle_deg = np.degrees(angle_rad)
        width = 2 * self.lengths[0] 
        height = 2 * self.lengths[1]  
        return width, height, angle_deg
# 绘制原始数据分布图 
def draw_point(data):
    plt.figure()
    plt.axis('equal')
    plt.scatter(data[:, 0], data[:, 1], s=16., c='k')
    plt.title("Original Data")
    plt.show()
# 获取椭球内数据点的数量 
def get_num(ellipsoid):
    return ellipsoid.n_samples if ellipsoid.n_samples > 0 else 0
# 可视化每次椭球分割结果 
def ge_plot_1(ellipsoid_list, plt_type=0, prev_ellipsoid_list=None):
    plt.figure()
    plt.axis('equal')
    fill_color = (0.6, 0.8, 1.0)  
    edge_color = (0.4, 0.6, 1.0)  
    colors = [fill_color] * len(ellipsoid_list)   
    new_indices = []
    if prev_ellipsoid_list is not None:
        hash_prev = [hash(str(e.center.tobytes())) for e in prev_ellipsoid_list]
        new_indices = [i for i, e in enumerate(ellipsoid_list) 
                       if hash(str(e.center.tobytes())) not in hash_prev]
    for i, ell in enumerate(ellipsoid_list):
        line_style = '-'  
        line_width = 1.2  
        if plt_type == 0:
            if ell.n_samples > 0:
                plt.scatter(ell.data[:, 0], ell.data[:, 1],
                          color=fill_color,  # 强制淡蓝色 
                          s=15, alpha=1, zorder=2,
                          edgecolors='w', linewidths=0.3)
        width, height, angle = ell.get_ellipse_params()
        ellipse = Ellipse(
            xy=ell.center,
            width=width,
            height=height,
            angle=angle,
            edgecolor=edge_color,
            facecolor=colors[i] if plt_type==1 else 'none',  # 填充色配置 
            linestyle=line_style,
            linewidth=line_width,
            zorder=3 if i in new_indices else 2 
        )
        plt.gca().add_patch(ellipse) 
        plt.scatter(ell.center[0], ell.center[1],
                  marker='o', c='red', s=10, linewidths=0.5, zorder=4)
        
    plt.show()  
# 绘制基础椭球图层（所有普通椭球）
def draw_base_ellipsoids(ellipsoids, ax):
    default_edge_color = (0.4, 0.6, 1.0) 
    plt.axis('equal')
    for i, ell in enumerate(ellipsoids):
        if ell.n_samples > 0:
            plt.scatter(ell.data[:, 0], ell.data[:, 1], 
                      color='grey', s=20, alpha=1, zorder=1,
                      edgecolors='none')
        width, height, angle = ell.get_ellipse_params()
        ellipse = Ellipse(
            xy=ell.center,  
            width=width,
            height=height,
            angle=angle,
            edgecolor=default_edge_color,
            facecolor='none',
            linewidth=1,
            alpha=0.7,  
            zorder=2
        )
        plt.gca().add_patch(ellipse)
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
# 基于离群值检测的递归椭球分割算法
def recursive_split_outlier_detection(initial_ellipsoids, t=1.0, max_iterations=10, data_shape_0=None):
    if data_shape_0 is None:
        pass
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
            if any(sub_ell.n_samples < np.ceil(np.sqrt(data_shape_0)) * 0.1 for sub_ell in sub_ells):
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
    if axes_sum * total_mahalanobis_distance == 0:
        return 0
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
    # 此函数包含人工交互，时间不计入算法耗时
    densities = np.asarray(densities)       
    min_dists = np.asarray(min_dists)     
    plt.figure()
    scatter = plt.scatter(densities, min_dists, 
                          c='black',  
                          s=40, 
                          edgecolors='w',
                          linewidths=0.5,
                          alpha=1)
    plt.xlabel('Density')
    plt.ylabel('Minimum Distance')
    plt.title('Decision Graph (Please Select Centers)')
    selected = []
    def on_select(eclick, erelease):
        xmin, xmax = sorted([eclick.xdata, erelease.xdata])       
        ymin, ymax = sorted([eclick.ydata, erelease.ydata])       
        mask = (densities >= xmin) & (densities <= xmax) & (min_dists >= ymin) & (min_dists <= ymax)
        selected.extend(np.where(mask)[0].tolist())       
        plt.close()       
    # 创建矩形选择器 
    rs = RectangleSelector(plt.gca(), on_select,
                          useblit=True,
                          button=[1],
                          minspanx=5, minspany=5,
                          spancoords='pixels',
                          interactive=True,
                          )
    plt.tight_layout()       
    print("请在决策图中框选聚类中心点...")
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
        # 防止用户“漏选”报错，实际上手动选择中心是不会出现独立的噪声簇
    
        labels[unassigned] = len(centers)
    return labels 
# 可视化最终椭球结果 
def ellipse_draw_cluster(ellipsoid_list, cluster_labels, dic_colors):
    plt.figure()
    plt.axis('equal')
    n_clusters = len(np.unique(cluster_labels)) 
    colors = [dic_colors[i % 30] for i in range(n_clusters)]  
    # 创建标签到颜色的映射
    label_to_color = {label: colors[label % len(colors)] for label in np.unique(cluster_labels)} 
    for i, ell in enumerate(ellipsoid_list):
        # 获取当前椭球的聚类标签
        c = cluster_labels[i]
        color = label_to_color.get(c, 'gray') 
        if ell.n_samples > 0:
            plt.scatter(ell.data[:, 0], ell.data[:, 1], 
                      color=color, s=20, alpha=1, zorder=3,
                      edgecolors='none', linewidths=0)
    plt.show()
# 主函数 
if __name__ == "__main__":
    dic_colors = {0: (.8, 0, 0), 1: (0, .8, 0),
                  2: (0, 0, .8), 3: (.8, .8, 0),
                  4: (.8, 0, .8), 5: (0, .8, .8),
                  6: (0, 0, 0), 7: (0.8, 0.8, 0.8),
                  8: (0.6, 0, 0), 9: (0, 0.6, 0),
                  10: (1, 0, .8), 11: (0, 1, .8),
                  12: (1, 1, .8), 13: (0.4, 0, .8),
                  14: (0, 0.4, .8), 15: (0.4, 0.4, .8),
                  16: (1, 0.4, .8), 17: (1, 0, 1),
                  18: (1, 0, .8), 19: (.8, 0.2, 0), 20: (0, 0.7, 0),
                  21: (0.9, 0, .8), 22: (.8, .8, 0.1),
                  23: (.8, 0.5, .8), 24: (0, .1, .8),
                  25: (0.9, 0, .8), 26: (.8, .8, 0.1),
                  27: (.8, 0.5, .8), 28: (0, .1, .8),
                  29: (0, .1, .8)
                  }
    np.set_printoptions(threshold=1e16)
    # 请修改为您本地的路径
    data = np.loadtxt('/Users/quanle/Documents/Master/Thesis/Code/GE-DPC-main/real_dataset_and_label/real_datasets/msplice_2.txt')
    # 初始绘图 (排除时间)
    print("绘制初始数据分布...")
    draw_point(data)
    
    print("开始椭球生成...")
    t1_start = time.time()
    num = np.ceil(np.sqrt(data.shape[0]))
    # 初始化第一个椭球并计算属性 
    initial_ellipsoid = Ellipsoid(data)
    ellipsoid_list = [initial_ellipsoid]
    # 安全分割流程 
    iteration_count = 0
    while True:
        current_count = len(ellipsoid_list)
        iteration_count += 1
        ellipsoid_list = splits(ellipsoid_list, num)
        if len(ellipsoid_list) == current_count:
            break
    # 使用基于离群值检测的分割方法
    ellipsoid_list = recursive_split_outlier_detection(ellipsoid_list, t=1.7, data_shape_0=data.shape[0])
    t1_end = time.time()
    time_gb_gen = t1_end - t1_start
    print(f"递归分割阶段结束，共生成椭球数量: {len(ellipsoid_list)}")
    
    print("开始属性计算...")
    t2_start = time.time()
    # 计算椭球属性 
    densities = []
    for ell in ellipsoid_list:
        quality = calculate_ellipsoid_density(ell)
        densities.append(quality)       
    # 转换为NumPy数组 
    densities = np.array(densities)     
    dist_matrix = ellipse_distance(ellipsoid_list)
    min_dists, nearest = ellipse_min_dist(dist_matrix, densities)
    t2_end = time.time()
    time_attr_calc = t2_end - t2_start
    # 密度峰值聚类选择中心 (暂停计时)
    selected = ellipse_draw_decision(densities, min_dists)
    if not selected:
        print("未选择中心，使用默认中心（例如密度最大的）。")
        # 简单回退策略
        # selected = [np.argmax(densities)]
        exit()
        
    print("开始核心聚类...")
    t3_start = time.time()
    labels = ellipse_cluster(densities, selected, nearest, min_dists)
    
    t3_end = time.time()
    time_core_cluster = t3_end - t3_start
    time_total_cluster = time_attr_calc + time_core_cluster
    time_total_effective = time_gb_gen + time_total_cluster
    print("\n" + "="*40)
    print(f"1. 椭球生成时间:         {time_gb_gen:.4f} s")
    print(f"2. 属性计算耗时:         {time_attr_calc:.4f} s")
    print(f"3. 核心聚类耗时:         {time_core_cluster:.4f} s")
    print("-" * 40)
    print(f"4. 聚类总时间 (2+3):     {time_total_cluster:.4f} s")
    print(f"5. 总有效运行时间 (1+4): {time_total_effective:.4f} s")
    print("="*40 + "\n")
    # 结果可视化 (排除时间)
    print('正在绘制聚类结果...')
    ellipse_draw_cluster(ellipsoid_list, labels, dic_colors)
    print('完成!')

