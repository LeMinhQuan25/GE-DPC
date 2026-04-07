import time 
import numpy as np
import matplotlib.pyplot  as plt
from scipy.spatial.distance  import pdist, squareform
from matplotlib.widgets  import RectangleSelector
from sklearn.metrics  import accuracy_score, normalized_mutual_info_score, adjusted_rand_score, confusion_matrix 
from sklearn.preprocessing  import LabelEncoder 
from scipy.optimize  import linear_sum_assignment 
 
class GranularBall:
    def __init__(self, points):
        self.points  = points 
        self.size  = len(points)  
        if self.size  > 0:
            self.center  = np.mean(points,  axis=0)  # 粒球中心 
            self.radius  = self.calculate_radius()    # 粒球半径 
            self.dm  = self.calculate_dm()    # 平均距离 
        else:
            self.center  = None 
            self.radius  = 0.0
            self.dm  = 0.0
    def calculate_radius(self):
        if self.size  == 0:
            return 0.0 
        distances = np.linalg.norm(self.points  - self.center,  axis=1)
        return np.max(distances) 
    def calculate_dm(self):
        if self.size  == 0:
            return 0.0
        distances = np.linalg.norm(self.points  - self.center,  axis=1)
        return np.sum(distances)  / self.size  
    def find_farthest_points(self):
        if self.size  < 2:
            return None, None 
        # 选择一个中心起始点
        idx1 = np.argmin(np.linalg.norm(self.points  - self.center,  axis=1))
        point1 = self.points[idx1]    
        # 找到离point1最远的point2 
        distances = np.linalg.norm(self.points  - point1, axis=1)
        idx2 = np.argmax(distances)    
        point2 = self.points[idx2]    
        # 找到离point2最远的point3 
        distances = np.linalg.norm(self.points  - point2, axis=1)
        idx3 = np.argmax(distances)    
        point3 = self.points[idx3]    
        # 返回距离最远的两个点 
        return point2, point3 
    def split(self):
        # 分裂前检查样本数 
        if self.size  < 6:  # 确保分裂后每个子球至少有2个点 
            return [self]
        # 使用最远点对算法找到两个初始中心 
        center1, center2 = self.find_farthest_points()    
        if center1 is None or center2 is None:
            return [self]
        # 计算所有点到两个中心的距离 
        dist_to_c1 = np.linalg.norm(self.points  - center1, axis=1)
        dist_to_c2 = np.linalg.norm(self.points  - center2, axis=1)
        # 根据距离分配簇标签 
        labels = np.where(dist_to_c1  < dist_to_c2, 0, 1)
        cluster1 = self.points[labels  == 0]
        cluster2 = self.points[labels  == 1]
        # 确保每个簇至少有3个点 
        if len(cluster1) < 3 or len(cluster2) < 3:
            return [self]
        return [GranularBall(cluster1), GranularBall(cluster2)]
def algorithm1_generation_of_granular_balls(D):
    GB_sets = [GranularBall(D)]
    # 第一阶段: 基于DM的分裂 
    while True:
        prev_count = len(GB_sets)
        new_GB_sets = []
        changed = False 
        for ball in GB_sets:
            if ball.size  < 3:
                new_GB_sets.append(ball)    
                continue 
            DMA = ball.dm 
            child_balls = ball.split()    
            
            if len(child_balls) == 1:  # 未实际分裂 
                new_GB_sets.append(ball)    
                continue 
            ball1, ball2 = child_balls 
            n1, n2 = ball1.size,  ball2.size    
            total_size = n1 + n2 
            DMweight = (n1 / total_size) * ball1.dm  + (n2 / total_size) * ball2.dm    
            if DMweight < DMA:
                new_GB_sets.extend([ball1,  ball2])
                changed = True 
            else:
                new_GB_sets.append(ball)    
        GB_sets = new_GB_sets 
        if not changed:
            break 
    # 第二阶段: 基于半径的分裂 
    MIN_RADIUS = 1e-5  # 最小半径阈值 
    while True:  # 循环，直到没有变化 
        new_GB_sets = []
        changed = False 
        # 过滤空粒球和无效半径 
        radii = [ball.radius for ball in GB_sets if ball.size  > 0 and ball.radius  > 0]
        if len(radii) == 0:
            break 
        mean_r = np.mean(radii)    
        median_r = np.median(radii)    
        threshold = 2 * max(mean_r, median_r)
        for ball in GB_sets:
            if ball.size  == 0:  # 空粒球直接保留 
                new_GB_sets.append(ball)    
                continue 
            # 同时满足分裂条件和最小半径限制 
            if ball.radius  >= threshold and ball.radius  > MIN_RADIUS:
                child_balls = ball.split()    
                # 检查是否实际分裂 
                if len(child_balls) > 1: 
                    changed = True 
                new_GB_sets.extend(child_balls)    
            else:
                new_GB_sets.append(ball)       
        GB_sets = new_GB_sets 
        if not changed:
            break 
    return GB_sets
 
def cluster_acc(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    row_ind, col_ind = linear_sum_assignment(cm, maximize=True)
    return cm[row_ind, col_ind].sum() / y_true.size    
def calculate_center_and_radius(points):
    if len(points) == 0:
        return None, 0.0
    center = points.mean(axis=0) 
    distances = np.linalg.norm(points  - center, axis=1)
    return center, np.max(distances) 
def distances(data, p):
    return np.linalg.norm(data  - p, axis=1)
def ball_distance(centers):
    return squareform(pdist(centers))
def ball_min_dist(ball_dist, ball_dens):
    n = ball_dist.shape[0] 
    min_dist = np.zeros(n) 
    nearest = np.zeros(n,  dtype=int)
    # 按密度降序排序索引 
    sorted_idx = np.argsort(-ball_dens) 
    for i in range(1, n):
        idx = sorted_idx[i]
        higher_dens_idx = sorted_idx[:i]
        min_dist[idx] = np.min(ball_dist[idx,  higher_dens_idx])
        min_idx = np.argmin(ball_dist[idx,  higher_dens_idx])
        nearest[idx] = higher_dens_idx[min_idx]
    min_dist[sorted_idx[0]] = np.max(min_dist) 
    return min_dist, nearest 
def ball_find_centers(ball_dens, ball_min_dist, selection_range):
    (x_range, y_range) = selection_range
    xmin, xmax = x_range
    ymin, ymax = y_range
    centers = []
    for i in range(len(ball_dens)):
        if xmin <= ball_dens[i] <= xmax and ymin <= ball_min_dist[i] <= ymax:
            centers.append(i) 
    return np.array(centers) 
def ball_cluster(ball_dens, ball_centers, ball_nearest):
    k = len(ball_centers)
    n = len(ball_dens)
    labels = -np.ones(n,  dtype=int)
    # 标记聚类中心
    for i, center in enumerate(ball_centers):
        labels[center] = i 
    # 按密度降序处理所有点
    sorted_idx = np.argsort(-ball_dens) 
    for idx in sorted_idx:
        if labels[idx] == -1:
            labels[idx] = labels[ball_nearest[idx]]
    return labels 
def draw_points(data, title='Original Data Distribution'):
    plt.figure(figsize=(8,  6))
    plt.scatter(data[:,  0], data[:, 1], s=20, c='k', alpha=0.6)
    plt.xlabel('Feature  1')
    plt.ylabel('Feature  2')
    plt.title(title) 
    plt.grid(alpha=0.3) 
    plt.show() 
def plot_granular_balls(gb_objects, plt_type=0):
    plt.figure()
    ax = plt.gca() 
    colors = plt.cm.tab10(np.linspace(0,  1, len(gb_objects)))
    for i, ball in enumerate(gb_objects):
        points = ball.points  
        if len(points) == 0:
            continue
        # 绘制数据点
        if plt_type == 0:
            plt.scatter(points[:,  0], points[:, 1], s=15, color=colors[i], alpha=0.7)
        # 绘制粒球边界 
        if plt_type == 0 or plt_type == 1:
            center = ball.center 
            radius = ball.radius  
            circle = plt.Circle(center, radius, color=colors[i], fill=False, linewidth=1.2, alpha=0.8)
            ax.add_patch(circle) 
        # 绘制粒球中心 
        plt.scatter(center[0],  center[1], marker='x' if plt_type == 0 else 'o', 
                    color='r' if plt_type == 0 else colors[i], s=60)
    plt.xlabel('Feature  1')
    plt.ylabel('Feature  2')
    plt.title('Granular  Ball Structure Visualization')
    plt.grid(alpha=0.2) 
    plt.axis('equal') 
    plt.show() 
def plot_decision_graph(ball_dens, ball_min_dist):
    fig, ax = plt.subplots()
    scatter = ax.scatter(ball_dens,  ball_min_dist, c='blue', s=50, alpha=0.6)
    ax.set_xlabel('Density',  fontsize=12)
    ax.set_ylabel('Minimum  Distance', fontsize=12)
    ax.set_title('Density-Peak  Decision Graph (Select Centers)', fontsize=14)
    ax.grid(True,  alpha=0.3)
    selection_box = []
    def onselect(eclick, erelease):
        xmin, xmax = sorted([eclick.xdata, erelease.xdata]) 
        ymin, ymax = sorted([eclick.ydata, erelease.ydata]) 
        selection_box.append(((xmin,  xmax), (ymin, ymax)))
        plt.close() 
    rs = RectangleSelector(ax, onselect, useblit=True,
                          button=[1], minspanx=5, minspany=5,
                          spancoords='pixels', interactive=True)
    plt.show(block=True)
    return selection_box 
def plot_clustering_results(gb_objects, ball_labs, centers):
    plt.figure()
    ax = plt.gca() 
    # 为不同聚类定义颜色
    cluster_colors = {}
    unique_labels = np.unique(ball_labs) 
    color_map = plt.cm.get_cmap('tab20',  len(unique_labels))
    for i, label in enumerate(unique_labels):
        cluster_colors[label] = color_map(i)
    # 绘制粒球和中心点
    for i, ball in enumerate(gb_objects):
        if len(ball.points)  == 0:
            continue
        color = cluster_colors[ball_labs[i]]
        # 绘制数据点
        plt.scatter(ball.points[:,  0], ball.points[:,  1], s=15, color=color, alpha=0.7)
        # 绘制粒球边界
        circle = plt.Circle(ball.center,  ball.radius,  color=color, fill=False, linewidth=1.5, alpha=0.7)
        ax.add_patch(circle) 
        # 高亮显示聚类中心对应的粒球 
        if i in centers:
            highlight = plt.Circle(ball.center,  ball.radius*1.05,  color='gold', fill=False, linewidth=3, alpha=0.9)
            ax.add_patch(highlight) 
    plt.xlabel('Feature  1')
    plt.ylabel('Feature  2')
    plt.title('Granular  Ball Clustering Results', fontsize=16)
    plt.grid(alpha=0.2) 
    plt.axis('equal') 
    plt.tight_layout() 
    plt.show() 
 
if __name__ == "__main__":
    # 配置参数 
    np.set_printoptions(precision=4,  suppress=True)
    plt.rcParams['font.size']  = 12 
    # 请修改为您本地的数据路径
    dataset_path = 'C:\\Users\\庄\\Desktop\\new real datasets\\banknote.txt' 
    label_path = 'C:\\Users\\庄\\Desktop\\new real datasets label\\banknote_label.txt' 
     
    X = np.loadtxt(dataset_path)    
    y = np.loadtxt(label_path)    
     
    # 标签编码
    le = LabelEncoder()
    y_true = le.fit_transform(y) 
     
    # 可视化原始数据 (不计入算法时间)
    # draw_points(X, "Original Dataset Distribution")
     
    t_gen_start = time.time()
    
    gb_objects = algorithm1_generation_of_granular_balls(X)
    
    t_gen_end = time.time()
    time_gen = t_gen_end - t_gen_start
    
    print(f"生成粒球数量: {len(gb_objects)}")
     
    # 可视化粒球结构 (不计入算法时间)
    # plot_granular_balls(gb_objects, plt_type=0)
     
    t_attr_start = time.time()
     
    # 提取粒球属性
    centers = [ball.center for ball in gb_objects if ball.size  > 0]
    radii = [ball.radius for ball in gb_objects if ball.size  > 0]
    sizes = [ball.size for ball in gb_objects if ball.size  > 0]
    dms = [ball.dm for ball in gb_objects if ball.size  > 0]
     
    # 转换为数组 
    centers_arr = np.array(centers) 
    radii_arr = np.array(radii) 
    sizes_arr = np.array(sizes) 
    dms_arr = np.array(dms) 
     
    t_attr_end = time.time()
    time_attr = t_attr_end - t_attr_start
     
    # --- 3.1: 计算距离和密度 (计时) ---
    t_cluster_p1_start = time.time()
    
    # 定义密度为粒球内点数 
    ball_dens = sizes_arr 
    # 计算粒球间距离
    ball_dist = ball_distance(centers_arr)
    # 计算最小密度距离 
    ball_min_dist_arr, ball_nearest_arr = ball_min_dist(ball_dist, ball_dens)
    
    t_cluster_p1_end = time.time()
     
    # --- 3.2: 用户交互 (不计时) ---
    print("\n请在弹出的决策图中框选聚类中心...")
    selection_box = plot_decision_graph(ball_dens, ball_min_dist_arr)
    print("聚类中心已选择，继续计算...")
    
    # 如果用户没选，给个默认值防止报错
    if not selection_box:
        print("未选择任何聚类中心，使用默认阈值")
        density_threshold = np.percentile(ball_dens,  85)
        dist_threshold = np.percentile(ball_min_dist_arr,  90)
        selection_box = [((density_threshold, np.max(ball_dens)),  
                         (dist_threshold, np.max(ball_min_dist_arr)))] 
     
    # --- 3.3: 确定中心与分配标签 (计时) ---
    t_cluster_p2_start = time.time()
    
    # 确定聚类中心 
    ball_centers = ball_find_centers(ball_dens, ball_min_dist_arr, selection_box[0])
     
    # 分配聚类标签 
    labels = ball_cluster(ball_dens, ball_centers, ball_nearest_arr)
     
    # !!! 修改点：在此处停止计时，排除后续映射与评估时间 !!!
    t_cluster_p2_end = time.time()
    
    # 将粒球标签映射到数据点 (不计入耗时)
    print("正在进行数据映射（不计入聚类时间）...")
    y_pred = np.zeros(len(X),  dtype=int)
    for i, ball in enumerate(gb_objects):
        if ball.size  > 0:
            for point in ball.points: 
                # 使用all匹配
                idx = np.where((X  == point).all(axis=1))[0]
                if len(idx) > 0:
                    y_pred[idx] = labels[i]
    
    # 计算评估指标 (不计入耗时)
    acc = cluster_acc(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)
    
    # 计算聚类总耗时 (Part 1 + Part 2)
    time_cluster = (t_cluster_p1_end - t_cluster_p1_start) + (t_cluster_p2_end - t_cluster_p2_start)
    
    print("\n" + "="*50)
    print(f"聚类准确率 (ACC): {acc:.3f}")
    print(f"标准化互信息 (NMI): {nmi:.3f}")
    print(f"调整兰德指数 (ARI): {ari:.3f}")
    print("="*50)
    
    print("程序运行时间统计 (已剔除人工交互及绘图时间):")
    print(f"1. 粒球生成时间: {time_gen:.14f} 秒")
    print(f"2. 属性计算时间: {time_attr:.14f} 秒")
    print(f"3. 聚类过程时间: {time_cluster:.14f} 秒")
    total_valid_time1 = time_attr + time_cluster
    total_valid_time2 = time_gen + time_attr + time_cluster
    print("-" * 40)
    print(f": {total_valid_time1:.14f} 秒")
    print(f"程序总有效运行时间: {total_valid_time2:.14f} 秒")
    print("="*50)
     
    # 最后展示结果图 (不计入时间)
    plot_clustering_results(gb_objects, labels, ball_centers)

