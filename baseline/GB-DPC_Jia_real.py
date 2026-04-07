import time
import numpy as np 
import matplotlib.pyplot  as plt 
from scipy.spatial.distance  import pdist, squareform, cdist
import math 
from sklearn.cluster  import KMeans 
from sklearn.metrics  import accuracy_score, normalized_mutual_info_score, adjusted_rand_score 
from sklearn.preprocessing  import LabelEncoder 
from sklearn.metrics  import confusion_matrix 
from scipy.optimize  import linear_sum_assignment 
from matplotlib.widgets  import RectangleSelector 
 
class GranularBall:
    def __init__(self, data):
        self.data  = data 
        self.size  = len(data)
        self.center  = self._compute_center()
        self.average_radius  = self._compute_average_radius()  
        self.max_radius  = self._compute_max_radius()
        self.left  = None 
        self.right  = None 
        self.is_leaf  = True 
        self.quality  = 0.0  
    def _compute_center(self):
        return np.mean(self.data,  axis=0) if self.size  > 0 else None 
    def _compute_average_radius(self):
        if self.size  == 0: 
            return 0 
        return np.mean(np.linalg.norm(self.data  - self.center,  axis=1))
    def _compute_max_radius(self):
        if self.size  == 0: 
            return 0
        return np.max(np.linalg.norm(self.data  - self.center,  axis=1))
    def split(self, min_samples):
        if self.size  < 2: 
            return False 
        # 计算点对距离并找到最远点对 
        dist_matrix = cdist(self.data,  self.data)       
        max_idx = np.argmax(dist_matrix)       
        i, j = np.unravel_index(max_idx,  dist_matrix.shape)       
        x_alpha = self.data[i]     # 存储最远点对 
        x_beta = self.data[j]      # 存储最远点对 
        # 根据距离划分数据 
        dist_to_alpha = np.linalg.norm(self.data  - x_alpha, axis=1)
        dist_to_beta = np.linalg.norm(self.data  - x_beta, axis=1)
        labels = (dist_to_alpha <= dist_to_beta).astype(int)  # 明确使用≤关系 
        data1 = self.data[labels  == 1]  # X_α集合 
        data2 = self.data[labels  == 0]  # X_β集合 
        # 确保子粒球满足最小样本要求 
        if len(data1) < min_samples or len(data2) < min_samples: 
            return False 
        self.left  = GranularBall(data1)
        self.right  = GranularBall(data2)
        self.is_leaf  = False 
        return True 
class GBGenerator:
    def __init__(self, min_samples=2, delta=0.8, gamma=0.5):
        self.min_samples  = min_samples 
        self.delta  = delta            # 动态比例阈值 δ∈(0,1]
        self.gamma  = gamma            # 特异性权重系数 
        self.root  = None 
        self.leaves  = []             # 最终GB集合 
        self.total_size  = 0          # 总样本数 
    def fit(self, X):
        self.root  = GranularBall(X)
        self.total_size  = len(X)  
        # 1. 构建完整二叉树
        self._build_tree(self.root)       
        # 2. 自底向上剪枝 
        self._post_prune(self.root)    
        self.leaves  = self._collect_pruned_balls()
        return self 
    def _build_tree(self, ball):
        # 分裂条件 |X| > δ·√|U|
        threshold = self.delta  * math.sqrt(self.total_size)   
        # 停止条件：样本数 ≤ δ√|U| 或 小于最小样本数 
        if ball.size  <= max(self.min_samples,  threshold):    
            return 
        # 尝试分裂
        if ball.split(self.min_samples):       
            self._build_tree(ball.left)       
            self._build_tree(ball.right)       
    def _compute_quality(self, ball):
        if ball.size  == 0: 
            return 0.0
        # 覆盖度 DC = f1(|{x∈X:∥x−CX∥≤RX_Ave}|)
        coverage = np.sum(np.linalg.norm(ball.data  - ball.center,  axis=1) <= ball.max_radius)    
        # 特异度 DS = f2(RX_Ave)
        specificity = math.exp(-self.gamma  * ball.max_radius)    
        # 综合质量 D(Ω_X) = DC(Ω_X) · DS(Ω_X)
        quality = coverage * specificity 
        ball.quality  = quality 
        return quality 
    def _post_prune(self, node):
        if node is None:
            return 0.0, []  # (BQ, BC)
        # 叶节点：BQ = D(Ω_X), BC = {Ω_X}
        if node.is_leaf:    
            d_val = self._compute_quality(node)
            return d_val, [node]
        # 递归计算子节点 
        bq_left, bc_left = self._post_prune(node.left)    
        bq_right, bc_right = self._post_prune(node.right)    
        children_bq = bq_left + bq_right 
        # 计算当前节点质量 
        d_current = self._compute_quality(node)
        # 决策：保留当前节点 OR 合并子节点
        if d_current >= children_bq:
            return d_current, [node]
        else:
            return children_bq, bc_left + bc_right 
    def _collect_pruned_balls(self):
        _, bc = self._post_prune(self.root)    
        return bc
def cluster_acc(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    row_ind, col_ind = linear_sum_assignment(cm, maximize=True)
    return cm[row_ind, col_ind].sum() / y_true.size    
def calculate_center_and_radius(gb):
    data_no_label = gb[:,:]  # 取坐标
    center = data_no_label.mean(axis=0)   # 压缩行，对列取均值 
    radius = np.max((((data_no_label  - center) ** 2).sum(axis=1) ** 0.5))
    return center, radius
def gb_plot(gb_list, plt_type=0):
    plt.figure(figsize=(10,  8))
    plt.axis() 
    for gb in gb_list:
        center, radius = calculate_center_and_radius(gb)
        if plt_type == 0:  # 绘制所有点 
            plt.plot(gb[:,  0], gb[:, 1], '.', c='k', markersize=5)
        if plt_type == 0 or plt_type == 1:  # 绘制粒球 
            theta = np.arange(0,  2 * np.pi,  0.01)
            x = center[0] + radius * np.cos(theta) 
            y = center[1] + radius * np.sin(theta) 
            plt.plot(x,  y, c='r', linewidth=0.8)
        plt.plot(center[0],  center[1], 'x' if plt_type == 0 else '.', color='r')
    plt.title('Granular  Balls Visualization')
    plt.xlabel('Feature  1')
    plt.ylabel('Feature  2')
    plt.grid(alpha=0.3) 
    plt.show() 
def get_ball_quality(gb, center):
    N = gb.shape[0] 
    ball_quality = N 
    mean_r = np.mean(((gb  - center) **2)**0.5)
    return ball_quality, mean_r
def ball_density2(radiusAD, ball_qualitysA, mean_rs):
    N = radiusAD.shape[0] 
    ball_dens2 = np.zeros(shape=N) 
    for i in range(N):
        if radiusAD[i] == 0:
            ball_dens2[i] = 0
        else:
            ball_dens2[i] = ball_qualitysA[i] / (radiusAD[i] * radiusAD[i] * mean_rs[i])
    return ball_dens2 
def ball_distance(centersAD):
    Y1 = pdist(centersAD)
    ball_distAD = squareform(Y1)
    return ball_distAD
def ball_min_dist(ball_distS, ball_densS):
    N3 = ball_distS.shape[0] 
    ball_min_distAD = np.zeros(shape=N3) 
    ball_nearestAD = np.zeros(shape=N3) 
    index_ball_dens = np.argsort(-ball_densS) 
    for i3, index in enumerate(index_ball_dens):
        if i3 == 0:
            continue 
        index_ball_higher_dens = index_ball_dens[:i3]
        ball_min_distAD[index] = np.min([ball_distS[index,  j] for j in index_ball_higher_dens])
        ball_index_near = np.argmin([ball_distS[index,  j] for j in index_ball_higher_dens])
        ball_nearestAD[index] = int(index_ball_higher_dens[ball_index_near])
    ball_min_distAD[index_ball_dens[0]] = np.max(ball_min_distAD) 
    if np.max(ball_min_distAD)  < 1:
        ball_min_distAD = ball_min_distAD * 10 
    return ball_min_distAD, ball_nearestAD 
def ball_draw_decision(ball_densS, ball_min_distS):
    Bval1_start = time.time()    
    fig, ax = plt.subplots(figsize=(10,  8))
    # 绘制决策图 
    scatter = ax.scatter(ball_densS,  ball_min_distS, 
                        c='k', s=30, alpha=0.6)
    ax.set_xlabel('Density',  fontsize=12)
    ax.set_ylabel('Minimum  Distance', fontsize=12)
    ax.set_title('Density-Peak  Decision Graph (Select Centers)', 
                fontsize=14, pad=15)
    ax.grid(True,  alpha=0.3)
    # 添加颜色条解释密度 
    cbar = plt.colorbar(scatter,  ax=ax)
    cbar.set_label('Density  Level', rotation=270, labelpad=15)
    lst = []  # 存储用户选择结果 
    # 定义回调函数 
    def onselect(eclick, erelease):
        # 获取选择范围 
        xmin, xmax = sorted([eclick.xdata, erelease.xdata])  
        ymin, ymax = sorted([eclick.ydata, erelease.ydata])  
        lst.append(((xmin,  xmax), (ymin, ymax)))
        plt.close()    # 关闭窗口继续执行 
    # 创建矩形选择器  
    rs = RectangleSelector(ax, onselect,
                          useblit=True,
                          button=[1],  
                          minspanx=5, minspany=5,  
                          spancoords='pixels',
                          interactive=True)
    plt.show(block=True)    # 阻塞直到窗口关闭 
    Bval1 = time.time()  - Bval1_start 
    return lst, Bval1 
def ball_find_centers(ball_densS, ball_min_distS, lst):
    # 提取用户选择的阈值范围 
    if not lst:
        return np.array([])
    (x_range, y_range) = lst[0]
    xmin, xmax = x_range 
    ymin, ymax = y_range 
    centers = []
    N4 = ball_densS.shape[0]  
    for i4 in range(N4):
        cond_dens = (ball_densS[i4] >= xmin) and (ball_densS[i4] <= xmax)
        cond_dist = (ball_min_distS[i4] >= ymin) and (ball_min_distS[i4] <= ymax)
        if cond_dens and cond_dist:
            centers.append(i4)  
    return np.array(centers)  
def ball_cluster(ball_densS, ball_centers, ball_nearest, ball_min_distS):
    K1 = len(ball_centers)
    if K1 == 0:
        print('No cluster centers selected!')
        return None
    N5 = ball_densS.shape[0] 
    ball_labs = -1 * np.ones(N5).astype(int) 
    for i5, cen1 in enumerate(ball_centers):
        ball_labs[cen1] = int(i5+1)
    ball_index_density = np.argsort(-ball_densS) 
    for i5, index2 in enumerate(ball_index_density):
        if ball_labs[index2] == -1:
            ball_labs[index2] = ball_labs[int(ball_nearest[index2])]
    return ball_labs
def ball_draw_cluster(centersA, radiusA, ball_labs, dic_colors, gb_list, ball_centers):
    plt.figure(figsize=(12,  10))
    N6 = centersA.shape[0] 
    # 绘制所有数据点和粒球边界 
    for i6 in range(N6):
        # 绘制粒球边界
        center = centersA[i6]
        radius = radiusA[i6]
        theta = np.arange(0,  2 * np.pi,  0.01)
        x = center[0] + radius * np.cos(theta) 
        y = center[1] + radius * np.sin(theta) 
        if i6 < len(ball_labs):
             c = dic_colors.get(ball_labs[i6], 'k')
        else:
             c = 'k'
        plt.plot(x,  y, c=c, linewidth=1.5, alpha=0.7)
        
        # 绘制粒球内的点
        plt.scatter(gb_list[i6][:,  0], gb_list[i6][:, 1], 
                   color=c, s=20, alpha=0.8)
    # 绘制粒球中心 
    plt.scatter(centersA[:,  0], centersA[:, 1], marker='X', 
               s=100, c='gold', edgecolors='k', linewidths=1, label='Ball Centers')
    # 绘制聚类中心
    plt.scatter(centersA[ball_centers][:,  0], centersA[ball_centers][:, 1], 
               marker='*', s=200, c='red', edgecolors='k', linewidths=1.5, label='Cluster Centers')
    plt.title('Granular  Ball Clustering Result')
    plt.xlabel('Feature  1')
    plt.ylabel('Feature  2')
    plt.grid(alpha=0.2) 
    plt.legend() 
    plt.show() 
 
if __name__ == "__main__": 
    dic_colors = {
        0: (.8, 0, 0), 1: (0, .8, 0), 2: (0, 0, .8), 3: (.8, .8, 0),
        4: (.8, 0, .8), 5: (0, .8, .8), 6: (0, 0, 0), 7: (0.8, 0.8, 0.8),
        8: (0.6, 0, 0), 9: (0, 0.6, 0), 10: (1, 0, .8), 11: (0, 1, .8),
        12: (1, 1, .8), 13: (0.4, 0, .8), 14: (0, 0.4, .8), 15: (0.4, 0.4, .8),
        16: (1, 0.4, .8), 17: (1, 0, 1), 18: (1, 0, .8), 19: (.8, 0.2, 0), 
        20: (0, 0.7, 0), 21: (0.9, 0, .8), 22: (.8, .8, 0.1),
        23: (.8, 0.5, .8), 24: (0, .1, .8), 25: (0.9, 0, .8), 
        26: (.8, .8, 0.1), 27: (.8, 0.5, .8), 28: (0, .1, .8),
        29: (0, .1, .8)
    }
    np.set_printoptions(threshold=1e16) 
    # 请修改为您本地的数据路径
    X = np.loadtxt('C:\\Users\\庄\\Desktop\\new real datasets\\htru2.txt' )    
    y = np.loadtxt('C:\\Users\\庄\\Desktop\\new real datasets label\\htru2_label.txt' )    
    
    y = y.astype(int)  
    le = LabelEncoder()
    y_true = le.fit_transform(y)  
    
    print(f"Dataset shape: {X.shape}") 
    print(f"Labels shape: {y.shape}") 
    
    t_gen_start = time.time()
    
    gb_generator = GBGenerator(min_samples=2, delta=0.8, gamma=0.5)
    gb_generator.fit(X) 
    leaves = gb_generator.leaves  
    # 构建粒球列表 
    gb_list = [leaf.data for leaf in leaves]
    t_gen_end = time.time()
    time_generation = t_gen_end - t_gen_start
    print(f"Generated {len(gb_list)} granular balls")
    
    # 属性计算作为聚类时间的一部分
    t_attr_start = time.time()
    # 计算粒球属性
    centersAD = np.array([leaf.center  for leaf in leaves])
    radiusAD = np.array([leaf.max_radius  for leaf in leaves])
    ball_qualitysA = np.array([leaf.size  for leaf in leaves])
    mean_rs = np.array([leaf.average_radius  for leaf in leaves])
    # 密度和距离计算 
    ball_dens2 = ball_density2(radiusAD, ball_qualitysA, mean_rs)
    ball_distAD = ball_distance(centersAD)
    ball_min_distAD, ball_nearestAD = ball_min_dist(ball_distAD, ball_dens2)
    t_attr_end = time.time()
    time_attributes = t_attr_end - t_attr_start

    print("Please select cluster centers in the pop-up window...")
    # 交互时间不计入
    lst, Bval1 = ball_draw_decision(ball_dens2, ball_min_distAD)
    
    if not lst:
        print("No centers selected, exiting.")
    else:
        t_cluster_core_start = time.time()
        
        ball_centers = ball_find_centers(ball_dens2, ball_min_distAD, lst)
        print(f"Selected cluster centers indices: {ball_centers}")
        
        # 聚类分配 
        labels = ball_cluster(ball_dens2, ball_centers, ball_nearestAD, ball_min_distAD)
        
        # !!! 修改点：在此处停止计时，排除后续映射与评估时间 !!!
        t_cluster_core_end = time.time()
        time_cluster_core = t_cluster_core_end - t_cluster_core_start
        
        # 预测标签分配 (移出计时范围)
        print("正在进行数据映射（不计入Clustering Process Time）...")
        y_pred = np.zeros(len(X),  dtype=int)
        for i, leaf in enumerate(leaves):
            for point in leaf.data: 
                # 这里为了精确匹配，使用all
                idx = np.where((X  == point).all(axis=1))[0]
                if len(idx) > 0:
                    y_pred[idx] = labels[i]
        
        # 评估 (不计入计时范围)
        acc = cluster_acc(y_true, y_pred)
        nmi = normalized_mutual_info_score(y_true, y_pred)
        ari = adjusted_rand_score(y_true, y_pred)   
        
        print("\n===== Clustering Performance =====")
        print(f"ACC: {acc:.3f}")
        print(f"NMI: {nmi:.3f}")
        print(f"ARI: {ari:.3f}")                    
        
        # 计算总聚类时间 = 属性计算时间 + 核心聚类标记时间
        total_clustering_time = time_attributes + time_cluster_core
        
        # 计算总有效时间
        total_valid_time = time_generation + total_clustering_time

        print("\n===== Execution Time Statistics =====")
        print(f"1. Granular Ball Generation Time    : {time_generation:.14f} s")
        print(f"   (Attribute Calculation Time      : {time_attributes:.14f} s)")
        print(f"   (Core Clustering Labeling Time   : {time_cluster_core:.14f} s)")
        print(f"2. Total Clustering Time (Attr+Core): {total_clustering_time:.14f} s")
        print("-" * 50)
        print(f"Total Valid Execution Time          : {total_valid_time:.14f} s")
        print("(Generation + Clustering. Interactive selection & Eval excluded)")
        print("-" * 50)

        # 可视化聚类结果 (最后展示，不计入时间)
        ball_draw_cluster(centersAD, radiusAD, labels, dic_colors, 
                         [leaf.data for leaf in leaves], ball_centers)

