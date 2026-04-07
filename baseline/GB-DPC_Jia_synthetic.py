import time
import numpy as np 
import matplotlib.pyplot  as plt 
from scipy.spatial.distance  import pdist, squareform, cdist
from matplotlib.widgets  import RectangleSelector 
from sklearn.cluster  import k_means
from sklearn.metrics  import accuracy_score, normalized_mutual_info_score 
from sklearn.preprocessing  import LabelEncoder 
import math

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
        self.x_alpha = self.data[i]    # 存储最远点对 
        self.x_beta = self.data[j]     # 存储最远点对 
        # 根据距离划分数据
        dist_to_alpha = np.linalg.norm(self.data  - self.x_alpha, axis=1)
        dist_to_beta = np.linalg.norm(self.data  - self.x_beta, axis=1)
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
    def __init__(self, min_samples=2, delta=0.3, gamma=1):
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
# 计算两点距离 
def distances(data, p):
    return ((data - p) ** 2).sum(axis=1) ** 0.5
# 返回粒球中心和半径 
def calculate_center_and_radius(gb):
    data_no_label = gb[:,:]  # 取坐标
    center = data_no_label.mean(axis=0)   # 压缩行，对列取均值 
    radius = np.max((((data_no_label  - center) ** 2).sum(axis=1) ** 0.5))
    return center, radius
# 判断粒球的标签和纯度
def get_num(gb):
    return gb.shape[0]   # 矩阵的行数 
# 绘制粒球 
def gb_plot(gb_list, plt_type=0):
    plt.figure()  
    plt.axis('equal')
    fill_color = (0.6, 0.8, 1.0)  
    edge_color = (0.4, 0.6, 1.0)  
    for gb in gb_list:
        # 这里的gb可能是ndarray也可能是GranularBall对象，做个兼容
        if hasattr(gb, 'data'):
            data = gb.data
            center = gb.center
            radius = gb.max_radius
        else:
            data = gb
            center, radius = calculate_center_and_radius(gb)
        if plt_type == 0:  
            plt.scatter(data[:,  0], data[:, 1],
                      color=fill_color,  
                      s=15, alpha=1, zorder=2,
                      edgecolors='w', linewidths=0.3)  
            
        if plt_type == 0 or plt_type == 1:  
            circle = plt.Circle(center, radius, 
                               edgecolor=edge_color,  
                               facecolor='none',  
                               linewidth=1.2,  
                               zorder=3)
            plt.gca().add_patch(circle)  
        # 绘制粒球中心
        plt.scatter(center[0],  center[1],
                  marker='o', c='red', s=15, linewidths=0.5, zorder=4)
    plt.xlabel('x')  
    plt.ylabel('y')  
    plt.show() 
# 粒球分裂 
def splits_ball(gb, splitting_method):
    splits_k = 2
    ball_list = []
    # 数组去重
    len_no_label = np.unique(gb,  axis=0)
    if splitting_method == '2-means':
        if len_no_label.shape[0]  < splits_k:
            splits_k = len_no_label.shape[0] 
        label = k_means(X=gb, n_clusters=splits_k, n_init=3, random_state=8)[1]
    for single_label in range(0, splits_k):
        ball_list.append(gb[label  == single_label, :])
    return ball_list 
# 计算粒球质量
def get_ball_quality(gb, center):
    N = gb.shape[0] 
    ball_quality = N 
    mean_r = np.mean(((gb  - center) ** 2) ** 0.5)
    return ball_quality, mean_r 
# 计算粒球密度
def ball_density2(radiusAD, ball_qualitysA, mean_rs):
    N = radiusAD.shape[0] 
    ball_dens2 = np.zeros(shape=N) 
    for i in range(N):
        if radiusAD[i] == 0:
            ball_dens2[i] = 0 
        else:
            ball_dens2[i] = ball_qualitysA[i] / (radiusAD[i] * radiusAD[i] * mean_rs[i])
    return ball_dens2 
# 计算粒球距离 
def ball_distance(centersAD):
    Y1 = pdist(centersAD)
    ball_distAD = squareform(Y1)
    return ball_distAD
# 计算最小密度峰距离
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
# 绘制决策图 
def ball_draw_decision(ball_densS, ball_min_distS):
    # 此函数包含人工交互，时间不计入算法耗时
    plt.figure()
    scatter = plt.scatter(ball_densS,  ball_min_distS, 
                         c='k', s=30, alpha=0.6)
    plt.xlabel('Density',  fontsize=12)
    plt.ylabel('Minimum  Distance', fontsize=12)
    plt.title('Density-Peak  Decision Graph (Select Centers)', 
                fontsize=14, pad=15)
    plt.grid(True,  alpha=0.3)
    cbar = plt.colorbar(scatter)
    cbar.set_label('Density  Level', rotation=270, labelpad=15)
    lst = []  # 存储用户选择结果 
    def onselect(eclick, erelease):
        xmin, xmax = sorted([eclick.xdata, erelease.xdata])  
        ymin, ymax = sorted([eclick.ydata, erelease.ydata])  
        lst.append(((xmin,  xmax), (ymin, ymax)))
        plt.close() 
    rs = RectangleSelector(plt.gca(), onselect,
                          useblit=True,
                          button=[1],
                          minspanx=5, minspany=5,
                          spancoords='pixels',
                          interactive=True)
    print("请在决策图中框选聚类中心点...")
    plt.show(block=True) 
    return lst, 0 
# 找粒球中心点
def ball_find_centers(ball_densS, ball_min_distS, lst):
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
# 聚类分配
def ball_cluster(ball_densS, ball_centers, ball_nearest, ball_min_distS):
    K1 = len(ball_centers)
    if K1 == 0:
        print('no centers')
        return 
    N5 = ball_densS.shape[0] 
    ball_labs = -1 * np.ones(N5).astype(int) 
    for i5, cen1 in enumerate(ball_centers):
        ball_labs[cen1] = int(i5+1)
    ball_index_density = np.argsort(-ball_densS) 
    for i5, index2 in enumerate(ball_index_density):
        if ball_labs[index2] == -1:
            ball_labs[index2] = ball_labs[int(ball_nearest[index2])]
    return ball_labs
# 绘制聚类结果
def ball_draw_cluster(centersA, radiusA, ball_labs, dic_colors, gb_list, ball_centers):
    plt.figure()
    N6 = centersA.shape[0] 
    # 绘制所有数据点 
    for i6 in range(N6):
        # 这里的gb_list可能是ndarray列表，也可能是GranularBall对象列表
        if isinstance(gb_list[i6], GranularBall):
            points = gb_list[i6].data
        else:
            points = gb_list[i6]
        color_idx = int(ball_labs[i6])
        if color_idx in dic_colors:
            c = dic_colors[color_idx]
        else:
            c = 'k'
        plt.scatter(points[:,  0], points[:, 1],
                   color=c,
                   s=20, alpha=1, zorder=5)
    plt.axis('equal')
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
    # 请修改为您本地的路径
    data_mat = np.loadtxt('C:\\Users\\庄\\Desktop\\clusteringDatasets\\jain.txt', delimiter=',') 
    data = data_mat 
    
    print("开始粒球生成...")
    t1_start = time.time()
    generator = GBGenerator(min_samples=2, delta=0.3, gamma=1)
    generator.fit(data) 
    # 获取生成的粒球数据结构
    gb_list = generator.leaves 
    t1_end = time.time()
    time_gb_gen = t1_end - t1_start
    print(f"生成粒球数量: {len(gb_list)}")
    # 绘制粒球图 (排除时间)
    print("正在绘制粒球图...")
    # gb_plot(gb_list, plt_type=0)
    
    print("开始属性计算...")
    t2_start = time.time()
    centers = [ball.center for ball in generator.leaves] 
    radiusA = np.array([ball.max_radius  for ball in generator.leaves] ) 
    ball_qualitysA = np.array([ball.size  for ball in generator.leaves] ) 
    mean_rs = np.array([ball.average_radius  for ball in generator.leaves] ) 
    centersA = np.array(centers) 
    # 计算粒球密度 
    ball_densS = ball_density2(radiusA, ball_qualitysA, mean_rs)
    # 计算粒球之间的距离 
    ball_distS = ball_distance(centersA)
    # 计算最小密度峰距离 
    ball_min_distS, ball_nearest = ball_min_dist(ball_distS, ball_densS)
    t2_end = time.time()
    time_attr_calc = t2_end - t2_start
    
    # 绘制决策图选择中心 (暂停计时)
    lst, _ = ball_draw_decision(ball_densS, ball_min_distS)
    if not lst:
        print("未选择中心，程序结束。")
        exit()
   
    print("开始核心聚类...")
    t3_start = time.time()
    # 查找中心点 
    ball_centers = ball_find_centers(ball_densS, ball_min_distS, lst)
    # 聚类分配
    ball_labs = ball_cluster(ball_densS, ball_centers, ball_nearest, ball_min_distS)
    
    # !!! 修改点：在此处停止计时 !!!
    t3_end = time.time()
    time_core_cluster = t3_end - t3_start
    
    time_total_cluster = time_attr_calc + time_core_cluster
    time_total_effective = time_gb_gen + time_total_cluster

    print("\n" + "="*40)
    print(f"1. 粒球生成时间:         {time_gb_gen:.4f} s")
    print(f"2. 属性计算耗时:         {time_attr_calc:.4f} s")
    print(f"3. 核心聚类耗时:         {time_core_cluster:.4f} s")
    print("-" * 40)
    print(f"4. 聚类总时间 (2+3):     {time_total_cluster:.4f} s")
    print(f"5. 总有效运行时间 (1+4): {time_total_effective:.4f} s")
    print("="*40 + "\n")
    
    # 绘制聚类结果 (排除时间)
    print('正在绘制聚类结果...')
    ball_draw_cluster(centersA, radiusA, ball_labs, dic_colors, gb_list, ball_centers)
    print('完成!')

