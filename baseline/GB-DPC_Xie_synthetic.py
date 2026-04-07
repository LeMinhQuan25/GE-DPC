import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from matplotlib.widgets import RectangleSelector
from matplotlib.patches import Circle

class GranularBall:
    def __init__(self, points):
        self.points = points
        self.size = len(points) 
        if self.size > 0:
            self.center = np.mean(points, axis=0)  # 粒球中心 
            self.radius = self.calculate_radius()     # 粒球半径 
            self.dm = self.calculate_dm()     # 平均距离
        else:
            # 空粒球默认值 
            self.center = None 
            self.radius = 0.0
            self.dm = 0.0
    def calculate_radius(self):
        if self.size == 0:
            return 0.0 
        distances = np.linalg.norm(self.points - self.center, axis=1)
        return np.max(distances) 
    def calculate_dm(self):
        if self.size == 0:
            return 0.0
        distances = np.linalg.norm(self.points - self.center, axis=1)
        s_j = np.sum(distances)  
        return s_j / self.size 
    def find_farthest_points(self):
        if self.size < 2:
            return None, None 
        # 选择一个中心起始点
        idx1 = np.argmin(np.linalg.norm(self.points - np.mean(self.points, axis=0), axis=1))
        point1 = self.points[idx1]  
        # 找到离point1最远的point2 
        distances = np.linalg.norm(self.points - point1, axis=1)
        idx2 = np.argmax(distances)  
        point2 = self.points[idx2]  
        # 找到离point2最远的point3 
        distances = np.linalg.norm(self.points - point2, axis=1)
        idx3 = np.argmax(distances)  
        point3 = self.points[idx3]  
        # 返回距离最远的两个点 
        return point2, point3 
    def split(self):
        # 分裂前检查样本数 
        if self.size < 6:  # 确保分裂后每个子球至少有2个点 
            return [self]
        # 使用最远点对算法找到两个初始中心 
        center1, center2 = self.find_farthest_points()  
        if center1 is None or center2 is None:
            return [self]
        # 计算所有点到两个中心的距离 
        dist_to_c1 = np.linalg.norm(self.points - center1, axis=1)
        dist_to_c2 = np.linalg.norm(self.points - center2, axis=1)
        # 根据距离分配簇标签 
        labels = np.where(dist_to_c1 < dist_to_c2, 0, 1)
        cluster1 = self.points[labels == 0]
        cluster2 = self.points[labels == 1]
        # 确保每个簇至少有2个点 
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
            if ball.size < 3:
                new_GB_sets.append(ball)  
                continue 
            DMA = ball.dm   
            child_balls = ball.split()  
            if len(child_balls) == 1:  # 未实际分裂 
                new_GB_sets.append(ball)  
                continue 
            ball1, ball2 = child_balls 
            n1, n2 = ball1.size, ball2.size  
            total_size = n1 + n2
            DMweight = (n1 / total_size) * ball1.dm + (n2 / total_size) * ball2.dm  
            if DMweight < DMA:
                new_GB_sets.extend([ball1, ball2])
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
        radii = [ball.radius for ball in GB_sets if ball.size > 0 and ball.radius > 0]
        if len(radii) == 0:
            break 
        mean_r = np.mean(radii)   
        median_r = np.median(radii)   
        threshold = 2 * max(mean_r, median_r)
        for ball in GB_sets:
            if ball.size == 0:  # 空粒球直接保留 
                new_GB_sets.append(ball)   
                continue 
            # 同时满足分裂条件和最小半径限制 
            if ball.radius >= threshold and ball.radius > MIN_RADIUS:
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
#绘制原始数据分布图 
def draw_point(data):
    N = data.shape[0] 
    plt.figure()  
    plt.axis('equal')  
    for i in range(N):
        plt.scatter(data[i][0], data[i][1], s=16., c='k')
        plt.xlabel('x') 
        plt.ylabel('y') 
        plt.title('Original Data Distribution')
    plt.show() 
# 判断粒球的标签和纯度
def get_num(gb):
    # 矩阵的行数 
    return gb.shape[0] 
# 返回粒球中心和半径 (兼容新粒球对象)
def calculate_center_and_radius(gb):
    if isinstance(gb, GranularBall):
        return gb.center, gb.radius  
    else:
        data_no_label = gb[:, :]  # 取坐标
        center = data_no_label.mean(axis=0)   # 压缩行，对列取均值 
        radius = np.max((((data_no_label - center) ** 2).sum(axis=1) ** 0.5))
        return center, radius
def gb_plot(gb_list, plt_type=0):
    plt.figure()  
    plt.axis('equal')  
    fill_color = (0.6, 0.8, 1.0)  
    edge_color = (0.4, 0.6, 1.0) 
    for gb in gb_list:
        # 处理不同类型的粒球表示
        if isinstance(gb, GranularBall):
            center, radius = gb.center, gb.radius  
            points = gb.points  
        else:
            center, radius = calculate_center_and_radius(gb)
            points = gb
        if plt_type == 0:   
            plt.scatter(points[:, 0], points[:, 1],
                        color=fill_color, s=15, alpha=1, zorder=2,
                        edgecolors='w', linewidths=0.3)
        if plt_type == 0 or plt_type == 1:  
            circle = Circle(center, radius, 
                            edgecolor=edge_color, 
                            facecolor='none', 
                            linewidth=1.2, 
                            zorder=3)
            plt.gca().add_patch(circle)  
        # 绘制粒球中心 
        plt.scatter(center[0], center[1],
                    marker='o', c='red', s=15, linewidths=0.5, zorder=4)
    plt.xlabel('x')  
    plt.ylabel('y')  
    plt.show() 
#计算粒球的密度
def ball_density2(radiusAD, ball_qualitysA, mean_rs):
    N = radiusAD.shape[0] 
    ball_dens2 = np.zeros(shape=N) 
    for i in range(N):
        if radiusAD[i] == 0:
            ball_dens2[i] = 0
        else:
            # 粒球密度 = 点数 / (半径² * 平均距离)
            ball_dens2[i] = ball_qualitysA[i] / (radiusAD[i] * radiusAD[i] * mean_rs[i])
    return ball_dens2
#计算粒球的相对距离 
def ball_distance(centersAD):
    Y1 = pdist(centersAD)
    ball_distAD = squareform(Y1)
    return ball_distAD
#计算最小密度峰距离以及该点
def ball_min_dist(ball_distS, ball_densS):
    N3 = ball_distS.shape[0] 
    ball_min_distAD = np.zeros(shape=N3) 
    ball_nearestAD = np.zeros(shape=N3) 
    #按密度从大到小排号 
    index_ball_dens = np.argsort(-ball_densS) 
    for i3, index in enumerate(index_ball_dens):
        if i3 == 0:
            continue 
        index_ball_higher_dens = index_ball_dens[:i3]
        ball_min_distAD[index] = np.min([ball_distS[index, j] for j in index_ball_higher_dens])
        ball_index_near = np.argmin([ball_distS[index, j] for j in index_ball_higher_dens])
        ball_nearestAD[index] = int(index_ball_higher_dens[ball_index_near])
    ball_min_distAD[index_ball_dens[0]] = np.max(ball_min_distAD) 
    if np.max(ball_min_distAD) < 1:
        ball_min_distAD = ball_min_distAD * 10 
    return ball_min_distAD, ball_nearestAD 
#画决策图  
def ball_draw_decision(ball_densS, ball_min_distS):
    # 此处为人工交互，不计时
    fig, ax = plt.subplots()  
    # 绘制决策图 
    scatter = ax.scatter(ball_densS, ball_min_distS, 
                         c='k', s=30, alpha=0.6)
    ax.set_xlabel('Density', fontsize=12)
    ax.set_ylabel('Minimum Distance', fontsize=12)
    ax.set_title('Density-Peak Decision Graph (Select Centers)', 
                 fontsize=14, pad=15)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('auto') 
    #添加颜色条解释密度 
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Density Level', rotation=270, labelpad=15)
    lst = []  # 存储用户选择结果 
    # 定义回调函数 
    def onselect(eclick, erelease):
        # 获取选择范围 
        xmin, xmax = sorted([eclick.xdata, erelease.xdata])  
        ymin, ymax = sorted([eclick.ydata, erelease.ydata])  
        lst.append(((xmin, xmax), (ymin, ymax)))
        plt.close()    # 关闭窗口继续执行 
    # 创建矩形选择器  
    rs = RectangleSelector(ax, onselect,
                           useblit=True,
                           button=[1],  
                           minspanx=5, minspany=5,  
                           spancoords='pixels',
                           interactive=True)
    print("请在决策图中框选聚类中心点...")
    plt.show(block=True)    # 阻塞直到窗口关闭
    return lst, 0  # 返回0时间，因为不计入算法耗时
# 找粒球中心点
def ball_find_centers(ball_densS, ball_min_distS, lst):
    # 提取用户选择的阈值范围 
    (x_range, y_range) = lst[0]
    xmin, xmax = x_range 
    ymin, ymax = y_range 
    centers = []
    N4 = ball_densS.shape[0]  
    for i4 in range(N4):
        # 同时满足x轴和y轴范围的条件 
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
    plt.axis('equal')  
    N6 = centersA.shape[0] 
    for i6 in range(N6):
        # 处理不同类型的粒球表示
        if isinstance(gb_list[i6], GranularBall):
            points = gb_list[i6].points
        else:
            points = gb_list[i6]
        color_idx = int(ball_labs[i6])
        if color_idx in dic_colors:
            c = dic_colors[color_idx]
        else:
            c = 'k'
        for point in points:
            plt.plot(point[0], point[1], marker='o', markersize=4.0, color=c)
    plt.xlabel('x')  
    plt.ylabel('y')  
    plt.show() 
    
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
                  29: (0, .1, .8)}
    np.set_printoptions(threshold=1e16) 
    # 请修改为您本地的路径
    data_mat = np.loadtxt('C:\\Users\\庄\\Desktop\\Synthetic_datasets\\2d-4c-2_1.txt') 
    # 绘制原始数据 (不计入时间)
    print("绘制原始数据分布...")
    # draw_point(data_mat)
    
    print("开始粒球生成...")
    t1_start = time.time()
    GB_sets = algorithm1_generation_of_granular_balls(data_mat)
    # 过滤空粒球
    GB_sets = [ball for ball in GB_sets if ball.size > 0]
    t1_end = time.time()
    time_gb_gen = t1_end - t1_start
    print(f"生成了 {len(GB_sets)} 个粒球")
    # 可视化生成的粒球 (不计入时间)
    # gb_plot(GB_sets)
    
   
    print("开始属性计算...")
    t2_start = time.time()
    # 准备后续计算所需的数据 
    centers = [ball.center for ball in GB_sets]
    radiuss = [ball.radius for ball in GB_sets]
    ball_qualitys = [ball.size for ball in GB_sets]  # 粒球质量 = 点数 
    mean_rs = [ball.dm for ball in GB_sets]  # 平均距离 
    # 转换为数组 
    centersA = np.array(centers) 
    radiusA = np.array(radiuss) 
    ball_qualitysA = np.array(ball_qualitys) 
    mean_rs = np.array(mean_rs) 
    # 计算粒球密度
    ball_densS = ball_density2(radiusA, ball_qualitysA, mean_rs)
    # 计算粒球间距离
    ball_distS = ball_distance(centersA)
    # 计算最小密度峰距离
    ball_min_distS, ball_nearest = ball_min_dist(ball_distS, ball_densS)
    t2_end = time.time()
    time_attr_calc = t2_end - t2_start
    
    # 绘制决策图并选择中心点 (暂停计时)
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
    
    # !!! 修改点：在此处停止计时，排除后续任何映射或初始化操作 !!!
    t3_end = time.time()
    time_core_cluster = t3_end - t3_start
    
    # ================= 结果统计 =================
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
    
    # 绘制聚类结果 (不计入时间)
    print('绘制聚类结果...')
    ball_draw_cluster(centersA, radiusA, ball_labs, dic_colors, GB_sets, ball_centers)
    print('完成!')
