import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from matplotlib.widgets import RectangleSelector
from sklearn.cluster import k_means
from sklearn.metrics import accuracy_score, normalized_mutual_info_score 
from sklearn.preprocessing import LabelEncoder 
from sklearn.metrics import confusion_matrix 
from scipy.optimize import linear_sum_assignment 
from sklearn.metrics import adjusted_rand_score

#ACC计算
def cluster_acc(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    row_ind, col_ind = linear_sum_assignment(cm, maximize=True)
    return cm[row_ind, col_ind].sum() / y_true.size  
#绘制原始数据分布图
def draw_point(data):
    N = data.shape[0]
    plt.figure()
    plt.axis()
    for i in range(N):
        plt.scatter(data[i][0],data[i][1],s=16.,c='k')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('origin graph')
    plt.show()
# 判断粒球的标签和纯度
def get_num(gb):
    # 矩阵的行数
    num = gb.shape[0]
    return num
# 返回粒球中心和半径
def calculate_center_and_radius(gb):
    data_no_label = gb[:,:]#取坐标
    center = data_no_label.mean(axis=0)#压缩行，对列取均值  取出平均的 x,y
    radius = np.max((((data_no_label - center) ** 2).sum(axis=1) ** 0.5))  #（x1-x1）**2 + (y1-y2)**2   所有点到中心的距离平均
    return center, radius
def gb_plot(gb_list, plt_type=0):
    plt.figure()
    plt.axis()
    for gb in gb_list:
        center, radius = calculate_center_and_radius(gb)  # 返回中心和半径
        if plt_type == 0:  # 绘制所有点
            plt.plot(gb[:, 0], gb[:, 1], '.', c='k', markersize=5)
        if plt_type == 0 or plt_type == 1:  # 绘制粒球
            theta = np.arange(0, 2 * np.pi, 0.01)
            x = center[0] + radius * np.cos(theta)
            y = center[1] + radius * np.sin(theta)
            plt.plot(x, y, c='r', linewidth=0.8)
        plt.plot(center[0], center[1], 'x' if plt_type == 0 else '.', color='r')  # 绘制粒球中心
    plt.show()
def splits(gb_list, num, splitting_method):
    gb_list_new = []
    for gb in gb_list:
        p = get_num(gb)
        if p < num:
            gb_list_new.append(gb)#该粒球包含的点数小于等于num，那
        else:
            gb_list_new.extend(splits_ball(gb, splitting_method))#反之，进行划分，本来是[[1],[2],[3]]  变成[...,[1],[2],[3]]
    return gb_list_new
def splits_ball(gb, splitting_method):
    splits_k = 2
    ball_list = []
    # 数组去重
    len_no_label = np.unique(gb, axis=0)
    if splitting_method == '2-means':
        if len_no_label.shape[0] < splits_k:
            splits_k = len_no_label.shape[0]
        # n_init:用不同聚类中心初始化运行算法的次数
        #random_state，通过固定它的值，每次可以分割得到同样的训练集和测试集
        label = k_means(X=gb, n_clusters=splits_k, n_init=3, random_state=8)[1]  # 返回标签,标签代表被划分到哪个新粒球中去
    for single_label in range(0, splits_k):
        ball_list.append(gb[label == single_label, :])#按照新打的标签分类
    return ball_list
# 距离
def distances(data, p):
    return ((data - p) ** 2).sum(axis=1) ** 0.5
#计算所有点到粒球中心的平均距离：
def get_ball_quality(gb, center):
    N = gb.shape[0]
    ball_quality =  N
    mean_r = np.mean(((gb - center) **2)**0.5)
    return ball_quality, mean_r
#计算粒球的密度---计算密度的方法二：粒球的密度=粒球的质量/粒球的体积
#粒球的质量=所有点到中心点的平均距离  体积=粒球半径的维数次方radiusA, dimen, ball_qualitysA
def ball_density2(radiusAD, ball_qualitysA, mean_rs):
    N = radiusAD.shape[0]
    ball_dens2 = np.zeros(shape=N)
    for i in range(N):
        if radiusAD[i] == 0:
            ball_dens2[i] = 0
        else:
            ball_dens2[i] = ball_qualitysA[i] / (radiusAD[i] * radiusAD[i] * mean_rs[i])
    return ball_dens2
#计算粒球的相对距离
def ball_distance(centersAD):
    Y1 = pdist(centersAD)
    ball_distAD = squareform(Y1)
    return ball_distAD
#计算最小密度峰距离以及该点ball_min_dist3
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
        ball_min_distAD[index] = np.min([ball_distS[index, j]for j in index_ball_higher_dens])
        ball_index_near = np.argmin([ball_distS[index, j]for j in index_ball_higher_dens])
        ball_nearestAD[index] = int(index_ball_higher_dens[ball_index_near])
    ball_min_distAD[index_ball_dens[0]] = np.max(ball_min_distAD)
    if np.max(ball_min_distAD) < 1:
        ball_min_distAD = ball_min_distAD * 10
    return ball_min_distAD, ball_nearestAD
#画图  
def ball_draw_decision(ball_densS, ball_min_distS):
    Bval1_start = time.time()  
    fig, ax = plt.subplots(figsize=(8,  6))
    # 绘制决策图 
    scatter = ax.scatter(ball_densS,  ball_min_distS, 
                        c='k', s=30, alpha=0.6)
    ax.set_xlabel('Density',  fontsize=12)
    ax.set_ylabel('Minimum  Distance', fontsize=12)
    ax.set_title('Density-Peak  Decision Graph (Select Centers)', 
                fontsize=14, pad=15)
    ax.grid(True,  alpha=0.3)
    #添加颜色条解释密度 
    cbar = plt.colorbar(scatter,  ax=ax)
    cbar.set_label('Density  Level', rotation=270, labelpad=15)
    lst = []  # 存储用户选择结果 
    # 定义回调函数 
    def onselect(eclick, erelease):
        # 获取选择范围 
        xmin, xmax = sorted([eclick.xdata, erelease.xdata]) 
        ymin, ymax = sorted([eclick.ydata, erelease.ydata]) 
        lst.append(((xmin,  xmax), (ymin, ymax)))
        plt.close()   # 关闭窗口继续执行 
    # 创建矩形选择器  
    rs = RectangleSelector(ax, onselect,
                          useblit=True,
                          button=[1],  
                          minspanx=5, minspany=5,  
                          spancoords='pixels',
                          interactive=True)
    plt.show(block=True)   # 阻塞直到窗口关闭
    Bval1 = time.time()  - Bval1_start 
    return lst, Bval1 
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
def  ball_draw_cluster(centersA, radiusA, ball_labs, dic_colors, gb_list, ball_centers):
    plt.figure()
    N6 = centersA.shape[0]
    for i6 in range(N6):
        for j6, point in enumerate(gb_list[i6]):
            plt.plot(point[0], point[1], marker='o', markersize=4.0, color=dic_colors[ball_labs[i6]])
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
                  29: (0, .1, .8)
                  }
    np.set_printoptions(threshold=1e16)
    # 请修改为您本地的路径
    X = np.loadtxt('C:\\Users\\庄\\Desktop\\new real datasets\\banknote.txt')    
    y = np.loadtxt('C:\\Users\\庄\\Desktop\\new real datasets label\\banknote_label.txt' )    

    y = y.astype(int) 
    le = LabelEncoder()
    y_true = le.fit_transform(y) 
    print("数据形状:", X.shape) 
    print("标签形状:", y.shape) 
    
    time_gb_start = time.time()
    num = np.ceil(np.sqrt(X.shape[0])) 
    splitting_method = '2-means'
    # 初始化粒球 
    gb_list = [X]
    # gb_plot(gb_list, plt_type=0) 
 
    # 粒球分裂过程
    while True:
        current_count = len(gb_list)
        gb_list = splits(gb_list, num, splitting_method)
        if len(gb_list) == current_count:
            break 
    
    time_gb_end = time.time()
    time_gb_gen = time_gb_end - time_gb_start
   
    time_attr_start = time.time()
    
    # 计算粒球属性
    centersAD = []
    radiusAD = []
    ball_qualitysA = []
    mean_rs = []
    for gb in gb_list:
        center, radius = calculate_center_and_radius(gb)
        ball_quality, mean_r = get_ball_quality(gb, center)
        centersAD.append(center) 
        radiusAD.append(radius) 
        ball_qualitysA.append(ball_quality) 
        mean_rs.append(mean_r) 
    centersAD = np.array(centersAD) 
    radiusAD = np.array(radiusAD) 
    ball_qualitysA = np.array(ball_qualitysA) 
    mean_rs = np.array(mean_rs) 
 
    # 密度和距离计算
    ball_dens2 = ball_density2(radiusAD, ball_qualitysA, mean_rs)
    ball_distAD = ball_distance(centersAD)
    ball_min_distAD, ball_nearestAD = ball_min_dist(ball_distAD, ball_dens2)
    
    time_attr_end = time.time()
    time_attr_calc = time_attr_end - time_attr_start
    
    # 用户选择聚类中心 (人工交互时间，不计入算法耗时)
    lst, Bval1 = ball_draw_decision(ball_dens2, ball_min_distAD)

    time_core_start = time.time()
    
    # 1. 确定中心
    ball_centers = ball_find_centers(ball_dens2, ball_min_distAD, lst)
 
    # 2. 调用聚类函数 (分配粒球标签)
    labels = ball_cluster(ball_dens2, ball_centers, ball_nearestAD, ball_min_distAD)

    # !!! 修改点：在此处停止计时，排除耗时的映射过程 !!!
    time_core_end = time.time()
    time_core_cluster = time_core_end - time_core_start
     
    # 3. 将粒球标签映射回原始点 (这一步是为了算ACC，非常耗时，但属于验证步骤)
    print("正在进行数据映射（不计入Clustering Process Time）...")
    y_pred = np.zeros(len(X),  dtype=int)
    for i, gb in enumerate(gb_list):
        for point in gb:
            # 寻找点在原数据中的索引 (耗时操作)
            mask = np.all(np.isclose(X,  point, atol=1e-6), axis=1)
            idx = np.where(mask)[0] 
            if idx.size  > 0:
                y_pred[idx] = labels[i]
    
    # 评估 (不计入运行时间)
    print("正在计算评估指标...")
    acc = cluster_acc(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)    
    print(f"ACC: {acc:.3f}")
    print(f"NMI: {nmi:.3f}")
    print(f"ARI: {ari:.3f}")                    
    
    # 计算总时间
    time_total_cluster = time_attr_calc + time_core_cluster
    time_total_effective = time_gb_gen + time_total_cluster

    print("-" * 30)
    print(f"粒球生成时间: {time_gb_gen:.14f} 秒")
    print(f"属性计算耗时: {time_attr_calc:.14f} 秒")
    print(f"核心聚类耗时: {time_core_cluster:.14f} 秒 (已剔除映射时间)")
    print(f"聚类总时间 (属性+核心): {time_total_cluster:.14f} 秒")
    print(f"总有效运行时间 (1+4): {time_total_effective:.14f} 秒")
    print("-" * 30)

    # 后续绘图
    ball_draw_cluster(centersAD, radiusAD, labels, dic_colors, gb_list, ball_centers)

