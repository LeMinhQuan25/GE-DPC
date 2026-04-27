import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

datasets = ["iris","seed","segment","landsat"]
t_values = [1.0,1.2,1.4,1.6,1.8,2.0]

ACC = np.array([
    [0.85,0.86,0.87,0.88,0.88,0.87],
    [0.82,0.83,0.84,0.85,0.84,0.83],
    [0.60,0.62,0.64,0.65,0.66,0.65],
    [0.55,0.57,0.59,0.60,0.61,0.60],
])

X, Y = np.meshgrid(t_values, np.arange(len(datasets)))

fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(X, Y, ACC, cmap='jet', edgecolor='k')

# Heatmap projection (giống paper)
ax.contourf(X, Y, ACC, zdir='z', offset=0, cmap='jet', alpha=0.8)

ax.set_xlabel('t')
ax.set_ylabel('Datasets')
ax.set_zlabel('ACC')

ax.set_yticks(range(len(datasets)))
ax.set_yticklabels(datasets)

fig.colorbar(surf, shrink=0.5, aspect=10)

plt.title("(a) ACC")
plt.show()