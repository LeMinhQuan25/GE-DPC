import numpy as np
import matplotlib.pyplot as plt

# =========================
# DATA - Total time, unit: ms
# =========================
datasets = [
    "Iris", "Seeds", "Segment", "Landsat", "Msplice", "Rice",
    "Banknote", "Htru2", "Breast Cancer", "HCV", "Dry Bean", "Rice C&O"
]

x = np.arange(len(datasets))

methods = {
    "DPC": np.array([
        1.1, 1.4, 72.1, 46.1, 128.4, 189.1,
        27.9, 349.9, 5.6, 5.5, 259.6, 170.5
    ]),

    "GB-DPC (Cheng)": np.array([
        4098.8, 3821.8, 14073.7, 13912.4, 16513.7, 22304.6,
        10341.7, 36778.2, 71.6, 46.5, 406.2, 156.8
    ]),

    "GE-DPC (paper)": np.array([
        10.0, 12.0, 84.0, 94.0, 173.0, 222.0,
        67.0, 1003.0, 85.0, 57.0, 848.0, 174.0
    ]),

    "GE-DPC new": np.array([
        7.0, 10.7, 65.2, 64.6, 82.3, 112.3,
        44.2, 443.8, 35.2, 34.7, 394.4, 106.0
    ]),
}

colors = {
    "DPC": "#7F7F7F",
    "GB-DPC (Cheng)": "#4C78A8",
    "GE-DPC (paper)": "#FFD600",
    "GE-DPC new": "#FF1744",
}

# =========================
# PLOT
# =========================
plt.figure(figsize=(16, 7))

for method, values in methods.items():
    plt.plot(
        x,
        values,
        marker="o",
        linewidth=2.8,
        markersize=8,
        label=method,
        color=colors[method],
    )

plt.yscale("log")

plt.title("Total Time", fontsize=20, fontweight="bold")
plt.xlabel("Dataset", fontsize=15)
plt.ylabel("Total time (ms)", fontsize=15)

plt.xticks(x, datasets, rotation=30, ha="right", fontsize=12)
plt.yticks(fontsize=12)

plt.grid(True, linestyle="--", alpha=0.35, which="both")
plt.legend(loc="upper left", fontsize=11, frameon=True)

plt.tight_layout()
plt.show()