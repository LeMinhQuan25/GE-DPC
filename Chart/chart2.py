import numpy as np
import matplotlib.pyplot as plt

datasets = ["Iris", "Seeds", "Segment", "Landsat", "Msplice", "Rice", "Banknote", "Htru2"]
x = np.arange(len(datasets))

# Total time, unit: ms
methods = {
    "GB-DPC (Cheng)": np.array([4.0988, 3.8218, 14.0737, 13.9124, 16.5137, 22.3046, 10.3417, 36.7782]) * 1000,
    "GB-DPC + Xie":   np.array([0.0030, 0.0044, 0.0480, 0.0411, 0.0695, 0.0829, 0.0309, 0.5941]) * 1000,
    "GB-DPC + Jia":   np.array([0.0029, 0.0045, 0.0441, 0.0377, 0.0680, 0.1602, 0.0250, 5.8815]) * 1000,
    "GE-DPC (paper)": np.array([10.0, 12.0, 84.0, 94.0, 173.0, 222.0, 67.0, 1003.0]),
    "GE-DPC new":     np.array([7.0, 10.7, 65.2, 64.6, 82.3, 112.3, 44.2, 443.8]),
}

colors = {
    "GB-DPC (Cheng)": "#4C78A8",
    "GB-DPC + Xie": "#72B7B2",
    "GB-DPC + Jia": "#54A24B",
    "GE-DPC (paper)": "#FFD600",
    "GE-DPC new": "#FF1744",
}

plt.figure(figsize=(14, 7))

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

plt.title("Total Time",
          fontsize=20, fontweight="bold")

plt.xlabel("Dataset", fontsize=15)
plt.ylabel("Total time (ms)", fontsize=15)

plt.xticks(x, datasets, rotation=30, ha="right", fontsize=12)
plt.yticks(fontsize=12)

plt.grid(True, linestyle="--", alpha=0.35, which="both")
plt.legend(loc="upper left", fontsize=11, frameon=True)

plt.tight_layout()

# plt.savefig("total_time_5_methods_logscale.png", dpi=300, bbox_inches="tight")
# plt.savefig("total_time_5_methods_logscale.pdf", bbox_inches="tight")

plt.show()