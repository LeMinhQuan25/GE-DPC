import numpy as np
import matplotlib.pyplot as plt

# =========================
# DATA - Total time (seconds)
# =========================
datasets = [
    "Iris", "Seeds", "Segment", "Landsat", "Msplice", "Rice",
    "Banknote", "HTRU2", "Breast Cancer", "HCV", "Dry Bean", "Rice C&O"
]

methods = {
    "DPC": np.array([
        1.1, 1.4, 72.1, 46.1, 128.4, 189.1,
        27.9, 349.9, 5.6, 5.5, 259.6, 170.5
    ]) / 1000,

    "GB-DPC": np.array([
        4098.8, 3821.8, 14073.7, 13912.4, 16513.7, 22304.6,
        10341.7, 36778.2, 71.6, 46.5, 406.2, 156.8
    ]) / 1000,

    "GE-DPC": np.array([
        10.0, 12.0, 84.0, 94.0, 173.0, 222.0,
        67.0, 1003.0, 85.0, 57.0, 848.0, 174.0
    ]) / 1000,

    "AQD-GE-DPC": np.array([
        5.3, 7.4, 69.9, 57.0, 164.6, 193.0,
        46.0, 399.6, 44.6, 26.9, 783.1, 192.3
    ]) / 1000,
}

colors = {
    "DPC": "#7F7F7F",
    "GB-DPC": "#2E75B6",
    "GE-DPC": "#FFD700",
    "AQD-GE-DPC": "#FF0040"
}

# =========================
# PLOT SETTING
# =========================
x = np.arange(len(datasets))
width = 0.18
y_cap = 1.2   # giới hạn trục Y để nhìn rõ time nhỏ

fig, ax = plt.subplots(figsize=(16, 7))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

# =========================
# DRAW BARS
# =========================
for i, (method, values) in enumerate(methods.items()):
    offset = (i - (len(methods) - 1) / 2) * width

    # Cắt chiều cao cột ở mức y_cap, nhưng vẫn ghi số thật
    plot_values = np.minimum(values, y_cap)

    bars = ax.bar(
        x + offset,
        plot_values,
        width=width,
        label=method,
        color=colors[method],
        edgecolor="black",
        linewidth=0.35
    )

    # Ghi số dọc trên mỗi cột
    for j, v in enumerate(values):
        xpos = x[j] + offset

        if v > y_cap:
            ypos = y_cap + 0.03
        else:
            ypos = v + 0.015

        ax.text(
            xpos,
            ypos,
            f"{v:.3f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
            rotation=90,
            color="#333333",
            clip_on=False
        )

# =========================
# STYLE
# =========================
ax.set_title("Runtime", fontsize=20, fontweight="bold", pad=18)
ax.set_ylabel("Runtime (s)", fontsize=15)

ax.set_xticks(x)
ax.set_xticklabels(datasets, rotation=25, ha="right", fontsize=11)

ax.set_ylim(0, 1.35)
ax.set_yticks(np.arange(0, 1.31, 0.2))

ax.grid(axis="y", linestyle="--", alpha=0.35)
ax.set_axisbelow(True)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.legend(
    loc="lower center",
    bbox_to_anchor=(0.5, -0.22),
    ncol=4,
    frameon=False,
    fontsize=12
)

plt.tight_layout()
plt.show()