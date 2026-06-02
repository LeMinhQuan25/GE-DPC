import numpy as np
import matplotlib.pyplot as plt

# =========================
# DATA - Total time, unit: ms
# =========================
datasets = [
    "Iris", "Seeds", "Segment", "Landsat", "Msplice", "Rice",
    "Banknote", "Htru2", "Breast Cancer", "HCV", "Dry Bean", "Rice C&O"
]

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

    "Ours": np.array([
        7.0, 10.7, 65.2, 64.6, 82.3, 112.3,
        44.2, 443.8, 35.2, 34.7, 394.4, 94.5
    ]),
}

colors = {
    "DPC": "#6C757D",             # gray
    "GB-DPC (Cheng)": "#2E86AB", # blue
    "GE-DPC (paper)": "#F4C430", # yellow
    "Ours": "#E63946",           # red
}

x = np.arange(len(datasets))
width = 0.18

# =========================
# BROKEN Y-AXIS PLOT
# =========================
fig, (ax_top, ax_bottom) = plt.subplots(
    2, 1,
    sharex=True,
    figsize=(17, 8),
    gridspec_kw={"height_ratios": [1.1, 2.6], "hspace": 0.05}
)

for i, (method, values) in enumerate(methods.items()):
    offset = (i - 1.5) * width

    ax_top.bar(
        x + offset, values,
        width=width,
        label=method,
        color=colors[method],
        edgecolor="black",
        linewidth=0.4
    )

    ax_bottom.bar(
        x + offset, values,
        width=width,
        color=colors[method],
        edgecolor="black",
        linewidth=0.4
    )

# =========================
# Y-axis ranges
# =========================
ax_bottom.set_ylim(0, 1200)
ax_top.set_ylim(3500, 39000)

# =========================
# Add value labels
# =========================
for i, (method, values) in enumerate(methods.items()):
    offset = (i - 1.5) * width

    for j, v in enumerate(values):
        xpos = x[j] + offset

        if v <= 1200:
            ax = ax_bottom
            ypos = v + 15
        else:
            ax = ax_top
            ypos = v + 300

        ax.text(
            xpos, ypos,
            f"{v:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
            rotation=90
        )

# =========================
# Style
# =========================
ax_top.set_title("Runtime Comparison", fontsize=20, fontweight="bold", pad=12)
ax_bottom.set_ylabel("Runtime (ms)", fontsize=14)

for ax in [ax_top, ax_bottom]:
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    ax.spines["right"].set_visible(False)

ax_top.spines["bottom"].set_visible(False)
ax_bottom.spines["top"].set_visible(False)

# Broken-axis diagonal marks
d = 0.008
kwargs = dict(transform=ax_top.transAxes, color="black", clip_on=False)
ax_top.plot((-d, +d), (-d, +d), **kwargs)
ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)

kwargs.update(transform=ax_bottom.transAxes)
ax_bottom.plot((-d, +d), (1 - d, 1 + d), **kwargs)
ax_bottom.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

# X-axis
ax_bottom.set_xticks(x)
ax_bottom.set_xticklabels(datasets, rotation=25, ha="right", fontsize=11)

# Legend giống chart ACC, đặt dưới
handles, labels = ax_top.get_legend_handles_labels()
fig.legend(
    handles, labels,
    loc="lower center",
    ncol=4,
    frameon=False,
    fontsize=12,
    bbox_to_anchor=(0.5, -0.02)
)

plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.show()