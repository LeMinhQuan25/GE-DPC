import numpy as np
import matplotlib.pyplot as plt

# =========================
# DATA - GỘP CHUNG TẤT CẢ DATASET
# =========================
datasets = [
    "Iris", "Seeds", "Segment", "Landsat", "Msplice", "Rice", "Banknote", "HTRU2",
    "Breast Cancer", "HCV", "Dry Bean", "Rice C&O"
]

acc = {
    "DPC":             [0.553, 0.790, 0.619, 0.527, 0.699, 0.918, 0.552, 0.744, 0.515, 0.552, 0.597, 0.918],
    "GB-DPC (Cheng)": [0.840, 0.829, 0.753, 0.491, 0.740, 0.856, 0.696, 0.882, 0.821, 0.409, 0.380, 0.622],
    "GE-DPC (paper)": [0.873, 0.857, 0.688, 0.625, 0.692, 0.854, 0.605, 0.973, 0.641, 0.895, 0.602, 0.908],
    "GE-DPC new":     [0.873, 0.857, 0.609, 0.614, 0.614, 0.854, 0.743, 0.933, 0.677, 0.727, 0.652, 0.908],
}

nmi = {
    "DPC":             [0.472, 0.555, 0.589, 0.392, 0.339, 0.588, 0.006, 0.042, 0.065, 0.009, 0.668, 0.588],
    "GB-DPC (Cheng)": [0.722, 0.588, 0.671, 0.424, 0.350, 0.442, 0.163, 0.002, 0.411, 0.067, 0.378, 0.189],
    "GE-DPC (paper)": [0.771, 0.681, 0.651, 0.541, 0.290, 0.450, 0.325, 0.649, 0.038, 0.022, 0.610, 0.552],
    "GE-DPC new":     [0.771, 0.681, 0.591, 0.541, 0.275, 0.450, 0.262, 0.392, 0.119, 0.040, 0.638, 0.552],
}

# =========================
# STYLE
# =========================
colors = {
    "DPC": "#7F7F7F",
    "GB-DPC (Cheng)": "#4C78A8",
    "GE-DPC (paper)": "#FFD600",
    "GE-DPC new": "#FF1744",
}

def plot_chart(data, title, ylabel):
    methods = list(data.keys())
    x = np.arange(len(datasets))
    width = 0.18

    fig, ax = plt.subplots(figsize=(18, 7))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for i, method in enumerate(methods):
        offset = (i - (len(methods) - 1) / 2) * width
        bars = ax.bar(
            x + offset,
            data[method],
            width,
            label=method,
            color=colors[method],
            edgecolor="black",
            linewidth=0.6
        )

        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.012,
                f"{h:.3f}",
                ha="center",
                va="bottom",
                color="black",
                fontsize=8,
                rotation=90
            )

    ax.set_title(title, fontsize=18, color="black", pad=15, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=14, color="black")
    ax.set_ylim(0, 1.10)

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=28, ha="right", color="black")
    ax.tick_params(axis="y", colors="black")

    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=4,
        frameon=False,
        fontsize=11
    )

    plt.tight_layout()
    plt.show()

# =========================
# DRAW - SHOW LẦN LƯỢT
# =========================
plot_chart(acc, "ACC", "ACC")
plot_chart(nmi, "NMI", "NMI")