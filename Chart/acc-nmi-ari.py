import numpy as np
import matplotlib.pyplot as plt

# =========================
# DATA - ALL DATASETS
# =========================
datasets = [
    "Iris", "Seeds", "Segment", "Landsat", "Msplice", "Rice",
    "Banknote", "HTRU2", "Breast Cancer", "HCV", "Dry Bean", "Rice C/O"
]

acc = {
    "DPC":        [0.553, 0.790, 0.619, 0.527, 0.699, 0.918, 0.552, 0.744, 0.515, 0.552, 0.597, 0.918],
    "GB-DPC":     [0.840, 0.829, 0.753, 0.491, 0.740, 0.856, 0.696, 0.882, 0.821, 0.409, 0.380, 0.622],
    "GE-DPC":     [0.873, 0.857, 0.688, 0.625, 0.692, 0.854, 0.605, 0.973, 0.641, 0.895, 0.602, 0.908],
    "AQG-GE-DPC": [0.873, 0.857, 0.671, 0.695, 0.741, 0.916, 0.884, 0.933, 0.680, 0.898, 0.674, 0.916],
}

nmi = {
    "DPC":        [0.472, 0.555, 0.589, 0.392, 0.339, 0.588, 0.006, 0.042, 0.065, 0.009, 0.668, 0.588],
    "GB-DPC":     [0.722, 0.588, 0.671, 0.424, 0.350, 0.442, 0.163, 0.002, 0.411, 0.067, 0.378, 0.189],
    "GE-DPC":     [0.771, 0.681, 0.651, 0.541, 0.290, 0.450, 0.325, 0.649, 0.038, 0.022, 0.610, 0.552],
    "AQG-GE-DPC": [0.771, 0.681, 0.655, 0.579, 0.343, 0.582, 0.527, 0.392, 0.126, 0.065, 0.674, 0.582],
}

ari = {
    "DPC":        [0.314, 0.519, 0.412, 0.234, 0.303, 0.699, -0.001, -0.094, -0.027, -0.019, 0.479, 0.699],
    "GB-DPC":     [0.642, 0.554, 0.602, 0.296, 0.339, 0.504, 0.152, 0.026, 0.410, -0.049, 0.172, 0.052],
    "GE-DPC":     [0.696, 0.649, 0.553, 0.419, 0.297, 0.501, 0.213, 0.800, 0.020, 0.026, 0.494, 0.665],
    "AQG-GE-DPC": [0.696, 0.649, 0.510, 0.484, 0.343, 0.693, 0.590, 0.592, 0.082, 0.076, 0.542, 0.693],
}

# =========================
# STYLE
# =========================
colors = {
    "DPC": "#7F7F7F",
    "GB-DPC": "#4C78A8",
    "GE-DPC": "#FFD600",
    "AQG-GE-DPC": "#FF1744",
}


def plot_chart(data, title, ylabel, ymin=0.0, ymax=1.10, label_from_zero=False):
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
            x_pos = bar.get_x() + bar.get_width() / 2

            # ACC/NMI: label nằm trên đỉnh cột bình thường.
            # ARI: nếu giá trị âm, label vẫn nằm phía trên trục X=0.
            if label_from_zero:
                y_pos = max(h, 0) + 0.012
            else:
                y_pos = h + 0.012

            ax.text(
                x_pos,
                y_pos,
                f"{h:.3f}",
                ha="center",
                va="bottom",
                color="black",
                fontsize=14,
                rotation=90,
                clip_on=False
            )

    ax.set_title(title, fontsize=30, color="black", pad=15, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=20, color="black")
    ax.set_ylim(ymin, ymax)

    # Vẽ đường trục 0, đặc biệt cần cho ARI có giá trị âm
    ax.axhline(0, color="black", linewidth=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=28, ha="right", color="black", fontsize=16)
    ax.tick_params(axis="y", colors="black", labelsize=16)

    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=4,
        frameon=False,
        fontsize=18
    )

    plt.tight_layout()
    plt.show()


# =========================
# DRAW CHARTS
# =========================
plot_chart(acc, "ACC", "ACC", ymin=0.0, ymax=1.10)
plot_chart(nmi, "NMI", "NMI", ymin=0.0, ymax=1.10)

# ARI có giá trị âm, nên label_from_zero=True để số âm vẫn nằm phía trên trục X=0
plot_chart(ari, "ARI", "ARI", ymin=-0.20, ymax=1.05, label_from_zero=True)