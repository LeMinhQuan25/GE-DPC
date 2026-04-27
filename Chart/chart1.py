import numpy as np
import matplotlib.pyplot as plt

# =========================
# DATA
# =========================
datasets = ["Iris","Seeds","Segment","Landsat","Msplice","Rice","Banknote","HTRU2"]

# ACC
acc = {
    "GB-DPC (Cheng)": [0.840,0.829,0.753,0.491,0.740,0.856,0.696,0.882],
    "GB-DPC + Xie":   [0.747,0.662,0.680,0.465,0.635,0.683,0.542,0.900],
    "GB-DPC + Jia":   [0.927,0.490,0.682,0.638,0.712,0.843,0.687,0.966],
    "GE-DPC (paper)": [0.873,0.857,0.688,0.625,0.692,0.854,0.605,0.973],
    "GE-DPC new":     [0.873,0.857,0.609,0.614,0.614,0.854,0.743,0.933],
}

# NMI
nmi = {
    "GB-DPC (Cheng)": [0.722,0.588,0.671,0.424,0.350,0.442,0.163,0.002],
    "GB-DPC + Xie":   [0.681,0.526,0.684,0.361,0.218,0.196,0.001,0.003],
    "GB-DPC + Jia":   [0.796,0.296,0.680,0.577,0.331,0.398,0.273,0.592],
    "GE-DPC (paper)": [0.771,0.681,0.651,0.541,0.290,0.450,0.325,0.649],
    "GE-DPC new":     [0.771,0.681,0.591,0.541,0.275,0.450,0.262,0.392],
}

# =========================
# STYLE
# =========================
colors = {
    "GB-DPC (Cheng)": "#4C78A8",
    "GB-DPC + Xie": "#72B7B2",
    "GB-DPC + Jia": "#54A24B",
    "GE-DPC (paper)": "#FFD600",
    "GE-DPC new": "#FF1744",
}

def plot_chart(data, title, ylabel):
    methods = list(data.keys())
    x = np.arange(len(datasets))
    width = 0.15

    fig, ax = plt.subplots(figsize=(16,7))

    # Dark background
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for i, method in enumerate(methods):
        offset = (i - 2) * width
        bars = ax.bar(
            x + offset,
            data[method],
            width,
            label=method,
            color=colors[method],
            edgecolor="#111",
            linewidth=0.7
        )

        # value label
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2,
                h + 0.01,
                f"{h:.3f}",
                ha='center',
                va='bottom',
                color='black',
                fontsize=8,
                rotation=90
            )

    ax.set_title(title, fontsize=18, color='black', pad=15)
    ax.set_ylabel(ylabel, fontsize=14, color='black')
    ax.set_ylim(0,1.1)

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=25, ha='right', color='black')

    ax.tick_params(axis='y', colors='black')

    ax.grid(axis='y', alpha=0.3)

    for spine in ["top","right"]:
        ax.spines[spine].set_visible(False)

    ax.legend(loc='upper center', bbox_to_anchor=(0.5,-0.15), ncol=3, frameon=False)

    plt.tight_layout()
    plt.show()

# =========================
# DRAW
# =========================
plot_chart(acc, "ACC Comparison among Five Methods", "ACC")
plot_chart(nmi, "NMI Comparison among Five Methods", "NMI")