import numpy as np
from scipy.stats import wilcoxon

# GE-DPC paper + additional datasets
datasets = [
    "Iris", "Seeds", "Segment", "Landsat", "Msplice", "Rice", "Banknote", "Htru2",
    "Breast Cancer Wisconsin", "HCV", "Dry Bean", "Rice (Cammeo and Osmancik)"
]

# GE-DPC paper results
ge_acc = np.array([0.873, 0.857, 0.688, 0.625, 0.692, 0.854, 0.605, 0.973,
                   0.641, 0.895, 0.602, 0.908])
ge_nmi = np.array([0.771, 0.681, 0.651, 0.541, 0.290, 0.450, 0.325, 0.649,
                   0.038, 0.022, 0.610, 0.552])
ge_ari = np.array([0.696, 0.649, 0.553, 0.419, 0.297, 0.501, 0.213, 0.800,
                   0.020, 0.026, 0.494, 0.665])
ge_time = np.array([10.0, 12.0, 84.0, 94.0, 173.0, 222.0, 67.0, 1003.0,
                    85.0, 57.0, 848.0, 174.0])

# AQG-GE-DPC results
aqg_acc = np.array([0.873, 0.857, 0.609, 0.614, 0.614, 0.854, 0.743, 0.933,
                    0.677, 0.727, 0.652, 0.908])
aqg_nmi = np.array([0.771, 0.681, 0.591, 0.541, 0.275, 0.450, 0.262, 0.392,
                    0.119, 0.040, 0.638, 0.552])
aqg_ari = np.array([0.696, 0.649, 0.441, 0.446, 0.270, 0.501, 0.236, 0.592,
                    0.076, -0.098, 0.494, 0.665])
aqg_time = np.array([7.0, 10.7, 65.2, 64.6, 82.3, 112.3, 44.2, 443.8,
                     35.2, 34.7, 394.4, 106.0])

def run_wilcoxon(metric_name, ge_values, aqg_values, alternative="two-sided"):
    stat, p_value = wilcoxon(
        ge_values,
        aqg_values,
        alternative=alternative,
        zero_method="wilcox"
    )
    print(f"{metric_name}: statistic={stat:.4f}, p-value={p_value:.6f}")

print("Two-sided Wilcoxon signed-rank test")
run_wilcoxon("ACC", ge_acc, aqg_acc)
run_wilcoxon("NMI", ge_nmi, aqg_nmi)
run_wilcoxon("ARI", ge_ari, aqg_ari)
run_wilcoxon("Total time", ge_time, aqg_time)

print("\nOne-sided test for runtime: GE-DPC > AQG-GE-DPC")
run_wilcoxon("Total time", ge_time, aqg_time, alternative="greater")