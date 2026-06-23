import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

# ============================================================
# Dataset order
# ============================================================
datasets = [
    "Iris",
    "Seeds",
    "Segment",
    "Landsat",
    "Msplice",
    "Rice",
    "Banknote",
    "Htru2",
    "Breast Cancer Wisconsin",
    "HCV",
    "Dry Bean",
    "Rice (Cammeo and Osmancik)",
]

# ============================================================
# Latest experimental results
# ============================================================
results = {
    "DPC": {
        "ACC": [0.553, 0.790, 0.619, 0.527, 0.699, 0.918,
                0.552, 0.744, 0.515, 0.552, 0.597, 0.918],
        "NMI": [0.472, 0.555, 0.589, 0.392, 0.339, 0.588,
                0.006, 0.042, 0.065, 0.009, 0.668, 0.588],
        "ARI": [0.314, 0.519, 0.412, 0.234, 0.303, 0.699,
                -0.001, -0.094, -0.027, -0.019, 0.479, 0.699],
    },

    "GB-DPC": {
        "ACC": [0.840, 0.829, 0.753, 0.491, 0.740, 0.856,
                0.696, 0.882, 0.821, 0.409, 0.380, 0.622],
        "NMI": [0.722, 0.588, 0.671, 0.424, 0.350, 0.442,
                0.163, 0.002, 0.411, 0.067, 0.378, 0.189],
        "ARI": [0.642, 0.554, 0.602, 0.296, 0.339, 0.504,
                0.152, 0.026, 0.410, -0.049, 0.172, 0.052],
    },

    "GE-DPC": {
        "ACC": [0.873, 0.857, 0.688, 0.625, 0.692, 0.854,
                0.605, 0.973, 0.641, 0.895, 0.602, 0.908],
        "NMI": [0.771, 0.681, 0.651, 0.541, 0.290, 0.450,
                0.325, 0.649, 0.038, 0.022, 0.610, 0.552],
        "ARI": [0.696, 0.649, 0.553, 0.419, 0.297, 0.501,
                0.213, 0.800, 0.020, 0.026, 0.494, 0.665],
    },

    "AQG-GE-DPC": {
        "ACC": [0.873, 0.857, 0.671, 0.695, 0.741, 0.916,
                0.884, 0.933, 0.680, 0.898, 0.674, 0.916],
        "NMI": [0.771, 0.681, 0.655, 0.579, 0.343, 0.582,
                0.527, 0.392, 0.126, 0.065, 0.674, 0.582],
        "ARI": [0.696, 0.649, 0.510, 0.484, 0.343, 0.693,
                0.590, 0.592, 0.082, 0.076, 0.542, 0.693],
    },
}

# Convert lists to NumPy arrays
for method in results:
    for metric in results[method]:
        results[method][metric] = np.asarray(
            results[method][metric], dtype=float
        )

# ============================================================
# Holm correction for multiple comparisons
# ============================================================
def holm_adjust(p_values):
    """
    Holm-Bonferroni adjusted p-values.
    """
    p_values = np.asarray(p_values, dtype=float)
    m = len(p_values)

    order = np.argsort(p_values)
    sorted_p = p_values[order]

    adjusted_sorted = np.empty(m, dtype=float)
    running_max = 0.0

    for i, p_value in enumerate(sorted_p):
        adjusted = (m - i) * p_value
        running_max = max(running_max, adjusted)
        adjusted_sorted[i] = min(running_max, 1.0)

    adjusted = np.empty(m, dtype=float)
    adjusted[order] = adjusted_sorted

    return adjusted


# ============================================================
# Paired Wilcoxon signed-rank tests
# ============================================================
def run_all_wilcoxon_tests(
    proposed_method="AQG-GE-DPC",
    baselines=("GE-DPC", "GB-DPC", "DPC"),
    metrics=("ACC", "NMI", "ARI"),
    alpha=0.05,
):
    rows = []

    for baseline in baselines:
        for metric in metrics:
            proposed_values = results[proposed_method][metric]
            baseline_values = results[baseline][metric]

            if len(proposed_values) != len(baseline_values):
                raise ValueError(
                    f"Length mismatch for {baseline} and {metric}."
                )

            differences = proposed_values - baseline_values

            wins = int(np.sum(differences > 0))
            ties = int(np.sum(np.isclose(differences, 0.0)))
            losses = int(np.sum(differences < 0))

            # Two-sided test:
            # H0: median paired difference equals zero
            # H1: median paired difference differs from zero
            statistic, p_value = wilcoxon(
                proposed_values,
                baseline_values,
                alternative="two-sided",
                zero_method="wilcox",
                method="auto",
            )

            rows.append({
                "Comparison": f"{proposed_method} vs {baseline}",
                "Metric": metric,
                "Statistic": statistic,
                "p-value": p_value,
                "Wins": wins,
                "Ties": ties,
                "Losses": losses,
                "Significant": "Yes" if p_value < alpha else "No",
            })

    output = pd.DataFrame(rows)

    # Adjustment across all 9 tests
    output["Holm-adjusted p-value"] = holm_adjust(
        output["p-value"].to_numpy()
    )
    output["Significant after Holm"] = np.where(
        output["Holm-adjusted p-value"] < alpha,
        "Yes",
        "No",
    )

    return output


# ============================================================
# Run and display
# ============================================================
test_results = run_all_wilcoxon_tests()

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 160)

print("\nTwo-sided paired Wilcoxon signed-rank tests")
print(test_results.to_string(
    index=False,
    formatters={
        "Statistic": "{:.4f}".format,
        "p-value": "{:.6f}".format,
        "Holm-adjusted p-value": "{:.6f}".format,
    },
))

# Save for paper/report
test_results.to_csv(
    "wilcoxon_results.csv",
    index=False,
    float_format="%.6f",
)

# Optional compact LaTeX table
latex_table = test_results[
    [
        "Comparison",
        "Metric",
        "p-value",
        "Wins",
        "Ties",
        "Losses",
        "Significant",
    ]
].to_latex(
    index=False,
    float_format="%.4f",
    escape=False,
)

print("\nLaTeX table:")
print(latex_table)