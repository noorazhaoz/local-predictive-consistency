import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------
# Style
# ----------------------------
def set_ml_paper_style():
    mpl.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 1.0,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "legend.frameon": False,
    })

set_ml_paper_style()
sns.set_style("white")

# ----------------------------
# Lambdas to visualize
# ----------------------------
LAM_LIST = [0.0, 10, 100]   # baseline + moderate + strong

# ----------------------------
# Load IF scores (REGRESSION)
# ----------------------------
def load_if_scores(lam):
    """
    Expect CSV saved by --save_if_csv
    Column: if_score_urw_log
    """
    path = f"results/laplacian/lap_depth1_lambda{lam}.csv"
    df = pd.read_csv(path)
    return df["if_score_urw_log"].values

# ----------------------------
# Build dataframe for plotting
# ----------------------------
df_plot = pd.concat(
    [
        pd.DataFrame({
            "lambda": str(lam),
            "log_if": np.log1p(load_if_scores(lam))
        })
        for lam in LAM_LIST
    ],
    ignore_index=True
)

# ----------------------------
# Color palette
# ----------------------------
PALETTE = {
    "0.0": "#1f77b4",   # baseline
    "10":  "#2ca02c",
    "100": "#d62728",
}

# ----------------------------
# Plot
# ----------------------------
fig, ax = plt.subplots(figsize=(7.2, 3.8), constrained_layout=True)

sns.violinplot(
    data=df_plot,
    x="lambda",
    y="log_if",
    order=[str(l) for l in LAM_LIST],
    palette=PALETTE,
    inner="box",
    bw_adjust=0.6,
    cut=0,
    linewidth=1.0,
    ax=ax
)

# y-axis: keep tail under control
ax.set_ylim(-0.02, 0.4)

ax.spines["bottom"].set_position(("outward", 6))
ax.tick_params(axis="x", pad=6)

ax.set_xlabel(r"$\lambda$")
ax.set_ylabel(r"$\log(1+\mathrm{IF})$")
ax.set_title("Standard Graph Laplacian (Regression): Distribution of individual fairness scores")

sns.despine(ax=ax, top=True, right=True)

# ----------------------------
# Save
# ----------------------------
fig.savefig("results/laplacian/plots/lap_regression_if_distribution.pdf")
plt.show()
