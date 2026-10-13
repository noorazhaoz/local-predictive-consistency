import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

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
sns.set_style("white")  # no grid

LAM_LIST = [0.0, 10,  100]

def load_if_scores(lam):
    path = f"urw_depth1_lambda{lam}.csv"   # ✅ 保留 0.0
    df = pd.read_csv(path)
    return df["if_score_forwardKL_prob"].values

# build dataframe
df_plot = pd.concat([
    pd.DataFrame({
        "lambda": str(lam),                 # ✅ legend/x-axis show 0.0
        "log_if": np.log1p(load_if_scores(lam))
    })
    for lam in LAM_LIST
], ignore_index=True)

# color palette (optional)
PALETTE = {"0.0":"#1f77b4", "10":"#2ca02c", "30":"#ff7f0e", "100":"#d62728"}

fig, ax = plt.subplots(figsize=(7.2, 3.8), constrained_layout=True)

sns.violinplot(
    data=df_plot,
    x="lambda",
    y="log_if",
    order=[str(l) for l in LAM_LIST],
    palette=PALETTE,
    inner="box",      # ✅ show median & IQR inside
    bw_adjust=0.6,
    cut=0,            # ✅ do not extend beyond data range
    linewidth=1.0,
    ax=ax
)



# keep the tail under control but preserve readability near 0
ax.set_ylim(-0.02, 1.4)

# make x-axis readable even when mass is near 0
ax.spines["bottom"].set_position(("outward", 6))
ax.tick_params(axis="x", pad=6)

ax.set_xlabel(r"$\lambda$")
ax.set_ylabel(r"$\log(1+\mathrm{IF\ score})$")
ax.set_title("URW: Distribution of individual fairness scores across $\\lambda$")

sns.despine(ax=ax, top=True, right=True)
fig.savefig("urw/urw_distribution.pdf")

plt.show()
