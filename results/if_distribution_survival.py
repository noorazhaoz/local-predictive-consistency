import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import os

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
# Dataset / method settings
# ----------------------------
DATASET = "rossi"
METHOD = "lap"          # for naming only

# ----------------------------
# Lambdas to visualize (baseline + moderate + strong)
# NOTE: pick the 3 lambdas you want to show
# ----------------------------
LAM_LIST = [0.0, 1e-3, 1.0]

# ----------------------------
# Load IF scores (SURVIVAL)
# ----------------------------

def fmt_lambda(lam: float) -> str:
    # exact zero
    if lam == 0:
        return "0.0"

    # exact one (or very close)
    if np.isclose(lam, 1.0):
        return "1.0"

    # other values → scientific, strip leading zeros in exponent
    s = f"{lam:.0e}"          # e.g. "1e-04"
    s = s.replace("e-0", "e-").replace("e+0", "e+")
    return s



def load_if_scores(lam: float):
    lam_str = fmt_lambda(lam)

    path = f"urw/{DATASET}/if_deepsurv_depth1_lambda{lam_str}.csv"

    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing per-instance IF file: {path}")

    df = pd.read_csv(path)

    col = "if_score_urw_eta"
    if col not in df.columns:
        raise KeyError(f"Missing column '{col}' in {path}. Available: {list(df.columns)}")

    return df[col].values


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
# Color palette (optional)
# ----------------------------
PALETTE = {
    str(LAM_LIST[0]): "#1f77b4",   # baseline
    str(LAM_LIST[1]): "#2ca02c",
    str(LAM_LIST[2]): "#d62728",
}

# ----------------------------
# Plot
# ----------------------------
fig, ax = plt.subplots(figsize=(7.2, 4), constrained_layout=True)

sns.violinplot(
    data=df_plot,
    x="lambda",
    y="log_if",
    order=[str(l) for l in LAM_LIST],
    palette=PALETTE,
    inner="box",
    bw_adjust=0.6,
    cut=3,
    linewidth=1.0,
    ax=ax
)
ax.margins(y=0.05)
ax.spines["bottom"].set_position(("outward", 6))
ax.tick_params(axis="x", pad=6)

ax.set_xlabel(r"$\lambda$")
ax.set_ylabel(r"$\log(1+\mathrm{IF})$")
ax.set_title(f"Survival ({DATASET}, {METHOD.upper()}): Distribution of individual fairness scores")

sns.despine(ax=ax, top=True, right=True)

# ----------------------------
# Save
# ----------------------------
out_path = f"lap/{DATASET}/survival_{METHOD}_if.pdf"
os.makedirs(os.path.dirname(out_path), exist_ok=True)
fig.savefig(out_path)
plt.show()
