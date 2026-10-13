import glob
import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter, LogFormatterMathtext

# ----------------------------
# 0) Colors (paper-friendly)
# ----------------------------
CONS_COLORS = {
    "gr": "#ff7f0e",   # orange
    "s":  "#17becf",   # cyan
}

# ----------------------------
# 1) Paper style (no grid)
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

        "lines.linewidth": 2.0,
        "lines.markersize": 5.0,

        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 1.0,

        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.minor.size": 2,
        "ytick.minor.size": 2,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.minor.width": 0.8,
        "ytick.minor.width": 0.8,

        "legend.frameon": False,
    })

set_ml_paper_style()

# ----------------------------
# 2) Axis helper: log lambda as 10^k
# ----------------------------
def setup_log_lambda_axis(ax):
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(LogLocator(base=10.0))
    ax.xaxis.set_major_formatter(LogFormatterMathtext())
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
    ax.xaxis.set_minor_formatter(NullFormatter())

# ----------------------------
# 3) Load metrics (one csv per lambda)
# ----------------------------
def load_metrics(pattern):
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matched: {pattern}")

    rows = []
    for f in files:
        df = pd.read_csv(f)
        s = df.mean(numeric_only=True)  # if multiple rows, average

        # 从文件名里解析 lambda
        # 例如: metrics_laplacian_compas_depth1_lambda0.01.csv
        lam_str = f.split("lambda")[-1].replace(".csv", "")
        lam = float(lam_str)

        s["lambda"] = lam
        rows.append(s)

    return pd.DataFrame(rows).sort_values("lambda").reset_index(drop=True)

# ----------------------------
# 4) Config
# ----------------------------
# TODO: adjust path if needed
pattern = "results_unified/metrics_urw_compas_aggforward_depth1_lambda*.csv"
save_path = "results_unified/urw_compas_forward_consistency.pdf"

COL_GR = "GR_Consistency"
COL_S  = "S_Consistency"

# ----------------------------
# 5) Read data
# ----------------------------
df = load_metrics(pattern)

if COL_GR not in df.columns:
    raise KeyError(f"Missing required column: {COL_GR}. Available columns: {list(df.columns)}")

lam = df["lambda"].values
pos_mask = lam > 0
lam_pos = lam[pos_mask]

if lam_pos.size == 0:
    raise ValueError("No positive lambdas found (log axis needs lambda > 0).")

has_zero = np.any(lam == 0)
idx0 = np.where(lam == 0)[0][0] if has_zero else None
eps = lam_pos.min() * 0.7  # pseudo x-position for lambda=0 on log axis

y_gr = df[COL_GR].values
y_s = df[COL_S].values if COL_S in df.columns else None

# GR 至少要在正 lambda 上有可用值
if not np.isfinite(y_gr[pos_mask]).any():
    raise ValueError(f"{COL_GR} has no finite values for positive lambdas.")

# S 是否真的可画：不仅列存在，而且正 lambda 上至少有一个有限值
has_s = False
if y_s is not None:
    has_s = np.isfinite(y_s[pos_mask]).any()

# ----------------------------
# 6) Plot helper
# ----------------------------
def plot_one(ax, lam, pos_mask, has_zero, idx0, eps, y, title, ylabel, color, annotate_zero=False):
    y = np.asarray(y, dtype=float)

    # 只画正 lambda 且 y 有限的点
    valid_mask = pos_mask & np.isfinite(y)

    if not np.any(valid_mask):
        ax.set_visible(False)
        return

    # main curve
    ax.plot(lam[valid_mask], y[valid_mask], marker="o", color=color)

    # baseline lambda = 0
    if has_zero and idx0 is not None and np.isfinite(y[idx0]):
        ax.scatter([eps], [y[idx0]], marker="s", s=60, zorder=4, color=color)
        if annotate_zero:
            ax.annotate(
                r"$\lambda=0$",
                (eps, y[idx0]),
                xytext=(6, 6),
                textcoords="offset points"
            )

    setup_log_lambda_axis(ax)
    ax.set_title(title)
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(ylabel)

    yy = y[valid_mask]
    y_min, y_max = yy.min(), yy.max()
    pad = 0.08 * max(y_max - y_min, 1e-6)
    ax.set_ylim(y_min - pad, y_max + pad)

# ----------------------------
# 7) Plot
# ----------------------------
if has_s:
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4), constrained_layout=True)

    plot_one(
        axes[0], lam, pos_mask, has_zero, idx0, eps,
        y_gr,
        "GR-consistency", "GR-consistency",
        CONS_COLORS["gr"],
        annotate_zero=True
    )

    plot_one(
        axes[1], lam, pos_mask, has_zero, idx0, eps,
        y_s,
        "S-consistency", "S-consistency",
        CONS_COLORS["s"],
        annotate_zero=False
    )
else:
    fig, ax = plt.subplots(1, 1, figsize=(4.6, 3.4), constrained_layout=True)

    plot_one(
        ax, lam, pos_mask, has_zero, idx0, eps,
        y_gr,
        "GR-consistency", "GR-consistency",
        CONS_COLORS["gr"],
        annotate_zero=True
    )

# ----------------------------
# 8) Save + show
# ----------------------------
os.makedirs(os.path.dirname(save_path), exist_ok=True)
fig.savefig(save_path)
plt.show()