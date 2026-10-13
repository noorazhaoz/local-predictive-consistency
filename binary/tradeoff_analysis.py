import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter, FuncFormatter
import glob
import pandas as pd
from matplotlib.ticker import LogFormatterMathtext

PANEL_COLORS = {
    "bal": "#1f77b4",   # 蓝
    "auc": "#2ca02c",   # 绿
    "ifm": "#d62728",   # 红
    "ifd": "#9467bd",   # 紫
}

def set_ml_paper_style():
    mpl.rcParams.update({
        # figure / save
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,

        # font
        "font.family": "DejaVu Sans",   # 默认就行，最稳
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,

        # lines
        "lines.linewidth": 2.0,
        "lines.markersize": 5.0,

        # axes look
        "axes.grid": False,             # ✅ 不要网格
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 1.0,

        # ticks (paper 常见：向内、短一些)
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

        # legend
        "legend.frameon": False,
    })

set_ml_paper_style()

import numpy as np

def setup_log_lambda_axis(ax, x_positive):
    """
    x_positive: 仅包含 >0 的 lambda 数组，用来确定 log ticks 范围
    """
    ax.set_xscale("log")

    # major ticks at decades
    ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=12))

    # show major tick labels as 0.1, 1, 10, 100...

    ax.xaxis.set_major_formatter(LogFormatterMathtext())

    # minor ticks (2..9)*10^k, but no labels
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2,10)*0.1, numticks=100))
    ax.xaxis.set_minor_formatter(NullFormatter())

def auto_yscale(ax, y, pad_ratio=0.08, min_range=1e-4):
    """
    Automatically tighten y-axis based on data range.
    pad_ratio: relative padding (8% default)
    min_range: avoid zero-height axis
    """
    y = np.asarray(y)
    y_min = y.min()
    y_max = y.max()

    rng = max(y_max - y_min, min_range)
    pad = rng * pad_ratio

    ax.set_ylim(y_min - pad, y_max + pad)

# ----------------------------
# 1) Load URW metrics across lambdas
# ----------------------------
def load_urw_metrics(pattern):
    """
    pattern example:
      results/urw/metrics_urw_lambda*.csv
    Each file contains ONE ROW of metrics (or multiple rows; we take mean).
    Lambda is parsed from filename: ...lambda{val}.csv
    """
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files: {pattern}")

    rows = []
    for f in files:
        df = pd.read_csv(f)
        # robust: if multiple rows, average
        s = df.mean(numeric_only=True)

        lam = float(f.split("lambda")[-1].replace(".csv", ""))
        s["lambda"] = lam
        rows.append(s)

    out = pd.DataFrame(rows).sort_values("lambda").reset_index(drop=True)
    return out

# TODO: change to your actual path
urw = load_urw_metrics("results/metrics_laplacian_depth1_lambda*.csv")

# ----------------------------
# 2) Column name mapping (edit to match yours)
# ----------------------------
COL = {
    "bal_acc": "test_bal_acc",   # e.g. "bal_acc" / "balanced_accuracy"
    "auc": "test_auc",                     # e.g. "auc"
    "if_mean": "IF_mean(forwardKL,prob; monitor)",             # e.g. "if_mean"
    "if_median": "IF_median(forwardKL,prob; monitor)",         # e.g. "if_median"
}

# ----------------------------
# 3) Prepare data
# ----------------------------
lam = urw["lambda"].values

# split positive lambdas and baseline
pos_mask = lam > 0
lam_pos = lam[pos_mask]

if lam_pos.size == 0:
    raise ValueError("No positive lambdas found. Log axis requires lambdas > 0.")

# baseline (lambda=0) if exists
has_zero = np.any(lam == 0)
if has_zero:
    # if multiple rows with lambda==0, take the first after sorting (should be one)
    idx0 = np.where(lam == 0)[0][0]
else:
    idx0 = None

# pseudo x-position for baseline marker on log axis
eps = lam_pos.min() * 0.7

# helper to get y arrays
def get_y(colname):
    if colname not in urw.columns:
        raise KeyError(f"Column '{colname}' not found in urw.columns: {list(urw.columns)}")
    return urw[colname].values

y_bal = get_y(COL["bal_acc"])
y_auc = get_y(COL["auc"])
y_ifm = get_y(COL["if_mean"])
y_ifd = get_y(COL["if_median"])

# ----------------------------
# 4) Plot helpers
# ----------------------------
def plot_metric(ax, y, title, ylabel, color, ylim=None, annotate_zero=True):
    # curve for positive lambdas
    ax.plot(lam_pos, y[pos_mask], marker="o",color=color)

    # baseline point at lambda=0
    if has_zero:
        y0 = y[idx0]

        # small vertical offset ONLY for visualization (AUC case)
        if ylabel == "AUC":
            y0 = y0 + 0.001

        ax.scatter(
            [eps], [y0],
            marker="s",
            s=60,
            zorder=4
        )

        if annotate_zero:
            ax.annotate(
                r"$\lambda=0$",
                (eps, y0),
                xytext=(6, 6),
                textcoords="offset points"
            )

    setup_log_lambda_axis(ax, lam_pos)

    ax.set_title(title)
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(ylabel)

    if ylim is not None:
        ax.set_ylim(*ylim)
    else:
        ax.margins(y=0.08)

    # small x padding (works in log too)
    ax.margins(x=0.02)

# ----------------------------
# 5) Main 2x2 figure
# ----------------------------
fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.0), constrained_layout=True)

plot_metric(axes[0, 0], y_bal, "Balanced Acc", "Balanced Acc", color=PANEL_COLORS["bal"])
ax = axes[0, 0]
y = y_bal[pos_mask]
ax.set_ylim(
    y.min() - 0.01,
    y.max() + 0.01
)

plot_metric(axes[0, 1], y_auc, "AUC", "AUC",color=PANEL_COLORS["auc"])
ax = axes[0, 1]
y = y_auc[pos_mask]
ax.set_ylim(
    y.min() - 0.01,
    y.max() + 0.01
)

plot_metric(axes[1, 0], y_ifm, "IF mean", "IF score (mean)",color=PANEL_COLORS["ifm"])
auto_yscale(axes[1, 0], y_ifm[pos_mask])

plot_metric(axes[1, 1], y_ifd, "IF median", "IF score (median)",color=PANEL_COLORS["ifd"])
auto_yscale(axes[1, 1], y_ifd[pos_mask])


# optional: keep suptitle short; or delete if you prefer no global title
fig.suptitle("Standard Graph Laplacian (Binary): Metrics across $\\lambda$", y=1.02)

# ----------------------------
# 6) Save (optional)
# ----------------------------
fig.savefig("results/lap/lap_binary_main.pdf")
# fig.savefig("urw_binary_main.png")

plt.show()
