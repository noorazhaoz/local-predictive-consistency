import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter
from matplotlib.ticker import LogFormatterMathtext
import glob
import pandas as pd
import numpy as np

# ----------------------------
# Style (same as yours)
# ----------------------------
PANEL_COLORS = {
    "rmse": "#1f77b4",
    "mae":  "#2ca02c",
    "ifm":  "#d62728",
    "ifd":  "#9467bd",
}

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

def setup_log_lambda_axis(ax, x_positive):
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=12))
    ax.xaxis.set_major_formatter(LogFormatterMathtext())
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2,10)*0.1, numticks=100))
    ax.xaxis.set_minor_formatter(NullFormatter())

def auto_yscale(ax, y, pad_ratio=0.08, min_range=1e-6):
    y = np.asarray(y)
    y_min = float(np.min(y))
    y_max = float(np.max(y))
    rng = max(y_max - y_min, min_range)
    pad = rng * pad_ratio
    ax.set_ylim(y_min - pad, y_max + pad)

# ----------------------------
# 1) Load metrics across lambdas
# ----------------------------
def load_metrics(pattern):
    """
    Each CSV expected to contain ONE ROW (or multiple rows; we average numeric).
    Lambda is parsed from filename: ...lambda{val}.csv
    """
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files: {pattern}")

    rows = []
    for f in files:
        df = pd.read_csv(f)
        s = df.mean(numeric_only=True)

        # parse lambda from filename (robust)
        # supports: ..._lambda0.3.csv or ...lambda0.3.csv
        m = None
        for key in ["_lambda", "lambda"]:
            if key in f:
                try:
                    m = float(f.split(key)[-1].replace(".csv", "").replace(".CSV",""))
                    break
                except:
                    pass
        if m is None:
            raise ValueError(f"Cannot parse lambda from filename: {f}")

        s["lambda"] = m
        rows.append(s)

    out = pd.DataFrame(rows).sort_values("lambda").reset_index(drop=True)
    return out

def auto_yscale_include_baseline(ax, y_pos, y0, pad_ratio=0.08, min_range=1e-6):
    y_min = float(np.min(y_pos))
    y_max = float(np.max(y_pos))
    y_min = min(y_min, float(y0))
    y_max = max(y_max, float(y0))

    rng = max(y_max - y_min, min_range)
    pad = rng * pad_ratio
    ax.set_ylim(y_min - pad, y_max + pad)


dfm = load_metrics("results/laplacian/metrics_laplacian_depth1_lambda*.csv")

# ----------------------------
# 2) Column mapping (REGRESSION)
# ----------------------------
COL = {
    "rmse": "test_RMSE_real",
    "mae":  "test_MAE_real",
    "if_mean":   "IF_mean(URW_L1,log_pred)",
    "if_median": "IF_median(URW_L1,log_pred)",
}

for k, c in COL.items():
    if c not in dfm.columns:
        raise KeyError(f"[Missing] {c} not in columns. Available: {list(dfm.columns)}")

lam = dfm["lambda"].values
print("All lambdas:", lam)
print("Min lambda:", lam.min(), "Max lambda:", lam.max())

y_rmse = dfm[COL["rmse"]].values
y_mae  = dfm[COL["mae"]].values
y_ifm  = dfm[COL["if_mean"]].values
y_ifd  = dfm[COL["if_median"]].values

pos_mask = lam > 0
lam_pos = lam[pos_mask]
if lam_pos.size == 0:
    raise ValueError("No positive lambdas found. Log axis requires lambdas > 0.")

has_zero = np.any(lam == 0)
idx0 = np.where(lam == 0)[0][0] if has_zero else None

# pseudo x-position for baseline marker on log axis
eps = lam_pos.min() * 0.7

def plot_metric(ax, y, title, ylabel, color, annotate_zero=True):
    # curve for positive lambdas
    ax.plot(lam_pos, y[pos_mask], marker="o", color=color)

    # baseline point at lambda=0
    has_zero = np.any(lam == 0)

    if has_zero:
        print(title, "baseline y0 =", y[idx0], "pos-range =", y[pos_mask].min(), y[pos_mask].max())

        y0 = y[idx0]
        ax.scatter([eps], [y0], marker="s", s=60, zorder=4, color=color)
        if annotate_zero:
            ax.annotate(r"$\lambda=0$", (eps, y0), xytext=(6, 6), textcoords="offset points")

    setup_log_lambda_axis(ax, lam_pos)
    ax.set_title(title)
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(ylabel)
    ax.margins(x=0.02)

# ----------------------------
# 3) Plot 2x2
# ----------------------------
fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.0), constrained_layout=True)

# (1) RMSE
plot_metric(
    axes[0, 0],
    y_rmse,
    "RMSE (real scale)",
    "RMSE (Salary)",
    PANEL_COLORS["rmse"]
)
if has_zero:
    y0_rmse = y_rmse[idx0]
    auto_yscale_include_baseline(axes[0, 0], y_rmse[pos_mask], y0_rmse)
else:
    auto_yscale(axes[0, 0], y_rmse[pos_mask])

# (2) MAE
plot_metric(
    axes[0, 1],
    y_mae,
    "MAE (real scale)",
    "MAE (Salary)",
    PANEL_COLORS["mae"]
)
if has_zero:
    y0_mae = y_mae[idx0]
    auto_yscale_include_baseline(axes[0, 1], y_mae[pos_mask], y0_mae)
else:
    auto_yscale(axes[0, 1], y_mae[pos_mask])

# (3) IF mean
plot_metric(
    axes[1, 0],
    y_ifm,
    "Individual Fairness (mean)",
    "IF mean (log-pred)",
    PANEL_COLORS["ifm"]
)
if has_zero:
    y0_ifm = y_ifm[idx0]
    auto_yscale_include_baseline(axes[1, 0], y_ifm[pos_mask], y0_ifm)
else:
    auto_yscale(axes[1, 0], y_ifm[pos_mask])

# (4) IF median
plot_metric(
    axes[1, 1],
    y_ifd,
    "Individual Fairness (median)",
    "IF median (log-pred)",
    PANEL_COLORS["ifd"]
)
if has_zero:
    y0_ifd = y_ifd[idx0]
    auto_yscale_include_baseline(axes[1, 1], y_ifd[pos_mask], y0_ifd)
else:
    auto_yscale(axes[1, 1], y_ifd[pos_mask])


fig.suptitle("Regression (Standard Graph Laplacian): Performance–Fairness across $\\lambda$", y=1.02)

# Save
# TODO: 改你想存的路径
out_pdf = "results/laplacian/plots/regression_lap_tradeoff.pdf"
# out_png = "results/regression_urw_tradeoff_2x2.png"
fig.savefig(out_pdf)
# fig.savefig(out_png)
plt.show()

