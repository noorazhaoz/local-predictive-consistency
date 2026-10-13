import re
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter, LogFormatterMathtext
import glob
import pandas as pd
import numpy as np

PANEL_COLORS = {
    "cindex": "#1f77b4",
    "ibs":    "#2ca02c",
    "ifm":    "#d62728",
    "ifd":    "#9467bd",
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
        "legend.frameon": False,
    })
set_ml_paper_style()

def setup_log_lambda_axis(ax):
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=12))
    ax.xaxis.set_major_formatter(LogFormatterMathtext())
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2,10)*0.1, numticks=100))
    ax.xaxis.set_minor_formatter(NullFormatter())

def auto_yscale(ax, y, pad_ratio=0.08, min_range=1e-6):
    y = np.asarray(y, dtype=float)
    y_min = float(np.min(y))
    y_max = float(np.max(y))
    rng = max(y_max - y_min, min_range)
    pad = rng * pad_ratio
    ax.set_ylim(y_min - pad, y_max + pad)

def auto_yscale_include_baseline(ax, y_pos, y0, pad_ratio=0.08, min_range=1e-6):
    y_pos = np.asarray(y_pos, dtype=float)
    y_min = min(float(np.min(y_pos)), float(y0))
    y_max = max(float(np.max(y_pos)), float(y0))
    rng = max(y_max - y_min, min_range)
    pad = rng * pad_ratio
    ax.set_ylim(y_min - pad, y_max + pad)

# ---------- robust lambda parser (supports 0.001, 1e-3, 3E-4 etc.) ----------
_LAM_RE = re.compile(r"lambda([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")

def parse_lambda_from_filename(path: str) -> float:
    m = _LAM_RE.search(path)
    if m is None:
        raise ValueError(f"Cannot parse lambda from filename: {path}")
    return float(m.group(1))

def load_metrics(pattern):
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No files: {pattern}")

    rows = []
    for f in files:
        df = pd.read_csv(f)
        s = df.mean(numeric_only=True)

        lam = parse_lambda_from_filename(f)
        s["lambda"] = lam
        rows.append(s)

    out = pd.DataFrame(rows)
    # IMPORTANT: numeric sort, not string sort
    out = out.sort_values("lambda").reset_index(drop=True)
    return out

# ----------------------------
# Load
# ----------------------------
dfm = load_metrics("lap/employee/metrics_deepsurv_laplacianL_depth1_lambda*.csv")

COL = {
    "cindex":    "test_c_index",
    "ibs":       "ibs",
    "if_mean":   "IF_mean(|(I-P)eta|,test)",
    "if_median": "IF_median(|(I-P)eta|,test)",
}
for c in COL.values():
    if c not in dfm.columns:
        raise KeyError(f"[Missing] {c} not in columns. Available: {list(dfm.columns)}")

lam = dfm["lambda"].values.astype(float)
print(dfm["lambda"].tolist())

y_cindex = dfm[COL["cindex"]].values.astype(float)
y_ibs    = dfm[COL["ibs"]].values.astype(float)
y_ifm    = dfm[COL["if_mean"]].values.astype(float)
y_ifd    = dfm[COL["if_median"]].values.astype(float)

# ----------------------------
# Log axis handling
# ----------------------------
pos_mask = lam > 0
lam_pos = lam[pos_mask]
if lam_pos.size == 0:
    raise ValueError("No positive lambdas found. Log axis requires lambdas > 0.")

has_zero = np.any(lam == 0)
idx0 = np.where(lam == 0)[0][0] if has_zero else None

# baseline x position: slightly smaller than smallest positive lambda
eps = lam_pos.min() / 1.5

def plot_metric(ax, y, title, ylabel, color, annotate_zero=True):
    # curve for positive lambdas (includes 1e-6, 1e-5, ...)
    ax.plot(lam_pos, y[pos_mask], marker="o", color=color)

    if has_zero:
        y0 = y[idx0]
        ax.scatter([eps], [y0], marker="s", s=60, zorder=4, color=color)
        if annotate_zero:
            ax.annotate(r"$\lambda=0$", (eps, y0),
                        xytext=(6, 6), textcoords="offset points")

    setup_log_lambda_axis(ax)
    ax.set_title(title)
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(ylabel)
    ax.margins(x=0.02)

# ----------------------------
# Plot 2x2
# ----------------------------
fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.0), constrained_layout=True)

# C-index
plot_metric(axes[0,0], y_cindex, "Performance (C-index)", "C-index", PANEL_COLORS["cindex"])
auto_yscale_include_baseline(axes[0,0], y_cindex[pos_mask], y_cindex[idx0] if has_zero else y_cindex[pos_mask][0])

# IBS
plot_metric(axes[0,1], y_ibs, "Performance (IBS)", "IBS", PANEL_COLORS["ibs"])
auto_yscale_include_baseline(axes[0,1], y_ibs[pos_mask], y_ibs[idx0] if has_zero else y_ibs[pos_mask][0])

# IF mean
plot_metric(axes[1,0], y_ifm, "Individual Fairness (mean)", "IF mean (eta scale)", PANEL_COLORS["ifm"])
auto_yscale_include_baseline(axes[1,0], y_ifm[pos_mask], y_ifm[idx0] if has_zero else y_ifm[pos_mask][0])

# IF median
plot_metric(axes[1,1], y_ifd, "Individual Fairness (median)", "IF median (eta scale)", PANEL_COLORS["ifd"])
auto_yscale_include_baseline(axes[1,1], y_ifd[pos_mask], y_ifd[idx0] if has_zero else y_ifd[pos_mask][0])

fig.suptitle(r"Survival: Performance–Fairness across $\lambda$", y=1.02)

out_pdf = "lap/employee/survival_lap_tradeoff.pdf"
fig.savefig(out_pdf)
plt.show()
