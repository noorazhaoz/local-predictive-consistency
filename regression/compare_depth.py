import os
import re
import glob
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, LogFormatterMathtext

# ----------------------------
# ML-paper style
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

# ===================================================
# Config
# ===================================================
BASE_DIR = "results/urw"
DEPTHS = [1, 2, 3]
PATTERN = os.path.join(BASE_DIR, "metrics_urw_depth*_lambda*.csv")
OUT_DIR = os.path.join(BASE_DIR, "plots", "compare_depths_1_2_3")
os.makedirs(OUT_DIR, exist_ok=True)

# ===================================================
# 1) Load all files and tag depth
# ===================================================
files = sorted(glob.glob(PATTERN))
print(f"Found {len(files)} files under: {PATTERN}")
if len(files) == 0:
    raise FileNotFoundError(f"No files matched: {PATTERN}")

depth_re = re.compile(r"depth(\d+)")
dfs = []

for f in files:
    m = depth_re.search(os.path.basename(f))
    if not m:
        continue
    depth = int(m.group(1))
    if depth not in DEPTHS:
        continue

    df = pd.read_csv(f)
    df["filename"] = f
    df["depth"] = depth
    dfs.append(df)

if len(dfs) == 0:
    raise ValueError(
        f"No CSVs found for DEPTHS={DEPTHS}. "
        f"Check filenames like metrics_urw_depth3_lambda0.1.csv"
    )

df_all = pd.concat(dfs, ignore_index=True)

if "lambda_cons" not in df_all.columns:
    raise KeyError("Column 'lambda_cons' not found in CSVs.")

df_all = df_all.sort_values(["depth", "lambda_cons"]).reset_index(drop=True)
print("\nCombined DF (head):\n", df_all.head())

# ===================================================
# 2) Metric map (col -> (title, ylabel, outname))
#    (Titles shortened for paper style; depth info comes from legend)
# ===================================================
METRICS = {
    # ---- Performance ----
    "test_RMSE_real": ("RMSE vs $\\lambda$", "RMSE", "rmse_vs_lambda_depth123"),
    "test_MAE_real":  ("MAE vs $\\lambda$",  "MAE",  "mae_vs_lambda_depth123"),
    "test_R2_real":   ("$R^2$ vs $\\lambda$",   "$R^2$",   "r2_vs_lambda_depth123"),
    "test_MSE_real":  ("MSE vs $\\lambda$",  "MSE",  "mse_vs_lambda_depth123"),

    # ---- Group fairness gaps ----
    "group_gap_abs_real_female_minus_male": ("Abs. group gap (real) vs $\\lambda$", "Abs. gap (real)", "gap_abs_real_vs_lambda_depth123"),
    "group_gap_abs_log_female_minus_male":  ("Abs. group gap (log) vs $\\lambda$",  "Abs. gap (log)",  "gap_abs_log_vs_lambda_depth123"),
    "group_gap_real_female_minus_male":     ("Group gap (real) vs $\\lambda$", "Gap (real)", "gap_real_vs_lambda_depth123"),
    "group_gap_log_female_minus_male":      ("Group gap (log) vs $\\lambda$",  "Gap (log)",  "gap_log_vs_lambda_depth123"),

    # ---- Tradeoff ----
    "tradeoff_abs_gap_over_std_real": ("Trade-off (real) vs $\\lambda$", "AbsGap/Std (real)", "trade_real_vs_lambda_depth123"),
    "tradeoff_abs_gap_over_std_log":  ("Trade-off (log) vs $\\lambda$",  "AbsGap/Std (log)",  "trade_log_vs_lambda_depth123"),

    # ---- Individual fairness (URW IF) ----
    "IF_mean(URW_L1,log_pred)":   ("IF mean vs $\\lambda$",   "IF mean",   "if_mean_vs_lambda_depth123"),
    "IF_median(URW_L1,log_pred)": ("IF median vs $\\lambda$", "IF median", "if_median_vs_lambda_depth123"),
}

# ===================================================
# 3) Plot helper: one metric, three depths on one plot
# ===================================================
def plot_metric_depth_compare(df_all: pd.DataFrame, col: str, title: str, ylabel: str, outname: str):
    if col not in df_all.columns:
        print(f"[Skip] Missing column: {col}")
        return

    fig, ax = plt.subplots(figsize=(6.2, 4.0))

    any_plotted = False
    for depth in DEPTHS:
        dfd = df_all[df_all["depth"] == depth].copy()
        if dfd.empty:
            continue

        dfd = dfd.sort_values("lambda_cons")
        x = dfd["lambda_cons"].to_numpy()
        y = dfd[col].to_numpy()

        mask = np.isfinite(x) & np.isfinite(y) & (x > 0)
        x, y = x[mask], y[mask]
        if len(x) == 0:
            continue

        ax.plot(x, y, marker="o", label=f"depth={depth}")
        any_plotted = True

    if not any_plotted:
        print(f"[Skip] No valid data plotted for: {col}")
        plt.close(fig)
        return

    ax.set_xscale("log")

    # log ticks that look like papers
    ax.xaxis.set_major_locator(LogLocator(base=10.0))
    ax.xaxis.set_major_formatter(LogFormatterMathtext())
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=(2, 3, 4, 5, 6, 7, 8, 9)))

    ax.set_xlabel("$\\lambda$")
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    # subtle horizontal guide lines (optional; comment out if you want pure no-grid)
    # ax.grid(axis="y", alpha=0.15)

    ax.legend(loc="best", ncols=1)

    fig.tight_layout()
    outpath = os.path.join(OUT_DIR, f"{outname}.pdf")
    fig.savefig(outpath)

    plt.close(fig)
    print(f"[Saved] {outpath}")

# ===================================================
# 4) Make plots
# ===================================================
for col, (title, ylabel, outname) in METRICS.items():
    plot_metric_depth_compare(df_all, col, title, ylabel, outname)

print(f"\nDone. All depth-comparison plots saved to: {OUT_DIR}")