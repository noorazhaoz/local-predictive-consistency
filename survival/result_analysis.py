import glob
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------
# 0. Config: change these if needed
# --------------------------------------

RESULT_GLOB = "results/lap/flchain/metrics_deepsurv_laplacianL_depth*_lambda*.csv"

# Where to save analysis outputs (table + plots)
OUTDIR = "results/lap/flchain"
os.makedirs(OUTDIR, exist_ok=True)

# --------------------------------------
# Helpers
# --------------------------------------
def require_col(df: pd.DataFrame, name: str):
    if name not in df.columns:
        raise KeyError(
            f"Required column '{name}' not found.\n"
            f"Available columns:\n{list(df.columns)}"
        )

def save_line_plot(x, y, xlabel, ylabel, title, outpath, xlog=False):
    plt.figure(figsize=(6, 4))
    plt.plot(x, y, marker="o")
    if xlog:
        plt.xscale("log")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, which="major", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

# --------------------------------------
# 1. Load all metrics CSVs
# --------------------------------------
files = sorted(glob.glob(RESULT_GLOB))

print("Found files:")
for f in files:
    print("  ", f)

assert len(files) > 0, f"No metrics files found with pattern: {RESULT_GLOB}"

dfs = []
for f in files:
    df = pd.read_csv(f)
    df["filename"] = os.path.basename(f)
    dfs.append(df)

df_all = pd.concat(dfs, ignore_index=True)

# --------------------------------------
# 2. Sort by lambda
# --------------------------------------
require_col(df_all, "lambda_cons")
df_all["lambda_cons"] = df_all["lambda_cons"].astype(float)
df_all = df_all.sort_values(by="lambda_cons").reset_index(drop=True)

print("\nCombined metrics (head):")
print(df_all.head())

# --------------------------------------
# 3. Required columns (DeepSurv outputs)
# --------------------------------------
COL_CINDEX = "test_c_index"
COL_BRIER  = "brier_mean"
COL_IBS    = "ibs"
COL_IF_MEAN = "IF_mean(|(I-P)eta|,test)"
COL_IF_MED  = "IF_median(|(I-P)eta|,test)"
COL_LIF     = "LIF_energy(eta,test)"
COL_GAPABS  = "group_gap_abs_eta"

required = [
    COL_CINDEX,
    COL_BRIER,
    COL_IBS,
    COL_IF_MEAN,
    COL_IF_MED,
    COL_LIF,
    COL_GAPABS,
]
for c in required:
    require_col(df_all, c)

# Extract arrays
lambdas = df_all["lambda_cons"].values.astype(float)

c_index    = df_all[COL_CINDEX].astype(float).values
brier_mean = df_all[COL_BRIER].astype(float).values
ibs        = df_all[COL_IBS].astype(float).values
IF_mean    = df_all[COL_IF_MEAN].astype(float).values
IF_median  = df_all[COL_IF_MED].astype(float).values
gap_abs    = df_all[COL_GAPABS].astype(float).values
lif_energy = df_all[COL_LIF].astype(float).values

# --------------------------------------
# 4. Print & save summary table
# --------------------------------------
summary_cols = [
    "lambda_cons",
    COL_CINDEX,
    COL_BRIER,
    COL_IBS,
    COL_IF_MEAN,
    COL_IF_MED,
    COL_GAPABS,
    COL_LIF,
    "theta",
    "tau_used",
    "graph_train_subset",
    "svd_rank",
    "use_metric",
    "metric_rank",
    "num_eval_times",
]
summary_cols = [c for c in summary_cols if c in df_all.columns]  # robust
summary = df_all[summary_cols].copy()

print("\n=== DeepSurv trade-off table (sorted by lambda) ===")
print(summary.to_string(index=False))

out_table_path = os.path.join(OUTDIR, "deepsurv_tradeoff_table.csv")
summary.to_csv(out_table_path, index=False)
print(f"\n[Saved] {out_table_path}")

# --------------------------------------
# 5. Sanity warnings
# --------------------------------------
if np.nanmax(brier_mean) > 2.0:
    print(
        "\n[Warn] brier_mean has unusually large values (>2). "
        "Brier score should typically be in [0, 1]. "
        "Check that you pass S(t|x) with correct shape (n_samples, n_times)."
    )

# --------------------------------------
# 6. Plot curves (log-lambda for lambda>0)
# --------------------------------------
mask_pos = lambdas > 0
if mask_pos.sum() < 2:
    print("\n[Warn] Not enough positive lambda values to plot log-scale curves (need >=2).")
else:
    lam_pos = lambdas[mask_pos]

    save_line_plot(
        lam_pos, c_index[mask_pos],
        xlabel=r"$\lambda$ (log scale)",
        ylabel="Test C-index (↑)",
        title="C-index vs λ (log)",
        outpath=os.path.join(OUTDIR, "loglambda_vs_cindex.pdf"),
        xlog=True
    )

    save_line_plot(
        lam_pos, brier_mean[mask_pos],
        xlabel=r"$\lambda$ (log scale)",
        ylabel="Mean Brier score (↓)",
        title="Mean Brier vs λ (log)",
        outpath=os.path.join(OUTDIR, "loglambda_vs_brier.pdf"),
        xlog=True
    )

    save_line_plot(
        lam_pos, ibs[mask_pos],
        xlabel=r"$\lambda$ (log scale)",
        ylabel="IBS (↓)",
        title="IBS vs λ (log)",
        outpath=os.path.join(OUTDIR, "loglambda_vs_ibs.pdf"),
        xlog=True
    )

    save_line_plot(
        lam_pos, IF_mean[mask_pos],
        xlabel=r"$\lambda$ (log scale)",
        ylabel="IF_mean (↓)",
        title="Individual Fairness vs λ (log)",
        outpath=os.path.join(OUTDIR, "loglambda_vs_IF_mean.pdf"),
        xlog=True
    )

    save_line_plot(
        lam_pos, gap_abs[mask_pos],
        xlabel=r"$\lambda$ (log scale)",
        ylabel="Group gap |Δ η| (↓)",
        title="Group gap vs λ (log)",
        outpath=os.path.join(OUTDIR, "loglambda_vs_groupgap.pdf"),
        xlog=True
    )

    save_line_plot(
        lam_pos, lif_energy[mask_pos],
        xlabel=r"$\lambda$ (log scale)",
        ylabel="LIF energy on η (↓)",
        title="LIF energy vs λ (log)",
        outpath=os.path.join(OUTDIR, "loglambda_vs_lifenergy.pdf"),
        xlog=True
    )

print("\n[Saved plots] in:", OUTDIR)
