#!/usr/bin/env python3
# plot_balacc_and_vr_separate_single_dataset.py
#
# One dataset only.
#
# Outputs:
#   1) Balanced Accuracy vs lambda
#      - URW (Forward KL)
#      - URW (Reverse KL)
#      - Combinatorial Laplacian
#
#   2) Forward fairness figure: VR(eps) vs lambda
#      - URW (Forward KL)
#      - Combinatorial Laplacian
#
#   3) Reverse fairness figure: VR(eps) vs lambda
#      - URW (Reverse KL)
#      - Combinatorial Laplacian
#
# Expected inputs:
#   Metrics CSVs:
#       lap
#       urw_forward
#       urw_reverse
#
#   VR-by-eps CSVs:
#       compas_forward_split_vr_by_eps.csv
#       compas_reverse_split_vr_by_eps.csv
#
# Edit only the CONFIG section below.

import os
import re
import glob
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


# ============================
# Style
# ============================
def set_ml_paper_style():
    mpl.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,

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


# ============================
# CONFIG: EDIT HERE
# ============================
DATASET = "compas"
TASK = "binary"

# -------- Metrics files (per-lambda) --------
# Adjust these glob patterns to your actual folders / filenames.
METRICS_GLOB = {
    "lap": "results_unified/metrics_laplacian_compas_depth1_lambda*.csv",
    "urw_forward": "results_unified/metrics_urw_compas_aggforward_depth1_lambda*.csv",
    "urw_reverse": "results_unified/metrics_urw_compas_aggreverse_depth1_lambda*.csv",
}

BAL_ACC_COL = "test_bal_acc"

# -------- VR by eps files --------
VR_BY_EPS_CSV_FORWARD = "results_unified/compas_forward_split_vr_by_eps.csv"
VR_BY_EPS_CSV_REVERSE = "results_unified/compas_reverse_split_vr_by_eps.csv"

# Four eps panels (2x2)
EPS_LIST = [0.001, 0.01, 0.05, 0.1]

OUTDIR = "plots_compas"

OUTPDF_BALACC = os.path.join(
    OUTDIR, f"balacc_vs_lambda_{DATASET}_{TASK}.pdf"
)
OUTPDF_VR_FORWARD = os.path.join(
    OUTDIR, f"vr_vs_lambda_forwardKL_plus_lap_{DATASET}_{TASK}.pdf"
)
OUTPDF_VR_REVERSE = os.path.join(
    OUTDIR, f"vr_vs_lambda_reverseKL_plus_lap_{DATASET}_{TASK}.pdf"
)

# ============================
# Labels / styles
# ============================
METHOD_LABEL = {
    "lap": "Combinatorial Laplacian",
    "urw_forward": "URW (Forward KL)",
    "urw_reverse": "URW (Reverse KL)",
}

METHOD_STYLE = {
    "urw_forward": dict(marker="s", linestyle="-", color="#1f77b4"),  # blue
    "lap": dict(marker="o", linestyle="-", color="#ff7f0e"),          # orange
    "urw_reverse": dict(marker="^", linestyle="-", color="#2ca02c"),  # green
}

PLOT_METHODS_BALACC = ["urw_forward", "urw_reverse", "lap"]
PLOT_METHODS_FORWARD = ["urw_forward", "lap"]
PLOT_METHODS_REVERSE = ["urw_reverse", "lap"]


# ============================
# Helpers
# ============================
LAMBDA_RE = re.compile(r"_lambda(?P<lam>[0-9.eE+-]+)\.csv$", re.IGNORECASE)


def parse_lambda_from_filename(path: str):
    fname = os.path.basename(path)
    m = LAMBDA_RE.search(fname)
    if not m:
        return None
    try:
        return float(m.group("lam"))
    except ValueError:
        return None


def load_metrics(pattern: str, method: str):
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"[{method}] No files matched: {pattern}")

    rows = []
    for f in files:
        lam = parse_lambda_from_filename(f)
        if lam is None:
            continue

        df = pd.read_csv(f)
        if BAL_ACC_COL not in df.columns:
            raise KeyError(f"[{method}] Missing '{BAL_ACC_COL}' in {f}")

        rows.append({
            "method": method,
            "lambda": float(lam),
            "bal_acc": float(df[BAL_ACC_COL].iloc[0]),
        })

    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError(f"[{method}] Loaded 0 rows from pattern={pattern}")

    return out.sort_values("lambda").reset_index(drop=True)


def _parse_eps_from_col(col: str):
    if not col.startswith("vr_eps"):
        return None
    try:
        return float(col.replace("vr_eps", ""))
    except Exception:
        return None


def load_vr_by_eps(vr_csv: str, eps: float):
    df = pd.read_csv(vr_csv)

    if "method" not in df.columns or "lambda" not in df.columns:
        raise KeyError(
            f"VR file must contain 'method' and 'lambda'. "
            f"Got columns={list(df.columns)}"
        )

    cand_cols = [c for c in df.columns if c.startswith("vr_eps")]
    if not cand_cols:
        raise KeyError(f"No 'vr_eps*' columns found in {vr_csv}")

    eps_map = []
    for c in cand_cols:
        v = _parse_eps_from_col(c)
        if v is not None:
            eps_map.append((c, v))

    if not eps_map:
        raise KeyError(f"Failed to parse epsilon values from columns in {vr_csv}")

    # nearest available epsilon column
    col, eps_found = min(eps_map, key=lambda t: abs(t[1] - eps))

    out = df[["method", "lambda", col]].copy()
    out = out.rename(columns={col: "vr"})
    out["lambda"] = out["lambda"].astype(float)
    out["vr"] = out["vr"].astype(float)

    return out, col, eps_found


def auto_pad_limits(vals, pad_ratio=0.08, min_range=1e-4):
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0, 1.0

    vmin = float(np.min(vals))
    vmax = float(np.max(vals))
    rng = max(vmax - vmin, min_range)
    pad = rng * pad_ratio
    return vmin - pad, vmax + pad


# ============================
# Plot 1: Balanced Accuracy
# ============================
def plot_balacc(metrics_df: pd.DataFrame, outpdf: str):
    fig, ax = plt.subplots(figsize=(6.8, 4.6))

    for method in PLOT_METHODS_BALACC:
        sub = metrics_df[metrics_df["method"] == method].sort_values("lambda")
        if sub.empty:
            continue

        ax.plot(
            sub["lambda"],
            sub["bal_acc"],
            label=METHOD_LABEL.get(method, method),
            **METHOD_STYLE.get(method, {})
        )

    ax.set_xscale("symlog", linthresh=0.1)
    ax.set_xlabel(r"Regularization strength $\lambda$")
    ax.set_ylabel("Balanced Accuracy")
    ax.set_title(f"{DATASET.capitalize()} ({TASK}): Balanced Accuracy vs $\\lambda$")
    ax.legend(loc="lower left")

    os.makedirs(os.path.dirname(outpdf), exist_ok=True)
    fig.savefig(outpdf)
    plt.show()
    print(f"[OK] saved: {outpdf}")


# ============================
# Plot 2/3: VR figures
# ============================
def plot_vr_figure(
    vr_csv: str,
    plot_methods: list,
    fig_title: str,
    outpdf: str,
):
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.6), constrained_layout=True)
    axes = axes.ravel()

    for ax, eps in zip(axes, EPS_LIST):
        vr_df, used_col, eps_found = load_vr_by_eps(vr_csv, eps)
        vr_df = vr_df[vr_df["method"].isin(plot_methods)].copy()

        for method in plot_methods:
            sub = vr_df[vr_df["method"] == method].sort_values("lambda")
            if sub.empty:
                continue

            ax.plot(
                sub["lambda"],
                sub["vr"],
                label=METHOD_LABEL.get(method, method),
                **METHOD_STYLE.get(method, {})
            )

        ax.set_xscale("symlog", linthresh=0.1)
        ax.set_xlabel(r"$\lambda$")
        ax.set_ylabel(r"Violation Rate $\mathrm{VR}(\varepsilon)$")

        if abs(eps_found - eps) < 1e-12:
            ax.set_title(rf"$\varepsilon={eps:g}$")
        else:
            ax.set_title(rf"$\varepsilon={eps:g}$ (used {used_col})")

        if not vr_df.empty:
            y0, y1 = auto_pad_limits(vr_df["vr"].values, pad_ratio=0.10, min_range=1e-3)
            ax.set_ylim(-0.1, 1)

        ax.legend(loc="upper right", bbox_to_anchor=(0.98, 0.98))

    fig.suptitle(fig_title, y=1.02)
    os.makedirs(os.path.dirname(outpdf), exist_ok=True)
    fig.savefig(outpdf)
    plt.show()
    print(f"[OK] saved: {outpdf}")


# ============================
# Main
# ============================
if __name__ == "__main__":
    # ---- Load performance metrics ----
    metrics_list = []
    for method, pattern in METRICS_GLOB.items():
        metrics_list.append(load_metrics(pattern, method))

    metrics = pd.concat(metrics_list, ignore_index=True)
    metrics = metrics[metrics["method"].isin(PLOT_METHODS_BALACC)].copy()

    # ---- Figure 1: performance ----
    plot_balacc(metrics, OUTPDF_BALACC)

    # ---- Figure 2: forward fairness ----
    plot_vr_figure(
        vr_csv=VR_BY_EPS_CSV_FORWARD,
        plot_methods=PLOT_METHODS_FORWARD,
        fig_title=f"{DATASET.capitalize()} ({TASK}): Forward-KL Fairness Violation vs $\\lambda$",
        outpdf=OUTPDF_VR_FORWARD,
    )

    # ---- Figure 3: reverse fairness ----
    plot_vr_figure(
        vr_csv=VR_BY_EPS_CSV_REVERSE,
        plot_methods=PLOT_METHODS_REVERSE,
        fig_title=f"{DATASET.capitalize()} ({TASK}): Reverse-KL Fairness Violation vs $\\lambda$",
        outpdf=OUTPDF_VR_REVERSE,
    )