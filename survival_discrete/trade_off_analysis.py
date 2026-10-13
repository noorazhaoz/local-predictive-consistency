#!/usr/bin/env python3
# plot_survival_lambda_perf_and_vr.py
#
# For survival datasets:
#   Fig1: performance vs lambda (URW vs Lap)  [C-index / IBS]
#   Fig2: VR(eps) vs lambda (URW vs Lap), 4 eps panels (2x2)
#
# Inputs:
# 1) ratio table (long): results/vr_if_sweep_ALL_depth1.csv
#    columns: dataset, penalty, lambda, eps_if, VR_IF(ours) ...
# 2) per-lambda metrics CSVs for each (dataset, method)
#
# Outputs (per dataset):
#   - perf_<dataset>_survival_<metric>_vs_lambda.pdf
#   - vr_<dataset>_survival_vs_lambda_multieps.pdf

import os
import re
import glob
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


# ----------------------------
# Style (reuse yours)
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

        "legend.frameon": False,
    })

set_ml_paper_style()


# ----------------------------
# Config
# ----------------------------
RATIO_CSV = "results/vr_if_sweep_ALL_depth1.csv"   # long VR table from vr_survival_if_sweep.py
OUTDIR = "results/plots"

DATASETS = ["flchain","whas","rossi"]  # <- survival datasets
EPS_LIST = [0.02, 0.1, 0.2, 0.3]
METHOD_KEEP = ["urw", "lap"]

# performance metrics: (column_name_in_metrics_csv, label, higher_is_better)
PERF_COLS = [
    ("test_c_index_from_risk=-logS(t*)", "C-index (higher is better)", True),
    ("ibs",     "IBS (lower is better)",      False),
]

# metrics CSV patterns for each dataset & method
METRICS_GLOB = {
    "flchain": {
        "lap": "results/lap/flchain/metrics_*_lap*_lambda*.csv",
        "urw": "results/urw/flchain/metrics_*_urw*_lambda*.csv",
    },
    "rossi": {
        "lap": "results/lap/rossi/metrics_*_lap*_lambda*.csv",
        "urw": "results/urw/rossi/metrics_*_urw*_lambda*.csv",
    },
    "whas": {
        "lap": "results/lap/whas/metrics_*_lap*_lambda*.csv",
        "urw": "results/urw/whas/metrics_*_urw*_lambda*.csv",
    },
}

TASK = "survival"


# ----------------------------
# Helpers
# ----------------------------
LAMBDA_RE = re.compile(r"_lambda(?P<lam>[0-9.eE+-]+)\.csv$", re.IGNORECASE)

def parse_lambda_from_filename(path: str):
    fname = os.path.basename(path)
    m = LAMBDA_RE.search(fname)
    if not m:
        return None, None
    lam_str = m.group("lam")
    try:
        lam = float(lam_str)
    except ValueError:
        lam = None
    return lam_str, lam

def auto_pad_limits(vals, pad_ratio=0.08, min_range=1e-6):
    vals = np.asarray(vals, dtype=float)
    vmin, vmax = float(np.min(vals)), float(np.max(vals))
    rng = max(vmax - vmin, min_range)
    pad = rng * pad_ratio
    return vmin - pad, vmax + pad

def normalize_penalty(p: str) -> str:
    s = str(p).strip().lower()
    if s.startswith("urw"):
        return "urw"
    if s.startswith("lap"):
        return "lap"
    if "laplacian" in s:
        return "lap"
    return s

def load_metrics_across_lambdas(pattern: str, method: str, perf_col: str):
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"[{method}] No metrics files matched: {pattern}")

    rows = []
    for f in files:
        lam_str, lam = parse_lambda_from_filename(f)
        if lam is None:
            continue

        df = pd.read_csv(f)
        if df.empty:
            continue

        if perf_col not in df.columns:
            raise KeyError(f"[{method}] Missing '{perf_col}' in {f}. Columns={list(df.columns)}")

        rows.append({
            "method": method,
            "lambda": float(lam),
            "perf": float(df[perf_col].mean()),
        })

    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError(f"[{method}] Loaded 0 rows from pattern: {pattern}")

    return out.sort_values("lambda").reset_index(drop=True)


# ----------------------------
# Load ratio table (VR source)
# ----------------------------
ratio = pd.read_csv(RATIO_CSV)

need_cols = {"dataset", "penalty", "lambda", "eps_if"}
missing = need_cols - set(ratio.columns)
if missing:
    raise KeyError(f"Missing columns in ratio table: {missing}. Available={list(ratio.columns)}")

# find VR column
vr_col_candidates = [c for c in ratio.columns if c.lower().startswith("vr")]
if len(vr_col_candidates) == 0:
    raise KeyError("Cannot find VR column in ratio table (expected something like 'VR_IF(ours)').")

VR_COL = None
for c in ratio.columns:
    if c.strip().lower() in ["vr_if(ours)", "vr_if", "vr(ours)", "vr"]:
        VR_COL = c
        break
if VR_COL is None:
    VR_COL = vr_col_candidates[0]

ratio["dataset"] = ratio["dataset"].astype(str).str.strip().str.lower()
ratio["method"]  = ratio["penalty"].apply(normalize_penalty)
ratio["lambda"]  = pd.to_numeric(ratio["lambda"], errors="coerce")
ratio["eps_if"]  = pd.to_numeric(ratio["eps_if"], errors="coerce")
ratio[VR_COL]    = pd.to_numeric(ratio[VR_COL], errors="coerce")

ratio = ratio.dropna(subset=["lambda", "eps_if", VR_COL]).copy()
ratio = ratio[ratio["method"].isin(METHOD_KEEP)].copy()


# ----------------------------
# Plot per dataset
# ----------------------------
os.makedirs(OUTDIR, exist_ok=True)

METHOD_LABEL = {"lap": "Combinatorial Laplacian", "urw": "URW"}
METHOD_STYLE = {
    "lap": dict(marker="o", linestyle="-"),
    "urw": dict(marker="s", linestyle="-"),
}

for dataset in DATASETS:
    # ----------------------------
    # (A) Performance vs lambda
    # ----------------------------
    for perf_col, perf_label, higher_is_better in PERF_COLS:
        metrics_all = []
        for method in METHOD_KEEP:
            pat = METRICS_GLOB.get(dataset, {}).get(method, None)
            if pat is None:
                raise KeyError(f"Missing METRICS_GLOB for dataset={dataset}, method={method}")
            metrics_all.append(load_metrics_across_lambdas(pat, method, perf_col))
        metrics_all = pd.concat(metrics_all, ignore_index=True)

        fig, ax = plt.subplots(figsize=(6.8, 4.6))

        ys_all = []
        for method in ["urw", "lap"]:
            sub = metrics_all[metrics_all["method"] == method].sort_values("lambda")
            if sub.empty:
                continue
            ax.plot(
                sub["lambda"].to_numpy(float),
                sub["perf"].to_numpy(float),
                label=METHOD_LABEL.get(method, method),
                **METHOD_STYLE.get(method, {})
            )
            ys_all.append(sub["perf"].to_numpy(float))

        ax.set_xscale("symlog", linthresh=1.0)
        ax.set_xticks([0, 1, 10, 100])
        ax.set_xticklabels(["0", "1", "10", "100"])

        ax.set_xlabel(r"Regularization strength $\lambda$")
        ax.set_ylabel(perf_label)
        ax.legend(loc="best")

        ax.set_title(f"{dataset.upper()} ({TASK}): {perf_label} vs $\\lambda$")

        # optional: nicer y-limits
        if ys_all:
            ys_all = np.concatenate(ys_all)
            ax.set_ylim(*auto_pad_limits(ys_all, pad_ratio=0.06))

        out_pdf = os.path.join(OUTDIR, f"perf_{dataset}_{TASK}_{perf_col}_vs_lambda.pdf")
        fig.savefig(out_pdf)
        plt.close(fig)
        print(f"[OK] saved: {out_pdf}")

    # ----------------------------
    # (B) VR(eps) vs lambda (4 panels)
    # ----------------------------
    r_d = ratio[ratio["dataset"] == dataset].copy()
    if r_d.empty:
        print(f"[WARN] No ratio rows for dataset={dataset}. Skipping VR plot.")
        continue

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.6), constrained_layout=True)
    axes = axes.ravel()

    for ax, eps in zip(axes, EPS_LIST):
        # tolerant match for eps
        eps_rows = r_d[np.isclose(r_d["eps_if"].values, eps, atol=1e-12, rtol=0)].copy()
        if eps_rows.empty:
            uniq = np.array(sorted(r_d["eps_if"].unique()))
            eps_used = float(uniq[np.argmin(np.abs(uniq - eps))])
            eps_rows = r_d[r_d["eps_if"] == eps_used].copy()
        else:
            eps_used = eps

        # aggregated VR vs lambda per method
        # (in case multiple seeds exist in ratio table, we average)
        vr_part = (
            eps_rows
            .groupby(["method", "lambda"], as_index=False)[VR_COL]
            .mean()
            .rename(columns={VR_COL: "vr"})
            .sort_values(["method", "lambda"])
        )

        if vr_part.empty:
            ax.set_title(rf"$\varepsilon={eps_used:g}$ (no data)")
            continue

        for method in ["urw", "lap"]:
            sub = vr_part[vr_part["method"] == method].sort_values("lambda")
            if sub.empty:
                continue
            ax.plot(
                sub["lambda"].to_numpy(float),
                sub["vr"].to_numpy(float),
                label=METHOD_LABEL.get(method, method),
                **METHOD_STYLE.get(method, {})
            )

        ax.set_xscale("symlog", linthresh=0.1)
        ax.set_xlabel(r"$\lambda$")
        ax.set_ylabel(r"Violation Rate $\mathrm{VR}(\varepsilon)$")
        ax.set_title(rf"$\varepsilon={eps_used:g}$")
        ax.set_ylim(*auto_pad_limits(vr_part["vr"].values, pad_ratio=0.10, min_range=1e-3))
        ax.legend(loc="best")

    fig.suptitle(f"{dataset.upper()} ({TASK}): Fairness Violation vs $\\lambda$ (multiple $\\varepsilon$)", y=1.02)

    out_pdf = os.path.join(OUTDIR, f"vr_{dataset}_{TASK}_vs_lambda_multieps.pdf")
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"[OK] saved: {out_pdf}")
