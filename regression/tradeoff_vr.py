#!/usr/bin/env python3
# plot_regression_rmse_and_vr_vs_lambda.py
#
# Fig1: RMSE vs lambda (lap & urw)
# Fig2: Violation Rate VR(eps) vs lambda (lap & urw), 4 eps panels

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
# Config
# ----------------------------
DATASET = "insurance"
TASK = "regression"

METRICS_GLOB = {
    "lap": "results/laplacian/metrics_insurance_laplacian_depth1_lambda*.csv",
    "urw": "results/urw/metrics_insurance_urw_depth1_lambda*.csv",
}

# performance column (lower is better)
PER_COL = "test_RMSE_real"   # <- 你原来就是这个

# VR wide-by-eps csv (must contain: method, lambda, vr_eps*)
VR_BY_EPS_CSV = "results/insurance_regression_vr_by_eps.csv"

EPS_LIST = [0.01, 0.05, 0.1,0.2]

OUTDIR = "results/plots"
OUTPDF_RMSE = os.path.join(OUTDIR, f"rmse_vs_lambda_{DATASET}_{TASK}.pdf")
OUTPNG_RMSE = os.path.join(OUTDIR, f"rmse_vs_lambda_{DATASET}_{TASK}.png")
OUTPDF_VR   = os.path.join(OUTDIR, f"vr_vs_lambda_{DATASET}_{TASK}_multieps.pdf")
OUTPNG_VR   = os.path.join(OUTDIR, f"vr_vs_lambda_{DATASET}_{TASK}_multieps.png")


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

def load_metrics_across_lambdas(pattern: str, method: str):
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"[{method}] No metrics files matched: {pattern}")

    rows = []
    for f in files:
        lam_str, lam = parse_lambda_from_filename(f)
        if lam_str is None or lam is None:
            continue

        df = pd.read_csv(f)
        if df.empty:
            continue

        if PER_COL not in df.columns:
            raise KeyError(f"[{method}] Missing '{PER_COL}' in {f}. Columns={list(df.columns)}")

        rows.append({
            "method": method,
            "lambda_str": lam_str,
            "lambda": float(lam),
            "rmse": float(df[PER_COL].mean()),
        })

    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError(f"[{method}] Loaded 0 rows from pattern: {pattern}")

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

    need_cols = {"method", "lambda"}
    if not need_cols.issubset(df.columns):
        raise KeyError(f"VR file must contain {need_cols}. Got cols={list(df.columns)[:40]}")

    cand_cols = [c for c in df.columns if c.startswith("vr_eps")]
    if not cand_cols:
        raise KeyError(f"No 'vr_eps*' columns found in {vr_csv}")

    eps_map = []
    for c in cand_cols:
        v = _parse_eps_from_col(c)
        if v is not None:
            eps_map.append((c, v))

    if not eps_map:
        raise KeyError(f"Found vr_eps columns but failed to parse eps values. cols={cand_cols[:20]}")

    col, eps_found = min(eps_map, key=lambda t: abs(t[1] - eps))
    out = df[["method", "lambda", col]].copy()
    out = out.rename(columns={col: "vr"})
    return out, col, eps_found

def auto_pad_limits(vals, pad_ratio=0.08, min_range=1e-4):
    vals = np.asarray(vals, dtype=float)
    vmin, vmax = float(np.min(vals)), float(np.max(vals))
    rng = max(vmax - vmin, min_range)
    pad = rng * pad_ratio
    return vmin - pad, vmax + pad


# ----------------------------
# Load metrics
# ----------------------------
metrics_all = []
for m, pat in METRICS_GLOB.items():
    metrics_all.append(load_metrics_across_lambdas(pat, m))
metrics_all = pd.concat(metrics_all, ignore_index=True)
metrics_all = metrics_all[metrics_all["method"].isin(["lap", "urw"])].copy()

if metrics_all.empty:
    raise RuntimeError("No metrics loaded after filtering to ['lap','urw'].")


# ----------------------------
# Fig1: RMSE vs lambda
# ----------------------------
METHOD_LABEL = {"lap": "Combinatorial Laplacian", "urw": "URW"}
METHOD_STYLE = {
    "lap": dict(marker="o", linestyle="-"),
    "urw": dict(marker="s", linestyle="-"),
}

fig1, ax1 = plt.subplots(figsize=(6.8, 4.6))

for method in ["urw", "lap"]:
    sub = metrics_all[metrics_all["method"] == method].sort_values("lambda")
    if sub.empty:
        continue

    ax1.plot(
        sub["lambda"].to_numpy(float),
        sub["rmse"].to_numpy(float),
        label=METHOD_LABEL.get(method, method),
        **METHOD_STYLE.get(method, {})
    )

ax1.set_xscale("symlog", linthresh=0.1)

ax1.set_xlabel(r"Regularization strength $\lambda$")
ax1.set_ylabel("RMSE (lower is better)")
ax1.legend(loc="best")
ax1.set_title(f"{DATASET.capitalize()} ({TASK}): RMSE vs $\\lambda$")

os.makedirs(OUTDIR, exist_ok=True)
fig1.savefig(OUTPDF_RMSE)
fig1.savefig(OUTPNG_RMSE)
plt.show()

print(f"[OK] saved: {OUTPDF_RMSE}")
print(f"[OK] saved: {OUTPNG_RMSE}")


# ----------------------------
# Fig2: VR(eps) vs lambda (4 panels)
# ----------------------------
fig2, axes = plt.subplots(2, 2, figsize=(11.2, 7.6), constrained_layout=True)
axes = axes.ravel()

for ax, eps in zip(axes, EPS_LIST):
    vr_df, used_col, eps_found = load_vr_by_eps(VR_BY_EPS_CSV, eps)
    vr_df = vr_df[vr_df["method"].isin(["lap", "urw"])].copy()

    for method in ["urw", "lap"]:
        sub = vr_df[vr_df["method"] == method].sort_values("lambda")
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

    if abs(eps_found - eps) < 1e-12:
        ax.set_title(rf"$\varepsilon={eps:g}$")
    else:
        ax.set_title(rf"$\varepsilon={eps:g}$ (used {used_col})")

    # per-panel y-limits (eps大时VR可能很小)
    if not vr_df.empty:
        y0, y1 = auto_pad_limits(vr_df["vr"].values, pad_ratio=0.10, min_range=1e-3)
        ax.set_ylim(-0.1, 1.1)

    ax.legend(loc="best")

fig2.suptitle(f"{DATASET.capitalize()} ({TASK}): Fairness Violation vs $\\lambda$ (multiple $\\varepsilon$)", y=1.02)
fig2.savefig(OUTPDF_VR)
# fig2.savefig(OUTPNG_VR)
plt.show()

print(f"[OK] saved: {OUTPDF_VR}")
# print(f"[OK] saved: {OUTPNG_VR}")
