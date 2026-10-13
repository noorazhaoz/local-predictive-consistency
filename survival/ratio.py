#!/usr/bin/env python3

import os
import re
import glob
import numpy as np
import pandas as pd


# -------------------------
# Config
# -------------------------
RESULTS_ROOT = "results"
DEPTH = 1
DATASETS = ["flchain","whas","rossi"]
PENALTIES = ["urw", "lap"]

# eps grid
EPS_IF_LIST = "0,0.01,0.02,0.05,0.1,0.2,0.3"          # e.g. "0,0.01,0.02"
EPS_IF_MIN = 0.0
EPS_IF_MAX = 0.2
EPS_IF_NUM = 41
EPS_IF_SCALE = "linear"


# -------------------------
# Lambda parsing
# -------------------------
LAMBDA_RE = re.compile(r"_lambda(?P<lam>[0-9]+(?:\.[0-9]+)?(?:[eE][-+]?[0-9]+)?)", re.IGNORECASE)

def parse_lambda_from_filename(fname: str) -> float:
    m = LAMBDA_RE.search(fname)
    if not m:
        raise ValueError(f"Cannot parse lambda from: {fname}")
    return float(m.group("lam"))


def get_single_if_col(df: pd.DataFrame) -> str:
    if_cols = [c for c in df.columns if c.startswith("if_score") and c.endswith("_eta")]
    if len(if_cols) != 1:
        raise ValueError(f"Expected exactly one if_score_*_eta column, got {if_cols}")
    return if_cols[0]


def parse_eps_list(s: str):
    vals = [float(x.strip()) for x in s.split(",") if x.strip() != ""]
    return np.array(sorted(set(vals)), dtype=float)


def make_eps_grid():
    if EPS_IF_LIST is not None:
        return parse_eps_list(EPS_IF_LIST), f"list:{EPS_IF_LIST}"

    if EPS_IF_SCALE == "linear":
        grid = np.linspace(EPS_IF_MIN, EPS_IF_MAX, EPS_IF_NUM)
    else:
        grid = np.geomspace(EPS_IF_MIN, EPS_IF_MAX, EPS_IF_NUM)

    return grid, f"{EPS_IF_SCALE}:{EPS_IF_MIN}-{EPS_IF_MAX}-n{EPS_IF_NUM}"


def compute_vr(if_vec, eps):
    x = np.asarray(if_vec, dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return np.nan
    return float(np.mean(x > eps))


def compute_for_one_folder(folder, dataset, penalty, depth, eps_grid):
    pattern = os.path.join(folder, f"if_deepsurv_depth{depth}_lambda*.csv")
    paths = sorted(glob.glob(pattern))
    if not paths:
        return None

    rows = []
    for p in paths:
        df = pd.read_csv(p)
        if_col = get_single_if_col(df)
        if_vec = df[if_col].to_numpy(float)

        lam = parse_lambda_from_filename(os.path.basename(p))

        for eps in eps_grid:
            rows.append({
                "dataset": dataset,
                "penalty": penalty,
                "depth": depth,
                "lambda": lam,
                "eps_if": eps,
                "VR_IF(ours)": compute_vr(if_vec, eps),
                "n": int(np.sum(~np.isnan(if_vec))),
            })

    return pd.DataFrame(rows).sort_values(["lambda", "eps_if"])


# -------------------------
# RUN
# -------------------------
eps_grid, eps_spec = make_eps_grid()
all_out = []

for dataset in DATASETS:
    for penalty in PENALTIES:
        folder = os.path.join(RESULTS_ROOT, penalty, dataset)

        out_df = compute_for_one_folder(
            folder, dataset, penalty, DEPTH, eps_grid
        )
        if out_df is None:
            continue

        out_df["eps_grid_spec"] = eps_spec

        save_path = os.path.join(folder, f"vr_if_sweep_depth{DEPTH}.csv")
        os.makedirs(folder, exist_ok=True)
        out_df.to_csv(save_path, index=False)
        print("[Saved]", save_path)

        all_out.append(out_df)

if not all_out:
    raise RuntimeError("No outputs produced. Check folder paths & file names.")

merged = pd.concat(all_out, axis=0, ignore_index=True)
merged_path = os.path.join(RESULTS_ROOT, f"vr_if_sweep_ALL_depth{DEPTH}.csv")
merged.to_csv(merged_path, index=False)
print("[Saved]", merged_path)

# wide table
wide_eps = merged.pivot_table(
    index=["dataset", "penalty", "depth", "lambda"],
    columns="eps_if",
    values="VR_IF(ours)",
    aggfunc="mean"
).reset_index()

# flatten eps columns
new_cols = []
for c in wide_eps.columns:
    if isinstance(c, float):
        new_cols.append(f"vr_eps{c:g}")
    else:
        new_cols.append(str(c))
wide_eps.columns = new_cols

wide_path = os.path.join(RESULTS_ROOT, f"vr_if_sweep_ALL_depth{DEPTH}_by_eps.csv")
wide_eps.to_csv(wide_path, index=False)
print("[Saved]", wide_path)

print("\nVR sweep finished.\n")
