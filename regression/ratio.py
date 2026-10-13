#!/usr/bin/env python3
# vr_employee_regression_lib.py
#
# Library-style VR computation (no CLI, no __main__).
#
# VR(eps) = mean_i 1[ IF_i > eps ]
#
import os
import re
import numpy as np
import pandas as pd
from typing import Dict, Any, Iterable, Tuple, List, Optional

# -----------------------------
# Default method specs (edit / extend)
# -----------------------------
DEFAULT_METHOD_SPECS: Dict[str, Dict[str, Any]] = {
    "lap": {
        "subdir": "laplacian",
        "prefix": "insurance_lap_depth1_lambda",
        "if_col": "if_score_urw_log",
        "index_col": "index",
    },
    # "nrw": {
    #     "subdir": "nrw",
    #     "prefix": "nrw_depth1_lambda",
    #     "if_col": "if_score_nrw_log",
    #     "index_col": "index",
    # },
    "urw": {
        "subdir": "urw",
        "prefix": "insurance_urw_depth1_lambda",
        "if_col": "if_score_urw_log",
        "index_col": "index",
    },
#     # 你要算 p-norm：把下面这块改成你的真实命名即可（先留模板）
#     "pnorm": {
#         "subdir": "",                 # e.g., "pnorm" or "urw_pnorm"
#         "prefix": "urw_p",   # e.g., "urw_pnorm_depth1_lambda"
#         "if_col": "if_score_urw_log",    # e.g., "if_score_urw_pnorm_log"
#         "index_col": "index",
#     },
}

LAMBDA_RE = re.compile(r"_lambda(?P<lam>[0-9.eE+-]+)\.csv$", re.IGNORECASE)

def parse_lambda(fname: str) -> Tuple[Optional[str], Optional[float]]:
    m = LAMBDA_RE.search(fname)
    if not m:
        return None, None
    lam_str = m.group("lam")
    try:
        lam_val = float(lam_str)
    except ValueError:
        lam_val = None
    return lam_str, lam_val

def compute_vr_from_scores(if_scores: np.ndarray, eps: float) -> Tuple[float, int]:
    x = np.asarray(if_scores, dtype=float)
    x = x[~np.isnan(x)]
    n = int(x.size)
    if n == 0:
        return np.nan, 0
    return float(np.mean(x > eps)), n

def iter_csv_files(root_dir: str) -> Iterable[str]:
    for dirpath, _, filenames in os.walk(root_dir):
        for fn in filenames:
            if fn.lower().endswith(".csv"):
                yield os.path.join(dirpath, fn)

def _collect_candidate_files(method_dir: str, prefix: str) -> List[str]:
    cands = []
    for path in iter_csv_files(method_dir):
        fn = os.path.basename(path)
        low = fn.lower()
        # skip obvious outputs
        if low.startswith("metrics") or "_vr_" in low:
            continue
        if not fn.startswith(prefix):
            continue
        if LAMBDA_RE.search(fn) is None:
            continue
        cands.append(path)
    return sorted(cands)

def compute_vr_tables(
    results_dir: str = "results_pnorm",
    eps_list: Iterable[float] = (0.01, 0.02, 0.05),
    method_specs: Optional[Dict[str, Dict[str, Any]]] = None,
    out_prefix: str = "insurance_regression",
    strict: bool = False,
    include_file: bool = False,
    dataset: str = "employee",
    task: str = "regression",
    save: bool = True,
) -> Dict[str, Any]:
    """
    Compute VR tables for given eps values across methods/lambdas.

    Returns dict with:
      - long_df, wide_by_method_df, wide_by_eps_df
      - saved_paths (if save=True)
      - skipped (list of (item, reason))
    """
    if method_specs is None:
        method_specs = DEFAULT_METHOD_SPECS

    results_dir = os.path.abspath(results_dir)
    if not os.path.isdir(results_dir):
        raise FileNotFoundError(f"results_dir not found: {results_dir}")

    eps_list = [float(e) for e in eps_list]
    rows = []
    skipped = []

    for method, spec in method_specs.items():
        subdir = spec["subdir"]
        pref = spec["prefix"]
        if_col = spec["if_col"]
        idx_col = spec.get("index_col", None)

        method_dir = os.path.join(results_dir, subdir)
        if not os.path.isdir(method_dir):
            skipped.append((method_dir, "missing method subdir"))
            continue

        candidates = _collect_candidate_files(method_dir, pref)

        for path in candidates:
            fn = os.path.basename(path)
            lam_str, lam_val = parse_lambda(fn)
            if lam_str is None:
                skipped.append((path, "cannot-parse-lambda"))
                continue

            df = pd.read_csv(path)

            if if_col not in df.columns:
                msg = f"missing IF column: {if_col}"
                if strict:
                    raise KeyError(f"{path}: {msg}. Columns={list(df.columns)}")
                skipped.append((path, msg))
                continue

            if idx_col is not None and idx_col not in df.columns:
                skipped.append((path, f"missing index col: {idx_col} (VR still computed)"))

            if_scores = df[if_col].to_numpy()

            for eps in eps_list:
                vr, n = compute_vr_from_scores(if_scores, eps)
                r = {
                    "dataset": dataset,
                    "task": task,
                    "method": method,
                    "lambda_str": lam_str,
                    "lambda": lam_val,
                    "eps": float(eps),
                    "vr": vr,
                    "n": n,
                }
                if include_file:
                    r["file"] = os.path.relpath(path, results_dir)
                rows.append(r)

    long_df = pd.DataFrame(rows)
    if long_df.empty:
        raise RuntimeError(
            "No VR rows produced.\n"
            f"Checked dir: {results_dir}\n"
            f"Skipped examples: {skipped[:15]}"
        )

    long_df = long_df.sort_values(
        by=["method", "lambda", "lambda_str", "eps"],
        kind="mergesort"
    ).reset_index(drop=True)

    # wide: one row per lambda, columns=method x eps
    wide_by_method_df = long_df.pivot_table(
        index=["lambda_str", "lambda"],
        columns=["method", "eps"],
        values="vr",
        aggfunc="mean"
    ).reset_index()

    # flatten columns
    new_cols = []
    for c in wide_by_method_df.columns:
        if isinstance(c, tuple):
            m, e = c
            if m in ("lambda_str", "lambda"):
                new_cols.append(m)
            else:
                new_cols.append(f"{m}_eps{e}")
        else:
            new_cols.append(c)
    wide_by_method_df.columns = new_cols

    # wide: one row per (method, lambda), columns=eps
    wide_by_eps_df = long_df.pivot_table(
        index=["method", "lambda_str", "lambda"],
        columns=["eps"],
        values="vr",
        aggfunc="mean"
    ).reset_index()

    wide_by_eps_df.columns = [
        f"vr_eps{c}" if isinstance(c, float) else str(c)
        for c in wide_by_eps_df.columns
    ]

    saved_paths = {}
    if save:
        p_long = os.path.join(results_dir, f"{out_prefix}_vr_long.csv")
        p_wm   = os.path.join(results_dir, f"{out_prefix}_vr_by_method.csv")
        p_we   = os.path.join(results_dir, f"{out_prefix}_vr_by_eps.csv")

        long_df.to_csv(p_long, index=False)
        wide_by_method_df.to_csv(p_wm, index=False)
        wide_by_eps_df.to_csv(p_we, index=False)

        saved_paths = {"long": p_long, "by_method": p_wm, "by_eps": p_we}

    return {
        "long_df": long_df,
        "wide_by_method_df": wide_by_method_df,
        "wide_by_eps_df": wide_by_eps_df,
        "saved_paths": saved_paths,
        "skipped": skipped,
    }
res = compute_vr_tables(
    results_dir="results",
    eps_list=[0.001,0.01,0.05,0.1,0.2,0.3],
    out_prefix="insurance_regression",
    strict=True,
    include_file=True,
    save=True,
)

print(res["saved_paths"])
print("skipped:", len(res["skipped"]))
res["wide_by_method_df"].head()
