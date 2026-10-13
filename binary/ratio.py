import os
import re
import numpy as np
import pandas as pd
from typing import Optional, Dict, List

# ------------------------------------------------------------
# Parse ..._lambda{X}.csv at the end
# ------------------------------------------------------------
LAMBDA_RE = re.compile(r"_lambda(?P<lam>[0-9.eE+-]+)\.csv$", re.IGNORECASE)


def _parse_lambda(fname: str):
    m = LAMBDA_RE.search(fname)
    if not m:
        return None, None
    lam_str = m.group("lam")
    try:
        lam_val = float(lam_str)
    except ValueError:
        lam_val = None
    return lam_str, lam_val


def _compute_vr(if_scores: np.ndarray, eps: float):
    """
    VR(eps) = fraction of instances with IF > eps
    """
    x = np.asarray(if_scores, dtype=float)
    x = x[~np.isnan(x)]
    n = int(x.size)
    if n == 0:
        return np.nan, 0
    return float(np.mean(x > eps)), n


def _as_list(x):
    if isinstance(x, str):
        return [x]
    return list(x)


def _match_files(all_csv: List[str], prefixes) -> List[str]:
    prefixes = _as_list(prefixes)
    files = []
    for pref in prefixes:
        files.extend([f for f in all_csv if f.startswith(pref)])
    return sorted(set(files))


def compute_vr_tables_binary(
    results_dir: str,
    eps_list,
    *,
    out_prefix: str,
    save: bool = True,
    strict: bool = False,
    include_file: bool = True,
    method_specs: Optional[Dict] = None,
):
    """
    Compute VR tables for a SINGLE evaluation protocol.

    Example:
      - forward split: {lap, urw_forward}
      - reverse split: {lap, urw_reverse}
    """

    if method_specs is None:
        raise ValueError("method_specs must be provided explicitly for forward/reverse split.")

    if not os.path.isdir(results_dir):
        raise FileNotFoundError(f"results_dir not found: {results_dir}")

    eps_list = [float(e) for e in eps_list]
    all_csv = sorted([f for f in os.listdir(results_dir) if f.endswith(".csv")])

    rows = []
    skipped = []
    matched_files = {m: [] for m in method_specs.keys()}

    for method, spec in method_specs.items():
        prefixes = spec["prefix"]
        if_candidates = _as_list(spec["if_col"])
        idx_col = spec.get("index_col", "orig_index")

        files = _match_files(all_csv, prefixes)
        matched_files[method] = files

        for fname in files:
            lam_str, lam_val = _parse_lambda(fname)
            if lam_str is None:
                skipped.append((fname, "cannot-parse-lambda"))
                continue

            path = os.path.join(results_dir, fname)
            df = pd.read_csv(path)

            picked_if = next((c for c in if_candidates if c in df.columns), None)
            if picked_if is None:
                msg = f"missing IF column. tried={if_candidates}"
                if strict:
                    raise KeyError(f"{fname}: {msg}. Columns={list(df.columns)}")
                skipped.append((fname, msg))
                continue

            if idx_col not in df.columns:
                skipped.append((fname, f"missing index col: {idx_col} (VR still computed)"))

            if_scores = df[picked_if].to_numpy()

            for eps in eps_list:
                vr, n = _compute_vr(if_scores, eps)
                row = {
                    "task": "binary",
                    "protocol": out_prefix,
                    "method": method,
                    "lambda_str": lam_str,
                    "lambda": lam_val,
                    "eps": eps,
                    "vr": vr,
                    "n": n,
                }
                if include_file:
                    row["file"] = fname
                    row["if_col_used"] = picked_if
                rows.append(row)

    out_long = pd.DataFrame(rows)
    if out_long.empty:
        raise RuntimeError(
            "No VR rows produced.\n"
            f"Checked dir: {results_dir}\n"
            f"CSV count: {len(all_csv)}\n"
            f"Matched files: {matched_files}\n"
            f"Skipped examples: {skipped[:15]}"
        )

    out_long = out_long.sort_values(
        by=["method", "lambda", "lambda_str", "eps"],
        kind="mergesort"
    ).reset_index(drop=True)

    # Wide 1: rows=lambda, cols=(method, eps)
    wide_method = out_long.pivot_table(
        index=["lambda_str", "lambda"],
        columns=["method", "eps"],
        values="vr",
        aggfunc="mean"
    ).reset_index()

    new_cols = []
    for c in wide_method.columns:
        if isinstance(c, tuple):
            a, b = c
            if a in ("lambda_str", "lambda"):
                new_cols.append(a)
            else:
                new_cols.append(f"{a}_eps{b}")
        else:
            new_cols.append(str(c))
    wide_method.columns = new_cols

    # Wide 2: rows=(method, lambda), cols=eps
    wide_eps = out_long.pivot_table(
        index=["method", "lambda_str", "lambda"],
        columns=["eps"],
        values="vr",
        aggfunc="mean"
    ).reset_index()
    wide_eps.columns = [
        f"vr_eps{c}" if isinstance(c, float) else str(c)
        for c in wide_eps.columns
    ]

    debug = {
        "matched_files": matched_files,
        "skipped": skipped,
        "all_csv_count": len(all_csv),
    }

    if save:
        os.makedirs(results_dir, exist_ok=True)
        p_long = os.path.join(results_dir, f"{out_prefix}_vr_long.csv")
        p_wm = os.path.join(results_dir, f"{out_prefix}_vr_by_method.csv")
        p_we = os.path.join(results_dir, f"{out_prefix}_vr_by_eps.csv")
        out_long.to_csv(p_long, index=False)
        wide_method.to_csv(p_wm, index=False)
        wide_eps.to_csv(p_we, index=False)

    return out_long, wide_method, wide_eps, debug


def compute_pairwise_ratio_table(
    out_long: pd.DataFrame,
    *,
    numerator_method: str,
    denominator_method: str,
    ratio_name: Optional[str] = None,
    save_path: Optional[str] = None,
):
    """
    Build ratio table:
        ratio = VR(numerator_method) / VR(denominator_method)

    Matched on (lambda_str, lambda, eps).
    """

    if ratio_name is None:
        ratio_name = f"{numerator_method}_over_{denominator_method}"

    need_cols = {"method", "lambda_str", "lambda", "eps", "vr"}
    missing = need_cols - set(out_long.columns)
    if missing:
        raise KeyError(f"out_long missing columns required for ratio: {missing}")

    num = out_long.loc[
        out_long["method"] == numerator_method,
        ["lambda_str", "lambda", "eps", "vr"]
    ].copy()

    den = out_long.loc[
        out_long["method"] == denominator_method,
        ["lambda_str", "lambda", "eps", "vr"]
    ].copy()

    num = num.rename(columns={"vr": f"vr_{numerator_method}"})
    den = den.rename(columns={"vr": f"vr_{denominator_method}"})

    ratio_df = num.merge(
        den,
        on=["lambda_str", "lambda", "eps"],
        how="inner",
        validate="one_to_one",
    )

    den_col = f"vr_{denominator_method}"
    num_col = f"vr_{numerator_method}"

    ratio_df[ratio_name] = np.where(
        ratio_df[den_col].notna() & (ratio_df[den_col] != 0),
        ratio_df[num_col] / ratio_df[den_col],
        np.nan,
    )

    ratio_df = ratio_df.sort_values(
        by=["lambda", "lambda_str", "eps"],
        kind="mergesort"
    ).reset_index(drop=True)

    ratio_wide = ratio_df.pivot_table(
        index=["lambda_str", "lambda"],
        columns="eps",
        values=ratio_name,
        aggfunc="mean"
    ).reset_index()

    ratio_wide.columns = [
        f"{ratio_name}_eps{c}" if isinstance(c, float) else str(c)
        for c in ratio_wide.columns
    ]

    if save_path is not None:
        ratio_df.to_csv(save_path, index=False)

    return ratio_df, ratio_wide


def build_method_specs(dataset_name: str):
    """
    dataset_name: 'adult' or 'compas'
    Adjust prefixes only; everything else stays the same.
    """
    forward_specs = {
        "lap": {
            "prefix": [
                f"if_{dataset_name}_depth1_lambda",
            ],
            "if_col": [
                "if_score_forwardKL_prob",
            ],
            "index_col": "orig_index",
        },

        "urw_forward": {
            "prefix": [
                f"if_{dataset_name}_aggforward_depth1_lambda",
            ],
            "if_col": [
                "if_score_forwardKL_node||ref",
            ],
            "index_col": "orig_index",
        },
    }

    reverse_specs = {
        "lap": {
            "prefix": [
                f"if_{dataset_name}_depth1_lambda",
            ],
            "if_col": [
                "if_score_reverseKL_ref||node",
            ],
            "index_col": "orig_index",
        },

        "urw_reverse": {
            "prefix": [
                f"if_{dataset_name}_aggreverse_depth1_lambda",
            ],
            "if_col": [
                "if_score_reverseKL_ref||node",
            ],
            "index_col": "orig_index",
        },
    }

    return forward_specs, reverse_specs


def run_one_dataset(results_dir: str, dataset_name: str, eps_list):
    forward_specs, reverse_specs = build_method_specs(dataset_name)

    # -----------------------------
    # Forward split
    # -----------------------------
    out_long_fwd, wide_method_fwd, wide_eps_fwd, debug_fwd = compute_vr_tables_binary(
        results_dir=results_dir,
        eps_list=eps_list,
        out_prefix=f"{dataset_name}_forward_split",
        save=True,
        include_file=True,
        strict=False,
        method_specs=forward_specs,
    )

    ratio_long_fwd, ratio_wide_fwd = compute_pairwise_ratio_table(
        out_long_fwd,
        numerator_method="urw_forward",
        denominator_method="lap",
        ratio_name="urw_forward_over_lap",
        save_path=os.path.join(results_dir, f"{dataset_name}_forward_split_ratio_long.csv"),
    )
    ratio_wide_fwd.to_csv(
        os.path.join(results_dir, f"{dataset_name}_forward_split_ratio_by_eps.csv"),
        index=False
    )

    # -----------------------------
    # Reverse split
    # -----------------------------
    out_long_rev, wide_method_rev, wide_eps_rev, debug_rev = compute_vr_tables_binary(
        results_dir=results_dir,
        eps_list=eps_list,
        out_prefix=f"{dataset_name}_reverse_split",
        save=True,
        include_file=True,
        strict=False,
        method_specs=reverse_specs,
    )

    ratio_long_rev, ratio_wide_rev = compute_pairwise_ratio_table(
        out_long_rev,
        numerator_method="urw_reverse",
        denominator_method="lap",
        ratio_name="urw_reverse_over_lap",
        save_path=os.path.join(results_dir, f"{dataset_name}_reverse_split_ratio_long.csv"),
    )
    ratio_wide_rev.to_csv(
        os.path.join(results_dir, f"{dataset_name}_reverse_split_ratio_by_eps.csv"),
        index=False
    )

    # -----------------------------
    # Prints
    # -----------------------------
    print("\n" + "=" * 100)
    print(f"DATASET: {dataset_name.upper()}")
    print("=" * 100)

    print("\nFORWARD SPLIT")
    print("-" * 100)
    print("Matched files:")
    print(debug_fwd["matched_files"])
    print("\nSkipped (first 20):")
    print(debug_fwd["skipped"][:20])
    print("\nForward VR by eps:")
    print(wide_eps_fwd.head())
    print("\nForward ratio by eps:")
    print(ratio_wide_fwd.head())

    print("\nREVERSE SPLIT")
    print("-" * 100)
    print("Matched files:")
    print(debug_rev["matched_files"])
    print("\nSkipped (first 20):")
    print(debug_rev["skipped"][:20])
    print("\nReverse VR by eps:")
    print(wide_eps_rev.head())
    print("\nReverse ratio by eps:")
    print(ratio_wide_rev.head())

    print("\nSaved files:")
    print(f"  - {dataset_name}_forward_split_vr_long.csv")
    print(f"  - {dataset_name}_forward_split_vr_by_method.csv")
    print(f"  - {dataset_name}_forward_split_vr_by_eps.csv")
    print(f"  - {dataset_name}_forward_split_ratio_long.csv")
    print(f"  - {dataset_name}_forward_split_ratio_by_eps.csv")
    print(f"  - {dataset_name}_reverse_split_vr_long.csv")
    print(f"  - {dataset_name}_reverse_split_vr_by_method.csv")
    print(f"  - {dataset_name}_reverse_split_vr_by_eps.csv")
    print(f"  - {dataset_name}_reverse_split_ratio_long.csv")
    print(f"  - {dataset_name}_reverse_split_ratio_by_eps.csv")


# ------------------------------------------------------------
# Main run
# ------------------------------------------------------------
if __name__ == "__main__":
    results_dir = "results_unified"
    eps_list = [0.0003, 0.0008, 0.001, 0.003,0.01,0.05,0.1]

    datasets = ["adult", "compas"]

    for ds in datasets:
        run_one_dataset(
            results_dir=results_dir,
            dataset_name=ds,
            eps_list=eps_list,
        )

    print("\n" + "=" * 100)
    print("ALL DONE")
    print("=" * 100)
    print("Saved all files into:", results_dir)