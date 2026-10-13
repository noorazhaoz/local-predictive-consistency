#!/usr/bin/env python3
"""
compute_pairwise_folder_metrics.py

Unified independent pairwise-fairness evaluation for:
  1) WHAS DeepSurv        -> scalar prediction = eta
  2) Employers regression -> scalar prediction = pred_log

The key principle is that ALL methods are evaluated on the SAME held-out
comparison graph.  We do not compare method-specific training objectives.

Outputs per run:
    method, lambda, utility, edge_mean_abs, edge_rms, edge_max, n_edges

Supported method folders:
    --root URW=/path/to/urw
    --root Lap=/path/to/lap
    --root NLap=/path/to/nlap

The script scans folders recursively for per-instance CSVs and metrics CSVs.

Examples
--------
WHAS:
python compute_pairwise_folder_metrics.py \
    --task deepsurv \
    --dataset whas \
    --root URW=results/urw/whas \
    --root Lap=results/lap/whas \
    --root NLap=results/nlap/whas \
    --seed 8 \
    --tau 5 \
    --theta 0.001 \
    --svd_rank 1 \
    --use_metric \
    --metric_rank 64 \
    --out pairwise_whas.csv

Employers:
python compute_pairwise_folder_metrics.py \
    --task regression \
    --dataset employers \
    --data_path ../data/Employers_data.csv \
    --root URW=../regression/results/urw \
    --root Lap=../regression/results/lap \
    --root NLap=../regression/results/nrw \
    --seed 0 \
    --tau 5 \
    --theta 0.001 \
    --svd_rank 1 \
    --use_metric \
    --metric_rank 64 \
    --out pairwise_employers.csv
"""

import argparse
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD

try:
    from sksurv.datasets import load_whas500
except Exception:
    load_whas500 = None


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True, choices=["deepsurv", "regression"])
    p.add_argument("--dataset", required=True, choices=["whas", "employers"])
    p.add_argument("--depth", type=int, default=1, help="Only use result files for this model depth.")
    p.add_argument(
        "--root", action="append", required=True,
        help="METHOD=FOLDER, e.g. --root URW=results/urw/whas"
    )
    p.add_argument("--data_path", default="../data/Employers_data.csv")

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--test_size", type=float, default=None)

    p.add_argument("--theta", type=float, default=1e-3)
    p.add_argument("--tau", type=float, default=5.0)
    p.add_argument("--svd_rank", type=int, default=1)

    p.add_argument("--use_metric", action="store_true")
    p.add_argument("--metric_rank", type=int, default=64)

    p.add_argument("--out", default="pairwise_folder_metrics.csv")
    return p.parse_args()


# ---------------------------------------------------------------------
# Generic graph helpers
# ---------------------------------------------------------------------
def e_dist_sq(A, B):
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    A2 = (A ** 2).sum(axis=1, keepdims=True)
    B2 = (B ** 2).sum(axis=1, keepdims=True).T
    return np.maximum(A2 - 2.0 * (A @ B.T) + B2, 0.0)


def build_W_common(X, theta, tau, task):
    """
    Build the COMMON held-out comparison graph used to evaluate all methods.

    IMPORTANT:
    The original project uses different tau conventions across task pipelines:

      - regression:
            D2 <= tau
        where D2 is squared Euclidean distance.

      - deepsurv:
            D2 <= tau**2
        i.e. tau is interpreted as an ordinary Euclidean-distance threshold.

    We intentionally preserve those task-specific conventions here so that
    the independent pairwise evaluation uses the same comparison graph scale
    as the corresponding original experiment.

    Self-loops are included in W for compatibility, but are removed when
    extracting unique pairwise edges.
    """
    D2 = e_dist_sq(X, X)

    if task == "regression":
        mask = D2 <= float(tau)
    elif task == "deepsurv":
        mask = D2 <= float(tau) ** 2
    else:
        raise ValueError(f"Unsupported task={task}")

    W = np.exp(-float(theta) * D2) * mask.astype(np.float64)

    np.fill_diagonal(W, 0.0)
    W = 0.5 * (W + W.T)
    np.fill_diagonal(W, 1.0)
    return W


def unique_nonself_edges(W):
    A = np.asarray(W) > 0
    np.fill_diagonal(A, False)
    i, j = np.where(np.triu(A, k=1))
    return i, j


def pairwise_metrics(pred, edge_i, edge_j):
    pred = np.asarray(pred, dtype=float).ravel()
    if len(edge_i) == 0:
        return {
            "edge_mean_abs": np.nan,
            "edge_rms": np.nan,
            "edge_max": np.nan,
            "n_edges": 0,
        }

    dif = np.abs(pred[edge_i] - pred[edge_j])
    return {
        "edge_mean_abs": float(np.mean(dif)),
        "edge_rms": float(np.sqrt(np.mean(dif ** 2))),
        "edge_max": float(np.max(dif)),
        "n_edges": int(len(dif)),
    }


# ---------------------------------------------------------------------
# Representation reconstruction
# ---------------------------------------------------------------------
def fit_fair_metric_space(X_tr, X_te, sens_tr, seed, svd_rank, use_metric, metric_rank):
    """
    Reconstruct the same fair-projected / optional metric-transformed space:
      1. Standardize using TRAIN only.
      2. Estimate sensitive subspace from group prototypes.
      3. Project onto complement.
      4. Optionally learn TruncatedSVD metric on fair TRAIN representation.
    """
    scaler = StandardScaler()
    X_tr_std = scaler.fit_transform(X_tr).astype(np.float32)
    X_te_std = scaler.transform(X_te).astype(np.float32)

    d = X_tr_std.shape[1]

    mu0 = X_tr_std[sens_tr == 0].mean(axis=0, keepdims=True).astype(np.float32)
    mu1 = X_tr_std[sens_tr == 1].mean(axis=0, keepdims=True).astype(np.float32)
    prot = np.vstack([mu0, mu1]).astype(np.float32)

    r_sens = min(int(svd_rank), prot.shape[0], d)
    svd = TruncatedSVD(n_components=r_sens, random_state=seed)
    svd.fit(prot)
    basis = svd.components_.T.astype(np.float32)

    proj = (
        basis
        @ np.linalg.pinv(basis.T @ basis)
        @ basis.T
    ).astype(np.float32)

    proj_compl = np.eye(d, dtype=np.float32) - proj

    X_tr_fair = (X_tr_std @ proj_compl).astype(np.float32)
    X_te_fair = (X_te_std @ proj_compl).astype(np.float32)

    effective_metric_rank = 0
    if use_metric:
        r = min(int(metric_rank), X_tr_fair.shape[1], X_tr_fair.shape[0] - 1)
        if r < 1:
            raise ValueError("Effective metric rank < 1.")
        svd_m = TruncatedSVD(n_components=r, random_state=seed)
        svd_m.fit(X_tr_fair)
        L_metric = svd_m.components_.astype(np.float32)

        X_te_metric = (X_te_fair @ L_metric.T).astype(np.float32)
        effective_metric_rank = int(L_metric.shape[0])
    else:
        X_te_metric = X_te_fair

    return X_te_metric, effective_metric_rank


def prepare_whas_test_space(args):
    if load_whas500 is None:
        raise ImportError("scikit-survival is required for --task deepsurv --dataset whas.")

    X_raw, y_struct = load_whas500()

    df = X_raw.copy()
    df["lenfol"] = y_struct["lenfol"].astype(np.float32)
    df["fstat"] = y_struct["fstat"].astype(int)

    sens_col = "gender"
    df = df.dropna(subset=["lenfol", "fstat", sens_col]).copy()

    t_full = df["lenfol"].astype(np.float32).values
    e_full = df["fstat"].astype(int).values
    sens_full = df[sens_col].astype(int).values

    X_df = df.drop(columns=["lenfol", "fstat", sens_col], errors="ignore")
    X_df = pd.get_dummies(X_df, drop_first=True)

    # Match the DeepSurv script's median imputation behaviour.
    # Simple direct median fill is equivalent here for deterministic numeric columns.
    for c in X_df.columns:
        if X_df[c].isna().any():
            X_df[c] = X_df[c].fillna(X_df[c].median())

    X = X_df.astype(np.float32).values
    idx_all = np.arange(len(df))

    test_size = 0.2 if args.test_size is None else float(args.test_size)

    # IMPORTANT: this mirrors the train/test-only DeepSurv code:
    # train_test_split without explicit stratify.
    (
        X_tr, X_te,
        t_tr, t_te,
        e_tr, e_te,
        s_tr, s_te,
        idx_tr, idx_te
    ) = train_test_split(
        X, t_full, e_full, sens_full, idx_all,
        test_size=test_size,
        random_state=args.seed,
    )

    X_te_metric, eff_rank = fit_fair_metric_space(
        X_tr, X_te, s_tr,
        seed=args.seed,
        svd_rank=args.svd_rank,
        use_metric=args.use_metric,
        metric_rank=args.metric_rank,
    )

    return {
        "X_test_metric": X_te_metric,
        "idx_test": np.asarray(idx_te, dtype=int),
        "effective_metric_rank": eff_rank,
    }


def prepare_employers_test_space(args):
    df = pd.read_csv(args.data_path).copy()
    df = df.drop(columns=["Employee_ID", "Name"], errors="ignore")

    if "Gender" not in df.columns or "Salary" not in df.columns:
        raise ValueError("Employers dataset must contain Gender and Salary.")

    sens_full = (
        df["Gender"].astype(str).str.lower() == "female"
    ).astype(int).values

    target_raw = np.clip(
        df["Salary"].astype(np.float32).values, 1e-6, None
    )
    y_log = np.log(target_raw).astype(np.float32)

    # Match regression sensitivity code.
    df["Gender_bin"] = sens_full
    X_full_df = pd.get_dummies(
        df.drop(columns=["Gender", "Salary"], errors="ignore"),
        drop_first=True,
    ).astype(np.float32)

    if "Gender_bin" not in X_full_df.columns:
        raise ValueError("Gender_bin missing after preprocessing.")

    X_model_df = X_full_df.drop(columns=["Gender_bin"]).copy()
    X = X_model_df.values.astype(np.float32)
    indices = np.arange(len(X))

    (
        X_tr, X_tmp,
        y_tr, y_tmp,
        sens_tr, sens_tmp,
        target_tr, target_tmp,
        idx_tr, idx_tmp,
    ) = train_test_split(
        X, y_log, sens_full, target_raw, indices,
        test_size=0.4,
        random_state=args.seed,
        stratify=sens_full,
    )

    (
        X_va, X_te,
        y_va, y_te,
        sens_va, sens_te,
        target_va, target_te,
        idx_va, idx_te,
    ) = train_test_split(
        X_tmp, y_tmp, sens_tmp, target_tmp, idx_tmp,
        test_size=0.5,
        random_state=args.seed,
        stratify=sens_tmp,
    )

    X_te_metric, eff_rank = fit_fair_metric_space(
        X_tr, X_te, sens_tr,
        seed=args.seed,
        svd_rank=args.svd_rank,
        use_metric=args.use_metric,
        metric_rank=args.metric_rank,
    )

    return {
        "X_test_metric": X_te_metric,
        "idx_test": np.asarray(idx_te, dtype=int),
        "effective_metric_rank": eff_rank,
    }


# ---------------------------------------------------------------------
# Result-file discovery
# ---------------------------------------------------------------------
LAMBDA_PATTERNS = [
    re.compile(r"lambda(?:_|-)?([0-9eE+.\-]+)", re.I),
    re.compile(r"lambda_cons(?:_|-)?([0-9eE+.\-]+)", re.I),
]


def parse_lambda_from_name(path):
    name = Path(path).stem
    for pat in LAMBDA_PATTERNS:
        m = pat.search(name)
        if m:
            raw = m.group(1).rstrip("._-")
            try:
                return float(raw)
            except Exception:
                pass
    return None


def parse_root_specs(specs):
    out = {}
    for s in specs:
        if "=" not in s:
            raise ValueError(f"--root must be METHOD=PATH, got {s!r}")
        method, path = s.split("=", 1)
        method = method.strip()
        path = path.strip()
        if not method or not path:
            raise ValueError(f"Invalid --root {s!r}")
        out[method] = Path(path)
    return out


def read_first_row_csv(path):
    try:
        df = pd.read_csv(path)
        if len(df) == 0:
            return None
        return df.iloc[0].to_dict()
    except Exception:
        return None


def path_matches_dataset(p: Path, task: str, dataset: str) -> bool:
    """
    Dataset-aware file filtering.

    Regression folder convention used in this project:
      - URW/Lap/NRW roots may contain BOTH Employers and Insurance results.
      - Insurance files/paths contain "insurance".
      - Employers files may NOT contain "employers" in the filename.

    Therefore for Employers we keep regression result files that do NOT contain
    "insurance" anywhere in their relative path/name.

    For WHAS DeepSurv we keep files containing "whas" in the path when present;
    if the caller already points directly at a WHAS folder, all files pass.
    """
    low = str(p).lower().replace("\\", "/")

    if task == "regression":
        if dataset == "employers":
            return "insurance" not in low
        if dataset == "insurance":
            return "insurance" in low

    if task == "deepsurv":
        if dataset == "whas":
            # If root is already .../whas this is naturally true.
            # Otherwise exclude clearly different survival datasets.
            bad = ("flchain", "rossi", "employee")
            return not any(x in low for x in bad) or "whas" in low

    return True


def discover_metrics_files(root, task, dataset, depth=1):
    """
    Discover only metrics files for the requested depth.

    Project naming examples:
      regression URW:  metrics_urw_depth1_lambda0.3.csv
      regression Lap:  metrics_laplacian_depth1_lambda0.3.csv
      regression NLap: metrics_nlap_depth1_lambda0.3.csv
      survival files follow analogous depth1 naming.

    Files from depth-sensitivity runs (depth2, depth3, ...) are excluded.
    """
    candidates = []
    depth_tag = f"depth{int(depth)}"

    for p in root.rglob("*.csv"):
        if not path_matches_dataset(p, task, dataset):
            continue

        low = p.name.lower()

        if not low.startswith("metrics_"):
            continue
        if "lambda" not in low:
            continue
        if depth_tag not in low:
            continue
        if any(x in low for x in ["sweep", "summary", "vr_"]):
            continue

        candidates.append(p)

    return candidates


def discover_instance_files(root, task, dataset, depth=1):
    """
    Discover per-instance prediction files for the requested depth.

    Regression naming in the current folders is heterogeneous:
      URW:  urw_depth1_lambda*.csv
      Lap:  lap_depth1_lambda*.csv
      NLap: if_nlap_lambda*.csv

    Therefore:
      - URW/Lap require depth{depth} in the filename.
      - NLap if_nlap files are accepted even when depth is omitted, because
        these are the dedicated depth-1 normalized-Laplacian runs.
      - metrics/sweep/summary files are excluded.
      - Employers is distinguished by excluding files/paths containing
        "insurance".
    """
    cands = []
    depth_tag = f"depth{int(depth)}"

    for p in root.rglob("*.csv"):
        if not path_matches_dataset(p, task, dataset):
            continue

        low = p.name.lower()

        if "lambda" not in low:
            continue
        if any(x in low for x in ["metrics_", "sweep", "summary", "vr_"]):
            continue

        # Known per-instance naming patterns.
        is_urw = low.startswith("urw_")
        is_lap = low.startswith("lap_")
        is_nlap = low.startswith("if_nlap_") or low.startswith("nlap_")
        is_surv_if = low.startswith("if_deepsurv_")

        if is_urw or is_lap or is_surv_if:
            if depth_tag not in low:
                continue
            cands.append(p)
            continue

        if is_nlap:
            # Current normalized-Laplacian regression exports omit depth in
            # the per-instance filename; these are the depth-1 runs.
            if int(depth) == 1:
                cands.append(p)
            elif depth_tag in low:
                cands.append(p)

    return cands


def metric_lambda_map(root, task, dataset, depth=1):
    mapping = {}

    for p in discover_metrics_files(root, task, dataset, depth=depth):
        lam = parse_lambda_from_name(p)
        row = read_first_row_csv(p)
        if row is None:
            continue

        # Extra content-level guard when dataset metadata exists.
        ds_value = None
        for key in ["dataset", "DATASET"]:
            if key in row and pd.notna(row[key]):
                ds_value = str(row[key]).lower()
                break

        if task == "regression" and dataset == "employers":
            if ds_value is not None and "insurance" in ds_value:
                continue
        elif task == "regression" and dataset == "insurance":
            if ds_value is not None and "insurance" not in ds_value:
                continue

        # Prefer explicit lambda in CSV.
        for key in ["lambda_cons", "lambda"]:
            if key in row and pd.notna(row[key]):
                try:
                    lam = float(row[key])
                    break
                except Exception:
                    pass

        if lam is not None:
            mapping[lam] = (p, row)

    return mapping


def instance_lambda_map(root, task, dataset, depth=1):
    mapping = {}
    for p in discover_instance_files(root, task, dataset, depth=depth):
        lam = parse_lambda_from_name(p)
        if lam is not None:
            mapping[lam] = p
    return mapping


def extract_utility(metric_row, task):
    if metric_row is None:
        return np.nan, "unknown"

    if task == "deepsurv":
        for key in ["test_c_index", "c_index", "test_C_index"]:
            if key in metric_row and pd.notna(metric_row[key]):
                return float(metric_row[key]), "C-index"
    else:
        for key in ["test_RMSE_real", "test_rmse", "RMSE", "rmse"]:
            if key in metric_row and pd.notna(metric_row[key]):
                return float(metric_row[key]), "RMSE"

    return np.nan, "unknown"


def align_predictions(instance_csv, task, canonical_idx):
    """
    Read predictions from heterogeneous per-instance CSV schemas.

    Preferred alignment:
      1) orig_index / idx when they correspond to canonical dataset indices;
      2) otherwise positional row order, provided n_rows == n_test.

    This is needed because some regression IF files save only row-order
    indices (0,...,n_test-1) or omit the original dataset index.
    """
    df = pd.read_csv(instance_csv)

    # Prediction column candidates used across project scripts.
    if task == "deepsurv":
        pred_candidates = [
            "eta",
            "pred_eta",
            "risk",
            "prediction",
            "pred",
        ]
    else:
        pred_candidates = [
            "pred_log",
            "y_pred_log",
            "prediction_log",
            "pred",
            "prediction",
            "yhat_log",
        ]

    pred_col = next((c for c in pred_candidates if c in df.columns), None)

    if pred_col is None:
        # Last-resort heuristic for regression: one column containing both
        # "pred" and "log".
        if task == "regression":
            matches = [
                c for c in df.columns
                if ("pred" in c.lower() and "log" in c.lower())
            ]
            if len(matches) == 1:
                pred_col = matches[0]

    if pred_col is None:
        raise ValueError(
            f"No prediction column recognized in {instance_csv}. "
            f"Columns={list(df.columns)}"
        )

    canonical_idx = np.asarray(canonical_idx, dtype=int)

    # Try true original-index alignment first.
    for idx_col in ["orig_index", "idx"]:
        if idx_col not in df.columns:
            continue

        try:
            vals = df[idx_col].astype(int).to_numpy()
        except Exception:
            continue

        # Only use this as an original-index key if it actually contains the
        # canonical test indices. A positional 0..n-1 'idx' should NOT be
        # mistaken for original dataset indices.
        if set(canonical_idx).issubset(set(vals)):
            tmp = df[[idx_col, pred_col]].copy()
            tmp[idx_col] = tmp[idx_col].astype(int)
            tmp = tmp.drop_duplicates(subset=[idx_col], keep="first").set_index(idx_col)
            return tmp.loc[canonical_idx, pred_col].astype(float).to_numpy()

    # Some scripts write an 'index' column that is merely 0..n_test-1.
    # In that case, or when no usable original index exists, rely on the
    # saved test-set row order.
    if len(df) == len(canonical_idx):
        return df[pred_col].astype(float).to_numpy()

    raise ValueError(
        f"Cannot align {instance_csv}: rows={len(df)}, expected n_test={len(canonical_idx)}, "
        f"and no usable original-index column was found. Columns={list(df.columns)}"
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    args = parse_args()
    roots = parse_root_specs(args.root)

    if args.task == "deepsurv":
        if args.dataset != "whas":
            raise ValueError("Current DeepSurv mode is configured for WHAS.")
        space = prepare_whas_test_space(args)
    else:
        if args.dataset != "employers":
            raise ValueError("Current regression mode is configured for employers.")
        space = prepare_employers_test_space(args)

    X_test_metric = space["X_test_metric"]
    canonical_idx = space["idx_test"]

    W = build_W_common(
        X_test_metric,
        theta=args.theta,
        tau=args.tau,
        task=args.task,
    )
    edge_i, edge_j = unique_nonself_edges(W)

    print("=" * 72)
    print("COMMON TEST COMPARISON GRAPH")
    print(f"task={args.task}")
    print(f"dataset={args.dataset}")
    print(f"depth={args.depth}")
    print(f"n_test={len(canonical_idx)}")
    print(f"n_edges={len(edge_i)}")
    print(f"tau={args.tau}")
    print(f"theta={args.theta}")
    print(f"use_metric={args.use_metric}")
    print(f"requested_metric_rank={args.metric_rank}")
    print(f"effective_metric_rank={space['effective_metric_rank']}")
    print("=" * 72)

    rows = []

    for method, root in roots.items():
        if not root.exists():
            print(f"[WARN] Missing folder for {method}: {root}")
            continue

        inst_map = instance_lambda_map(root, args.task, args.dataset, depth=args.depth)
        met_map = metric_lambda_map(root, args.task, args.dataset, depth=args.depth)

        print(
            f"[{method}] found {len(inst_map)} per-instance runs, "
            f"{len(met_map)} metrics runs"
        )

        for lam in sorted(inst_map):
            inst_path = inst_map[lam]

            try:
                pred = align_predictions(
                    inst_path,
                    task=args.task,
                    canonical_idx=canonical_idx,
                )
            except Exception as e:
                print(f"[WARN] {method} lambda={lam}: {e}")
                continue

            pm = pairwise_metrics(pred, edge_i, edge_j)

            metric_row = None
            metric_path = None
            if lam in met_map:
                metric_path, metric_row = met_map[lam]

            utility, utility_name = extract_utility(metric_row, args.task)

            row = {
                "dataset": args.dataset,
                "task": args.task,
                "method": method,
                "depth": int(args.depth),
                "lambda": float(lam),
                "utility_name": utility_name,
                "utility": utility,
                "edge_mean_abs": pm["edge_mean_abs"],
                "edge_rms": pm["edge_rms"],
                "edge_max": pm["edge_max"],
                "n_edges": pm["n_edges"],
                "n_test": len(canonical_idx),
                "tau": float(args.tau),
                "theta": float(args.theta),
                "use_metric": bool(args.use_metric),
                "effective_metric_rank": int(space["effective_metric_rank"]),
                "instance_csv": str(inst_path),
                "metrics_csv": str(metric_path) if metric_path else "",
            }
            rows.append(row)

    out = pd.DataFrame(rows)

    if len(out) == 0:
        raise RuntimeError(
            "No runs were evaluated. Check folder paths and per-instance CSV names."
        )

    # Sort for easy visual inspection.
    out = out.sort_values(["method", "lambda"]).reset_index(drop=True)
    out.to_csv(args.out, index=False)

    print("\nRESULTS")
    display_cols = [
        "method", "lambda", "utility_name", "utility",
        "edge_mean_abs", "edge_rms", "edge_max", "n_edges"
    ]
    print(out[display_cols].to_string(index=False))
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
