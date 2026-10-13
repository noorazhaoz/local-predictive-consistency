#!/usr/bin/env python3
"""
compute_waller_binary_metrics.py

Post-hoc Waller-style local consistency metrics for saved binary-classification
prediction/IF CSVs (Adult / COMPAS).

No model retraining is required.

Primary definition
------------------
For each test instance i and its k nearest neighbors N_k(i),

    consistency_i
      = 1 - | yhat_i - (1/k) * sum_{j in N_k(i)} yhat_j |,

where yhat is the HARD binary decision.

For binary labels this is exactly the fraction of the k neighbors receiving
the same predicted class as the focal instance.

From consistency_i = c_i, this script reports:

    mean_consistency  = mean_i c_i
    mean_inconsistency = mean_i (1 - c_i)

    LICC_delta = # { i : c_i < delta }

    PCS_delta  = # { i : c_i >= delta } / n

    BCC_delta  = (1/n) * sum_i [ c_i if c_i >= delta else 0 ]

    BCC_penalty_delta
               = (1/n) * sum_i [ c_i if c_i >= delta else penalty ]

Default:
    delta = 0.5
    penalty = -1
    k in {5,10,15}

Expected saved CSV columns
--------------------------
The binary URW code used in this project saves per-instance files containing
at least:

    orig_index
    y_true
    pred_prob
    pred_label

and raw feature columns prefixed by "raw_".

It may additionally contain:
    if_score_urw_monitor_KL_p||Pp
    if_score_forwardKL_node||ref
    if_score_reverseKL_ref||node

The script discovers these files recursively.

Dataset recognition
-------------------
Adult:
    usually contains raw_relationship_* columns

COMPAS:
    usually contains raw_c_charge_degree_* columns

The script uses --dataset to filter candidates. If filenames contain the
dataset name, that is used; otherwise the raw feature schema is inspected.

Distance space
--------------
By default, kNN is computed in the saved raw feature representation after:
    - dropping target/index fields;
    - optionally excluding sensitive sex_* columns (default);
    - standardizing all retained columns within the held-out cohort.

This is a post-hoc metric comparison. If you later want to reproduce a
specific alternative distance definition from Waller et al. exactly, use the
--distance option / adjust build_knn_features() accordingly.

Examples
--------
Adult:
python compute_waller_binary_metrics.py \
    --root results_unified \
    --dataset adult \
    --k 5,10,15 \
    --delta 0.5 \
    --distance euclidean \
    --out waller_adult_summary.csv \
    --instance_out waller_adult_instances.csv

COMPAS:
python compute_waller_binary_metrics.py \
    --root results_unified \
    --dataset compas \
    --k 5,10,15 \
    --delta 0.5 \
    --distance euclidean \
    --out waller_compas_summary.csv \
    --instance_out waller_compas_instances.csv
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--root",
        required=True,
        help="Folder containing saved per-instance binary CSVs; scanned recursively.",
    )
    p.add_argument(
        "--dataset",
        required=True,
        choices=["adult", "compas"],
    )

    p.add_argument(
        "--k",
        default="5,10,15",
        help="Comma-separated k values. Default: 5,10,15",
    )
    p.add_argument("--delta", type=float, default=0.5)
    p.add_argument("--penalty", type=float, default=-1.0)

    p.add_argument(
        "--distance",
        choices=["euclidean", "manhattan"],
        default="euclidean",
    )

    p.add_argument(
        "--include_sensitive",
        action="store_true",
        help=(
            "Include raw_sex_* columns in the kNN feature space. "
            "Default is to exclude them."
        ),
    )

    p.add_argument(
        "--no_standardize",
        action="store_true",
        help="Do not standardize retained kNN features.",
    )

    p.add_argument(
        "--out",
        default="waller_binary_summary.csv",
    )
    p.add_argument(
        "--instance_out",
        default="waller_binary_instances.csv",
    )

    return p.parse_args()


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def parse_float_list(s):
    return [int(float(x.strip())) for x in s.split(",") if x.strip()]


def parse_lambda(path: Path):
    low = path.stem.lower()

    patterns = [
        r"lambda(?:_|-)?([0-9eE+.\-]+)",
        r"lam(?:_|-)?([0-9eE+.\-]+)",
    ]

    for pat in patterns:
        m = re.search(pat, low)
        if m:
            raw = m.group(1).rstrip("._-")
            try:
                return float(raw)
            except ValueError:
                pass

    return np.nan


def parse_method(path: Path, df: pd.DataFrame):
    """
    Infer FKL/RKL/baseline from filename first, then saved IF columns.
    """
    low = str(path).lower()

    if "aggforward" in low:
        return "URW-FKL"

    if "aggreverse" in low:
        return "URW-RKL"

    if "forward" in low or "fkl" in low:
        return "URW-FKL"

    if "reverse" in low or "rkl" in low:
        return "URW-RKL"

    cols = set(df.columns)

    if "if_score_reverseKL_ref||node" in cols:
        # If both method-specific columns exist, filename is preferred.
        # A reverse-only saved output normally contains this column.
        if "if_score_forwardKL_node||ref" not in cols:
            return "URW-RKL"

    if "if_score_forwardKL_node||ref" in cols:
        return "URW-FKL"

    return "URW"


def looks_like_dataset(path: Path, df: pd.DataFrame, dataset: str):
    """
    Dataset filtering that works even when the dataset name is absent from
    the filename.
    """
    low = str(path).lower()

    # Strong filename/path evidence.
    if "adult" in low:
        return dataset == "adult"
    if "compas" in low:
        return dataset == "compas"

    cols = [c.lower() for c in df.columns]

    # Raw schema evidence.
    adult_markers = (
        "raw_relationship_",
        "raw_marital-status_",
        "raw_occupation_",
        "raw_workclass_",
    )
    compas_markers = (
        "raw_c_charge_degree_",
        "raw_priors_count",
        "raw_juv_fel_count",
        "raw_juv_misd_count",
    )

    is_adult = any(any(c.startswith(m) for m in adult_markers) for c in cols)
    is_compas = any(any(c.startswith(m) for m in compas_markers) for c in cols)

    if dataset == "adult":
        return is_adult and not is_compas

    return is_compas


def discover_instance_files(root: Path, dataset: str):
    """
    Discover saved per-instance prediction CSVs by required columns rather
    than relying on one exact filename convention.
    """
    out = []

    for p in root.rglob("*.csv"):
        low = p.name.lower()

        # Keep only the canonical binary outputs:
        # filenames explicitly tagged with aggforward / aggreverse.
        if ("aggforward" not in low) and ("aggreverse" not in low):
            continue

        # Exclude aggregate result files.
        if any(x in low for x in ["metrics_", "summary", "sweep", "pareto"]):
            continue

        try:
            df0 = pd.read_csv(p, nrows=3)
        except Exception:
            continue

        required = {"pred_prob", "y_true"}
        if not required.issubset(df0.columns):
            continue

        raw_cols = [c for c in df0.columns if c.startswith("raw_")]
        if len(raw_cols) == 0:
            continue

        if not looks_like_dataset(p, df0, dataset):
            continue

        out.append(p)

    return sorted(out)


def build_knn_features(
    df: pd.DataFrame,
    include_sensitive: bool,
    standardize: bool,
):
    raw_cols = [c for c in df.columns if c.startswith("raw_")]

    if not raw_cols:
        raise ValueError("No raw_* feature columns found.")

    exclude_exact = {
        "raw_y",
        "raw_orig_index",
        "raw_two_year_recid",
    }

    feat_cols = []

    for c in raw_cols:
        if c in exclude_exact:
            continue

        cl = c.lower()

        # Avoid accidental target leakage.
        if cl in {
            "raw_y_true",
            "raw_pred_prob",
            "raw_pred_label",
        }:
            continue

        if not include_sensitive and cl.startswith("raw_sex_"):
            continue

        feat_cols.append(c)

    if not feat_cols:
        raise ValueError("No usable raw feature columns remain.")

    Xdf = df[feat_cols].copy()

    # Saved raw feature tables should already be numeric after get_dummies,
    # but make this robust.
    for c in Xdf.columns:
        Xdf[c] = pd.to_numeric(Xdf[c], errors="coerce")

    # Cohort-level median fill only if necessary.
    if Xdf.isna().any().any():
        Xdf = Xdf.fillna(Xdf.median(numeric_only=True)).fillna(0)

    X = Xdf.to_numpy(dtype=np.float64)

    if standardize:
        X = StandardScaler().fit_transform(X)

    return X, feat_cols


def compute_knn_indices(X, k, metric):
    n = len(X)

    if n <= k:
        raise ValueError(f"k={k} requires more than {k} instances; got n={n}.")

    # n_neighbors=k+1 because each point is its own nearest neighbor.
    nn = NearestNeighbors(
        n_neighbors=k + 1,
        metric=metric,
        algorithm="auto",
    )
    nn.fit(X)

    _, inds = nn.kneighbors(X, return_distance=True)

    # Remove self robustly rather than simply dropping column 0.
    nbrs = np.empty((n, k), dtype=int)

    for i in range(n):
        row = [j for j in inds[i].tolist() if j != i]
        if len(row) < k:
            raise RuntimeError(
                f"Could not obtain {k} non-self neighbors for row {i}."
            )
        nbrs[i] = np.asarray(row[:k], dtype=int)

    return nbrs


def waller_consistency(pred_label, nbrs):
    pred_label = np.asarray(pred_label, dtype=int).ravel()

    nbr_labels = pred_label[nbrs]

    # Original formula:
    # 1 - | yhat_i - mean_j yhat_j |
    c_formula = 1.0 - np.abs(
        pred_label.astype(float) - nbr_labels.mean(axis=1)
    )

    # For binary hard labels this must equal neighbor agreement rate.
    c_agreement = (nbr_labels == pred_label[:, None]).mean(axis=1)

    if not np.allclose(c_formula, c_agreement, atol=1e-12):
        raise RuntimeError(
            "Binary consistency formula and agreement-rate calculation differ."
        )

    return c_formula


def aggregate_waller(c, delta, penalty):
    c = np.asarray(c, dtype=float)
    pass_mask = c >= float(delta)

    licc = int(np.sum(~pass_mask))
    pcs = float(np.mean(pass_mask))

    bcc_values = np.where(pass_mask, c, 0.0)
    bcc = float(np.mean(bcc_values))

    bcc_pen_values = np.where(pass_mask, c, float(penalty))
    bcc_penalty = float(np.mean(bcc_pen_values))

    return {
        "mean_consistency": float(np.mean(c)),
        "mean_inconsistency": float(np.mean(1.0 - c)),
        "LICC": licc,
        "LICC_rate": float(licc / max(len(c), 1)),
        "PCS": pcs,
        "BCC": bcc,
        "BCC_penalty": bcc_penalty,
    }


def get_urw_if(df, method):
    """
    Return a saved URW instance-level IF vector when available.
    """
    candidates = []

    if method == "URW-FKL":
        candidates += [
            "if_score_forwardKL_node||ref",
            "if_score_urw_monitor_KL_p||Pp",
        ]
    elif method == "URW-RKL":
        candidates += [
            "if_score_reverseKL_ref||node",
            "if_score_urw_monitor_KL_p||Pp",
        ]
    else:
        candidates += [
            "if_score_urw_monitor_KL_p||Pp",
            "if_score_forwardKL_node||ref",
            "if_score_reverseKL_ref||node",
        ]

    for c in candidates:
        if c in df.columns:
            return pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float), c

    return None, ""


def corr_safe(a, b, method="spearman"):
    if a is None:
        return np.nan

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return np.nan

    sa = pd.Series(a[mask])
    sb = pd.Series(b[mask])

    return float(sa.corr(sb, method=method))


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    args = parse_args()

    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(root)

    ks = parse_float_list(args.k)

    if not ks:
        raise ValueError("No k values supplied.")

    files = discover_instance_files(root, args.dataset)

    print("=" * 78)
    print(f"dataset={args.dataset}")
    print(f"root={root}")
    print(f"found per-instance files={len(files)}")
    print(f"k values={ks}")
    print(f"delta={args.delta}")
    print(f"penalty={args.penalty}")
    print(f"distance={args.distance}")
    print(f"include_sensitive={args.include_sensitive}")
    print(f"standardize={not args.no_standardize}")
    print("=" * 78)

    if not files:
        raise RuntimeError(
            "No eligible per-instance CSVs found. "
            "Expected y_true, pred_prob, and raw_* columns."
        )

    summary_rows = []
    instance_parts = []

    for path in files:
        df = pd.read_csv(path)

        lam = parse_lambda(path)
        method = parse_method(path, df)

        if not np.isfinite(lam):
            print(f"[WARN] Skip: cannot parse lambda from {path.name}")
            continue

        y_true = pd.to_numeric(df["y_true"], errors="coerce").to_numpy(dtype=float)
        pred_prob = pd.to_numeric(df["pred_prob"], errors="coerce").to_numpy(dtype=float)

        if "pred_label" in df.columns:
            pred_label = pd.to_numeric(
                df["pred_label"], errors="coerce"
            ).to_numpy(dtype=int)
        else:
            pred_label = (pred_prob >= 0.5).astype(int)

        if not (
            np.isfinite(y_true).all()
            and np.isfinite(pred_prob).all()
        ):
            raise ValueError(f"NaN/inf in predictions or labels: {path}")

        try:
            auc = float(roc_auc_score(y_true.astype(int), pred_prob))
        except Exception:
            auc = np.nan

        X, feat_cols = build_knn_features(
            df,
            include_sensitive=args.include_sensitive,
            standardize=not args.no_standardize,
        )

        urw_if, urw_if_col = get_urw_if(df, method)
        urw_if_mean = (
            float(np.nanmean(urw_if))
            if urw_if is not None
            else np.nan
        )

        base_instance = pd.DataFrame({
            "dataset": args.dataset,
            "method": method,
            "lambda": float(lam),
            "orig_index": (
                df["orig_index"].to_numpy()
                if "orig_index" in df.columns
                else np.arange(len(df))
            ),
            "y_true": y_true.astype(int),
            "pred_prob": pred_prob,
            "pred_label": pred_label,
            "URW_IF": urw_if if urw_if is not None else np.nan,
            "URW_IF_column": urw_if_col,
            "source_file": str(path),
        })

        for k in ks:
            nbrs = compute_knn_indices(X, k=k, metric=args.distance)
            c = waller_consistency(pred_label, nbrs)
            agg = aggregate_waller(
                c,
                delta=args.delta,
                penalty=args.penalty,
            )

            # Waller "unfairness" orientation for correlation with URW IF:
            # higher means less locally consistent.
            waller_incons = 1.0 - c

            row = {
                "dataset": args.dataset,
                "method": method,
                "lambda": float(lam),
                "k": int(k),
                "delta": float(args.delta),
                "penalty": float(args.penalty),
                "distance": args.distance,
                "include_sensitive": bool(args.include_sensitive),
                "standardized_features": bool(not args.no_standardize),
                "n_test": int(len(df)),
                "n_knn_features": int(len(feat_cols)),
                "AUC": auc,
                "URW_IF_mean": urw_if_mean,
                "URW_IF_column": urw_if_col,
                **agg,
                "Spearman_URWIF_vs_WallerInconsistency": corr_safe(
                    urw_if,
                    waller_incons,
                    method="spearman",
                ),
                "Pearson_URWIF_vs_WallerInconsistency": corr_safe(
                    urw_if,
                    waller_incons,
                    method="pearson",
                ),
                "source_file": str(path),
            }

            summary_rows.append(row)

            dfi = base_instance.copy()
            dfi["k"] = int(k)
            dfi["delta"] = float(args.delta)
            dfi["Waller_consistency"] = c
            dfi["Waller_inconsistency"] = waller_incons
            dfi["Waller_pass_delta"] = c >= float(args.delta)

            # Useful disagreement-oriented ranks:
            dfi["Waller_inconsistency_percentile"] = (
                pd.Series(waller_incons).rank(pct=True).to_numpy()
            )

            if urw_if is not None:
                dfi["URW_IF_percentile"] = (
                    pd.Series(urw_if).rank(pct=True).to_numpy()
                )
                dfi["percentile_gap_URW_minus_Waller"] = (
                    dfi["URW_IF_percentile"]
                    - dfi["Waller_inconsistency_percentile"]
                )
            else:
                dfi["URW_IF_percentile"] = np.nan
                dfi["percentile_gap_URW_minus_Waller"] = np.nan

            instance_parts.append(dfi)

        print(
            f"[OK] {method:8s} lambda={lam:g} "
            f"n={len(df)} AUC={auc:.6f} file={path.name}"
        )

    summary = pd.DataFrame(summary_rows)

    if len(summary) == 0:
        raise RuntimeError(
            "No runs were successfully evaluated. "
            "Check lambda tags in filenames."
        )

    summary = summary.sort_values(
        ["method", "lambda", "k"]
    ).reset_index(drop=True)

    summary.to_csv(args.out, index=False)

    instances = pd.concat(instance_parts, ignore_index=True)
    instances.to_csv(args.instance_out, index=False)

    show_cols = [
        "method",
        "lambda",
        "k",
        "AUC",
        "URW_IF_mean",
        "mean_consistency",
        "LICC_rate",
        "PCS",
        "BCC",
        "BCC_penalty",
        "Spearman_URWIF_vs_WallerInconsistency",
    ]

    print("\nSUMMARY")
    print(summary[show_cols].to_string(index=False))

    print(f"\nSaved summary   -> {args.out}")
    print(f"Saved instances -> {args.instance_out}")


if __name__ == "__main__":
    main()
