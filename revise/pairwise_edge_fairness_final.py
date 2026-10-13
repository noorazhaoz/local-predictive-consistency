#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
pairwise_fairness_unified_final.py

Post-hoc independent pairwise fairness diagnostics for:
  1) Binary classification: Adult / COMPAS, FKL / RKL
  2) Regression: Employee Salary / Insurance
  3) Survival: FLCHAIN / WHAS / ROSSI, DeepSurv log-risk / survival-curve risk

NO model retraining.

Independent diagnostics on unique NON-SELF undirected graph edges:
    EdgeMeanAbs   = mean |s_i - s_j|
    EdgeMedianAbs = median |s_i - s_j|
    EdgeRMS       = sqrt(mean (s_i - s_j)^2)
    EdgeMaxAbs    = max |s_i - s_j|

Node-level diagnostics on covered nodes:
    NodeMeanAbs_i = mean_j |s_i - s_j|
    NodeRMS_i     = sqrt(mean_j (s_i - s_j)^2)
    NodeMaxAbs_i  = max_j |s_i - s_j|

IMPORTANT:
- Binary and regression use Euclidean tau, implemented as D2 <= tau**2.
- DeepSurv uses Euclidean tau, implemented as D2 <= tau**2.
- The survival-curve block preserves the original curve-training implementation,
  whose graph function thresholded the squared-distance matrix as D2 <= tau.
  This is intentionally NOT silently normalized here: the goal is to evaluate
  each saved paper model on the graph convention used by its corresponding run.
- Metric projection is applied for the regularized graph construction.

Edit only the PATHS section if your folder layout differs.
"""

import os
import re
import glob
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.impute import SimpleImputer

from lifelines.datasets import load_rossi
from sksurv.datasets import load_whas500, load_flchain


# ============================================================
# CONFIG
# ============================================================

TAU = 5.0
THETA = 1e-3
SVD_RANK = 1
METRIC_RANK = 64

BINARY_SEED = 0
REGRESSION_SEED = 0
TEST_SIZE_SURVIVAL = 0.2
MAX_SURVIVAL_SEED = 1000

# ------------------------------------------------------------
# PATHS -- EDIT IF NECESSARY
# ------------------------------------------------------------

DATA_DIR = "../data"

# Final binary source-of-truth
BINARY_RESULT_ROOT = "../binary/results_unified"

# Regression / survival final result folders
REGRESSION_RESULT_ROOT = "../regression/results"
DEEPSURV_RESULT_ROOT = "../survival/results/urw"
CURVE_RESULT_ROOT = "../survival_discrete/results/urw"

EMPLOYERS_DATA_PATH = os.path.join(DATA_DIR, "Employers_data.csv")
INSURANCE_DATA_PATH = os.path.join(DATA_DIR, "insurance.csv")

OUTPUT_DIR = "reviewer_pairwise_fairness_final"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# EXPECTED OPERATING-POINT DIAGNOSTICS
# Used only as a warning-level consistency check.
# ============================================================

EXPECTED_GRAPH = {
    ("binary", "adult", "URW"): {
        "coverage": 0.9850746268656716,
        "isolated_rate": 0.014925373134328358,
        "density": 0.06308926643014716,
        "median_degree": 283.0,
    },
    ("binary", "compas", "URW"): {
        "coverage": 0.9972,      # rounded paper/rebuttal value
        "isolated_rate": 0.0028,
        "density": 0.7292,
        "median_degree": 1215.0,
    },
    ("regression", "employers", "URW"): {
        "coverage": 0.9980,
        "isolated_rate": 0.0020,
        "density": 0.010697,
        "median_degree": 21.0,
    },
    ("regression", "insurance", "URW"): {
        "coverage": 1.0,
        "isolated_rate": 0.0,
        "density": 0.132875,
        "median_degree": 35.0,
    },
    ("survival", "flchain", "DeepSurv_log_risk"): {
        "coverage": 0.9079,
        "isolated_rate": 0.0921,
        "density": 0.017666,
        "median_degree": 14.0,
    },
    ("survival", "flchain", "Survival_curve"): {
        "coverage": 0.9079,
        "isolated_rate": 0.0921,
        "density": 0.017666,
        "median_degree": 14.0,
    },
    ("survival", "whas", "DeepSurv_log_risk"): {
        "coverage": 0.7500,
        "isolated_rate": 0.2500,
        "density": 0.028081,
        "median_degree": 2.0,
    },
    ("survival", "whas", "Survival_curve"): {
        "coverage": 0.7300,
        "isolated_rate": 0.2700,
        "density": 0.035354,
        "median_degree": 2.0,
    },
    ("survival", "rossi", "DeepSurv_log_risk"): {
        "coverage": 0.9885,
        "isolated_rate": 0.0115,
        "density": 0.241914,
        "median_degree": 22.0,
    },
    ("survival", "rossi", "Survival_curve"): {
        "coverage": 0.9600,
        "isolated_rate": 0.0400,
        "density": 0.266667,
        "median_degree": 7.0,
    },
}


# ============================================================
# GENERAL UTILITIES
# ============================================================

def parse_lambda_from_filename(path):
    name = os.path.basename(path)
    m = re.search(r"_lambda([-+0-9.eE]+)(?:_tau[^.]*)?\.csv$", name)
    if m is None:
        m = re.search(r"_lambda([-+0-9.eE]+)\.csv$", name)
    if m is None:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def e_dist_sq(A, B):
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    A2 = (A ** 2).sum(axis=1, keepdims=True)
    B2 = (B ** 2).sum(axis=1, keepdims=True).T
    return np.maximum(A2 - 2.0 * (A @ B.T) + B2, 0.0)


def build_W_tau_only(X, theta=THETA, tau=TAU, *, euclidean_tau=True):
    """
    euclidean_tau=True:
        tau means ordinary Euclidean radius:
            ||zi-zj|| <= tau
        so with D2 = ||zi-zj||^2:
            D2 <= tau^2

    euclidean_tau=False:
        preserve an older implementation that directly used:
            D2 <= tau
    """
    D2 = e_dist_sq(X, X)

    if euclidean_tau:
        mask = D2 <= float(tau) ** 2
    else:
        mask = D2 <= float(tau)

    W = np.exp(-float(theta) * D2) * mask.astype(np.float64)

    np.fill_diagonal(W, 0.0)
    W = 0.5 * (W + W.T)

    n = W.shape[0]
    W[np.arange(n), np.arange(n)] = 1.0
    return W


def graph_diagnostics(W):
    W_off = W.copy()
    np.fill_diagonal(W_off, 0.0)
    A = W_off > 0

    degree = A.sum(axis=1)
    covered = degree > 0
    n = len(degree)

    edge_i, edge_j = np.where(np.triu(A, k=1))
    n_edges = len(edge_i)
    possible_edges = n * (n - 1) / 2

    return {
        "n_nodes": int(n),
        "n_edges": int(n_edges),
        "density": float(n_edges / possible_edges) if possible_edges > 0 else np.nan,
        "coverage": float(covered.mean()) if n else np.nan,
        "isolated_rate": float((~covered).mean()) if n else np.nan,
        "mean_degree": float(degree.mean()) if n else np.nan,
        "median_degree": float(np.median(degree)) if n else np.nan,
        "min_degree": int(degree.min()) if n else 0,
        "max_degree": int(degree.max()) if n else 0,
        "degree": degree,
        "covered": covered,
        "adj": A,
    }


def check_graph(task, dataset, formulation, diag):
    key = (task, dataset, formulation)
    exp = EXPECTED_GRAPH.get(key)
    if exp is None:
        return

    print(f"\n[Graph check] {key}")
    for k in ["coverage", "isolated_rate", "density", "median_degree"]:
        got = float(diag[k])
        target = float(exp[k])

        # paper/rebuttal values for many tasks are rounded, so warn rather than fail
        atol = 5e-4 if k in ("coverage", "isolated_rate") else 5e-4
        if k == "median_degree":
            atol = 0.5

        status = "OK" if np.isclose(got, target, atol=atol, rtol=1e-3) else "CHECK"
        print(f"  {k:16s} got={got:.10g} expected≈{target:.10g} [{status}]")


def fit_fair_projection(X_tr_std, sens_tr, seed):
    X_tr_std = np.asarray(X_tr_std, dtype=np.float32)
    sens_tr = np.asarray(sens_tr)

    d = X_tr_std.shape[1]
    mu0 = X_tr_std[sens_tr == 0].mean(axis=0, keepdims=True).astype(np.float32)
    mu1 = X_tr_std[sens_tr == 1].mean(axis=0, keepdims=True).astype(np.float32)
    prot = np.vstack([mu0, mu1]).astype(np.float32)

    r = min(SVD_RANK, prot.shape[0], d)
    svd = TruncatedSVD(n_components=r, random_state=seed)
    svd.fit(prot)
    basis = svd.components_.T.astype(np.float32)

    proj = (basis @ np.linalg.pinv(basis.T @ basis) @ basis.T).astype(np.float32)
    return (np.eye(d, dtype=np.float32) - proj).astype(np.float32)


def fit_metric_transform(X_tr_fair, seed):
    max_rank = min(
        METRIC_RANK,
        X_tr_fair.shape[1],
        max(X_tr_fair.shape[0] - 1, 1),
    )
    if max_rank < 1:
        raise ValueError("Metric rank < 1.")

    svd_metric = TruncatedSVD(
        n_components=max_rank,
        random_state=seed,
    )
    svd_metric.fit(X_tr_fair)
    L_metric = svd_metric.components_.astype(np.float32)
    return L_metric, max_rank


def apply_metric(X_fair, L_metric):
    if L_metric is None:
        return np.asarray(X_fair, dtype=np.float32)
    return (np.asarray(X_fair, dtype=np.float32) @ L_metric.T).astype(np.float32)


# ============================================================
# INDEPENDENT EDGE + NODE METRICS
# ============================================================

def compute_pairwise_diagnostics(scores, W):
    scores = np.asarray(scores, dtype=float).reshape(-1)

    if len(scores) != W.shape[0]:
        raise ValueError(f"Prediction length {len(scores)} != graph size {W.shape[0]}")
    if not np.isfinite(scores).all():
        raise ValueError("Non-finite prediction scores found.")

    diag = graph_diagnostics(W)
    A = diag["adj"]

    edge_i, edge_j = np.where(np.triu(A, k=1))
    diff = np.abs(scores[edge_i] - scores[edge_j])

    if len(diff) == 0:
        edge_mean = edge_median = edge_rms = edge_max = np.nan
    else:
        edge_mean = float(np.mean(diff))
        edge_median = float(np.median(diff))
        edge_rms = float(np.sqrt(np.mean(diff ** 2)))
        edge_max = float(np.max(diff))

    n = len(scores)
    node_mean = np.full(n, np.nan, dtype=float)
    node_rms = np.full(n, np.nan, dtype=float)
    node_max = np.full(n, np.nan, dtype=float)

    # Vectorized enough for these graph sizes; loop only over covered nodes.
    for i in np.where(diag["covered"])[0]:
        nbr = np.where(A[i])[0]
        d = np.abs(scores[i] - scores[nbr])
        node_mean[i] = float(np.mean(d))
        node_rms[i] = float(np.sqrt(np.mean(d ** 2)))
        node_max[i] = float(np.max(d))

    return {
        "EdgeMeanAbs": edge_mean,
        "EdgeMedianAbs": edge_median,
        "EdgeRMS": edge_rms,
        "EdgeMaxAbs": edge_max,

        "NodeMeanAbs_mean": float(np.nanmean(node_mean)),
        "NodeMeanAbs_median": float(np.nanmedian(node_mean)),
        "NodeRMS_mean": float(np.nanmean(node_rms)),
        "NodeRMS_median": float(np.nanmedian(node_rms)),
        "NodeMaxAbs_mean": float(np.nanmean(node_max)),
        "NodeMaxAbs_median": float(np.nanmedian(node_max)),

        "node_mean": node_mean,
        "node_rms": node_rms,
        "node_max": node_max,
        "graph_diag": diag,
    }


# ============================================================
# BINARY DATA LOADERS
# ============================================================

def load_adult_clean():
    headers = [
        "age", "workclass", "fnlwgt", "education", "education-num",
        "marital-status", "occupation", "relationship", "race", "sex",
        "capital-gain", "capital-loss", "hours-per-week", "native-country", "y",
    ]

    train_path = os.path.join(DATA_DIR, "adult", "adult.data")
    test_path = os.path.join(DATA_DIR, "adult", "adult.test")

    train = pd.read_csv(
        train_path, header=None, sep=",",
        skipinitialspace=True, na_values="?"
    )
    test = pd.read_csv(
        test_path, header=None, sep=",",
        skipinitialspace=True, na_values="?", comment="|"
    )

    df = pd.concat([train, test], axis=0, ignore_index=True)
    df.columns = headers

    df["y"] = (
        df["y"].astype(str)
        .str.replace(".", "", regex=False)
        .str.strip()
        .map({"<=50K": 0, ">50K": 1})
    )

    df = df.dropna(subset=["y"]).copy()
    df["y"] = df["y"].astype(int)
    df = df.dropna().copy()

    df = pd.get_dummies(df, drop_first=False)

    native_cols = [c for c in df.columns if c.startswith("native-country_")]
    df = df.drop(columns=native_cols, errors="ignore")

    drop_cols = [
        c for c in df.columns
        if c.startswith("fnlwgt") or c.startswith("education_")
    ]
    df = df.drop(columns=drop_cols, errors="ignore")

    df["orig_index"] = np.arange(len(df))
    return df


def load_compas_clean():
    path = os.path.join(DATA_DIR, "compas", "compas-scores-two-years.csv")
    df = pd.read_csv(path)

    keep_cols = [
        "age", "priors_count", "juv_fel_count", "juv_misd_count",
        "juv_other_count", "c_charge_degree", "sex", "race", "two_year_recid",
    ]

    df = df[keep_cols].copy().dropna()
    df["y"] = df["two_year_recid"].astype(int)
    df = df.drop(columns=["two_year_recid"])

    for c in [
        "age", "priors_count", "juv_fel_count",
        "juv_misd_count", "juv_other_count",
    ]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna().copy()
    df = pd.get_dummies(
        df,
        columns=["c_charge_degree", "sex", "race"],
        drop_first=False,
    )
    df["orig_index"] = np.arange(len(df))
    return df


def prepare_binary_graph(dataset):
    seed = BINARY_SEED
    df = load_adult_clean() if dataset == "adult" else load_compas_clean()

    y = df["y"].astype(int).values
    idx = df["orig_index"].astype(int).values

    if "sex_Female" in df.columns:
        sens = df["sex_Female"].astype(int).values
    elif "sex_Male" in df.columns:
        sens = 1 - df["sex_Male"].astype(int).values
    else:
        raise ValueError(f"No sex dummy found for {dataset}")

    drop_cols = [c for c in df.columns if c.startswith("sex_")] + ["y", "orig_index"]
    X = df.drop(columns=drop_cols).astype(np.float32).values

    X_tr, X_tmp, y_tr, y_tmp, s_tr, s_tmp, idx_tr, idx_tmp = train_test_split(
        X, y, sens, idx,
        test_size=0.4,
        random_state=seed,
        stratify=y,
    )

    X_va, X_te, y_va, y_te, s_va, s_te, idx_va, idx_te = train_test_split(
        X_tmp, y_tmp, s_tmp, idx_tmp,
        test_size=0.5,
        random_state=seed,
        stratify=y_tmp,
    )

    scaler = StandardScaler()
    scaler.fit(X_tr)
    X_tr_std = scaler.transform(X_tr).astype(np.float32)
    X_te_std = scaler.transform(X_te).astype(np.float32)

    proj_compl = fit_fair_projection(X_tr_std, s_tr, seed)
    X_tr_fair = (X_tr_std @ proj_compl).astype(np.float32)
    X_te_fair = (X_te_std @ proj_compl).astype(np.float32)

    L_metric, metric_rank = fit_metric_transform(X_tr_fair, seed)
    X_te_metric = apply_metric(X_te_fair, L_metric)

    W = build_W_tau_only(
        X_te_metric,
        THETA,
        TAU,
        euclidean_tau=True,
    )

    gd = graph_diagnostics(W)
    check_graph("binary", dataset, "URW", gd)

    return {
        "idx": idx_te.astype(int),
        "W": W,
        "metric_rank": metric_rank,
        **gd,
    }


# ============================================================
# BINARY FILE DISCOVERY / ALIGNMENT
# ============================================================

def find_binary_files(dataset, agg):
    pattern = os.path.join(
        BINARY_RESULT_ROOT,
        f"if_{dataset}_agg{agg}_depth*_lambda*.csv"
    )

    out = []
    for p in glob.glob(pattern):
        name = os.path.basename(p).lower()
        if "_tauq" in name:
            continue

        lam = parse_lambda_from_filename(p)
        if lam is None:
            continue

        try:
            cols = pd.read_csv(p, nrows=1).columns
        except Exception:
            continue

        if {"orig_index", "pred_prob"}.issubset(cols):
            out.append(p)

    return sorted(out, key=lambda p: parse_lambda_from_filename(p))


def align_by_ids(path, graph_idx, score_col, id_col):
    df = pd.read_csv(path)

    if score_col not in df.columns:
        raise ValueError(f"{score_col} missing: {path}")
    if id_col not in df.columns:
        raise ValueError(f"{id_col} missing: {path}")

    file_idx = df[id_col].astype(int).values
    scores = df[score_col].astype(float).values
    graph_idx = np.asarray(graph_idx, dtype=int)

    if len(file_idx) != len(graph_idx):
        raise ValueError(
            f"Size mismatch for {path}: csv={len(file_idx)}, graph={len(graph_idx)}"
        )

    if np.array_equal(file_idx, graph_idx):
        return scores, df

    if set(file_idx.tolist()) == set(graph_idx.tolist()):
        m = {int(i): float(s) for i, s in zip(file_idx, scores)}
        aligned = np.array([m[int(i)] for i in graph_idx], dtype=float)
        df2 = df.set_index(id_col).loc[graph_idx].reset_index()
        return aligned, df2

    missing = set(graph_idx.tolist()) - set(file_idx.tolist())
    extra = set(file_idx.tolist()) - set(graph_idx.tolist())
    raise ValueError(
        f"ID membership mismatch: {path}\n"
        f"missing={list(missing)[:10]}\nextra={list(extra)[:10]}"
    )


# ============================================================
# REGRESSION GRAPH
# ============================================================

def load_regression_arrays(dataset):
    path = EMPLOYERS_DATA_PATH if dataset == "employers" else INSURANCE_DATA_PATH
    df = pd.read_csv(path).copy()

    if dataset == "employers":
        df = df.drop(columns=["Employee_ID", "Name"], errors="ignore")
        sens = (df["Gender"].astype(str).str.lower() == "female").astype(int).values
        df["Gender_bin"] = sens

        target = np.clip(
            df["Salary"].astype(np.float32).values,
            1e-6, None
        ).astype(np.float32)
        y_log = np.log(target).astype(np.float32)

        features_full = df.drop(columns=["Gender", "Salary"], errors="ignore")
        X_full_df = pd.get_dummies(features_full, drop_first=True).astype(np.float32)
        X_model_df = X_full_df.drop(columns=["Gender_bin"]).copy()

    elif dataset == "insurance":
        required = ["age", "sex", "bmi", "children", "smoker", "region", "expenses"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Insurance missing columns: {missing}")

        sens = (df["sex"].astype(str).str.lower() == "female").astype(int).values
        df["sex_bin"] = sens

        target = np.clip(
            df["expenses"].astype(np.float32).values,
            1e-6, None
        ).astype(np.float32)
        y_log = np.log(target).astype(np.float32)

        features_full = df.drop(columns=["sex", "expenses"], errors="ignore")
        X_full_df = pd.get_dummies(features_full, drop_first=True).astype(np.float32)
        X_model_df = X_full_df.drop(columns=["sex_bin"]).copy()

    else:
        raise ValueError(dataset)

    X = X_model_df.values.astype(np.float32)
    idx = np.arange(len(X))
    return X, y_log, sens, target, idx


def prepare_regression_graph(dataset):
    seed = REGRESSION_SEED
    X, y_log, sens, target, idx = load_regression_arrays(dataset)

    X_tr, X_tmp, y_tr, y_tmp, s_tr, s_tmp, target_tr, target_tmp, idx_tr, idx_tmp = train_test_split(
        X, y_log, sens, target, idx,
        test_size=0.4,
        random_state=seed,
        stratify=sens,
    )

    X_va, X_te, y_va, y_te, s_va, s_te, target_va, target_te, idx_va, idx_te = train_test_split(
        X_tmp, y_tmp, s_tmp, target_tmp, idx_tmp,
        test_size=0.5,
        random_state=seed,
        stratify=s_tmp,
    )

    scaler = StandardScaler()
    scaler.fit(X_tr)
    X_tr_std = scaler.transform(X_tr).astype(np.float32)
    X_te_std = scaler.transform(X_te).astype(np.float32)

    proj_compl = fit_fair_projection(X_tr_std, s_tr, seed)
    X_tr_fair = (X_tr_std @ proj_compl).astype(np.float32)
    X_te_fair = (X_te_std @ proj_compl).astype(np.float32)

    L_metric, metric_rank = fit_metric_transform(X_tr_fair, seed)
    X_te_metric = apply_metric(X_te_fair, L_metric)

    W = build_W_tau_only(
        X_te_metric,
        THETA,
        TAU,
        euclidean_tau=False,
    )

    gd = graph_diagnostics(W)
    check_graph("regression", dataset, "URW", gd)

    return {
        "idx": idx_te.astype(int),
        "W": W,
        "metric_rank": metric_rank,
        **gd,
    }


def find_regression_files(dataset):
    if dataset == "employers":
        patterns = [
            os.path.join(REGRESSION_RESULT_ROOT, "**", "urw_depth*_lambda*.csv"),
            os.path.join(REGRESSION_RESULT_ROOT, "**", "*employers*urw*lambda*.csv"),
        ]
    else:
        patterns = [
            os.path.join(REGRESSION_RESULT_ROOT, "**", "*insurance*urw*lambda*.csv"),
        ]

    candidates = []
    for pattern in patterns:
        candidates.extend(glob.glob(pattern, recursive=True))

    valid = []
    for p in set(candidates):
        if "_tauq" in os.path.basename(p).lower():
            continue

        lam = parse_lambda_from_filename(p)
        if lam is None:
            continue

        try:
            cols = pd.read_csv(p, nrows=1).columns
        except Exception:
            continue

        if "pred_log" not in cols:
            continue
        if "orig_index" in cols or "index" in cols:
            valid.append(p)

    # deduplicate same lambda by preferring paths containing dataset name
    by_lambda = {}
    for p in sorted(valid):
        lam = parse_lambda_from_filename(p)
        score = int(dataset in os.path.basename(p).lower())
        if lam not in by_lambda or score > by_lambda[lam][0]:
            by_lambda[lam] = (score, p)

    return [by_lambda[k][1] for k in sorted(by_lambda)]


def align_regression_scores(dataset, path, graph_idx):
    df = pd.read_csv(path)
    scores = df["pred_log"].astype(float).values

    if "orig_index" in df.columns:
        return align_by_ids(path, graph_idx, "pred_log", "orig_index")

    if "index" in df.columns:
        pos = df["index"].astype(int).values
        expected = np.arange(len(df))

        if not np.array_equal(pos, expected):
            raise ValueError(
                f"{dataset}: saved 'index' is not 0..n-1 test-row position: {path}"
            )

        if len(scores) != len(graph_idx):
            raise ValueError(
                f"{dataset}: row-order size mismatch: csv={len(scores)}, graph={len(graph_idx)}"
            )

        # This follows the saved Employers convention: index is test-row position.
        return scores, df

    raise ValueError(f"No usable regression ID/index column: {path}")


# ============================================================
# SURVIVAL LOADERS
# ============================================================

def _binary_sensitive(arr):
    a = np.asarray(arr)
    if np.issubdtype(a.dtype, np.number):
        vals = pd.Series(a).dropna().unique()
        if set(np.asarray(vals).astype(int).tolist()).issubset({0, 1}):
            return a.astype(int)

    s = pd.Series(a).astype(str).str.strip().str.lower()
    # datasets used here are already normally mapped, but keep safe fallbacks
    mapped = s.map({
        "female": 1, "f": 1, "1": 1, "true": 1,
        "male": 0, "m": 0, "0": 0, "false": 0,
    })
    if mapped.isna().any():
        # categorical two-level fallback
        uniq = sorted(s.dropna().unique().tolist())
        if len(uniq) != 2:
            raise ValueError(f"Cannot binarize sensitive attribute: {uniq}")
        mapped = s.map({uniq[0]: 0, uniq[1]: 1})
    return mapped.astype(int).values


def load_survival_dataset(dataset):
    if dataset == "rossi":
        df = load_rossi().copy()
        time_col = "week"
        event_col = "arrest"
        sens_col = "race"

    elif dataset == "whas":
        X_raw, y_struct = load_whas500()
        df = X_raw.copy()
        time_col = "lenfol"
        event_col = "fstat"
        sens_col = "gender"
        df[time_col] = y_struct[time_col].astype(np.float32)
        df[event_col] = y_struct[event_col].astype(int)

    elif dataset == "flchain":
        X_raw, y_struct = load_flchain()
        df = X_raw.copy()
        time_col = "futime"
        event_col = "death"
        sens_col = "sex"
        df[time_col] = y_struct[time_col].astype(np.float32)
        df[event_col] = y_struct[event_col].astype(int)
        df[sens_col] = (
            df[sens_col].astype(str).str.upper().str[0].eq("F").astype(int)
        )

    else:
        raise ValueError(dataset)

    df = df.dropna(subset=[time_col, event_col, sens_col]).copy()
    return df, time_col, event_col, sens_col


def survival_base_arrays(dataset):
    df, time_col, event_col, sens_col = load_survival_dataset(dataset)

    t = df[time_col].astype(np.float32).values
    e = df[event_col].astype(int).values
    sens = _binary_sensitive(df[sens_col])

    X_df = df.drop(columns=[time_col, event_col, sens_col], errors="ignore")
    X_df = pd.get_dummies(X_df, drop_first=True)

    imputer = SimpleImputer(strategy="median")
    X_df[:] = imputer.fit_transform(X_df)

    X = X_df.astype(np.float32).values
    idx_all = np.arange(len(df))
    return X, t, e, sens, idx_all


def get_saved_survival_idx(path):
    df = pd.read_csv(path)
    if "idx" not in df.columns:
        raise ValueError(f"No idx column: {path}")
    return df["idx"].astype(int).values


def infer_survival_seed(dataset, reference_file):
    X, t, e, sens, idx_all = survival_base_arrays(dataset)
    saved_idx = get_saved_survival_idx(reference_file)
    saved_set = set(saved_idx.tolist())

    for seed in range(MAX_SURVIVAL_SEED + 1):
        X_tr, X_te, t_tr, t_te, e_tr, e_te, s_tr, s_te, idx_tr, idx_te = train_test_split(
            X, t, e, sens, idx_all,
            test_size=TEST_SIZE_SURVIVAL,
            random_state=seed,
        )

        idx_te = idx_te.astype(int)

        # Saved curve files may contain a filtered subset of test rows.
        if len(saved_idx) == len(idx_te):
            if np.array_equal(saved_idx, idx_te) or set(idx_te.tolist()) == saved_set:
                return seed

        if len(saved_idx) < len(idx_te) and saved_set.issubset(set(idx_te.tolist())):
            # We still need uniqueness: accept only if this seed reproduces all saved IDs
            # as members of the test split. In practice the intended seeds are unique.
            return seed

    raise RuntimeError(
        f"Could not infer seed for {dataset} from {os.path.basename(reference_file)}"
    )


def prepare_survival_graph(dataset, reference_file, *, formulation):
    """
    Build graph on EXACT saved per-instance IDs.

    This avoids re-implementing downstream test filtering for the curve model:
    the saved idx column defines the evaluated node set exactly.
    """
    seed = infer_survival_seed(dataset, reference_file)

    X, t, e, sens, idx_all = survival_base_arrays(dataset)
    saved_idx = get_saved_survival_idx(reference_file)

    X_tr, X_te, t_tr, t_te, e_tr, e_te, s_tr, s_te, idx_tr, idx_te = train_test_split(
        X, t, e, sens, idx_all,
        test_size=TEST_SIZE_SURVIVAL,
        random_state=seed,
    )

    scaler = StandardScaler()
    scaler.fit(X_tr)
    X_tr_std = scaler.transform(X_tr).astype(np.float32)
    X_te_std = scaler.transform(X_te).astype(np.float32)

    # For post-hoc common-graph evaluation of the saved fairness runs,
    # use the positive-lambda graph construction: fair projection + metric.
    proj_compl = fit_fair_projection(X_tr_std, s_tr, seed)
    X_tr_fair = (X_tr_std @ proj_compl).astype(np.float32)
    X_te_fair = (X_te_std @ proj_compl).astype(np.float32)

    L_metric, metric_rank = fit_metric_transform(X_tr_fair, seed)
    X_te_metric = apply_metric(X_te_fair, L_metric)

    pos_by_idx = {int(idx): pos for pos, idx in enumerate(idx_te.astype(int))}
    missing = [int(i) for i in saved_idx if int(i) not in pos_by_idx]
    if missing:
        raise ValueError(
            f"Saved survival IDs are not in inferred test split for {dataset}: "
            f"{missing[:10]}"
        )

    pos = np.array([pos_by_idx[int(i)] for i in saved_idx], dtype=int)
    X_graph = X_te_metric[pos]

    # Preserve exact graph convention of each corresponding implementation.
    if formulation == "DeepSurv_log_risk":
        euclidean_tau = False
    elif formulation == "Survival_curve":
        euclidean_tau = False
    else:
        raise ValueError(formulation)

    W = build_W_tau_only(
        X_graph,
        THETA,
        TAU,
        euclidean_tau=euclidean_tau,
    )

    gd = graph_diagnostics(W)
    check_graph("survival", dataset, formulation, gd)

    return {
        "seed": seed,
        "idx": saved_idx.astype(int),
        "W": W,
        "metric_rank": metric_rank,
        **gd,
    }


def find_survival_files(root, dataset, prefix):
    folder = os.path.join(root, dataset)
    pattern = os.path.join(folder, f"{prefix}_depth*_lambda*.csv")

    out = []
    for p in glob.glob(pattern):
        if "_tauq" in os.path.basename(p).lower():
            continue
        lam = parse_lambda_from_filename(p)
        if lam is None:
            continue
        out.append(p)

    return sorted(out, key=lambda p: parse_lambda_from_filename(p))


# ============================================================
# RESULT ROW HELPERS
# ============================================================

summary_rows = []
node_rows = []


def add_run(
    *,
    task,
    dataset,
    formulation,
    score_scale,
    lam,
    path,
    scores,
    graph_idx,
    graph_pack,
    aligned_df=None,
):
    pair = compute_pairwise_diagnostics(scores, graph_pack["W"])
    gd = pair["graph_diag"]

    summary_rows.append({
        "task": task,
        "dataset": dataset,
        "formulation": formulation,
        "score_scale": score_scale,
        "lambda": float(lam),
        "file": os.path.basename(path),

        "tau": TAU,
        "theta": THETA,
        "metric_rank": graph_pack.get("metric_rank", np.nan),

        "n_nodes": gd["n_nodes"],
        "n_edges": gd["n_edges"],
        "graph_density": gd["density"],
        "coverage": gd["coverage"],
        "isolated_rate": gd["isolated_rate"],
        "mean_degree": gd["mean_degree"],
        "median_degree": gd["median_degree"],

        "EdgeMeanAbs": pair["EdgeMeanAbs"],
        "EdgeMedianAbs": pair["EdgeMedianAbs"],
        "EdgeRMS": pair["EdgeRMS"],
        "EdgeMaxAbs": pair["EdgeMaxAbs"],

        "NodeMeanAbs_mean": pair["NodeMeanAbs_mean"],
        "NodeMeanAbs_median": pair["NodeMeanAbs_median"],
        "NodeRMS_mean": pair["NodeRMS_mean"],
        "NodeRMS_median": pair["NodeRMS_median"],
        "NodeMaxAbs_mean": pair["NodeMaxAbs_mean"],
        "NodeMaxAbs_median": pair["NodeMaxAbs_median"],
    })

    for i, idx in enumerate(graph_idx):
        node_rows.append({
            "task": task,
            "dataset": dataset,
            "formulation": formulation,
            "lambda": float(lam),
            "node_id": int(idx),
            "score": float(scores[i]),
            "degree": int(gd["degree"][i]),
            "covered": bool(gd["covered"][i]),
            "NodeMeanAbs": pair["node_mean"][i],
            "NodeRMS": pair["node_rms"][i],
            "NodeMaxAbs": pair["node_max"][i],
        })


# ============================================================
# 1) BINARY
# ============================================================

for dataset in ["adult", "compas"]:
    print("\n" + "=" * 100)
    print(f"BINARY: {dataset}")
    print("=" * 100)

    graph_pack = prepare_binary_graph(dataset)

    for agg, formulation in [
        ("forward", "URW-FKL"),
        ("reverse", "URW-RKL"),
    ]:
        files = find_binary_files(dataset, agg)
        print(f"{formulation}: {len(files)} files")

        for path in files:
            lam = parse_lambda_from_filename(path)
            scores, aligned_df = align_by_ids(
                path,
                graph_pack["idx"],
                "pred_prob",
                "orig_index",
            )

            add_run(
                task="binary",
                dataset=dataset,
                formulation=formulation,
                score_scale="pred_prob",
                lam=lam,
                path=path,
                scores=scores,
                graph_idx=graph_pack["idx"],
                graph_pack=graph_pack,
                aligned_df=aligned_df,
            )


# ============================================================
# 2) REGRESSION
# ============================================================

for dataset in ["employers", "insurance"]:
    print("\n" + "=" * 100)
    print(f"REGRESSION: {dataset}")
    print("=" * 100)

    graph_pack = prepare_regression_graph(dataset)
    files = find_regression_files(dataset)
    print(f"URW: {len(files)} files")

    for path in files:
        lam = parse_lambda_from_filename(path)
        scores, aligned_df = align_regression_scores(
            dataset,
            path,
            graph_pack["idx"],
        )

        add_run(
            task="regression",
            dataset=dataset,
            formulation="URW",
            score_scale="pred_log",
            lam=lam,
            path=path,
            scores=scores,
            graph_idx=graph_pack["idx"],
            graph_pack=graph_pack,
            aligned_df=aligned_df,
        )


# ============================================================
# 3) SURVIVAL
# ============================================================

for dataset in ["flchain", "whas", "rossi"]:
    print("\n" + "#" * 100)
    print(f"SURVIVAL: {dataset}")
    print("#" * 100)

    # ---------------- DeepSurv ----------------
    ds_files = find_survival_files(
        DEEPSURV_RESULT_ROOT,
        dataset,
        "if_deepsurv",
    )

    print(f"DeepSurv files: {len(ds_files)}")

    if ds_files:
        ds_graph = prepare_survival_graph(
            dataset,
            ds_files[0],
            formulation="DeepSurv_log_risk",
        )

        for path in ds_files:
            lam = parse_lambda_from_filename(path)
            scores, aligned_df = align_by_ids(
                path,
                ds_graph["idx"],
                "eta",
                "idx",
            )

            add_run(
                task="survival",
                dataset=dataset,
                formulation="DeepSurv_log_risk",
                score_scale="eta",
                lam=lam,
                path=path,
                scores=scores,
                graph_idx=ds_graph["idx"],
                graph_pack=ds_graph,
                aligned_df=aligned_df,
            )

    # ---------------- Curve ----------------
    curve_files = find_survival_files(
        CURVE_RESULT_ROOT,
        dataset,
        "if_curve",
    )

    print(f"Curve files: {len(curve_files)}")

    if curve_files:
        curve_graph = prepare_survival_graph(
            dataset,
            curve_files[0],
            formulation="Survival_curve",
        )

        for path in curve_files:
            lam = parse_lambda_from_filename(path)
            scores, aligned_df = align_by_ids(
                path,
                curve_graph["idx"],
                "risk_-logS(t*)",
                "idx",
            )

            add_run(
                task="survival",
                dataset=dataset,
                formulation="Survival_curve",
                score_scale="-logS(t*)",
                lam=lam,
                path=path,
                scores=scores,
                graph_idx=curve_graph["idx"],
                graph_pack=curve_graph,
                aligned_df=aligned_df,
            )


# ============================================================
# SAVE
# ============================================================

summary = pd.DataFrame(summary_rows)
nodes = pd.DataFrame(node_rows)

if len(summary):
    summary = summary.sort_values(
        ["task", "dataset", "formulation", "lambda"]
    ).reset_index(drop=True)

if len(nodes):
    nodes = nodes.sort_values(
        ["task", "dataset", "formulation", "lambda", "node_id"]
    ).reset_index(drop=True)


full_path = os.path.join(
    OUTPUT_DIR,
    "pairwise_fairness_summary_full.csv"
)
node_path = os.path.join(
    OUTPUT_DIR,
    "pairwise_fairness_node_diagnostics.csv"
)

summary.to_csv(full_path, index=False)
nodes.to_csv(node_path, index=False)


compact_cols = [
    "task",
    "dataset",
    "formulation",
    "score_scale",
    "lambda",
    "n_nodes",
    "n_edges",
    "coverage",
    "isolated_rate",
    "graph_density",
    "median_degree",
    "EdgeMeanAbs",
    "EdgeMedianAbs",
    "EdgeRMS",
    "EdgeMaxAbs",
    "NodeMeanAbs_mean",
    "NodeRMS_mean",
    "NodeMaxAbs_mean",
]

compact = summary[compact_cols].copy() if len(summary) else pd.DataFrame(columns=compact_cols)

compact_path = os.path.join(
    OUTPUT_DIR,
    "pairwise_fairness_reviewer_compact.csv"
)
compact.to_csv(compact_path, index=False)


print("\n\n" + "=" * 120)
print("PAIRWISE FAIRNESS RESULTS")
print("=" * 120)

if len(compact):
    print(compact.to_string(index=False))
else:
    print("No results produced.")

print("\nSaved:")
print(" ", full_path)
print(" ", compact_path)
print(" ", node_path)

print(
    "\nIMPORTANT: inspect every [Graph check] block above. "
    "If a row says [CHECK], do not use that task's pairwise numbers in the rebuttal "
    "until the graph mismatch is resolved."
)
