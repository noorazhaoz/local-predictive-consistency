# survival_graph_coverage_diagnostics.py
# ------------------------------------------------------------
# Graph coverage / isolated-node diagnostics for survival tasks
#
# Scope:
#   - lambda > 0 only
#   - fair-projected graph
#   - use_metric = False
#
# Formulations:
#   1) DeepSurv / scalar eta
#   2) Curve / survival curve
#
# Datasets:
#   - flchain
#   - whas
#   - rossi
#   - employee
# ------------------------------------------------------------

import os
import numpy as np
import pandas as pd

from lifelines.datasets import load_rossi
from sksurv.datasets import load_whas500, load_flchain

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.impute import SimpleImputer


# ============================================================
# CONFIG
# ============================================================

SEED = 0
TEST_SIZE = 0.2

TAU = 5.0
THETA = 1e-3
SVD_RANK = 1

# lambda > 0 regime only
# projection is identical for all positive lambda values
USE_METRIC = False

# CHANGE THIS:
EMPLOYEE_PATH = "../data/employee.csv"

OUTPUT_DIR = "reviewer_survival_graph_diagnostics"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# GRAPH UTILS
# ============================================================

def e_dist_sq(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A2 = (A ** 2).sum(axis=1, keepdims=True)
    B2 = (B ** 2).sum(axis=1, keepdims=True).T
    return np.maximum(A2 - 2.0 * (A @ B.T) + B2, 0.0)


def build_W_tau_only(
    X_for_metric: np.ndarray,
    theta: float,
    tau: float
) -> np.ndarray:

    D2 = e_dist_sq(
        X_for_metric,
        X_for_metric
    ).astype(np.float64)

    mask = (D2 <= float(tau)).astype(np.float64)

    W = (
        np.exp(-float(theta) * D2) * mask
    ).astype(np.float64)

    # Remove diagonal before symmetrization
    np.fill_diagonal(W, 0.0)

    W = 0.5 * (W + W.T)

    # Original implementation adds explicit self-loop
    n = W.shape[0]
    W[np.arange(n), np.arange(n)] = 1.0

    return W


def compute_graph_diagnostics(W: np.ndarray) -> dict:
    """
    IMPORTANT:
    Isolation is defined using NON-SELF neighbours only.
    """

    W_off = W.copy()
    np.fill_diagonal(W_off, 0.0)

    adj = W_off > 0

    degree = adj.sum(axis=1)

    covered = degree > 0
    isolated = degree == 0

    n = W.shape[0]

    edges_undirected = int(
        np.triu(adj, k=1).sum()
    )

    possible_edges = n * (n - 1) / 2

    if possible_edges > 0:
        density = edges_undirected / possible_edges
    else:
        density = np.nan

    return {
        "test_n": int(n),
        "edges_undirected": int(edges_undirected),
        "graph_density": float(density),

        "covered_nodes": int(covered.sum()),
        "coverage": float(covered.mean()),

        "isolated_nodes": int(isolated.sum()),
        "isolated_rate": float(isolated.mean()),

        "mean_degree": float(degree.mean()),
        "median_degree": float(np.median(degree)),
    }


# ============================================================
# DATA LOADERS
# ============================================================

def load_dataset(dataset: str):

    if dataset == "rossi":

        time_col = "week"
        event_col = "arrest"
        sens_col = "race"

        df = load_rossi().copy()

        df = df.dropna(
            subset=[time_col, event_col, sens_col]
        ).copy()

        return df, time_col, event_col, sens_col


    elif dataset == "whas":

        time_col = "lenfol"
        event_col = "fstat"
        sens_col = "gender"

        X_raw, y_struct = load_whas500()

        df = X_raw.copy()

        df[time_col] = (
            y_struct[time_col]
            .astype(np.float32)
        )

        df[event_col] = (
            y_struct[event_col]
            .astype(int)
        )

        df = df.dropna(
            subset=[time_col, event_col, sens_col]
        ).copy()

        return df, time_col, event_col, sens_col


    elif dataset == "employee":

        time_col = "stag"
        event_col = "event"
        sens_col = "gender"

        df = pd.read_csv(
            EMPLOYEE_PATH,
            engine="python",
            encoding="latin1"
        )

        g = (
            df[sens_col]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        df[sens_col] = g.map({
            "f": 1,
            "female": 1,
            "m": 0,
            "male": 0
        })

        df[time_col] = (
            df[time_col]
            .astype(np.float32)
        )

        df[event_col] = (
            df[event_col]
            .astype(int)
        )

        df[sens_col] = (
            df[sens_col]
            .astype(int)
        )

        df = df.dropna(
            subset=[time_col, event_col, sens_col]
        ).copy()

        return df, time_col, event_col, sens_col


    elif dataset == "flchain":

        time_col = "futime"
        event_col = "death"
        sens_col = "sex"

        X_raw, y_struct = load_flchain()

        df = X_raw.copy()

        df[time_col] = (
            y_struct[time_col]
            .astype(np.float32)
        )

        df[event_col] = (
            y_struct[event_col]
            .astype(int)
        )

        df[sens_col] = (
            df[sens_col]
            .astype(str)
            .str.upper()
            .str[0]
            .eq("F")
            .astype(int)
        )

        df = df.dropna(
            subset=[time_col, event_col, sens_col]
        ).copy()

        return df, time_col, event_col, sens_col

    else:
        raise ValueError(
            f"Unsupported dataset: {dataset}"
        )


# ============================================================
# PREPROCESS + FAIR PROJECTION
# ============================================================

def prepare_dataset(dataset: str):

    df, time_col, event_col, sens_col = load_dataset(dataset)

    t_full = (
        df[time_col]
        .astype(np.float32)
        .values
    )

    e_full = (
        df[event_col]
        .astype(int)
        .values
    )

    sens_full = (
        df[sens_col]
        .astype(int)
        .values
    )

    # Drop outcome + sensitive variables
    X_df = df.drop(
        columns=[
            time_col,
            event_col,
            sens_col
        ],
        errors="ignore"
    )

    X_df = pd.get_dummies(
        X_df,
        drop_first=True
    )

    # Match original survival scripts
    imputer = SimpleImputer(
        strategy="median"
    )

    X_df[:] = imputer.fit_transform(X_df)

    X = (
        X_df
        .astype(np.float32)
        .values
    )

    idx_all = np.arange(len(df))

    # Same train/test split as original code
    (
        X_tr,
        X_te,
        t_tr,
        t_te,
        e_tr,
        e_te,
        s_tr,
        s_te,
        idx_tr,
        idx_te
    ) = train_test_split(
        X,
        t_full,
        e_full,
        sens_full,
        idx_all,
        test_size=TEST_SIZE,
        random_state=SEED
    )

    # Standardize train-only
    scaler = StandardScaler()

    scaler.fit(X_tr)

    X_tr_std = (
        scaler
        .transform(X_tr)
        .astype(np.float32)
    )

    X_te_std = (
        scaler
        .transform(X_te)
        .astype(np.float32)
    )

    d = X_tr_std.shape[1]

    # ========================================================
    # lambda > 0 FAIR PROJECTION
    # ========================================================

    mu0 = (
        X_tr_std[s_tr == 0]
        .mean(axis=0, keepdims=True)
        .astype(np.float32)
    )

    mu1 = (
        X_tr_std[s_tr == 1]
        .mean(axis=0, keepdims=True)
        .astype(np.float32)
    )

    prot = np.vstack(
        [mu0, mu1]
    ).astype(np.float32)

    r_sens = min(
        SVD_RANK,
        prot.shape[0],
        d
    )

    svd = TruncatedSVD(
        n_components=r_sens,
        random_state=SEED
    )

    svd.fit(prot)

    basis = (
        svd.components_
        .T
        .astype(np.float32)
    )

    proj = (
        basis
        @ np.linalg.pinv(
            basis.T @ basis
        )
        @ basis.T
    ).astype(np.float32)

    proj_compl = (
        np.eye(
            d,
            dtype=np.float32
        )
        - proj
    ).astype(np.float32)

    X_tr_fair = (
        X_tr_std
        @ proj_compl
    ).astype(np.float32)

    X_te_fair = (
        X_te_std
        @ proj_compl
    ).astype(np.float32)

    return {
        "dataset": dataset,

        "X_tr_fair": X_tr_fair,
        "X_te_fair": X_te_fair,

        "t_tr": t_tr,
        "t_te": t_te,

        "e_tr": e_tr,
        "e_te": e_te,

        "s_tr": s_tr,
        "s_te": s_te,

        "idx_tr": idx_tr,
        "idx_te": idx_te,
    }


# ============================================================
# DEEPSURV GRAPH
# ============================================================

def diagnostics_deepsurv(pack: dict) -> dict:

    X_te_fair = pack["X_te_fair"]

    # DeepSurv original code:
    # graph on full TEST set
    W_te = build_W_tau_only(
        X_te_fair,
        THETA,
        TAU
    )

    stats = compute_graph_diagnostics(
        W_te
    )

    stats.update({
        "formulation": "DeepSurv_eta",
        "dataset": pack["dataset"],
        "graph_scope": "full_test",
        "lambda_regime": "lambda>0",
        "tau": TAU,
        "theta": THETA,
    })

    return stats


# ============================================================
# CURVE GRAPH
# ============================================================

def diagnostics_curve(pack: dict) -> dict:

    X_te_fair = pack["X_te_fair"]

    t_tr = pack["t_tr"]
    t_te = pack["t_te"]

    # Match original curve implementation:
    # filter test according to train max follow-up time
    tr_max = float(
        np.max(t_tr)
    )

    eps = 1e-12

    mask_ibs = (
        t_te < (tr_max - eps)
    )

    X_te_fair_filtered = (
        X_te_fair[mask_ibs]
    )

    # Curve original code builds graph AFTER filtering
    W_te = build_W_tau_only(
        X_te_fair_filtered,
        THETA,
        TAU
    )

    stats = compute_graph_diagnostics(
        W_te
    )

    stats.update({
        "formulation": "Curve",
        "dataset": pack["dataset"],
        "graph_scope": "ibs_filtered_test",
        "lambda_regime": "lambda>0",
        "tau": TAU,
        "theta": THETA,

        "original_test_n": int(
            len(t_te)
        ),

        "filtered_test_n": int(
            mask_ibs.sum()
        ),

        "filtered_out_n": int(
            (~mask_ibs).sum()
        ),
    })

    return stats


# ============================================================
# MAIN
# ============================================================

datasets = [
    "flchain",
    "whas",
    "rossi",
    "employee"
]

rows = []

for dataset in datasets:

    print("\n" + "=" * 70)
    print(f"DATASET: {dataset}")
    print("=" * 70)

    pack = prepare_dataset(dataset)

    # ------------------------
    # DeepSurv
    # ------------------------
    ds_stats = diagnostics_deepsurv(
        pack
    )

    rows.append(
        ds_stats
    )

    print("\n[DeepSurv]")
    print(ds_stats)

    # ------------------------
    # Curve
    # ------------------------
    curve_stats = diagnostics_curve(
        pack
    )

    rows.append(
        curve_stats
    )

    print("\n[Curve]")
    print(curve_stats)


# ============================================================
# SAVE
# ============================================================

df_out = pd.DataFrame(rows)

# Nice column order
preferred_cols = [

    "formulation",
    "dataset",
    "graph_scope",
    "lambda_regime",

    "test_n",

    "original_test_n",
    "filtered_test_n",
    "filtered_out_n",

    "edges_undirected",
    "graph_density",

    "covered_nodes",
    "coverage",

    "isolated_nodes",
    "isolated_rate",

    "mean_degree",
    "median_degree",

    "tau",
    "theta",
]

cols = (
    [c for c in preferred_cols
     if c in df_out.columns]
    +
    [c for c in df_out.columns
     if c not in preferred_cols]
)

df_out = df_out[cols]

out_path = os.path.join(
    OUTPUT_DIR,
    "survival_graph_coverage_lambda_positive.csv"
)

df_out.to_csv(
    out_path,
    index=False
)

print("\n")
print("=" * 70)
print("FINAL RESULTS")
print("=" * 70)

print(
    df_out.to_string(index=False)
)

print(
    f"\nSaved -> {out_path}"
)