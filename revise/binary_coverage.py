# ============================================================
# binary_pairwise_fairness_review.py
# ============================================================
#
# Post-hoc independent pairwise fairness evaluation for
# binary classification:
#
#   Adult
#   COMPAS
#
# Methods:
#   URW-FKL  -> aggforward
#   URW-RKL  -> aggreverse
#
# Prediction score:
#   pred_prob
#
# Graph reconstruction EXACTLY follows the main experiment:
#
#   standardized non-sensitive X
#       -> fair projection
#       -> SVD metric transform (--use_metric)
#       -> Euclidean tau graph
#
# Since e_dist_sq returns squared Euclidean distance:
#
#       D2 <= tau**2
#
# Independent edge metrics:
#
#   EdgeMeanAbs
#   EdgeMedianAbs
#   EdgeRMS
#   EdgeMaxAbs
#
# Node-level pairwise diagnostics:
#
#   NodeMeanAbs
#   NodeRMS
#   NodeMaxAbs
#
# ============================================================

import os
import re
import glob

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD


# ============================================================
# CONFIG
# ============================================================

DATA_DIR = "../data"

RESULT_ROOT = "../binary/results_unified"

OUTPUT_DIR = os.path.join(
    RESULT_ROOT,
    "reviewer_pairwise_binary"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


SEED = 0

TAU = 5.0

THETA = 1e-3

SVD_RANK = 1

METRIC_RANK = 64

USE_METRIC = True


# ============================================================
# DISTANCE / GRAPH
# ============================================================

def e_dist_sq(A, B):

    A = np.asarray(
        A,
        dtype=np.float64
    )

    B = np.asarray(
        B,
        dtype=np.float64
    )

    A2 = (
        A ** 2
    ).sum(
        axis=1,
        keepdims=True
    )

    B2 = (
        B ** 2
    ).sum(
        axis=1,
        keepdims=True
    ).T

    return np.maximum(
        A2
        - 2.0 * (A @ B.T)
        + B2,
        0.0
    )


def build_W_tau_only(
    X,
    theta=THETA,
    tau=TAU
):

    """
    Exact graph convention from training script.

    e_dist_sq returns squared Euclidean distances.

    Therefore:

        ||zi-zj|| <= tau

    is implemented as:

        D2 <= tau**2
    """

    D2 = e_dist_sq(
        X,
        X
    )

    mask = (
        D2 <= float(tau) ** 2
    )

    W = (
        np.exp(
            -float(theta) * D2
        )
        *
        mask.astype(
            np.float64
        )
    )

    # Remove diagonal first
    np.fill_diagonal(
        W,
        0.0
    )

    # Symmetrize
    W = 0.5 * (
        W + W.T
    )

    # Restore explicit self-loops
    n = W.shape[0]

    W[
        np.arange(n),
        np.arange(n)
    ] = 1.0

    return W


# ============================================================
# GRAPH DIAGNOSTICS
# ============================================================

def graph_diagnostics(W):

    W_off = W.copy()

    np.fill_diagonal(
        W_off,
        0.0
    )

    A = (
        W_off > 0
    )

    degree = A.sum(
        axis=1
    )

    covered = (
        degree > 0
    )

    n = len(degree)

    edge_i, edge_j = np.where(
        np.triu(
            A,
            k=1
        )
    )

    n_edges = len(
        edge_i
    )

    possible_edges = (
        n * (n - 1) / 2
    )

    density = (
        n_edges / possible_edges
        if possible_edges > 0
        else np.nan
    )

    return {

        "n_nodes":
            int(n),

        "n_edges":
            int(n_edges),

        "edge_density":
            float(density),

        "coverage":
            float(
                covered.mean()
            ),

        "isolated_rate":
            float(
                (~covered).mean()
            ),

        "avg_degree":
            float(
                degree.mean()
            ),

        "median_degree":
            float(
                np.median(
                    degree
                )
            ),

        "min_degree":
            int(
                degree.min()
            ),

        "max_degree":
            int(
                degree.max()
            ),

        "degree":
            degree,

        "covered":
            covered,
    }


# ============================================================
# FAIR PROJECTION
# ============================================================

def fit_fair_projection(
    X_train,
    sens_train,
    seed=SEED
):

    X_train = np.asarray(
        X_train,
        dtype=np.float32
    )

    sens_train = np.asarray(
        sens_train
    )

    d = X_train.shape[1]

    mu0 = (
        X_train[
            sens_train == 0
        ]
        .mean(
            axis=0,
            keepdims=True
        )
    )

    mu1 = (
        X_train[
            sens_train == 1
        ]
        .mean(
            axis=0,
            keepdims=True
        )
    )

    prototypes = np.vstack(
        [
            mu0,
            mu1
        ]
    ).astype(
        np.float32
    )

    rank = min(
        SVD_RANK,
        prototypes.shape[0],
        d
    )

    svd = TruncatedSVD(
        n_components=rank,
        random_state=seed
    )

    svd.fit(
        prototypes
    )

    basis = (
        svd.components_
        .T
        .astype(
            np.float32
        )
    )

    proj = (
        basis
        @ np.linalg.pinv(
            basis.T @ basis
        )
        @ basis.T
    ).astype(
        np.float32
    )

    proj_compl = (
        np.eye(
            d,
            dtype=np.float32
        )
        -
        proj
    )

    return proj_compl


# ============================================================
# METRIC TRANSFORMATION
# ============================================================

def fit_metric_transform(
    X_train_fair,
    seed=SEED
):

    """
    Reproduce:

        svd_metric = TruncatedSVD(...)
        svd_metric.fit(X_tr_fair_full)
        L_metric = svd_metric.components_

    from the original experiment.
    """

    fair_dim = (
        X_train_fair.shape[1]
    )

    metric_rank_used = min(
        METRIC_RANK,
        fair_dim,
        max(
            X_train_fair.shape[0] - 1,
            1
        )
    )

    svd_metric = TruncatedSVD(
        n_components=metric_rank_used,
        random_state=seed
    )

    svd_metric.fit(
        X_train_fair
    )

    L_metric = (
        svd_metric.components_
        .astype(
            np.float32
        )
    )

    return (
        L_metric,
        metric_rank_used
    )


# ============================================================
# ADULT LOADER
# ============================================================

def load_adult_clean():

    headers = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
        "y",
    ]

    train_path = os.path.join(
        DATA_DIR,
        "adult",
        "adult.data"
    )

    test_path = os.path.join(
        DATA_DIR,
        "adult",
        "adult.test"
    )

    train = pd.read_csv(
        train_path,
        header=None,
        sep=",",
        skipinitialspace=True,
        na_values="?"
    )

    test = pd.read_csv(
        test_path,
        header=None,
        sep=",",
        skipinitialspace=True,
        na_values="?",
        comment="|"
    )

    df = pd.concat(
        [
            train,
            test
        ],
        axis=0,
        ignore_index=True
    )

    df.columns = headers

    df["y"] = (
        df["y"]
        .astype(str)
        .str.replace(
            ".",
            "",
            regex=False
        )
        .str.strip()
    )

    df["y"] = (
        df["y"]
        .map({
            "<=50K": 0,
            ">50K": 1
        })
    )

    df = df.dropna(
        subset=["y"]
    )

    df["y"] = (
        df["y"]
        .astype(int)
    )

    df = df.dropna()

    df = pd.get_dummies(
        df,
        drop_first=False
    )

    native_cols = [
        c
        for c in df.columns
        if c.startswith(
            "native-country_"
        )
    ]

    df = df.drop(
        columns=native_cols,
        errors="ignore"
    )

    drop_cols = [
        c
        for c in df.columns
        if (
            c.startswith(
                "fnlwgt"
            )
            or
            c.startswith(
                "education_"
            )
        )
    ]

    df = df.drop(
        columns=drop_cols,
        errors="ignore"
    )

    df["orig_index"] = np.arange(
        len(df)
    )

    return df


# ============================================================
# COMPAS LOADER
# ============================================================

def load_compas_clean():

    path = os.path.join(
        DATA_DIR,
        "compas",
        "compas-scores-two-years.csv"
    )

    df = pd.read_csv(
        path
    )

    keep_cols = [
        "age",
        "priors_count",
        "juv_fel_count",
        "juv_misd_count",
        "juv_other_count",
        "c_charge_degree",
        "sex",
        "race",
        "two_year_recid",
    ]

    missing = [
        c
        for c in keep_cols
        if c not in df.columns
    ]

    if missing:

        raise ValueError(
            f"COMPAS missing columns: "
            f"{missing}"
        )

    df = (
        df[
            keep_cols
        ]
        .copy()
        .dropna()
    )

    df["y"] = (
        df["two_year_recid"]
        .astype(int)
    )

    df = df.drop(
        columns=[
            "two_year_recid"
        ]
    )

    for c in [
        "age",
        "priors_count",
        "juv_fel_count",
        "juv_misd_count",
        "juv_other_count",
    ]:

        df[c] = pd.to_numeric(
            df[c],
            errors="coerce"
        )

    df = df.dropna()

    df = pd.get_dummies(
        df,
        columns=[
            "c_charge_degree",
            "sex",
            "race"
        ],
        drop_first=False
    )

    df["orig_index"] = np.arange(
        len(df)
    )

    return df


# ============================================================
# PREPARE TEST GRAPH
# ============================================================

def prepare_binary_graph(
    dataset
):

    if dataset == "adult":

        df = load_adult_clean()

    elif dataset == "compas":

        df = load_compas_clean()

    else:

        raise ValueError(
            dataset
        )

    y_full = (
        df["y"]
        .astype(int)
        .values
    )

    orig_index_full = (
        df["orig_index"]
        .astype(int)
        .values
    )

    if "sex_Female" in df.columns:

        sens_full = (
            df["sex_Female"]
            .astype(int)
            .values
        )

    elif "sex_Male" in df.columns:

        sens_full = (
            1
            -
            df["sex_Male"]
            .astype(int)
            .values
        )

    else:

        raise ValueError(
            f"No sex dummy found "
            f"for {dataset}"
        )

    drop_for_X = (
        [
            c
            for c in df.columns
            if c.startswith(
                "sex_"
            )
        ]
        +
        [
            "y",
            "orig_index"
        ]
    )

    X = (
        df.drop(
            columns=drop_for_X
        )
        .astype(
            np.float32
        )
        .values
    )

    (
        X_tr,
        X_tmp,
        y_tr,
        y_tmp,
        s_tr,
        s_tmp,
        rid_tr,
        rid_tmp
    ) = train_test_split(

        X,
        y_full,
        sens_full,
        orig_index_full,

        test_size=0.4,

        random_state=SEED,

        stratify=y_full
    )

    (
        X_va,
        X_te,
        y_va,
        y_te,
        s_va,
        s_te,
        rid_va,
        rid_te
    ) = train_test_split(

        X_tmp,
        y_tmp,
        s_tmp,
        rid_tmp,

        test_size=0.5,

        random_state=SEED,

        stratify=y_tmp
    )

    # --------------------------------------------------------
    # Standardization
    # --------------------------------------------------------

    scaler = StandardScaler()

    scaler.fit(
        X_tr
    )

    X_tr = (
        scaler.transform(
            X_tr
        )
        .astype(
            np.float32
        )
    )

    X_te = (
        scaler.transform(
            X_te
        )
        .astype(
            np.float32
        )
    )

    # --------------------------------------------------------
    # Fair projection
    # --------------------------------------------------------

    proj_compl = (
        fit_fair_projection(
            X_tr,
            s_tr,
            SEED
        )
    )

    X_tr_fair = (
        X_tr
        @ proj_compl
    ).astype(
        np.float32
    )

    X_te_fair = (
        X_te
        @ proj_compl
    ).astype(
        np.float32
    )

    # --------------------------------------------------------
    # Metric transform
    # --------------------------------------------------------

    if USE_METRIC:

        (
            L_metric,
            metric_rank_used
        ) = fit_metric_transform(
            X_tr_fair,
            SEED
        )

        X_te_metric = (
            X_te_fair
            @ L_metric.T
        )

    else:

        L_metric = None

        metric_rank_used = 0

        X_te_metric = (
            X_te_fair
        )

    # --------------------------------------------------------
    # Test graph
    # --------------------------------------------------------

    W_te = build_W_tau_only(
        X_te_metric,
        THETA,
        TAU
    )

    diag = graph_diagnostics(
        W_te
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        f"GRAPH: {dataset}"
    )

    print(
        "=" * 80
    )

    print(
        "test n        =",
        diag["n_nodes"]
    )

    print(
        "metric rank   =",
        metric_rank_used
    )

    print(
        "edge density  =",
        diag["edge_density"]
    )

    print(
        "coverage      =",
        diag["coverage"]
    )

    print(
        "isolated rate =",
        diag["isolated_rate"]
    )

    print(
        "median degree =",
        diag["median_degree"]
    )

    return {

        "idx":
            rid_te.astype(int),

        "W":
            W_te,

        "metric_rank":
            metric_rank_used,

        **diag,
    }


# ============================================================
# RESULT FILE DISCOVERY
# ============================================================

def parse_lambda(path):

    name = os.path.basename(
        path
    )

    m = re.search(
        r"_lambda([-+0-9.eE]+)\.csv$",
        name
    )

    if m is None:
        return None

    return float(
        m.group(1)
    )


def find_prediction_files(
    dataset,
    agg
):

    pattern = os.path.join(
        RESULT_ROOT,
        f"if_{dataset}_agg{agg}_"
        f"depth*_lambda*.csv"
    )

    files = glob.glob(
        pattern
    )

    clean = []

    for path in files:

        name = (
            os.path.basename(
                path
            )
            .lower()
        )

        # Exclude tau sensitivity outputs
        if "_tauq" in name:
            continue

        lam = parse_lambda(
            path
        )

        if lam is None:
            continue

        df_head = pd.read_csv(
            path,
            nrows=2
        )

        required = {
            "orig_index",
            "pred_prob"
        }

        if required.issubset(
            df_head.columns
        ):

            clean.append(
                path
            )

    return sorted(
        clean,
        key=lambda p: parse_lambda(p)
    )


# ============================================================
# METRICS FILE LOOKUP
# ============================================================

def matching_metrics_file(
    dataset,
    agg,
    lam
):

    candidates = [

        os.path.join(
            RESULT_ROOT,
            f"metrics_urw_{dataset}_"
            f"agg{agg}_depth1_"
            f"lambda{lam}.csv"
        ),

        os.path.join(
            RESULT_ROOT,
            f"metrics_urw_{dataset}_"
            f"agg{agg}_depth1_"
            f"lambda{float(lam)}.csv"
        ),
    ]

    for p in candidates:

        if os.path.exists(
            p
        ):
            return p

    glob_pattern = os.path.join(
        RESULT_ROOT,
        f"metrics_urw_{dataset}_"
        f"agg{agg}_depth*_lambda*.csv"
    )

    for p in glob.glob(
        glob_pattern
    ):

        if "_tauq" in p:
            continue

        lp = parse_lambda(
            p
        )

        if (
            lp is not None
            and
            np.isclose(
                lp,
                lam
            )
        ):
            return p

    return None


# ============================================================
# ALIGN PREDICTIONS TO GRAPH
# ============================================================

def align_predictions(
    path,
    graph_idx
):

    df = pd.read_csv(
        path
    )

    file_idx = (
        df["orig_index"]
        .astype(int)
        .values
    )

    pred = (
        df["pred_prob"]
        .astype(float)
        .values
    )

    graph_idx = np.asarray(
        graph_idx,
        dtype=int
    )

    if len(file_idx) != len(
        graph_idx
    ):

        raise ValueError(
            f"\nSize mismatch:\n"
            f"{path}\n"
            f"file={len(file_idx)}, "
            f"graph={len(graph_idx)}"
        )

    if np.array_equal(
        file_idx,
        graph_idx
    ):

        return pred

    if (
        set(file_idx.tolist())
        ==
        set(graph_idx.tolist())
    ):

        pred_map = {

            int(i):
                float(p)

            for i, p
            in zip(
                file_idx,
                pred
            )
        }

        return np.array(
            [
                pred_map[
                    int(i)
                ]
                for i in graph_idx
            ],
            dtype=float
        )

    missing = (
        set(graph_idx)
        -
        set(file_idx)
    )

    extra = (
        set(file_idx)
        -
        set(graph_idx)
    )

    raise ValueError(
        "\nID membership mismatch:\n"
        f"{path}\n"
        f"missing={list(missing)[:10]}\n"
        f"extra={list(extra)[:10]}"
    )


# ============================================================
# EDGE + NODE PAIRWISE DIAGNOSTICS
# ============================================================

def pairwise_diagnostics(
    scores,
    W
):

    scores = np.asarray(
        scores,
        dtype=float
    ).reshape(-1)

    n = len(
        scores
    )

    W_off = W.copy()

    np.fill_diagonal(
        W_off,
        0.0
    )

    A = (
        W_off > 0
    )

    # --------------------------------------------------------
    # Unique undirected edges
    # --------------------------------------------------------

    edge_i, edge_j = np.where(
        np.triu(
            A,
            k=1
        )
    )

    diff = np.abs(
        scores[edge_i]
        -
        scores[edge_j]
    )

    # --------------------------------------------------------
    # Global edge statistics
    # --------------------------------------------------------

    if len(diff) == 0:

        edge_mean = np.nan
        edge_median = np.nan
        edge_rms = np.nan
        edge_max = np.nan

    else:

        edge_mean = float(
            np.mean(
                diff
            )
        )

        edge_median = float(
            np.median(
                diff
            )
        )

        edge_rms = float(
            np.sqrt(
                np.mean(
                    diff ** 2
                )
            )
        )

        edge_max = float(
            np.max(
                diff
            )
        )

    # --------------------------------------------------------
    # Node-level pairwise discrepancy
    #
    # For each covered node i:
    #
    # mean_j |p_i-p_j|
    # RMS_j  |p_i-p_j|
    # max_j  |p_i-p_j|
    #
    # Self-loops excluded.
    # --------------------------------------------------------

    node_mean = np.full(
        n,
        np.nan
    )

    node_rms = np.full(
        n,
        np.nan
    )

    node_max = np.full(
        n,
        np.nan
    )

    degree = A.sum(
        axis=1
    )

    for i in range(
        n
    ):

        nbr = np.where(
            A[i]
        )[0]

        if len(nbr) == 0:
            continue

        d = np.abs(
            scores[i]
            -
            scores[nbr]
        )

        node_mean[i] = np.mean(
            d
        )

        node_rms[i] = np.sqrt(
            np.mean(
                d ** 2
            )
        )

        node_max[i] = np.max(
            d
        )

    covered = (
        degree > 0
    )

    return {

        "edge_mean_abs":
            edge_mean,

        "edge_median_abs":
            edge_median,

        "edge_rms":
            edge_rms,

        "edge_max_abs":
            edge_max,

        "node_mean_abs_mean":
            float(
                np.nanmean(
                    node_mean
                )
            ),

        "node_mean_abs_median":
            float(
                np.nanmedian(
                    node_mean
                )
            ),

        "node_rms_mean":
            float(
                np.nanmean(
                    node_rms
                )
            ),

        "node_rms_median":
            float(
                np.nanmedian(
                    node_rms
                )
            ),

        "node_max_abs_mean":
            float(
                np.nanmean(
                    node_max
                )
            ),

        "node_max_abs_median":
            float(
                np.nanmedian(
                    node_max
                )
            ),

        "n_covered_nodes":
            int(
                covered.sum()
            ),

        "node_mean":
            node_mean,

        "node_rms":
            node_rms,

        "node_max":
            node_max,
    }


# ============================================================
# OPTIONAL:
# READ ORIGINAL URW RESIDUAL / KL IF COLUMN
# ============================================================

def detect_if_column(df):

    candidates = [

        c
        for c in df.columns

        if (
            c.startswith(
                "if_score_forwardKL"
            )
            or
            c.startswith(
                "if_score_reverseKL"
            )
        )
    ]

    if len(candidates) == 0:

        return None

    return candidates[0]


# ============================================================
# RUN
# ============================================================

summary_rows = []

node_rows = []


for dataset in [
    "adult",
    "compas"
]:

    graph = prepare_binary_graph(
        dataset
    )

    for agg in [
        "forward",
        "reverse"
    ]:

        formulation = (
            "FKL"
            if agg == "forward"
            else "RKL"
        )

        files = find_prediction_files(
            dataset,
            agg
        )

        print(
            "\n"
            + "-" * 80
        )

        print(
            dataset,
            formulation,
            "files:",
            len(files)
        )

        print(
            "-" * 80
        )

        for path in files:

            lam = parse_lambda(
                path
            )

            df_raw = pd.read_csv(
                path
            )

            pred = align_predictions(
                path,
                graph["idx"]
            )

            diag = pairwise_diagnostics(
                pred,
                graph["W"]
            )

            # ------------------------------------------------
            # Summary row
            # ------------------------------------------------

            row = {

                "dataset":
                    dataset,

                "formulation":
                    formulation,

                "agg":
                    agg,

                "lambda":
                    lam,

                "score":
                    "pred_prob",

                "tau":
                    TAU,

                "theta":
                    THETA,

                "use_metric":
                    USE_METRIC,

                "metric_rank":
                    graph[
                        "metric_rank"
                    ],

                "n_nodes":
                    graph[
                        "n_nodes"
                    ],

                "n_edges":
                    graph[
                        "n_edges"
                    ],

                "edge_density":
                    graph[
                        "edge_density"
                    ],

                "coverage":
                    graph[
                        "coverage"
                    ],

                "isolated_rate":
                    graph[
                        "isolated_rate"
                    ],

                "avg_degree":
                    graph[
                        "avg_degree"
                    ],

                "median_degree":
                    graph[
                        "median_degree"
                    ],

                "EdgeMeanAbs":
                    diag[
                        "edge_mean_abs"
                    ],

                "EdgeMedianAbs":
                    diag[
                        "edge_median_abs"
                    ],

                "EdgeRMS":
                    diag[
                        "edge_rms"
                    ],

                "EdgeMaxAbs":
                    diag[
                        "edge_max_abs"
                    ],

                "NodeMeanAbs_mean":
                    diag[
                        "node_mean_abs_mean"
                    ],

                "NodeMeanAbs_median":
                    diag[
                        "node_mean_abs_median"
                    ],

                "NodeRMS_mean":
                    diag[
                        "node_rms_mean"
                    ],

                "NodeRMS_median":
                    diag[
                        "node_rms_median"
                    ],

                "NodeMaxAbs_mean":
                    diag[
                        "node_max_abs_mean"
                    ],

                "NodeMaxAbs_median":
                    diag[
                        "node_max_abs_median"
                    ],

                "file":
                    os.path.basename(
                        path
                    ),
            }


            # ------------------------------------------------
            # Compare reconstructed graph with saved metrics
            # ------------------------------------------------

            metrics_path = matching_metrics_file(
                dataset,
                agg,
                lam
            )

            if metrics_path is not None:

                mdf = pd.read_csv(
                    metrics_path
                )

                if len(mdf) > 0:

                    mr = mdf.iloc[0]

                    for old_name, new_name in [

                        (
                            "graph_test_edge_density",
                            "saved_edge_density"
                        ),

                        (
                            "graph_test_isolated_rate",
                            "saved_isolated_rate"
                        ),

                        (
                            "graph_test_avg_degree",
                            "saved_avg_degree"
                        ),

                        (
                            "graph_test_median_degree",
                            "saved_median_degree"
                        ),

                        (
                            "use_metric",
                            "saved_use_metric"
                        ),

                        (
                            "metric_rank",
                            "saved_metric_rank"
                        ),
                    ]:

                        if old_name in mdf.columns:

                            row[new_name] = (
                                mr[
                                    old_name
                                ]
                            )


            summary_rows.append(
                row
            )


            # ------------------------------------------------
            # Node-level output
            # ------------------------------------------------

            if_col = detect_if_column(
                df_raw
            )

            # Reorder entire dataframe to graph idx
            df_indexed = (
                df_raw
                .set_index(
                    "orig_index"
                )
                .loc[
                    graph["idx"]
                ]
                .reset_index()
            )

            for i in range(
                graph["n_nodes"]
            ):

                nr = {

                    "dataset":
                        dataset,

                    "formulation":
                        formulation,

                    "lambda":
                        lam,

                    "orig_index":
                        int(
                            graph["idx"][i]
                        ),

                    "pred_prob":
                        float(
                            pred[i]
                        ),

                    "degree":
                        int(
                            graph[
                                "degree"
                            ][i]
                        ),

                    "covered":
                        bool(
                            graph[
                                "covered"
                            ][i]
                        ),

                    "pair_mean_abs":
                        diag[
                            "node_mean"
                        ][i],

                    "pair_rms":
                        diag[
                            "node_rms"
                        ][i],

                    "pair_max_abs":
                        diag[
                            "node_max"
                        ][i],
                }

                if if_col is not None:

                    nr[
                        "original_IF_score"
                    ] = float(
                        df_indexed[
                            if_col
                        ].iloc[i]
                    )

                node_rows.append(
                    nr
                )


# ============================================================
# SAVE
# ============================================================

summary = pd.DataFrame(
    summary_rows
)

summary = summary.sort_values(
    [
        "dataset",
        "formulation",
        "lambda"
    ]
).reset_index(
    drop=True
)


nodes = pd.DataFrame(
    node_rows
)

nodes = nodes.sort_values(
    [
        "dataset",
        "formulation",
        "lambda",
        "orig_index"
    ]
).reset_index(
    drop=True
)


summary_path = os.path.join(
    OUTPUT_DIR,
    "binary_pairwise_summary.csv"
)

node_path = os.path.join(
    OUTPUT_DIR,
    "binary_pairwise_node_diagnostics.csv"
)


summary.to_csv(
    summary_path,
    index=False
)

nodes.to_csv(
    node_path,
    index=False
)


# ============================================================
# PRINT MAIN SUMMARY
# ============================================================

print(
    "\n\n"
    + "=" * 120
)

print(
    "FINAL BINARY PAIRWISE SUMMARY"
)

print(
    "=" * 120
)


display_cols = [

    "dataset",

    "formulation",

    "lambda",

    "coverage",

    "isolated_rate",

    "edge_density",

    "EdgeMeanAbs",

    "EdgeMedianAbs",

    "EdgeRMS",

    "EdgeMaxAbs",

    "NodeMeanAbs_mean",

    "NodeRMS_mean",

    "NodeMaxAbs_mean",
]


print(
    summary[
        display_cols
    ].to_string(
        index=False
    )
)


print(
    "\nSaved:"
)

print(
    summary_path
)

print(
    node_path
)


# ============================================================
# GRAPH RECONSTRUCTION CHECK
# ============================================================

print(
    "\n\n"
    + "=" * 120
)

print(
    "GRAPH RECONSTRUCTION CHECK"
)

print(
    "=" * 120
)


check_cols = [

    "dataset",

    "formulation",

    "lambda",

    "edge_density",

    "saved_edge_density",

    "isolated_rate",

    "saved_isolated_rate",

    "median_degree",

    "saved_median_degree",

    "metric_rank",

    "saved_metric_rank",
]


existing_check_cols = [

    c
    for c in check_cols
    if c in summary.columns
]


print(
    summary[
        existing_check_cols
    ].to_string(
        index=False
    )
)