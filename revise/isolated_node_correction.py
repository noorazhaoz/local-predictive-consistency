import os
import re
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD


# ============================================================
# ONLY CHANGE THESE PATHS
# ============================================================

# 1. Adult raw data
ADULT_DATA_PATH = "../data/adult/adult.data"
ADULT_TEST_PATH = "../data/adult/adult.test"

# 2. COMPAS raw data
COMPAS_DATA_PATH = "../data/compas/compas-scores-two-years.csv"

# 3. Folder containing your saved IF CSV files
#    e.g.
#    if_adult_depth1_lambda0.1.csv
#    if_adult_aggforward_depth1_lambda0.1.csv
#    if_adult_aggreverse_depth1_lambda0.1.csv
#    etc.
RESULTS_DIR = "../binary/results_unified"

# 4. Folder where the new reviewer-analysis CSVs will be saved
OUTPUT_DIR = "reviewer_isolated_node_analysis"


# ============================================================
# ORIGINAL EXPERIMENT SETTINGS
# ============================================================

SEED = 0
TAU = 5.0
THETA = 1e-3
SVD_RANK = 1

EPS_LIST = [
    0.0003,
    0.0008,
    0.001,
    0.003,
    0.01,
    0.05,
    0.1,
]


# ============================================================
# GRAPH FUNCTIONS
# ============================================================

def e_dist_sq(A, B):
    A2 = (A ** 2).sum(axis=1, keepdims=True)
    B2 = (B ** 2).sum(axis=1, keepdims=True).T

    return np.maximum(
        A2 - 2.0 * (A @ B.T) + B2,
        0.0
    )


def build_W_tau_only(
    X_for_metric,
    theta=THETA,
    tau=TAU
):
    """
    Same graph definition as original experiment.

    Note:
        tau is applied to SQUARED Euclidean distance:
            D^2 <= tau

    Self-loop is added after graph construction.
    """

    D2 = e_dist_sq(
        X_for_metric,
        X_for_metric
    )

    W = (
        np.exp(-theta * D2)
        * (D2 <= tau).astype(np.float64)
    )

    # Remove diagonal
    np.fill_diagonal(
        W,
        0.0
    )

    # Symmetrize
    W = 0.5 * (
        W + W.T
    )

    # Add self-loops
    n = W.shape[0]

    W[
        np.arange(n),
        np.arange(n)
    ] = 1.0

    return W


# ============================================================
# LOAD ADULT
# ============================================================

def load_adult_clean(
    adult_data_path,
    adult_test_path
):

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
        "y"
    ]

    train = pd.read_csv(
        adult_data_path,
        header=None,
        sep=",",
        skipinitialspace=True,
        na_values="?"
    )

    test = pd.read_csv(
        adult_test_path,
        header=None,
        sep=",",
        skipinitialspace=True,
        na_values="?",
        comment="|"
    )

    df = pd.concat(
        [train, test],
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

    df["y"] = df["y"].map({
        "<=50K": 0,
        ">50K": 1
    })

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

    # Same preprocessing as original code
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
            c.startswith("fnlwgt")
            or c.startswith("education_")
        )
    ]

    df = df.drop(
        columns=drop_cols,
        errors="ignore"
    )

    df["orig_index"] = (
        np.arange(len(df))
    )

    return df


# ============================================================
# LOAD COMPAS
# ============================================================

def load_compas_clean(
    compas_path
):

    df = pd.read_csv(
        compas_path
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
        "two_year_recid"
    ]

    missing = [
        c
        for c in keep_cols
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"COMPAS missing columns: {missing}"
        )

    df = (
        df[keep_cols]
        .copy()
        .dropna()
    )

    df["y"] = (
        df["two_year_recid"]
        .astype(int)
    )

    df = df.drop(
        columns=["two_year_recid"]
    )

    numeric_cols = [
        "age",
        "priors_count",
        "juv_fel_count",
        "juv_misd_count",
        "juv_other_count"
    ]

    for c in numeric_cols:

        df[c] = pd.to_numeric(
            df[c],
            errors="coerce"
        )

    df = df.dropna()

    cat_cols = [
        "c_charge_degree",
        "sex",
        "race"
    ]

    df = pd.get_dummies(
        df,
        columns=cat_cols,
        drop_first=False
    )

    df["orig_index"] = (
        np.arange(len(df))
    )

    return df


# ============================================================
# RECONSTRUCT TEST GRAPH
# ============================================================

def reconstruct_test_graph(
    df,
    seed=SEED,
    tau=TAU,
    theta=THETA,
    svd_rank=SVD_RANK
):

    y_full = (
        df["y"]
        .astype(int)
        .values
    )

    orig_index_full = (
        df["orig_index"]
        .values
    )

    # Same sensitive-variable logic
    if "sex_Female" in df.columns:

        sens_full = (
            df["sex_Female"]
            .astype(int)
            .values
        )

    elif "sex_Male" in df.columns:

        sens_full = (
            1
            - df["sex_Male"]
            .astype(int)
            .values
        )

    else:

        raise ValueError(
            "Cannot identify sex dummy variable."
        )

    # Same feature construction
    drop_for_X = (
        [
            c
            for c in df.columns
            if c.startswith("sex_")
        ]
        + [
            "y",
            "orig_index"
        ]
    )

    X = (
        df
        .drop(
            columns=drop_for_X
        )
        .astype(np.float32)
        .values
    )

    # --------------------------------------------------------
    # Original 60/20/20 split
    # --------------------------------------------------------

    (
        X_tr,
        X_tmp,
        y_tr,
        y_tmp,
        sens_tr,
        sens_tmp,
        rid_tr,
        rid_tmp
    ) = train_test_split(
        X,
        y_full,
        sens_full,
        orig_index_full,
        test_size=0.4,
        random_state=seed,
        stratify=y_full
    )

    (
        X_va,
        X_te,
        y_va,
        y_te,
        sens_va,
        sens_te,
        rid_va,
        rid_te
    ) = train_test_split(
        X_tmp,
        y_tmp,
        sens_tmp,
        rid_tmp,
        test_size=0.5,
        random_state=seed,
        stratify=y_tmp
    )

    # --------------------------------------------------------
    # StandardScaler fitted on train only
    # --------------------------------------------------------

    scaler = StandardScaler()

    scaler.fit(
        X_tr
    )

    X_tr = (
        scaler
        .transform(X_tr)
        .astype(np.float32)
    )

    X_te = (
        scaler
        .transform(X_te)
        .astype(np.float32)
    )

    d = X_tr.shape[1]

    # --------------------------------------------------------
    # Fair-space projection
    # --------------------------------------------------------

    mu0 = (
        X_tr[
            sens_tr == 0
        ]
        .mean(
            axis=0,
            keepdims=True
        )
    )

    mu1 = (
        X_tr[
            sens_tr == 1
        ]
        .mean(
            axis=0,
            keepdims=True
        )
    )

    prot = np.vstack(
        [mu0, mu1]
    ).astype(np.float32)

    svd_rank_used = min(
        svd_rank,
        prot.shape[0],
        d
    )

    svd = TruncatedSVD(
        n_components=svd_rank_used,
        random_state=seed
    )

    svd.fit(
        prot
    )

    basis = (
        svd.components_.T
    )

    proj = (
        basis
        @ np.linalg.pinv(
            basis.T @ basis
        )
        @ basis.T
    )

    proj_compl = (
        np.eye(
            d,
            dtype=np.float32
        )
        - proj
    )

    X_te_fair = (
        X_te
        @ proj_compl
    )

    # --------------------------------------------------------
    # Test graph
    # --------------------------------------------------------

    W_te = build_W_tau_only(
        X_te_fair,
        theta=theta,
        tau=tau
    )

    return (
        W_te,
        rid_te
    )


# ============================================================
# GRAPH COVERAGE DIAGNOSTICS
# ============================================================

def graph_diagnostics(
    W
):

    n = W.shape[0]

    # --------------------------------------------------------
    # IMPORTANT:
    # Remove self-loop before checking isolation
    # --------------------------------------------------------

    W_off = (
        W.copy()
    )

    np.fill_diagonal(
        W_off,
        0.0
    )

    adjacency = (
        W_off > 0
    )

    degree = (
        adjacency
        .sum(axis=1)
    )

    covered_mask = (
        degree > 0
    )

    isolated_mask = (
        degree == 0
    )

    n_covered = int(
        covered_mask.sum()
    )

    n_isolated = int(
        isolated_mask.sum()
    )

    n_edges = int(
        np.triu(
            adjacency,
            k=1
        ).sum()
    )

    possible_edges = (
        n * (n - 1) / 2
    )

    graph_density = (
        n_edges / possible_edges
        if possible_edges > 0
        else np.nan
    )

    result = {
        "test_n": n,
        "edges_undirected": n_edges,
        "graph_density": graph_density,
        "covered_nodes": n_covered,
        "coverage": n_covered / n,
        "isolated_nodes": n_isolated,
        "isolated_rate": n_isolated / n,
        "mean_degree": float(
            degree.mean()
        ),
        "median_degree": float(
            np.median(degree)
        )
    }

    return (
        result,
        covered_mask,
        degree
    )


# ============================================================
# VR FUNCTIONS
# ============================================================

def compute_vr_all_and_covered(
    if_scores,
    covered_mask,
    eps
):

    x = np.asarray(
        if_scores,
        dtype=float
    )

    covered_mask = np.asarray(
        covered_mask,
        dtype=bool
    )

    if len(x) != len(
        covered_mask
    ):
        raise ValueError(
            "IF score length and covered mask "
            "length do not match."
        )

    valid = (
        ~np.isnan(x)
    )

    valid_covered = (
        valid
        & covered_mask
    )

    n_all = int(
        valid.sum()
    )

    n_covered = int(
        valid_covered.sum()
    )

    if n_all > 0:

        vr_all = float(
            np.mean(
                x[valid] > eps
            )
        )

    else:

        vr_all = np.nan

    if n_covered > 0:

        vr_covered = float(
            np.mean(
                x[
                    valid_covered
                ] > eps
            )
        )

    else:

        vr_covered = np.nan

    coverage = (
        n_covered / n_all
        if n_all > 0
        else np.nan
    )

    abs_change = (
        vr_covered
        - vr_all
        if (
            not np.isnan(vr_all)
            and not np.isnan(vr_covered)
        )
        else np.nan
    )

    rel_change = (
        vr_covered / vr_all - 1.0
        if (
            not np.isnan(vr_all)
            and not np.isnan(vr_covered)
            and vr_all > 0
        )
        else np.nan
    )

    return {
        "vr_all": vr_all,
        "vr_covered": vr_covered,
        "n_all": n_all,
        "n_covered": n_covered,
        "coverage": coverage,
        "absolute_change": abs_change,
        "relative_change": rel_change
    }


# ============================================================
# FILE MATCHING
# ============================================================

LAMBDA_RE = re.compile(
    r"_lambda(?P<lam>[0-9.eE+-]+)\.csv$",
    re.IGNORECASE
)


def parse_lambda(
    fname
):

    m = LAMBDA_RE.search(
        fname
    )

    if not m:
        return None, None

    lam_str = (
        m.group("lam")
    )

    try:

        lam_val = float(
            lam_str
        )

    except ValueError:

        lam_val = np.nan

    return (
        lam_str,
        lam_val
    )


# ============================================================
# METHOD DEFINITIONS
# ============================================================

def build_method_specs(
    dataset_name
):

    return {

        "lap_forward": {
            "prefix":
                f"if_{dataset_name}_depth1_lambda",

            "if_col":
                "if_score_forwardKL_prob"
        },

        "urw_forward": {
            "prefix":
                f"if_{dataset_name}_aggforward_depth1_lambda",

            "if_col":
                "if_score_forwardKL_node||ref"
        },

        "lap_reverse": {
            "prefix":
                f"if_{dataset_name}_depth1_lambda",

            "if_col":
                "if_score_reverseKL_ref||node"
        },

        "urw_reverse": {
            "prefix":
                f"if_{dataset_name}_aggreverse_depth1_lambda",

            "if_col":
                "if_score_reverseKL_ref||node"
        }
    }


# ============================================================
# PROCESS ONE DATASET
# ============================================================

def process_dataset(
    dataset_name,
    W_te,
    rid_te,
    results_dir,
    output_dir
):

    print(
        "\n"
        + "=" * 100
    )

    print(
        f"PROCESSING {dataset_name.upper()}"
    )

    print(
        "=" * 100
    )

    # --------------------------------------------------------
    # Graph diagnostics
    # --------------------------------------------------------

    (
        graph_stats,
        covered_mask,
        degree
    ) = graph_diagnostics(
        W_te
    )

    graph_stats[
        "dataset"
    ] = dataset_name

    print(
        "\nGraph diagnostics:"
    )

    print(
        pd.Series(
            graph_stats
        )
    )

    # --------------------------------------------------------
    # Scan IF CSV files
    # --------------------------------------------------------

    all_files = [
        f
        for f in os.listdir(
            results_dir
        )
        if f.endswith(".csv")
    ]

    method_specs = (
        build_method_specs(
            dataset_name
        )
    )

    rows = []

    for method, spec in (
        method_specs.items()
    ):

        matching_files = sorted([
            f
            for f in all_files
            if f.startswith(
                spec["prefix"]
            )
        ])

        print(
            f"\n{method}: "
            f"{len(matching_files)} matched files"
        )

        for fname in matching_files:

            path = os.path.join(
                results_dir,
                fname
            )

            df = pd.read_csv(
                path
            )

            # -----------------------------------------------
            # IF column exists?
            # -----------------------------------------------

            if (
                spec["if_col"]
                not in df.columns
            ):

                continue

            # -----------------------------------------------
            # orig_index must exist
            # -----------------------------------------------

            if "orig_index" not in df.columns:

                print(
                    f"SKIP {fname}: "
                    "orig_index missing"
                )

                continue

            file_ids = (
                df["orig_index"]
                .to_numpy()
            )

            # -----------------------------------------------
            # Critical ordering check
            # -----------------------------------------------

            if not np.array_equal(
                file_ids,
                rid_te
            ):

                raise ValueError(
                    f"\n{fname}\n"
                    "orig_index does not match "
                    "the reconstructed test split."
                )

            if_scores = (
                df[
                    spec["if_col"]
                ]
                .to_numpy()
            )

            (
                lam_str,
                lam_val
            ) = parse_lambda(
                fname
            )

            # -----------------------------------------------
            # Every original epsilon
            # -----------------------------------------------

            for eps in EPS_LIST:

                res = (
                    compute_vr_all_and_covered(
                        if_scores,
                        covered_mask,
                        eps
                    )
                )

                rows.append({
                    "dataset":
                        dataset_name,

                    "method":
                        method,

                    "lambda_str":
                        lam_str,

                    "lambda":
                        lam_val,

                    "eps":
                        eps,

                    "vr_all":
                        res[
                            "vr_all"
                        ],

                    "vr_covered":
                        res[
                            "vr_covered"
                        ],

                    "absolute_change":
                        res[
                            "absolute_change"
                        ],

                    "relative_change":
                        res[
                            "relative_change"
                        ],

                    "n_all":
                        res[
                            "n_all"
                        ],

                    "n_covered":
                        res[
                            "n_covered"
                        ],

                    "coverage":
                        res[
                            "coverage"
                        ],

                    "file":
                        fname,

                    "if_col_used":
                        spec[
                            "if_col"
                        ]
                })

    out = pd.DataFrame(
        rows
    )

    if out.empty:

        print(
            f"\nWARNING: No IF files found "
            f"for {dataset_name}"
        )

    else:

        out = (
            out
            .sort_values(
                [
                    "method",
                    "lambda",
                    "eps"
                ]
            )
            .reset_index(
                drop=True
            )
        )

        out_path = os.path.join(
            output_dir,
            f"{dataset_name}_VR_all_vs_covered.csv"
        )

        out.to_csv(
            out_path,
            index=False
        )

        print(
            f"\nSaved VR table:\n{out_path}"
        )

    return (
        graph_stats,
        out
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    # ========================================================
    # ADULT
    # ========================================================

    print(
        "\nReconstructing Adult graph..."
    )

    adult_df = (
        load_adult_clean(
            ADULT_DATA_PATH,
            ADULT_TEST_PATH
        )
    )

    (
        W_adult,
        rid_adult
    ) = reconstruct_test_graph(
        adult_df
    )

    (
        adult_graph_stats,
        adult_vr
    ) = process_dataset(
        dataset_name="adult",
        W_te=W_adult,
        rid_te=rid_adult,
        results_dir=RESULTS_DIR,
        output_dir=OUTPUT_DIR
    )

    # ========================================================
    # COMPAS
    # ========================================================

    print(
        "\nReconstructing COMPAS graph..."
    )

    compas_df = (
        load_compas_clean(
            COMPAS_DATA_PATH
        )
    )

    (
        W_compas,
        rid_compas
    ) = reconstruct_test_graph(
        compas_df
    )

    (
        compas_graph_stats,
        compas_vr
    ) = process_dataset(
        dataset_name="compas",
        W_te=W_compas,
        rid_te=rid_compas,
        results_dir=RESULTS_DIR,
        output_dir=OUTPUT_DIR
    )

    # ========================================================
    # SAVE GRAPH SUMMARY
    # ========================================================

    graph_summary = pd.DataFrame([
        adult_graph_stats,
        compas_graph_stats
    ])

    graph_summary = graph_summary[
        [
            "dataset",
            "test_n",
            "edges_undirected",
            "graph_density",
            "covered_nodes",
            "coverage",
            "isolated_nodes",
            "isolated_rate",
            "mean_degree",
            "median_degree"
        ]
    ]

    graph_path = os.path.join(
        OUTPUT_DIR,
        "graph_coverage_tau5.csv"
    )

    graph_summary.to_csv(
        graph_path,
        index=False
    )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "DONE"
    )

    print(
        "=" * 100
    )

    print(
        "\nGraph summary:"
    )

    print(
        graph_summary
    )

    print(
        "\nSaved:"
    )

    print(
        graph_path
    )

    print(
        os.path.join(
            OUTPUT_DIR,
            "adult_VR_all_vs_covered.csv"
        )
    )

    print(
        os.path.join(
            OUTPUT_DIR,
            "compas_VR_all_vs_covered.csv"
        )
    )