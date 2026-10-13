import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD


# ============================================================
# 1. Original graph functions
# ============================================================

def e_dist_sq(A, B):
    A2 = (A ** 2).sum(axis=1, keepdims=True)
    B2 = (B ** 2).sum(axis=1, keepdims=True).T
    return np.maximum(A2 - 2.0 * (A @ B.T) + B2, 0.0)


def build_W_tau_only(X_for_metric, theta=1e-3, tau=5.0):
    """
    Exact graph construction used in the original URW code.

    Important:
        tau is applied to squared Euclidean distance:
            D^2 <= tau
    """
    D2 = e_dist_sq(X_for_metric, X_for_metric)

    W = (
        np.exp(-theta * D2, dtype=np.float64)
        * (D2 <= tau).astype(np.float64)
    )

    # Remove diagonal first
    np.fill_diagonal(W, 0.0)

    # Symmetrize
    W = 0.5 * (W + W.T)

    # Add self-loop
    n = W.shape[0]
    W[np.arange(n), np.arange(n)] = 1.0

    return W


# ============================================================
# 2. Data loaders — copied from original URW implementation
# ============================================================

def load_adult_clean(adult_data_path, adult_test_path):

    headers = [
        "age", "workclass", "fnlwgt", "education", "education-num",
        "marital-status", "occupation", "relationship", "race", "sex",
        "capital-gain", "capital-loss", "hours-per-week",
        "native-country", "y"
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

    df = pd.concat([train, test], axis=0, ignore_index=True)
    df.columns = headers

    df["y"] = df["y"].astype(str)
    df["y"] = df["y"].str.replace(".", "", regex=False).str.strip()
    df["y"] = df["y"].map({
        "<=50K": 0,
        ">50K": 1
    })

    df = df.dropna(subset=["y"])
    df["y"] = df["y"].astype(int)

    df = df.dropna()
    df = pd.get_dummies(df, drop_first=False)

    native_cols = [
        c for c in df.columns
        if c.startswith("native-country_")
    ]
    df = df.drop(columns=native_cols, errors="ignore")

    drop_cols = [
        c for c in df.columns
        if c.startswith("fnlwgt") or c.startswith("education_")
    ]
    df = df.drop(columns=drop_cols, errors="ignore")

    df["orig_index"] = np.arange(len(df))

    return df


def load_compas_clean(compas_path):

    df = pd.read_csv(compas_path)

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

    df = df[keep_cols].copy().dropna()

    df["y"] = df["two_year_recid"].astype(int)
    df = df.drop(columns=["two_year_recid"])

    numerical_cols = [
        "age",
        "priors_count",
        "juv_fel_count",
        "juv_misd_count",
        "juv_other_count"
    ]

    for c in numerical_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

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

    df["orig_index"] = np.arange(len(df))

    return df


# ============================================================
# 3. Reconstruct exact test graph
# ============================================================

def reconstruct_test_graph(
    df,
    seed=0,
    tau=5.0,
    theta=1e-3,
    svd_rank=1
):

    y_full = df["y"].astype(int).values
    orig_index_full = df["orig_index"].values

    # Same sensitive variable logic as original code
    if "sex_Female" in df.columns:
        sens_full = df["sex_Female"].astype(int).values

    elif "sex_Male" in df.columns:
        sens_full = 1 - df["sex_Male"].astype(int).values

    else:
        raise ValueError("Cannot identify sex dummy column.")

    # Sensitive attribute removed from model/graph feature matrix
    drop_for_X = (
        [c for c in df.columns if c.startswith("sex_")]
        + ["y", "orig_index"]
    )

    X = (
        df
        .drop(columns=drop_for_X)
        .astype(np.float32)
        .values
    )

    # --------------------------------------------------------
    # Exact original 60 / 20 / 20 split
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
    # Exact original scaling
    # --------------------------------------------------------

    scaler = StandardScaler()

    scaler.fit(X_tr)

    X_tr = scaler.transform(X_tr).astype(np.float32)
    X_te = scaler.transform(X_te).astype(np.float32)

    d = X_tr.shape[1]

    # --------------------------------------------------------
    # Exact fair-space projection
    # --------------------------------------------------------

    mu0 = X_tr[sens_tr == 0].mean(
        axis=0,
        keepdims=True
    )

    mu1 = X_tr[sens_tr == 1].mean(
        axis=0,
        keepdims=True
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

    svd.fit(prot)

    basis = svd.components_.T

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

    # No learned metric in the original default setting
    X_te_fair = X_te @ proj_compl

    # --------------------------------------------------------
    # Build exact test graph
    # --------------------------------------------------------

    W_te = build_W_tau_only(
        X_te_fair,
        theta=theta,
        tau=tau
    )

    return W_te, rid_te


# ============================================================
# 4. Coverage / isolated-node diagnostics
# ============================================================

def graph_diagnostics(W):

    n = W.shape[0]

    # --------------------------------------------------------
    # IMPORTANT:
    # remove self-loop before checking isolation
    # --------------------------------------------------------

    W_off = W.copy()

    np.fill_diagonal(
        W_off,
        0.0
    )

    adjacency = (
        W_off > 0
    )

    degree = adjacency.sum(
        axis=1
    )

    covered = degree > 0
    isolated = degree == 0

    n_covered = int(
        covered.sum()
    )

    n_isolated = int(
        isolated.sum()
    )

    # Undirected edge count
    n_edges = int(
        np.triu(
            adjacency,
            k=1
        ).sum()
    )

    possible_edges = (
        n * (n - 1) / 2
    )

    density = (
        n_edges / possible_edges
        if possible_edges > 0
        else np.nan
    )

    result = {
        "test_n": n,

        "edges_undirected": n_edges,

        "graph_density": density,

        "covered_nodes": n_covered,

        "coverage": (
            n_covered / n
        ),

        "isolated_nodes": n_isolated,

        "isolated_rate": (
            n_isolated / n
        ),

        "mean_degree": float(
            degree.mean()
        ),

        "median_degree": float(
            np.median(degree)
        )
    }

    return result, covered, degree


# ============================================================
# 5. Optional: compute VR_all and VR_covered
# ============================================================

def compute_vr(
    if_csv_path,
    covered,
    rid_te,
    epsilon,
    if_col="if_score_urw_monitor_KL_p||Pp"
):

    df_if = pd.read_csv(
        if_csv_path
    )

    # --------------------------------------------------------
    # Safety check:
    # verify same test ordering
    # --------------------------------------------------------

    if not np.array_equal(
        df_if["orig_index"].values,
        rid_te
    ):
        raise ValueError(
            "orig_index ordering does not match reconstructed test split."
        )

    scores = (
        df_if[if_col]
        .to_numpy()
        .astype(float)
    )

    violation = (
        scores > epsilon
    )

    vr_all = float(
        violation.mean()
    )

    vr_covered = float(
        violation[covered].mean()
    )

    return {
        "VR_all": vr_all,
        "VR_covered": vr_covered,
        "VR_difference": (
            vr_covered - vr_all
        )
    }


# ============================================================
# 6. Run Adult and COMPAS
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # CHANGE THESE PATHS
    # --------------------------------------------------------

    ADULT_DATA = "../data/adult/adult.data"
    ADULT_TEST = "../data/adult/adult.test"

    COMPAS_DATA = (
        "../data/compas/"
        "compas-scores-two-years.csv"
    )

    # Original settings
    SEED = 0
    TAU = 5.0
    THETA = 1e-3
    SVD_RANK = 1

    all_results = []

    # ========================================================
    # Adult
    # ========================================================

    df_adult = load_adult_clean(
        ADULT_DATA,
        ADULT_TEST
    )

    W_adult, rid_adult = (
        reconstruct_test_graph(
            df_adult,
            seed=SEED,
            tau=TAU,
            theta=THETA,
            svd_rank=SVD_RANK
        )
    )

    adult_stats, adult_covered, adult_degree = (
        graph_diagnostics(
            W_adult
        )
    )

    adult_stats["dataset"] = "Adult"

    all_results.append(
        adult_stats
    )

    print("\nAdult")
    print(
        pd.Series(adult_stats)
    )

    # ========================================================
    # COMPAS
    # ========================================================

    df_compas = load_compas_clean(
        COMPAS_DATA
    )

    W_compas, rid_compas = (
        reconstruct_test_graph(
            df_compas,
            seed=SEED,
            tau=TAU,
            theta=THETA,
            svd_rank=SVD_RANK
        )
    )

    compas_stats, compas_covered, compas_degree = (
        graph_diagnostics(
            W_compas
        )
    )

    compas_stats["dataset"] = "COMPAS"

    all_results.append(
        compas_stats
    )

    print("\nCOMPAS")
    print(
        pd.Series(compas_stats)
    )

    # ========================================================
    # Save graph diagnostics
    # ========================================================

    out = pd.DataFrame(
        all_results
    )

    cols = [
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

    out = out[cols]

    out.to_csv(
        "graph_coverage_tau5.csv",
        index=False
    )

    print(
        "\nSaved: graph_coverage_tau5.csv"
    )