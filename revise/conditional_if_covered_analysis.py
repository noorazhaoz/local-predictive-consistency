# conditional_if_covered_final.py
# ============================================================
# FINAL coverage-conditional fairness analysis
#
# Regression:
#   employers, insurance
#   experiment seed = 0
#
# Survival:
#   flchain, whas, rossi
#   DeepSurv / eta
#   Curve
#
# Survival seed is automatically inferred from saved IF CSV IDs.
#
# Only lambda > 0 is analysed.
#
# Outputs:
#   1. graph_coverage_summary.csv
#   2. conditional_IF_mean_median_summary.csv
#   3. conditional_VR_summary.csv
#   4. conditional_IF_VR_all_vs_covered.csv
# ============================================================

import os
import re
import glob
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

REGRESSION_SEED = 0

TAU = 5.0
THETA = 1e-3
SVD_RANK = 1

TEST_SIZE_SURVIVAL = 0.2

# search range for reconstructing survival seed
MAX_SURVIVAL_SEED = 1000

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
# PATHS -- CHANGE IF NECESSARY
# ============================================================

EMPLOYERS_DATA_PATH = "../data/Employers_data.csv"
INSURANCE_DATA_PATH = "../data/insurance.csv"

REGRESSION_IF_ROOT = "../regression/results"

DEEPSURV_IF_ROOT = "../survival/results/urw"

CURVE_IF_ROOT = "../survival_discrete/results/urw"

OUTPUT_DIR = "reviewer_conditional_coverage_analysis"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# GENERAL UTILS
# ============================================================

def parse_lambda_from_filename(path):

    name = os.path.basename(path)

    m = re.search(
        r"_lambda([-+0-9.eE]+)\.csv$",
        name
    )

    if m is None:
        return None

    try:
        return float(m.group(1))
    except ValueError:
        return None


def positive_lambda_files(pattern):

    files = []

    for path in sorted(
        glob.glob(pattern)
    ):

        lam = parse_lambda_from_filename(
            path
        )

        if (
            lam is not None
            and lam > 0
        ):
            files.append(path)

    return files


def e_dist_sq(A, B):

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

    D2 = e_dist_sq(
        X,
        X
    ).astype(np.float64)

    W = (
        np.exp(
            -float(theta) * D2
        )
        *
        (
            D2 <= float(tau)
        )
    ).astype(np.float64)

    # remove diagonal first
    np.fill_diagonal(
        W,
        0.0
    )

    # symmetric graph
    W = 0.5 * (
        W + W.T
    )

    # restore original self-loop
    n = W.shape[0]

    W[
        np.arange(n),
        np.arange(n)
    ] = 1.0

    return W


def get_covered_mask(W):

    W_off = W.copy()

    # self-loop must NOT count as coverage
    np.fill_diagonal(
        W_off,
        0.0
    )

    degree = (
        W_off > 0
    ).sum(
        axis=1
    )

    covered = (
        degree > 0
    )

    return covered, degree


def graph_summary(
    *,
    W,
    task,
    dataset,
    formulation,
    seed
):

    covered, degree = get_covered_mask(
        W
    )

    n = len(covered)

    adj = W.copy()

    np.fill_diagonal(
        adj,
        0.0
    )

    adj = (
        adj > 0
    )

    edges = int(
        np.triu(
            adj,
            k=1
        ).sum()
    )

    possible_edges = (
        n * (n - 1) / 2
    )

    density = (
        edges / possible_edges
        if possible_edges > 0
        else np.nan
    )

    return {
        "task": task,
        "dataset": dataset,
        "formulation": formulation,
        "seed": int(seed),

        "test_n": int(n),

        "edges_undirected": int(edges),
        "graph_density": float(density),

        "covered_nodes": int(
            covered.sum()
        ),

        "coverage": float(
            covered.mean()
        ),

        "isolated_nodes": int(
            (~covered).sum()
        ),

        "isolated_rate": float(
            (~covered).mean()
        ),

        "mean_degree": float(
            degree.mean()
        ),

        "median_degree": float(
            np.median(
                degree
            )
        ),

        "tau": float(TAU),
        "theta": float(THETA),
    }


# ============================================================
# FAIRNESS SUMMARY
# ============================================================

def compute_if_summary(
    if_values,
    covered
):

    if_values = np.asarray(
        if_values,
        dtype=float
    )

    covered = np.asarray(
        covered,
        dtype=bool
    )

    valid = np.isfinite(
        if_values
    )

    vals_all = (
        if_values[
            valid
        ]
    )

    vals_cov = (
        if_values[
            valid & covered
        ]
    )

    base = {
        "n_all": int(
            vals_all.size
        ),

        "n_covered": int(
            vals_cov.size
        ),

        "IF_mean_all": (
            float(
                np.mean(vals_all)
            )
            if vals_all.size
            else np.nan
        ),

        "IF_mean_covered": (
            float(
                np.mean(vals_cov)
            )
            if vals_cov.size
            else np.nan
        ),

        "IF_median_all": (
            float(
                np.median(vals_all)
            )
            if vals_all.size
            else np.nan
        ),

        "IF_median_covered": (
            float(
                np.median(vals_cov)
            )
            if vals_cov.size
            else np.nan
        ),
    }

    rows = []

    for eps in EPS_LIST:

        vr_all = (
            float(
                np.mean(
                    vals_all > eps
                )
            )
            if vals_all.size
            else np.nan
        )

        vr_cov = (
            float(
                np.mean(
                    vals_cov > eps
                )
            )
            if vals_cov.size
            else np.nan
        )

        row = dict(base)

        row.update({
            "epsilon": float(eps),

            "VR_all": vr_all,

            "VR_covered": vr_cov,

            "VR_absolute_change": (
                vr_cov - vr_all
                if (
                    np.isfinite(vr_cov)
                    and np.isfinite(vr_all)
                )
                else np.nan
            ),

            "VR_relative_change": (
                (
                    vr_cov / vr_all
                    - 1.0
                )
                if (
                    np.isfinite(vr_cov)
                    and np.isfinite(vr_all)
                    and vr_all > 0
                )
                else np.nan
            ),
        })

        rows.append(row)

    return rows


# ============================================================
# ALIGN SAVED CSV WITH RECONSTRUCTED GRAPH
# ============================================================

def align_covered_mask(
    *,
    file_idx,
    graph_idx,
    covered,
    path,
    dataset,
    formulation
):

    file_idx = np.asarray(
        file_idx,
        dtype=int
    )

    graph_idx = np.asarray(
        graph_idx,
        dtype=int
    )

    covered = np.asarray(
        covered,
        dtype=bool
    )

    if len(file_idx) != len(graph_idx):

        raise ValueError(
            "\nSIZE MISMATCH\n"
            f"{dataset} / {formulation}\n"
            f"file={path}\n"
            f"IF CSV n={len(file_idx)}, graph n={len(graph_idx)}"
        )

    if len(np.unique(file_idx)) != len(file_idx):

        raise ValueError(
            f"Duplicate IDs in IF file: {path}"
        )

    if len(np.unique(graph_idx)) != len(graph_idx):

        raise ValueError(
            f"Duplicate graph IDs: {dataset}/{formulation}"
        )

    # exact order
    if np.array_equal(
        file_idx,
        graph_idx
    ):

        print(
            "[OK] Exact alignment: "
            f"{dataset} / {formulation} / "
            f"{os.path.basename(path)}"
        )

        return covered

    # same members, different order
    if (
        set(file_idx.tolist())
        ==
        set(graph_idx.tolist())
    ):

        coverage_by_idx = {
            int(idx): bool(cov)
            for idx, cov
            in zip(
                graph_idx,
                covered
            )
        }

        covered_aligned = np.array(
            [
                coverage_by_idx[
                    int(i)
                ]
                for i in file_idx
            ],
            dtype=bool
        )

        print(
            "[OK] Same observations; reordered mask: "
            f"{dataset} / {formulation} / "
            f"{os.path.basename(path)}"
        )

        return covered_aligned

    missing = sorted(
        set(graph_idx.tolist())
        -
        set(file_idx.tolist())
    )

    extra = sorted(
        set(file_idx.tolist())
        -
        set(graph_idx.tolist())
    )

    raise ValueError(
        "\nTEST MEMBERSHIP MISMATCH\n"
        f"{dataset} / {formulation}\n"
        f"File: {path}\n"
        f"Missing from IF CSV (first 20): {missing[:20]}\n"
        f"Extra in IF CSV (first 20): {extra[:20]}\n"
        "The inferred experiment configuration does not match "
        "the saved IF file."
    )


# ============================================================
# REGRESSION
# ============================================================

def prepare_regression(dataset):

    seed = REGRESSION_SEED

    if dataset == "employers":

        df = pd.read_csv(
            EMPLOYERS_DATA_PATH
        ).copy()

        df = df.drop(
            columns=[
                "Employee_ID",
                "Name"
            ],
            errors="ignore"
        )

        sens = (
            df["Gender"]
            .astype(str)
            .str.lower()
            .eq("female")
            .astype(int)
            .values
        )

        df["Gender_bin"] = sens

        X_full_df = pd.get_dummies(
            df.drop(
                columns=[
                    "Gender",
                    "Salary"
                ],
                errors="ignore"
            ),
            drop_first=True
        ).astype(np.float32)

        X_model_df = X_full_df.drop(
            columns=[
                "Gender_bin"
            ]
        )

    elif dataset == "insurance":

        df = pd.read_csv(
            INSURANCE_DATA_PATH
        ).copy()

        sens = (
            df["sex"]
            .astype(str)
            .str.lower()
            .eq("female")
            .astype(int)
            .values
        )

        df["sex_bin"] = sens

        X_full_df = pd.get_dummies(
            df.drop(
                columns=[
                    "sex",
                    "expenses"
                ],
                errors="ignore"
            ),
            drop_first=True
        ).astype(np.float32)

        X_model_df = X_full_df.drop(
            columns=[
                "sex_bin"
            ]
        )

    else:

        raise ValueError(
            dataset
        )

    X = (
        X_model_df
        .values
        .astype(np.float32)
    )

    indices = np.arange(
        len(X)
    )

    dummy = np.zeros(
        len(X),
        dtype=np.float32
    )

    (
        X_tr,
        X_tmp,
        _,
        _,
        sens_tr,
        sens_tmp,
        idx_tr,
        idx_tmp
    ) = train_test_split(
        X,
        dummy,
        sens,
        indices,

        test_size=0.4,
        random_state=seed,
        stratify=sens
    )

    (
        X_va,
        X_te,
        _,
        _,
        sens_va,
        sens_te,
        idx_va,
        idx_te
    ) = train_test_split(
        X_tmp,
        np.zeros(
            len(X_tmp),
            dtype=np.float32
        ),
        sens_tmp,
        idx_tmp,

        test_size=0.5,
        random_state=seed,
        stratify=sens_tmp
    )

    scaler = StandardScaler()

    scaler.fit(
        X_tr
    )

    X_tr_std = (
        scaler.transform(
            X_tr
        )
        .astype(np.float32)
    )

    X_te_std = (
        scaler.transform(
            X_te
        )
        .astype(np.float32)
    )

    d = X_tr_std.shape[1]

    mu0 = (
        X_tr_std[
            sens_tr == 0
        ]
        .mean(
            axis=0,
            keepdims=True
        )
        .astype(np.float32)
    )

    mu1 = (
        X_tr_std[
            sens_tr == 1
        ]
        .mean(
            axis=0,
            keepdims=True
        )
        .astype(np.float32)
    )

    prot = np.vstack(
        [
            mu0,
            mu1
        ]
    ).astype(np.float32)

    r = min(
        SVD_RANK,
        prot.shape[0],
        d
    )

    svd = TruncatedSVD(
        n_components=r,
        random_state=seed
    )

    svd.fit(
        prot
    )

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
    )

    X_te_fair = (
        X_te_std
        @ proj_compl
    ).astype(np.float32)

    W = build_W_tau_only(
        X_te_fair
    )

    covered, degree = get_covered_mask(
        W
    )

    return {
        "seed": seed,
        "idx": idx_te.astype(int),
        "W": W,
        "covered": covered,
        "degree": degree,
    }


def find_regression_if_files(dataset):

    if dataset == "employers":

        patterns = [
            os.path.join(
                REGRESSION_IF_ROOT,
                "**",
                "urw_depth*_lambda*.csv"
            )
        ]

    elif dataset == "insurance":

        patterns = [
            os.path.join(
                REGRESSION_IF_ROOT,
                "**",
                "*insurance*urw*lambda*.csv"
            )
        ]

    else:

        raise ValueError(
            f"Unsupported regression dataset: {dataset}"
        )

    candidates = []

    for pattern in patterns:

        candidates.extend(
            glob.glob(
                pattern,
                recursive=True
            )
        )

    valid = []

    for path in candidates:

        lam = parse_lambda_from_filename(
            path
        )

        # only lambda > 0
        if (
            lam is None
            or lam <= 0
        ):
            continue

        try:

            cols = pd.read_csv(
                path,
                nrows=1
            ).columns

            # These are the actual per-instance URW IF files
            if (
                "orig_index" in cols
                and
                "if_score_urw_log" in cols
            ):

                valid.append(
                    path
                )

        except Exception:
            pass

    return sorted(
        set(valid)
    )

# ============================================================
# SURVIVAL DATA LOADER
# ============================================================

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

        df[time_col] = (
            y_struct[
                time_col
            ]
            .astype(np.float32)
        )

        df[event_col] = (
            y_struct[
                event_col
            ]
            .astype(int)
        )

    elif dataset == "flchain":

        X_raw, y_struct = load_flchain()

        df = X_raw.copy()

        time_col = "futime"
        event_col = "death"
        sens_col = "sex"

        df[time_col] = (
            y_struct[
                time_col
            ]
            .astype(np.float32)
        )

        df[event_col] = (
            y_struct[
                event_col
            ]
            .astype(int)
        )

        df[sens_col] = (
            df[
                sens_col
            ]
            .astype(str)
            .str.upper()
            .str[0]
            .eq("F")
            .astype(int)
        )

    else:

        raise ValueError(
            dataset
        )

    df = df.dropna(
        subset=[
            time_col,
            event_col,
            sens_col
        ]
    ).copy()

    return (
        df,
        time_col,
        event_col,
        sens_col
    )


def survival_base_arrays(dataset):

    (
        df,
        time_col,
        event_col,
        sens_col
    ) = load_survival_dataset(
        dataset
    )

    t = (
        df[
            time_col
        ]
        .astype(np.float32)
        .values
    )

    e = (
        df[
            event_col
        ]
        .astype(int)
        .values
    )

    sens = (
        df[
            sens_col
        ]
        .astype(int)
        .values
    )

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

    # Match original survival scripts:
    # imputation occurs before train/test split
    imputer = SimpleImputer(
        strategy="median"
    )

    X_df[:] = imputer.fit_transform(
        X_df
    )

    X = (
        X_df
        .astype(np.float32)
        .values
    )

    idx_all = np.arange(
        len(df)
    )

    return (
        X,
        t,
        e,
        sens,
        idx_all
    )


# ============================================================
# SURVIVAL SEED INFERENCE
# ============================================================

def get_saved_if_idx(path):

    df = pd.read_csv(
        path
    )

    if "idx" not in df.columns:

        raise ValueError(
            f"No idx column in {path}"
        )

    return (
        df[
            "idx"
        ]
        .astype(int)
        .values
    )


def infer_survival_seed(
    *,
    dataset,
    formulation,
    reference_if_path
):

    (
        X,
        t,
        e,
        sens,
        idx_all
    ) = survival_base_arrays(
        dataset
    )

    saved_idx = get_saved_if_idx(
        reference_if_path
    )

    saved_set = set(
        saved_idx.tolist()
    )

    print(
        f"\n[Seed search] "
        f"{dataset} / {formulation}"
    )

    print(
        f"Reference: "
        f"{os.path.basename(reference_if_path)}"
    )

    for seed in range(
        MAX_SURVIVAL_SEED + 1
    ):

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
            t,
            e,
            sens,
            idx_all,

            test_size=TEST_SIZE_SURVIVAL,
            random_state=seed
        )

        if formulation == "DeepSurv_eta":

            candidate_idx = (
                idx_te.astype(int)
            )

        elif formulation == "Curve":

            tr_max = float(
                np.max(
                    t_tr
                )
            )

            mask_curve = (
                t_te
                <
                (
                    tr_max - 1e-12
                )
            )

            candidate_idx = (
                idx_te[
                    mask_curve
                ]
                .astype(int)
            )

        else:

            raise ValueError(
                formulation
            )

        if len(candidate_idx) != len(saved_idx):
            continue

        # exact order
        if np.array_equal(
            candidate_idx,
            saved_idx
        ):

            print(
                f"[FOUND] seed={seed}, exact order=True"
            )

            return seed

        # same membership
        if (
            set(candidate_idx.tolist())
            ==
            saved_set
        ):

            print(
                f"[FOUND] seed={seed}, "
                f"exact membership=True, order=False"
            )

            return seed

    raise RuntimeError(
        "\nCould not infer survival seed.\n"
        f"dataset={dataset}\n"
        f"formulation={formulation}\n"
        f"searched seeds 0..{MAX_SURVIVAL_SEED}\n"
        f"reference={reference_if_path}"
    )


# ============================================================
# BUILD SURVIVAL GRAPH
# ============================================================

def prepare_survival(
    *,
    dataset,
    formulation,
    seed
):

    (
        X,
        t,
        e,
        sens,
        idx_all
    ) = survival_base_arrays(
        dataset
    )

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
        t,
        e,
        sens,
        idx_all,

        test_size=TEST_SIZE_SURVIVAL,
        random_state=seed
    )

    scaler = StandardScaler()

    scaler.fit(
        X_tr
    )

    X_tr_std = (
        scaler.transform(
            X_tr
        )
        .astype(np.float32)
    )

    X_te_std = (
        scaler.transform(
            X_te
        )
        .astype(np.float32)
    )

    d = X_tr_std.shape[1]

    # lambda > 0 fair projection
    mu0 = (
        X_tr_std[
            s_tr == 0
        ]
        .mean(
            axis=0,
            keepdims=True
        )
        .astype(np.float32)
    )

    mu1 = (
        X_tr_std[
            s_tr == 1
        ]
        .mean(
            axis=0,
            keepdims=True
        )
        .astype(np.float32)
    )

    prot = np.vstack(
        [
            mu0,
            mu1
        ]
    ).astype(np.float32)

    r = min(
        SVD_RANK,
        prot.shape[0],
        d
    )

    svd = TruncatedSVD(
        n_components=r,
        random_state=seed
    )

    svd.fit(
        prot
    )

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
    )

    X_te_fair = (
        X_te_std
        @ proj_compl
    ).astype(np.float32)

    if formulation == "DeepSurv_eta":

        graph_idx = (
            idx_te.astype(int)
        )

        X_graph = (
            X_te_fair
        )

    elif formulation == "Curve":

        tr_max = float(
            np.max(
                t_tr
            )
        )

        mask_curve = (
            t_te
            <
            (
                tr_max - 1e-12
            )
        )

        graph_idx = (
            idx_te[
                mask_curve
            ]
            .astype(int)
        )

        X_graph = (
            X_te_fair[
                mask_curve
            ]
        )

    else:

        raise ValueError(
            formulation
        )

    W = build_W_tau_only(
        X_graph
    )

    covered, degree = get_covered_mask(
        W
    )

    return {
        "seed": seed,
        "idx": graph_idx,
        "W": W,
        "covered": covered,
        "degree": degree,
    }


# ============================================================
# PROCESS IF FILE
# ============================================================

def process_if_file(
    *,
    path,
    graph_idx,
    covered,
    if_col,
    task,
    dataset,
    formulation,
    seed
):

    lam = parse_lambda_from_filename(
        path
    )

    if (
        lam is None
        or lam <= 0
    ):
        return []

    df = pd.read_csv(
        path
    )

    if "orig_index" in df.columns:

        id_col = "orig_index"

    elif "idx" in df.columns:

        id_col = "idx"

    else:

        raise ValueError(
            f"No ID column: {path}"
        )

    if if_col not in df.columns:

        raise ValueError(
            f"Missing IF column '{if_col}': {path}"
        )

    file_idx = (
        df[
            id_col
        ]
        .astype(int)
        .values
    )

    covered_aligned = align_covered_mask(
        file_idx=file_idx,
        graph_idx=graph_idx,
        covered=covered,
        path=path,
        dataset=dataset,
        formulation=formulation
    )

    if_values = (
        df[
            if_col
        ]
        .astype(float)
        .values
    )

    summaries = compute_if_summary(
        if_values,
        covered_aligned
    )

    rows = []

    for summary in summaries:

        row = {
            "task": task,
            "dataset": dataset,
            "formulation": formulation,

            "seed": int(seed),
            "lambda": float(lam),

            "file": os.path.basename(
                path
            ),

            "coverage": float(
                covered_aligned.mean()
            ),

            "isolated_rate": float(
                (~covered_aligned).mean()
            ),
        }

        row.update(
            summary
        )

        rows.append(
            row
        )

    return rows


# ============================================================
# MAIN
# ============================================================

all_results = []
graph_rows = []


# ============================================================
# 1. REGRESSION
# ============================================================

for dataset in [
    "employers",
    "insurance"
]:

    print(
        "\n"
        + "=" * 90
    )

    print(
        f"REGRESSION: {dataset}"
    )

    print(
        "=" * 90
    )

    pack = prepare_regression(
        dataset
    )

    graph_rows.append(
        graph_summary(
            W=pack["W"],
            task="regression",
            dataset=dataset,
            formulation="URW_log",
            seed=pack["seed"]
        )
    )

    files = find_regression_if_files(
        dataset
    )

    print(
        f"Positive-lambda URW IF files: "
        f"{len(files)}"
    )

    for path in files:

        rows = process_if_file(
            path=path,

            graph_idx=pack["idx"],
            covered=pack["covered"],

            if_col="if_score_urw_log",

            task="regression",
            dataset=dataset,
            formulation="URW_log",
            seed=pack["seed"]
        )

        all_results.extend(
            rows
        )


# ============================================================
# 2. SURVIVAL
# ============================================================

for dataset in [
    "flchain",
    "whas",
    "rossi"
]:

    print(
        "\n"
        + "#" * 90
    )

    print(
        f"SURVIVAL DATASET: {dataset}"
    )

    print(
        "#" * 90
    )

    # ========================================================
    # DeepSurv / eta
    # ========================================================

    ds_folder = os.path.join(
        DEEPSURV_IF_ROOT,
        dataset
    )

    ds_files = positive_lambda_files(
        os.path.join(
            ds_folder,
            "if_deepsurv_depth*_lambda*.csv"
        )
    )

    print(
        f"\nDeepSurv positive-lambda IF files: "
        f"{len(ds_files)}"
    )

    if len(ds_files) > 0:

        ds_seed = infer_survival_seed(
            dataset=dataset,
            formulation="DeepSurv_eta",
            reference_if_path=ds_files[0]
        )

        ds_pack = prepare_survival(
            dataset=dataset,
            formulation="DeepSurv_eta",
            seed=ds_seed
        )

        graph_rows.append(
            graph_summary(
                W=ds_pack["W"],
                task="survival",
                dataset=dataset,
                formulation="DeepSurv_eta",
                seed=ds_seed
            )
        )

        for path in ds_files:

            rows = process_if_file(
                path=path,

                graph_idx=ds_pack["idx"],
                covered=ds_pack["covered"],

                if_col="if_score_urw_eta",

                task="survival",
                dataset=dataset,
                formulation="DeepSurv_eta",
                seed=ds_seed
            )

            all_results.extend(
                rows
            )

    else:

        print(
            "[WARN] No DeepSurv IF files found."
        )

    # ========================================================
    # Curve
    # ========================================================

    curve_folder = os.path.join(
        CURVE_IF_ROOT,
        dataset
    )

    curve_files = positive_lambda_files(
        os.path.join(
            curve_folder,
            "if_curve_depth*_lambda*.csv"
        )
    )

    print(
        f"\nCurve positive-lambda IF files: "
        f"{len(curve_files)}"
    )

    if len(curve_files) > 0:

        curve_seed = infer_survival_seed(
            dataset=dataset,
            formulation="Curve",
            reference_if_path=curve_files[0]
        )

        curve_pack = prepare_survival(
            dataset=dataset,
            formulation="Curve",
            seed=curve_seed
        )

        graph_rows.append(
            graph_summary(
                W=curve_pack["W"],
                task="survival",
                dataset=dataset,
                formulation="Curve",
                seed=curve_seed
            )
        )

        for path in curve_files:

            rows = process_if_file(
                path=path,

                graph_idx=curve_pack["idx"],
                covered=curve_pack["covered"],

                if_col="if_curve_L2_over_eval_grid",

                task="survival",
                dataset=dataset,
                formulation="Curve",
                seed=curve_seed
            )

            all_results.extend(
                rows
            )

    else:

        print(
            "[WARN] No Curve IF files found."
        )


# ============================================================
# SAVE RESULTS
# ============================================================

df_graph = pd.DataFrame(
    graph_rows
)

graph_path = os.path.join(
    OUTPUT_DIR,
    "graph_coverage_summary.csv"
)

df_graph.to_csv(
    graph_path,
    index=False
)


df_all = pd.DataFrame(
    all_results
)

all_path = os.path.join(
    OUTPUT_DIR,
    "conditional_IF_VR_all_vs_covered.csv"
)

df_all.to_csv(
    all_path,
    index=False
)


# ============================================================
# COMPACT IF MEAN / MEDIAN TABLE
# ============================================================

if len(df_all) > 0:

    df_if = (
        df_all[
            [
                "task",
                "dataset",
                "formulation",
                "seed",
                "lambda",

                "coverage",
                "isolated_rate",

                "n_all",
                "n_covered",

                "IF_mean_all",
                "IF_mean_covered",

                "IF_median_all",
                "IF_median_covered",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            [
                "task",
                "dataset",
                "formulation",
                "lambda"
            ]
        )
    )

else:

    df_if = pd.DataFrame()


if_path = os.path.join(
    OUTPUT_DIR,
    "conditional_IF_mean_median_summary.csv"
)

df_if.to_csv(
    if_path,
    index=False
)


# ============================================================
# COMPACT VR TABLE
# ============================================================

if len(df_all) > 0:

    df_vr = (
        df_all[
            [
                "task",
                "dataset",
                "formulation",
                "seed",
                "lambda",

                "coverage",
                "isolated_rate",

                "epsilon",

                "VR_all",
                "VR_covered",

                "VR_absolute_change",
                "VR_relative_change",
            ]
        ]
        .sort_values(
            [
                "task",
                "dataset",
                "formulation",
                "lambda",
                "epsilon"
            ]
        )
    )

else:

    df_vr = pd.DataFrame()


vr_path = os.path.join(
    OUTPUT_DIR,
    "conditional_VR_summary.csv"
)

df_vr.to_csv(
    vr_path,
    index=False
)


# ============================================================
# PRINT FINAL
# ============================================================

print(
    "\n\n"
    + "=" * 100
)

print(
    "FINAL GRAPH COVERAGE"
)

print(
    "=" * 100
)

if len(df_graph):

    print(
        df_graph.to_string(
            index=False
        )
    )

else:

    print(
        "No graph results."
    )


print(
    "\n"
    + "=" * 100
)

print(
    "CONDITIONAL IF SUMMARY"
)

print(
    "=" * 100
)

if len(df_if):

    print(
        df_if.to_string(
            index=False
        )
    )

else:

    print(
        "No IF results."
    )


print(
    "\n"
    + "=" * 100
)

print(
    "SAVED"
)

print(
    "=" * 100
)

print(graph_path)
print(all_path)
print(if_path)
print(vr_path)