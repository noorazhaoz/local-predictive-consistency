import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. PATHS
# ============================================================

PAIRWISE_CSV = Path(
    "reviewer_pairwise_fairness_final/"
    "pairwise_fairness_summary_full.csv"
)

# Can be either a .zip file or a folder
BINARY_RESULTS = Path("../binary/results_unified")
REGRESSION_RESULTS = Path("../regression/results")

SURVIVAL_LOGRISK_RESULTS = Path("../survival/results/urw")
SURVIVAL_CURVE_RESULTS = Path("../survival_discrete/results/urw")

OUT_DIR = Path("pareto_all_tasks")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. BASIC HELPERS
# ============================================================

def normalize_dataset(name):
    s = str(name).lower()

    if "adult" in s:
        return "adult"

    if "compas" in s:
        return "compas"

    if "employee" in s or "employer" in s or "salary" in s:
        return "employers"

    if "insurance" in s:
        return "insurance"

    if "flchain" in s:
        return "flchain"

    if "whas" in s:
        return "whas"

    if "rossi" in s:
        return "rossi"

    return s


def extract_lambda(filename):
    m = re.search(
        r"lambda([-+0-9.eE]+)\.csv$",
        filename
    )

    if m is None:
        return None

    return float(m.group(1))


def iter_csv_files(path):
    """
    Recursively iterate through CSV files in either:
      - a zip archive
      - a normal directory
    """

    path = Path(path)

    if path.suffix.lower() == ".zip":

        with zipfile.ZipFile(path) as z:

            for name in z.namelist():

                if not name.endswith(".csv"):
                    continue

                if "__MACOSX" in name:
                    continue

                if Path(name).name.startswith("._"):
                    continue

                try:
                    df = pd.read_csv(
                        z.open(name)
                    )
                except Exception:
                    continue

                yield name, df

    elif path.is_dir():

        for f in path.rglob("*.csv"):

            if "__MACOSX" in str(f):
                continue

            if f.name.startswith("._"):
                continue

            try:
                df = pd.read_csv(f)
            except Exception:
                continue

            yield str(f), df

    else:

        raise ValueError(
            f"Path is neither a zip file nor a directory: {path}"
        )


def safe_name(s):
    return (
        str(s)
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
    )


# ============================================================
# 3. LOAD INDEPENDENT PAIRWISE FAIRNESS RESULTS
# ============================================================

pairwise = pd.read_csv(
    PAIRWISE_CSV
)

pairwise["dataset"] = (
    pairwise["dataset"]
    .map(normalize_dataset)
)

pairwise["lambda"] = (
    pairwise["lambda"]
    .astype(float)
)


def normalize_formulation(row):

    task = str(
        row["task"]
    ).lower()

    form = str(
        row["formulation"]
    ).lower()

    if task == "binary":

        if "fkl" in form or "forward" in form:
            return "URW-FKL"

        if "rkl" in form or "reverse" in form:
            return "URW-RKL"

    if task == "regression":
        return "URW"

    if task == "survival":

        if "log" in form or "deep" in form:
            return "log-risk"

        if "curve" in form:
            return "survival-curve"

    return row["formulation"]


pairwise["formulation"] = (
    pairwise.apply(
        normalize_formulation,
        axis=1
    )
)


fair = pairwise[
    [
        "task",
        "dataset",
        "formulation",
        "lambda",
        "EdgeMeanAbs",
        "EdgeRMS",
    ]
].copy()


print("\n======================================")
print("PAIRWISE FAIRNESS GROUPS")
print("======================================")

print(
    fair[
        [
            "task",
            "dataset",
            "formulation",
        ]
    ]
    .drop_duplicates()
    .sort_values(
        [
            "task",
            "dataset",
            "formulation",
        ]
    )
    .to_string(index=False)
)


# ============================================================
# 4. CLASSIFICATION UTILITY
# ============================================================

binary_rows = []

for name, df in iter_csv_files(
    BINARY_RESULTS
):

    lower = name.lower()
    filename = Path(name).name

    if "metrics_urw_" not in lower:
        continue

    if "adult" in lower:
        dataset = "adult"

    elif "compas" in lower:
        dataset = "compas"

    else:
        continue

    if "aggforward" in lower:
        formulation = "URW-FKL"

    elif "aggreverse" in lower:
        formulation = "URW-RKL"

    else:
        continue

    lam = extract_lambda(
        filename
    )

    if lam is None:
        continue

    if len(df) == 0:
        continue

    if "test_auc" not in df.columns:
        continue

    row = df.iloc[0]

    binary_rows.append({
        "task": "binary",
        "dataset": dataset,
        "formulation": formulation,
        "lambda": lam,
        "utility_metric": "AUC",
        "utility": float(
            row["test_auc"]
        ),
        "utility_direction": "maximize",
    })


binary_utility = pd.DataFrame(
    binary_rows
)


# ============================================================
# 5. REGRESSION UTILITY
# ============================================================

reg_rows = []

employee_pat = re.compile(
    r"^metrics_urw_depth1_lambda([-+0-9.eE]+)\.csv$"
)

insurance_pat = re.compile(
    r"^metrics_insurance_urw_depth1_lambda([-+0-9.eE]+)\.csv$"
)


for name, df in iter_csv_files(
    REGRESSION_RESULTS
):

    full_path = str(name)
    lower = full_path.lower()
    filename = Path(name).name.lower()

    if "__macosx" in lower:
        continue

    if filename.startswith("._"):
        continue

    if "laplacian" in lower:
        continue

    m_emp = employee_pat.match(
        filename
    )

    m_ins = insurance_pat.match(
        filename
    )

    if m_emp is not None:

        dataset = "employers"
        lam = float(
            m_emp.group(1)
        )

    elif m_ins is not None:

        dataset = "insurance"
        lam = float(
            m_ins.group(1)
        )

    else:
        continue

    if len(df) == 0:
        continue

    if "test_RMSE_real" not in df.columns:
        continue

    row = df.iloc[0]

    path_parts = [
        p.lower()
        for p in Path(full_path).parts
    ]

    is_explicit_urw_dir = (
        "urw" in path_parts
    )

    reg_rows.append({
        "task": "regression",
        "dataset": dataset,
        "formulation": "URW",
        "lambda": lam,
        "utility_metric": "RMSE",
        "utility": float(
            row["test_RMSE_real"]
        ),
        "utility_direction": "minimize",
        "source_file": full_path,
        "prefer_urw_dir": int(
            is_explicit_urw_dir
        ),
    })


reg_utility = pd.DataFrame(
    reg_rows
)


if len(reg_utility) == 0:

    raise ValueError(
        "No regression utility results were found."
    )


reg_key_cols = [
    "task",
    "dataset",
    "formulation",
    "lambda",
]


# ------------------------------------------------------------
# Check whether duplicate copies disagree numerically
# ------------------------------------------------------------

for keys, group in reg_utility.groupby(
    reg_key_cols,
    dropna=False
):

    if len(group) <= 1:
        continue

    utilities = (
        group["utility"]
        .to_numpy(float)
    )

    if not np.allclose(
        utilities,
        utilities[0],
        rtol=1e-10,
        atol=1e-10,
    ):

        print("\n======================================")
        print("CONFLICTING REGRESSION DUPLICATES")
        print("======================================")

        print(
            group[
                [
                    "dataset",
                    "lambda",
                    "utility",
                    "source_file",
                ]
            ]
            .to_string(index=False)
        )

        raise ValueError(
            f"Duplicate regression files disagree for {keys}."
        )


# ------------------------------------------------------------
# Prefer copies explicitly stored under /urw/
# ------------------------------------------------------------

reg_utility = (
    reg_utility
    .sort_values(
        reg_key_cols
        +
        ["prefer_urw_dir"],
        ascending=[
            True,
            True,
            True,
            True,
            False,
        ],
    )
    .drop_duplicates(
        subset=reg_key_cols,
        keep="first",
    )
    .reset_index(drop=True)
)


remaining_dups = (
    reg_utility
    .duplicated(
        subset=reg_key_cols,
        keep=False,
    )
)

if remaining_dups.any():

    raise RuntimeError(
        "Regression duplicates remain after deduplication."
    )


print("\n======================================")
print("REGRESSION UTILITY AFTER DEDUPLICATION")
print("======================================")

print(
    reg_utility[
        [
            "dataset",
            "lambda",
            "utility",
            "source_file",
        ]
    ]
    .sort_values(
        [
            "dataset",
            "lambda",
        ]
    )
    .to_string(index=False)
)


# ============================================================
# 6. SURVIVAL LOG-RISK / DEEPSURV UTILITY
#    BOTH C-INDEX AND IBS
# ============================================================

logrisk_rows = []

for name, df in iter_csv_files(
    SURVIVAL_LOGRISK_RESULTS
):

    lower = name.lower()

    if "metrics" not in lower:
        continue

    if "lambda" not in lower:
        continue

    dataset = normalize_dataset(
        lower
    )

    if dataset not in {
        "flchain",
        "whas",
        "rossi",
    }:
        continue

    lam = extract_lambda(
        Path(name).name
    )

    if lam is None:
        continue

    if len(df) == 0:
        continue

    row = df.iloc[0]

    # --------------------------------------------------------
    # C-index
    # Higher is better
    # --------------------------------------------------------

    if "test_c_index" in df.columns:

        logrisk_rows.append({
            "task": "survival",
            "dataset": dataset,
            "formulation": "log-risk",
            "lambda": lam,
            "utility_metric": "C-index",
            "utility": float(
                row["test_c_index"]
            ),
            "utility_direction": "maximize",
        })

    # --------------------------------------------------------
    # IBS
    # Lower is better
    # --------------------------------------------------------

    if "ibs" in df.columns:

        logrisk_rows.append({
            "task": "survival",
            "dataset": dataset,
            "formulation": "log-risk",
            "lambda": lam,
            "utility_metric": "IBS",
            "utility": float(
                row["ibs"]
            ),
            "utility_direction": "minimize",
        })


logrisk_utility = pd.DataFrame(
    logrisk_rows
)


print("\n======================================")
print("LOG-RISK SURVIVAL UTILITY METRICS")
print("======================================")

if len(logrisk_utility) > 0:

    print(
        logrisk_utility[
            [
                "dataset",
                "formulation",
                "lambda",
                "utility_metric",
                "utility",
            ]
        ]
        .sort_values(
            [
                "dataset",
                "utility_metric",
                "lambda",
            ]
        )
        .to_string(index=False)
    )


# ============================================================
# 7. SURVIVAL-CURVE UTILITY
#    BOTH C-INDEX AND IBS
# ============================================================

curve_rows = []

curve_cindex_col = (
    "test_c_index_from_risk=-logS(t*)"
)


for name, df in iter_csv_files(
    SURVIVAL_CURVE_RESULTS
):

    lower = name.lower()

    if "metrics_curve" not in lower:
        continue

    dataset = normalize_dataset(
        lower
    )

    if dataset not in {
        "flchain",
        "whas",
        "rossi",
    }:
        continue

    lam = extract_lambda(
        Path(name).name
    )

    if lam is None:
        continue

    if len(df) == 0:
        continue

    row = df.iloc[0]

    # --------------------------------------------------------
    # C-index
    # Higher is better
    # --------------------------------------------------------

    if curve_cindex_col in df.columns:

        curve_rows.append({
            "task": "survival",
            "dataset": dataset,
            "formulation": "survival-curve",
            "lambda": lam,
            "utility_metric": "C-index",
            "utility": float(
                row[curve_cindex_col]
            ),
            "utility_direction": "maximize",
        })

    # --------------------------------------------------------
    # IBS
    # Lower is better
    # --------------------------------------------------------

    if "ibs" in df.columns:

        curve_rows.append({
            "task": "survival",
            "dataset": dataset,
            "formulation": "survival-curve",
            "lambda": lam,
            "utility_metric": "IBS",
            "utility": float(
                row["ibs"]
            ),
            "utility_direction": "minimize",
        })


curve_utility = pd.DataFrame(
    curve_rows
)


print("\n======================================")
print("SURVIVAL-CURVE UTILITY METRICS")
print("======================================")

if len(curve_utility) > 0:

    print(
        curve_utility[
            [
                "dataset",
                "formulation",
                "lambda",
                "utility_metric",
                "utility",
            ]
        ]
        .sort_values(
            [
                "dataset",
                "utility_metric",
                "lambda",
            ]
        )
        .to_string(index=False)
    )


# ============================================================
# 8. COMBINE ALL UTILITY RESULTS
# ============================================================

reg_utility_clean = (
    reg_utility[
        [
            "task",
            "dataset",
            "formulation",
            "lambda",
            "utility_metric",
            "utility",
            "utility_direction",
        ]
    ]
    .copy()
)


all_utility = pd.concat(
    [
        binary_utility,
        reg_utility_clean,
        logrisk_utility,
        curve_utility,
    ],
    ignore_index=True
)


# ============================================================
# 9. MERGE UTILITY WITH ACHIEVED PAIRWISE FAIRNESS
# ============================================================

all_results = all_utility.merge(
    fair,
    on=[
        "task",
        "dataset",
        "formulation",
        "lambda",
    ],
    how="inner",
)


all_results = (
    all_results
    .sort_values(
        [
            "task",
            "dataset",
            "formulation",
            "utility_metric",
            "lambda",
        ]
    )
    .reset_index(drop=True)
)


all_results.to_csv(
    OUT_DIR /
    "fairness_utility_all_metrics.csv",
    index=False
)


print("\n======================================")
print("ALL FAIRNESS-UTILITY GROUPS")
print("======================================")

print(
    all_results[
        [
            "task",
            "dataset",
            "formulation",
            "utility_metric",
        ]
    ]
    .drop_duplicates()
    .sort_values(
        [
            "task",
            "dataset",
            "formulation",
            "utility_metric",
        ]
    )
    .to_string(index=False)
)


# ============================================================
# 10. SAFETY CHECK:
#     ONE ROW PER TASK/DATASET/FORMULATION/METRIC/LAMBDA
# ============================================================

analysis_key = [
    "task",
    "dataset",
    "formulation",
    "utility_metric",
    "lambda",
]


analysis_dups = all_results.duplicated(
    subset=analysis_key,
    keep=False,
)


if analysis_dups.any():

    print("\n======================================")
    print("DUPLICATE ANALYSIS OPERATING POINTS")
    print("======================================")

    print(
        all_results.loc[
            analysis_dups,
            analysis_key
            +
            [
                "utility",
                "EdgeMeanAbs",
            ]
        ]
        .sort_values(
            analysis_key
        )
        .to_string(index=False)
    )

    raise ValueError(
        "Duplicate fairness-utility operating points remain."
    )


# ============================================================
# 11. PARETO FUNCTION
# ============================================================

def mark_pareto(
    df,
    fairness_col="EdgeMeanAbs",
    utility_col="utility",
):
    """
    Fairness:
        lower is always better.

    Utility:
        direction determined by utility_direction.

    Point i is dominated if another point j is:
        - no worse in fairness,
        - no worse in utility,
        - strictly better in at least one dimension.
    """

    out = df.copy()

    direction = (
        out["utility_direction"]
        .iloc[0]
    )

    F = (
        out[fairness_col]
        .to_numpy(float)
    )

    U = (
        out[utility_col]
        .to_numpy(float)
    )

    pareto = np.ones(
        len(out),
        dtype=bool
    )

    for i in range(
        len(out)
    ):

        for j in range(
            len(out)
        ):

            if i == j:
                continue

            fair_better_equal = (
                F[j] <= F[i]
            )

            strict_fair = (
                F[j] < F[i]
            )

            if direction == "maximize":

                util_better_equal = (
                    U[j] >= U[i]
                )

                strict_util = (
                    U[j] > U[i]
                )

            elif direction == "minimize":

                util_better_equal = (
                    U[j] <= U[i]
                )

                strict_util = (
                    U[j] < U[i]
                )

            else:

                raise ValueError(
                    f"Unknown utility direction: {direction}"
                )

            dominates = (
                fair_better_equal
                and util_better_equal
                and (
                    strict_fair
                    or strict_util
                )
            )

            if dominates:

                pareto[i] = False
                break

    out["pareto"] = pareto

    return out


# ============================================================
# 12. MARK RAW PARETO POINTS
#     ALL TASKS + ALL UTILITY METRICS
# ============================================================

pareto_parts = []


for keys, g in all_results.groupby(
    [
        "task",
        "dataset",
        "formulation",
        "utility_metric",
    ]
):

    pareto_parts.append(
        mark_pareto(g)
    )


all_pareto = pd.concat(
    pareto_parts,
    ignore_index=True
)


all_pareto = (
    all_pareto
    .sort_values(
        [
            "task",
            "dataset",
            "formulation",
            "utility_metric",
            "lambda",
        ]
    )
    .reset_index(drop=True)
)


all_pareto.to_csv(
    OUT_DIR /
    "pareto_all_tasks_all_metrics.csv",
    index=False
)


# ============================================================
# 13. NORMALIZE RELATIVE TO LAMBDA=0
#     ONLY FOR PRESENTATION
# ============================================================

def normalize_relative_to_baseline(df):

    normalized_parts = []

    for keys, g in df.groupby(
        [
            "task",
            "dataset",
            "formulation",
            "utility_metric",
        ]
    ):

        g = g.copy()

        baseline = g[
            np.isclose(
                g["lambda"],
                0.0
            )
        ]

        if len(baseline) != 1:

            raise ValueError(
                f"Expected exactly one lambda=0 baseline "
                f"for {keys}, found {len(baseline)}."
            )

        base = baseline.iloc[0]

        F0 = float(
            base["EdgeMeanAbs"]
        )

        U0 = float(
            base["utility"]
        )

        if np.isclose(
            F0,
            0.0
        ):

            raise ValueError(
                f"Baseline EdgeMeanAbs is zero for {keys}; "
                f"percentage fairness improvement is undefined."
            )

        if np.isclose(
            U0,
            0.0
        ):

            raise ValueError(
                f"Baseline utility is zero for {keys}; "
                f"percentage utility change is undefined."
            )

        # ----------------------------------------------------
        # Fairness improvement
        #
        # lower EdgeMeanAbs is better
        #
        # positive value =
        # reduction in pairwise discrepancy
        # ----------------------------------------------------

        g[
            "fairness_improvement_pct"
        ] = (
            100.0
            *
            (
                1.0
                -
                g["EdgeMeanAbs"] / F0
            )
        )

        # ----------------------------------------------------
        # Utility change
        #
        # Always constructed so:
        # positive = better predictive utility
        # negative = worse predictive utility
        # ----------------------------------------------------

        if (
            base["utility_direction"]
            == "maximize"
        ):

            g[
                "utility_change_pct"
            ] = (
                100.0
                *
                (
                    g["utility"] - U0
                )
                /
                U0
            )

        elif (
            base["utility_direction"]
            == "minimize"
        ):

            g[
                "utility_change_pct"
            ] = (
                100.0
                *
                (
                    1.0
                    -
                    g["utility"] / U0
                )
            )

        else:

            raise ValueError(
                base[
                    "utility_direction"
                ]
            )

        normalized_parts.append(
            g
        )

    return pd.concat(
        normalized_parts,
        ignore_index=True
    )


normalized_all = (
    normalize_relative_to_baseline(
        all_pareto
    )
)


normalized_all.to_csv(
    OUT_DIR /
    "pareto_all_tasks_all_metrics_normalized.csv",
    index=False
)


# ============================================================
# 14. PRINT RAW PARETO SUMMARY
# ============================================================

print("\n")
print("======================================")
print("RAW ACHIEVED-FAIRNESS PARETO LAMBDAS")
print("======================================")


for keys, g in all_pareto.groupby(
    [
        "task",
        "dataset",
        "formulation",
        "utility_metric",
    ]
):

    frontier = (
        g[
            g["pareto"]
        ]
        .sort_values(
            "lambda"
        )
    )

    vals = [
        float(x)
        for x
        in frontier["lambda"]
    ]

    print(
        f"{keys}: "
        f"Pareto={vals}, "
        f"n_pareto={len(frontier)}/{len(g)}"
    )


# ============================================================
# 15. COMPACT PARETO SUMMARY CSV
# ============================================================

summary_rows = []


for keys, g in all_pareto.groupby(
    [
        "task",
        "dataset",
        "formulation",
        "utility_metric",
    ]
):

    task, dataset, formulation, metric = keys

    frontier = (
        g[
            g["pareto"]
        ]
        .sort_values(
            "lambda"
        )
    )

    summary_rows.append({
        "task": task,
        "dataset": dataset,
        "formulation": formulation,
        "utility_metric": metric,
        "pareto_lambdas": ", ".join(
            f"{x:g}"
            for x
            in frontier["lambda"]
        ),
        "n_tested": len(g),
        "n_pareto": len(frontier),
    })


pareto_summary = pd.DataFrame(
    summary_rows
)


pareto_summary.to_csv(
    OUT_DIR /
    "pareto_summary_all_metrics.csv",
    index=False
)


# ============================================================
# 16. MAX FAIRNESS IMPROVEMENT SUMMARY
#     DESCRIPTIVE ONLY
# ============================================================

max_fair_rows = []


for keys, g in normalized_all.groupby(
    [
        "task",
        "dataset",
        "formulation",
        "utility_metric",
    ]
):

    task, dataset, formulation, metric = keys

    strongest = (
        g
        .sort_values(
            "fairness_improvement_pct",
            ascending=False
        )
        .iloc[0]
    )

    max_fair_rows.append({
        "task": task,
        "dataset": dataset,
        "formulation": formulation,
        "utility_metric": metric,
        "lambda": float(
            strongest["lambda"]
        ),
        "fairness_improvement_pct": float(
            strongest[
                "fairness_improvement_pct"
            ]
        ),
        "utility_change_pct": float(
            strongest[
                "utility_change_pct"
            ]
        ),
        "pareto": bool(
            strongest["pareto"]
        ),
    })


max_fair_summary = pd.DataFrame(
    max_fair_rows
)


max_fair_summary.to_csv(
    OUT_DIR /
    "max_fairness_improvement_all_metrics.csv",
    index=False
)


print("\n")
print("======================================")
print("MAX ACHIEVED FAIRNESS IMPROVEMENT")
print("======================================")


for _, row in max_fair_summary.iterrows():

    print(
        f"("
        f"{row['task']}, "
        f"{row['dataset']}, "
        f"{row['formulation']}, "
        f"{row['utility_metric']}"
        f"): "
        f"fairness improvement = "
        f"{row['fairness_improvement_pct']:.2f}%, "
        f"utility change = "
        f"{row['utility_change_pct']:.2f}%, "
        f"lambda={row['lambda']:g}, "
        f"Pareto={row['pareto']}"
    )


# ============================================================
# 17. NORMALIZED PLOT FUNCTION
#     OPTIONAL DIAGNOSTIC FIGURES
# ============================================================

def plot_normalized_tradeoff(
    g,
    output_file,
):

    g = g.copy()

    fig, ax = plt.subplots(
        figsize=(
            6.2,
            4.6
        )
    )

    # --------------------------------------------------------
    # All tested operating points
    # --------------------------------------------------------

    ax.scatter(
        g[
            "fairness_improvement_pct"
        ],
        g[
            "utility_change_pct"
        ],
        s=30,
        alpha=0.40,
        label="Tested operating points",
    )

    # --------------------------------------------------------
    # Pareto-efficient points
    # --------------------------------------------------------

    pg = (
        g[
            g["pareto"]
        ]
        .copy()
    )

    ax.scatter(
        pg[
            "fairness_improvement_pct"
        ],
        pg[
            "utility_change_pct"
        ],
        s=90,
        facecolors="none",
        linewidths=1.5,
        label="Pareto-efficient",
    )

    # --------------------------------------------------------
    # Baseline
    # --------------------------------------------------------

    baseline = g[
        np.isclose(
            g["lambda"],
            0.0
        )
    ]

    if len(baseline) == 1:

        ax.scatter(
            [0],
            [0],
            marker="x",
            s=70,
            linewidths=1.5,
            label="Baseline",
        )

    # --------------------------------------------------------
    # Reference lines
    # --------------------------------------------------------

    ax.axhline(
        0,
        linewidth=0.8,
        linestyle="--",
        alpha=0.6,
    )

    ax.axvline(
        0,
        linewidth=0.8,
        linestyle="--",
        alpha=0.6,
    )

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    ax.set_xlabel(
        "Pairwise-fairness improvement "
        "from baseline (%)"
    )

    ax.set_ylabel(
        "Predictive-utility change "
        "from baseline (%)"
    )

    dataset = (
        g["dataset"]
        .iloc[0]
    )

    formulation = (
        g["formulation"]
        .iloc[0]
    )

    utility_metric = (
        g["utility_metric"]
        .iloc[0]
    )

    ax.set_title(
        f"{dataset.upper()} — "
        f"{formulation} "
        f"({utility_metric})"
    )

    ax.legend(
        frameon=False,
        fontsize=8,
    )

    fig.tight_layout()

    fig.savefig(
        output_file.with_suffix(
            ".pdf"
        ),
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


# ============================================================
# 18. GENERATE OPTIONAL NORMALIZED FIGURES
# ============================================================

for keys, g in normalized_all.groupby(
    [
        "task",
        "dataset",
        "formulation",
        "utility_metric",
    ]
):

    task, dataset, formulation, metric = keys

    filename = (
        f"normalized_"
        f"{safe_name(task)}_"
        f"{safe_name(dataset)}_"
        f"{safe_name(formulation)}_"
        f"{safe_name(metric)}"
    )

    plot_normalized_tradeoff(
        g,
        OUT_DIR / filename,
    )


# ============================================================
# 19. FINISH
# ============================================================

print("\nDone.")

print(
    "Outputs saved to:",
    OUT_DIR.resolve()
)