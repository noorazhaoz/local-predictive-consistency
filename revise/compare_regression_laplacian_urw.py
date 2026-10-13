#!/usr/bin/env python3

import os
import glob
import argparse
import numpy as np
import pandas as pd


# ============================================================
# Utilities
# ============================================================

def read_metric_csvs(root):
    files = sorted(
        glob.glob(os.path.join(root, "**", "*.csv"), recursive=True)
    )

    rows = []

    for path in files:
        name = os.path.basename(path).lower()

        # We only want metric summary files, not per-instance IF files.
        if "metric" not in name:
            continue

        try:
            df = pd.read_csv(path)
        except Exception:
            continue

        if len(df) != 1:
            continue

        row = df.iloc[0].to_dict()
        row["_file"] = path
        row["_filename"] = name
        rows.append(row)

    if not rows:
        raise RuntimeError(
            f"No one-row metrics CSV files found under {root}"
        )

    return pd.DataFrame(rows)


def find_column(df, candidates, required=True):
    cols = list(df.columns)

    # Exact first
    for candidate in candidates:
        if candidate in cols:
            return candidate

    # Then substring / case-insensitive
    for candidate in candidates:
        candidate_lower = candidate.lower()

        for col in cols:
            if candidate_lower in str(col).lower():
                return col

    if required:
        raise KeyError(
            "Could not identify required column.\n"
            f"Candidates: {candidates}\n"
            f"Available columns:\n{cols}"
        )

    return None


def infer_method(filename):
    name = filename.lower()

    if "laplacian" in name or "lap_" in name or "_lap" in name:
        return "Laplacian"

    if "urw" in name:
        return "URW"

    return None


def pareto_frontier(
    df,
    fairness_col,
    utility_col,
    utility_direction="min"
):
    """
    Fairness is always lower-is-better.

    Regression utility:
        RMSE lower-is-better.
    """

    df = df.copy().reset_index(drop=True)

    pareto = np.ones(len(df), dtype=bool)

    for i in range(len(df)):

        fi = df.loc[i, fairness_col]
        ui = df.loc[i, utility_col]

        for j in range(len(df)):

            if i == j:
                continue

            fj = df.loc[j, fairness_col]
            uj = df.loc[j, utility_col]

            fair_not_worse = fj <= fi

            if utility_direction == "min":
                util_not_worse = uj <= ui
                strict = (fj < fi) or (uj < ui)

            else:
                util_not_worse = uj >= ui
                strict = (fj < fi) or (uj > ui)

            if fair_not_worse and util_not_worse and strict:
                pareto[i] = False
                break

    return df.loc[pareto].copy()


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        required=True,
        help="Folder containing both Laplacian and URW metrics."
    )

    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset substring, e.g. employee or insurance."
    )

    parser.add_argument(
        "--out_dir",
        default="comparison_regression"
    )

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = read_metric_csvs(args.root)

    dataset = args.dataset.lower()

    df = df[
        df["_filename"].str.contains(dataset, regex=False)
    ].copy()

    if len(df) == 0:
        raise RuntimeError(
            f"No metric files matching dataset '{args.dataset}'"
        )

    # --------------------------------------------------------
    # Method
    # --------------------------------------------------------

    df["method"] = df["_filename"].apply(infer_method)

    df = df.dropna(subset=["method"]).copy()

    print("\nFiles by method:")
    print(df.groupby("method").size())

    # --------------------------------------------------------
    # Lambda
    # --------------------------------------------------------

    lambda_col = find_column(
        df,
        [
            "lambda_cons",
            "lambda",
            "lam"
        ]
    )

    df["lambda"] = pd.to_numeric(
        df[lambda_col],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Utility = RMSE
    # --------------------------------------------------------

    rmse_col = find_column(
        df,
        [
            "test_rmse",
            "rmse"
        ]
    )

    # --------------------------------------------------------
    # Common graph diagnostic = LIF energy
    # --------------------------------------------------------

    lif_col = find_column(
        df,
        [
            "LIF_energy",
            "lif_energy",
            "LIF energy",
            "graph_energy"
        ]
    )

    print("\nDetected columns")
    print("----------------")
    print("lambda :", lambda_col)
    print("RMSE   :", rmse_col)
    print("LIF    :", lif_col)

    df[rmse_col] = pd.to_numeric(
        df[rmse_col],
        errors="coerce"
    )

    df[lif_col] = pd.to_numeric(
        df[lif_col],
        errors="coerce"
    )

    df = df.dropna(
        subset=["lambda", rmse_col, lif_col]
    ).copy()

    # --------------------------------------------------------
    # Clean table
    # --------------------------------------------------------

    all_points = df[
        [
            "method",
            "lambda",
            rmse_col,
            lif_col,
            "_file"
        ]
    ].copy()

    all_points = all_points.sort_values(
        ["method", "lambda"]
    )

    # --------------------------------------------------------
    # Within-method Pareto
    # --------------------------------------------------------

    within = []

    for method, g in all_points.groupby("method"):

        p = pareto_frontier(
            g,
            fairness_col=lif_col,
            utility_col=rmse_col,
            utility_direction="min"
        )

        p["pareto_scope"] = "within_method"
        within.append(p)

        print("\n===================================")
        print(method)
        print("===================================")

        print(
            p[
                [
                    "lambda",
                    rmse_col,
                    lif_col
                ]
            ]
            .sort_values(lif_col)
            .to_string(index=False)
        )

    within = pd.concat(within, ignore_index=True)

    # --------------------------------------------------------
    # GLOBAL Pareto
    # --------------------------------------------------------

    global_pareto = pareto_frontier(
        all_points,
        fairness_col=lif_col,
        utility_col=rmse_col,
        utility_direction="min"
    )

    global_pareto = global_pareto.sort_values(lif_col)

    print("\n\n===================================")
    print("GLOBAL PARETO FRONTIER")
    print("===================================")

    print(
        global_pareto[
            [
                "method",
                "lambda",
                rmse_col,
                lif_col
            ]
        ].to_string(index=False)
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    prefix = args.dataset.lower()

    all_points.to_csv(
        os.path.join(
            args.out_dir,
            f"{prefix}_regression_all_points.csv"
        ),
        index=False
    )

    within.to_csv(
        os.path.join(
            args.out_dir,
            f"{prefix}_regression_pareto_within_method.csv"
        ),
        index=False
    )

    global_pareto.to_csv(
        os.path.join(
            args.out_dir,
            f"{prefix}_regression_pareto_global.csv"
        ),
        index=False
    )

    print("\nSaved to:", args.out_dir)


if __name__ == "__main__":
    main()