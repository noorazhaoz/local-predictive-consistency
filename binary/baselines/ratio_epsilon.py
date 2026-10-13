#!/usr/bin/env python3
# vr_simple_twofiles.py
#
# Compute violation ratio for one or more CSV files:
#   VR(eps) = mean(IF > eps)
#
# Output: ONE csv
#   rows: file
#   cols: vr_eps*
#
# Example:
# python vr_simple_twofiles.py \
#   --files baseline.csv fair.csv \
#   --if_col if_score \
#   --eps 0.001 0.005 0.01 0.02 0.05 \
#   --out vr_results.csv

import argparse
import numpy as np
import pandas as pd

def compute_vr(x, eps):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan
    return float(np.mean(x > eps))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", required=True,
                    help="CSV files to evaluate")
    ap.add_argument("--if_col", type=str, required=True,
                    help="Column name containing IF values")
    ap.add_argument("--eps", type=float, nargs="+", required=True,
                    help="List of epsilon thresholds")
    ap.add_argument("--out", type=str, default="vr.csv")
    args = ap.parse_args()

    rows = []

    for f in args.files:
        df = pd.read_csv(f)

        if args.if_col not in df.columns:
            raise ValueError(f"{f} missing column {args.if_col}")

        scores = df[args.if_col].values

        row = {"file": f}
        for eps in args.eps:
            row[f"vr_eps{eps}"] = compute_vr(scores, eps)

        rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)
    print("[OK] saved:", args.out)

if __name__ == "__main__":
    main()
