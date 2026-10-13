#!/usr/bin/env python3
# vr_iflipper.py
#
# VR(eps) = mean_i 1[ IF_i > eps ]
#
# Only reads iFlipper IF files:
#   if_iflipper_depth1_rho0.1.csv
#   if_iflipper_depth1_rho0.7.csv
#   if_iflipper_depth1_rho0.9.csv
#
# Output:
#   iflipper_vr_by_rho.csv

import os
import re
import argparse
import numpy as np
import pandas as pd

# extract rho anywhere in filename, e.g. "..._rho0.1.csv"
RHO_RE = re.compile(r"_rho(?P<rho>[0-9.]+)(?:\.csv)?$|_rho(?P<rho2>[0-9.]+)\.")

IF_COL = "if_score_forwardKL_prob"

def parse_rho(fname: str):
    m = RHO_RE.search(fname)
    if not m:
        return None
    rho_str = m.group("rho") if m.group("rho") is not None else m.group("rho2")
    try:
        return float(rho_str)
    except:
        return None

def compute_vr(x, eps: float):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return np.nan, 0
    return float(np.mean(x > eps)), int(x.size)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", type=str, default=".",
                    help="Folder containing if_iflipper_depth1_rho*.csv")
    ap.add_argument("--eps", type=float, nargs="+", required=True,
                    help="e.g. 0.001 0.005 0.01 0.02 0.05")
    ap.add_argument("--out", type=str, default="iflipper_vr_by_rho.csv")
    args = ap.parse_args()

    # STRICTLY match IF files only (ignore metrics_*.csv)
    files = sorted([
        f for f in os.listdir(args.results_dir)
        if f.startswith("if_iflipper_depth1_rho") and f.endswith(".csv")
    ])

    print(f"[Info] matched {len(files)} IF files in {args.results_dir}")
    for f in files[:10]:
        print("  -", f)

    if not files:
        raise FileNotFoundError(
            f"No IF files matched. Expect filenames like 'if_iflipper_depth1_rho0.1.csv' in {args.results_dir}"
        )

    rows = []
    for f in files:
        rho = parse_rho(f)
        if rho is None:
            print(f"[WARN] cannot parse rho from {f}, skipped")
            continue

        path = os.path.join(args.results_dir, f)
        df = pd.read_csv(path)

        if IF_COL not in df.columns:
            raise KeyError(f"{f} missing IF column '{IF_COL}'. Columns={list(df.columns)[:20]}...")

        if_scores = df[IF_COL].to_numpy()

        for eps in args.eps:
            vr, n = compute_vr(if_scores, float(eps))
            rows.append({
                "rho": float(rho),
                "eps": float(eps),
                "vr": vr,
                "n": n,
                "file": f,
            })

    if not rows:
        raise RuntimeError("No VR rows produced (rows is empty). Check rho parsing and file matches.")

    out = pd.DataFrame(rows).sort_values(["rho", "eps"]).reset_index(drop=True)

    out_path = os.path.join(args.results_dir, args.out)
    out.to_csv(out_path, index=False)
    print(f"[OK] saved → {out_path}")

if __name__ == "__main__":
    main()
