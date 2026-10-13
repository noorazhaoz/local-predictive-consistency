import glob
import os
import numpy as np
import pandas as pd


data_dir = "results_unified"

files = sorted(glob.glob(os.path.join(data_dir, "lap_compas_depth1_lambda*.csv")))
print(files)
print(f"found {len(files)} files")

rows = []
for f in files:
    df = pd.read_csv(f)
    if "if_score_lapl_monitor_forwardKL_prob" not in df.columns:
        print(f"skip {os.path.basename(f)}: no if_score column")
        continue

    q70 = df["if_score_lapl_monitor_forwardKL_prob"].quantile(0.70)
    q80 = df["if_score_lapl_monitor_forwardKL_prob"].quantile(0.80)
    q90 = df["if_score_lapl_monitor_forwardKL_prob"].quantile(0.90)

    rows.append({
        "file": os.path.basename(f),
        "q70": q70,
        "q80": q80,
        "q90": q90,
    })

qdf = pd.DataFrame(rows).sort_values("file")
print(qdf)

eps_low  = float(qdf["q70"].median())
eps_mid  = float(qdf["q80"].median())
eps_high = float(qdf["q90"].median())

print("\nSuggested eps:")
print(f"eps_low  (median q70) = {eps_low:.6f}")
print(f"eps_mid  (median q80) = {eps_mid:.6f}")
print(f"eps_high (median q90) = {eps_high:.6f}")