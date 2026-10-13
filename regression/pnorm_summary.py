import os
import re
import glob
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# Settings
# ============================================================
RESULT_DIR = "results_pnorm_new"
OUTFILE = os.path.join(RESULT_DIR, "pnorm_summary.csv")

EPS_LIST = [0.01, 0.02, 0.05, 0.10]


# ============================================================
# Parse filename
# Example:
#   urw_p1_lambda0.001.csv
#   urw_p4_lambda0.3.csv
#   urw_pinf_lambda1.csv
# ============================================================
pattern = re.compile(
    r"urw_p(?P<p>1|4|inf)_lambda(?P<lam>[0-9.eE+-]+)\.csv$"
)


rows = []

files = sorted(glob.glob(os.path.join(RESULT_DIR, "urw_p*_lambda*.csv")))

if not files:
    raise FileNotFoundError(
        f"No files found in {RESULT_DIR} matching urw_p*_lambda*.csv"
    )

print(f"Found {len(files)} files.")


for path in files:
    name = os.path.basename(path)

    m = pattern.match(name)
    if m is None:
        print(f"[Skip] filename not matched: {name}")
        continue

    p_raw = m.group("p")
    lam = float(m.group("lam"))

    p_label = "inf" if p_raw == "inf" else int(p_raw)

    df = pd.read_csv(path)

    required = {
        "y_real",
        "pred_real",
        "if_score_urw_log",
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{name} is missing required columns: {sorted(missing)}"
        )

    y = df["y_real"].to_numpy(dtype=float)
    pred = df["pred_real"].to_numpy(dtype=float)
    if_score = df["if_score_urw_log"].to_numpy(dtype=float)

    # -------------------------
    # Predictive metrics
    # -------------------------
    mae = mean_absolute_error(y, pred)
    mse = mean_squared_error(y, pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y, pred)

    # -------------------------
    # IF metrics
    # -------------------------
    if_mean = np.mean(if_score)
    if_median = np.median(if_score)
    if_std = np.std(if_score, ddof=1)
    if_max = np.max(if_score)

    row = {
        "p": p_label,
        "lambda": lam,
        "MAE": mae,
        "RMSE": rmse,
        "MSE": mse,
        "R2": r2,
        "IF_mean": if_mean,
        "IF_median": if_median,
        "IF_std": if_std,
        "IF_max": if_max,
        "n_test": len(df),
        "source_file": name,
    }

    for eps in EPS_LIST:
        row[f"VR_eps_{eps:g}"] = np.mean(if_score > eps)

    rows.append(row)


# ============================================================
# Assemble summary
# ============================================================
summary = pd.DataFrame(rows)

# custom order: p=1, p=4, p=inf
p_order = {"1": 0, "4": 1, "inf": 2}

summary["_p_sort"] = summary["p"].astype(str).map(p_order)
summary = summary.sort_values(
    ["_p_sort", "lambda"]
).drop(columns="_p_sort").reset_index(drop=True)

summary.to_csv(OUTFILE, index=False)

print("\nSaved:")
print(OUTFILE)

print("\nSummary:")
print(summary.to_string(index=False))