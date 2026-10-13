import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

base = "regression/results/"


files_d1 = glob.glob(os.path.join(base, "urw_lambda*.csv"))


files_d_other = glob.glob(os.path.join(base, "urw_depth*_lambda*.csv"))

print("depth=1 files:", files_d1)
print("depth>=2 files:", files_d_other)

rows = []

# ---------- depth = 1 ----------
for f in files_d1:
    df = pd.read_csv(f)
    if "if_score_urw_log" not in df.columns:
        print(f"[Skip] {f} has no 'if_score_urw_log'")
        continue

    # 从文件名里解析 lambda，例如 urw_lambda10.0.csv
    name = os.path.basename(f)
    # 拿到 'lambda10.0.csv' 然后去掉 'lambda' 和 '.csv'
    lam_str = name.split("lambda")[-1].replace(".csv", "")
    lam = float(lam_str)

    if_mean = df["if_score_urw_log"].mean()
    if_median = df["if_score_urw_log"].median()

    rows.append({
        "depth": 1,
        "lambda_cons": lam,
        "IF_mean": if_mean,
        "IF_median": if_median,
    })

# ---------- depth >= 2 ----------
for f in files_d_other:
    df = pd.read_csv(f)
    if "if_score_urw_log" not in df.columns:
        print(f"[Skip] {f} has no 'if_score_urw_log'")
        continue

    name = os.path.basename(f)
    # 例如: urw_depth2_lambda10.0.csv
    parts = name.split("_")
    depth_part = [p for p in parts if p.startswith("depth")][0]   # 'depth2'
    depth = int(depth_part.replace("depth", ""))

    lam_str = name.split("lambda")[-1].replace(".csv", "")
    lam = float(lam_str)

    if_mean = df["if_score_urw_log"].mean()
    if_median = df["if_score_urw_log"].median()

    rows.append({
        "depth": depth,
        "lambda_cons": lam,
        "IF_mean": if_mean,
        "IF_median": if_median,
    })

summary = pd.DataFrame(rows)
summary = summary.sort_values(["depth", "lambda_cons"])
print(summary)

# -----------------------------
# 1.  IF_mean vs lambda
# -----------------------------
plt.figure(figsize=(7,5))
for d in sorted(summary["depth"].unique()):
    df_d = summary[summary["depth"] == d].sort_values("lambda_cons")
    plt.plot(df_d["lambda_cons"], df_d["IF_mean"],
             marker="o", label=f"depth = {d}")

plt.xscale("log")
plt.xlabel(r"$\lambda_{\mathrm{cons}}$")
plt.ylabel("Mean IF (log predictions)")
plt.title("URW – Mean IF vs $\\lambda$ for different depths")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# -----------------------------
# median IF
# -----------------------------
plt.figure(figsize=(7,5))
for d in sorted(summary["depth"].unique()):
    df_d = summary[summary["depth"] == d].sort_values("lambda_cons")
    plt.plot(df_d["lambda_cons"], df_d["IF_median"],
             marker="o", label=f"depth = {d}")

plt.xscale("log")
plt.xlabel(r"$\lambda_{\mathrm{cons}}$")
plt.ylabel("Median IF (log predictions)")
plt.title("URW – Median IF vs $\\lambda$ for different depths")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
