import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

def set_ml_paper_style():
    mpl.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "lines.linewidth": 2.0,
        "lines.markersize": 5.0,
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 1.0,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "legend.frameon": False,
    })

set_ml_paper_style()

VR_FILE = "results/adult_binary_vr_by_eps.csv"
METHODS_WANTED = ["urw", "lap"]  # case-insensitive

# ----------------------------
# Load
# ----------------------------
df = pd.read_csv(VR_FILE)

eps_cols = [c for c in df.columns if c.startswith("vr_eps")]
if not eps_cols:
    raise ValueError("No vr_eps* columns found.")

def col_to_eps(c: str) -> float:
    return float(c.replace("vr_eps", ""))

eps = np.array([col_to_eps(c) for c in eps_cols], dtype=float)

# sort eps
order = np.argsort(eps)
eps = eps[order]
eps_cols = [eps_cols[i] for i in order]

# ---- FIX: drop eps <= 0 for log-scale AUC ----
mask_pos = eps > 0
if not np.all(mask_pos):
    dropped = eps[~mask_pos]
    print(f"[WARN] Dropping eps <= 0 for log-AUC: {dropped.tolist()}")
eps = eps[mask_pos]
eps_cols = [c for c, keep in zip(eps_cols, mask_pos) if keep]

if len(eps) < 2:
    raise RuntimeError("Need at least 2 positive eps points to compute AUC over log(eps).")

# numeric conversion for VR columns
for c in eps_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# lambda column
if "lambda" in df.columns:
    df["lambda_num"] = pd.to_numeric(df["lambda"], errors="coerce")
elif "lambda_str" in df.columns:
    df["lambda_num"] = pd.to_numeric(df["lambda_str"], errors="coerce")
else:
    raise ValueError("No 'lambda' or 'lambda_str' column found in VR file.")

df = df.dropna(subset=["lambda_num"]).copy()

# normalize method labels
df["method_norm"] = df["method"].astype(str).str.strip().str.lower()

methods_present = set(df["method_norm"].unique())
methods_to_plot = [m.lower() for m in METHODS_WANTED if m.lower() in methods_present]
if not methods_to_plot:
    methods_to_plot = sorted(methods_present)

print("Methods in file:", sorted(methods_present))
print("Plotting methods:", methods_to_plot)
print("Using eps grid (positive):", eps.tolist())

# ----------------------------
# Compute AUC_VR per (method, lambda)
# AUC over log-eps using trapezoidal rule
# ----------------------------
log_eps = np.log(eps)
dlog = np.diff(log_eps)
log_span = float(log_eps[-1] - log_eps[0])  # for normalization if desired

rows = []
for method in methods_to_plot:
    d = df[df["method_norm"] == method].copy()
    if d.empty:
        continue

    for lam, g in d.groupby("lambda_num"):
        vr_vals = g[eps_cols].mean(axis=0).to_numpy(dtype=float)

        if np.all(np.isnan(vr_vals)):
            continue

        # fill any NaNs along eps order (shouldn't happen if your VR NaN ratio is 0)
        s = pd.Series(vr_vals).interpolate(limit_direction="both").to_numpy()

        auc = float(np.sum(0.5 * (s[:-1] + s[1:]) * dlog))

        # OPTIONAL (recommended): normalize by log-span -> "average VR over log-eps"
        auc_norm = auc / log_span

        rows.append({"method": method, "lambda": float(lam), "auc_vr": auc_norm})

auc_df = pd.DataFrame(rows)
if auc_df.empty:
    raise RuntimeError("auc_df is empty -> nothing to plot (unexpected given your diagnostics).")

auc_df = auc_df.sort_values(["method", "lambda"]).reset_index(drop=True)
print("[OK] auc_df head:")
print(auc_df.head(10))

# ----------------------------
# Plot
# ----------------------------
fig, ax = plt.subplots(figsize=(6.4, 4.2))

for method in methods_to_plot:
    d = auc_df[auc_df["method"] == method].copy()
    if d.empty:
        continue

    ax.plot(d["lambda"].to_numpy(float),
            d["auc_vr"].to_numpy(float),
            marker="o",
            label=method.upper())

ax.set_xscale("log")
ax.set_xlabel(r"Fairness weight $\lambda$ (log scale)")
ax.set_ylabel(r"Avg VR over $\log \varepsilon$ (lower is better)")
ax.set_title("Adult (binary): Overall violation vs fairness weight")
ax.axvspan(5, 50, color="grey", alpha=0.08)

ax.legend()

plt.show()
