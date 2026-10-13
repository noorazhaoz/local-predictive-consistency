import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# Settings
# ============================================================
SUMMARY_CSV = "results_pnorm_new/pnorm_summary.csv"
OUTDIR = "results_pnorm_new/figures"
os.makedirs(OUTDIR, exist_ok=True)

# ============================================================
# Load
# ============================================================
df = pd.read_csv(SUMMARY_CSV)

# Make sure p is treated consistently
df["p"] = df["p"].astype(str)
df["p"] = df["p"].replace({
    "1.0": "1",
    "4.0": "4",
    "infinity": "inf",
    "Infinity": "inf",
})

# Expected ordering
p_order = ["1", "4", "inf"]

# Lambda ordering
df = df.sort_values(["p", "lambda"]).reset_index(drop=True)

print(df)

# ============================================================
# Plotting x positions
# ============================================================
# Lambda=0 cannot be shown on a log scale.
# Place it slightly to the left of the smallest positive lambda.
positive_lambdas = sorted(df.loc[df["lambda"] > 0, "lambda"].unique())

if not positive_lambdas:
    raise ValueError("No positive lambda values found.")

min_positive = min(positive_lambdas)
zero_plot_value = min_positive / 3.0

df["lambda_plot"] = df["lambda"].copy()
df.loc[df["lambda"] == 0, "lambda_plot"] = zero_plot_value

# Tick positions and labels
all_lambdas = sorted(df["lambda"].unique())

tick_positions = [
    zero_plot_value if lam == 0 else lam
    for lam in all_lambdas
]

tick_labels = [
    "0" if lam == 0 else f"{lam:g}"
    for lam in all_lambdas
]

# ============================================================
# Figure 1: MAE vs lambda
# ============================================================
fig, ax = plt.subplots(figsize=(6.2, 4.3))

for p in p_order:
    sub = df[df["p"] == p].sort_values("lambda")

    if sub.empty:
        continue

    label = r"$p=\infty$" if p == "inf" else rf"$p={p}$"

    ax.plot(
        sub["lambda_plot"],
        sub["MAE"],
        marker="o",
        linewidth=1.8,
        markersize=5,
        label=label,
    )

ax.set_xscale("log")
ax.set_xticks(tick_positions)
ax.set_xticklabels(tick_labels)

ax.set_xlabel(r"Fairness regularization strength $\lambda$")
ax.set_ylabel("Mean absolute error (MAE)")
ax.set_title("Predictive error under tail-sensitive regularization")

ax.grid(True, alpha=0.25)
ax.legend(frameon=False)

fig.tight_layout()

mae_pdf = os.path.join(OUTDIR, "mae_compare_lambda.pdf")
mae_png = os.path.join(OUTDIR, "mae_compare_lambda.png")

fig.savefig(mae_pdf, bbox_inches="tight")
fig.savefig(mae_png, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"Saved: {mae_pdf}")
print(f"Saved: {mae_png}")

# ============================================================
# Figure 2: VR(eps=0.05) vs lambda
# ============================================================
vr_col = "VR_eps_0.05"

if vr_col not in df.columns:
    raise ValueError(
        f"Column '{vr_col}' not found. Available columns:\n{df.columns.tolist()}"
    )

fig, ax = plt.subplots(figsize=(6.2, 4.3))

for p in p_order:
    sub = df[df["p"] == p].sort_values("lambda")

    if sub.empty:
        continue

    label = r"$p=\infty$" if p == "inf" else rf"$p={p}$"

    ax.plot(
        sub["lambda_plot"],
        sub[vr_col],
        marker="o",
        linewidth=1.8,
        markersize=5,
        label=label,
    )

ax.set_xscale("log")
ax.set_xticks(tick_positions)
ax.set_xticklabels(tick_labels)

ax.set_xlabel(r"Fairness regularization strength $\lambda$")
ax.set_ylabel(r"Violation rate ($\epsilon=0.05$)")
ax.set_title("Local-consistency violation rate")

ax.set_ylim(bottom=0)

ax.grid(True, alpha=0.25)
ax.legend(frameon=False)

fig.tight_layout()

vr_pdf = os.path.join(OUTDIR, "pnorm_vr_eps0.05.pdf")
vr_png = os.path.join(OUTDIR, "pnorm_vr_eps0.05.png")

fig.savefig(vr_pdf, bbox_inches="tight")
fig.savefig(vr_png, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"Saved: {vr_pdf}")
print(f"Saved: {vr_png}")