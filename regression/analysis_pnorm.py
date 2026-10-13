import os, re, glob
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# ----------------------------
# Style (ML-paper)
# ----------------------------
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
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.minor.size": 2,
        "ytick.minor.size": 2,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.minor.width": 0.8,
        "ytick.minor.width": 0.8,

        "legend.frameon": False,
    })

set_ml_paper_style()

OUTDIR = "results_pnorm"
os.makedirs(OUTDIR, exist_ok=True)

# ---------------------------------------------------
# 1) Load all metric files
# ---------------------------------------------------
all_files = glob.glob(os.path.join(OUTDIR, "metrics_urw_depth*_p*_lambda*.csv"))
print(f"Found {len(all_files)} files")
if len(all_files) == 0:
    raise FileNotFoundError(f"No matching CSV files under {OUTDIR}/")

# robust parse from filename: ..._p{p}_lambda{L}.csv
_re = re.compile(r"_p(?P<p>[^_]+)_lambda(?P<lam>[0-9.eE+-]+)\.csv$")

def parse_p_token(p_token: str):
    s = str(p_token).strip().lower()
    if s in ["inf", "infty", "infinity"]:
        return np.inf
    s = s.replace("p", ".")
    return float(s)

def p_sort_key(p):
    return (np.isinf(p), p)

def p_label(p):
    if np.isinf(p):
        return r"$p=\infty$"
    if float(p).is_integer():
        return rf"$p={int(p)}$"
    return rf"$p={p}$"

# ---------------------------------------------------
# 2) Load metrics into one df
# ---------------------------------------------------
rows = []
for f in all_files:
    m = _re.search(f)
    if not m:
        print(f"[Skip] cannot parse filename: {f}")
        continue

    p_token = m.group("p")
    lam_token = m.group("lam")

    p = parse_p_token(p_token)
    lam = float(lam_token)

    df = pd.read_csv(f)
    if len(df) == 0:
        continue

    row = df.iloc[0].to_dict()
    row["p"] = p
    row["lambda"] = lam
    row["file"] = os.path.basename(f)
    rows.append(row)

if len(rows) == 0:
    raise FileNotFoundError("Parsed 0 metric rows. Check file naming.")

df_all = pd.DataFrame(rows)

need_cols = ["p", "lambda", "test_MAE_real", "test_RMSE_real", "IF_mean(URW_L1,log_pred)"]
for c in need_cols:
    if c not in df_all.columns:
        raise KeyError(f"Missing required column: {c}")

# ---------------------------------------------------
# 3) Effective lambda scaling across p
# ---------------------------------------------------
def compute_lambda_eff(row):
    p = row["p"]
    lam = row["lambda"]
    r_clip = row.get("r_clip", np.nan)

    if np.isinf(p):
        return lam

    if pd.isna(r_clip) or r_clip <= 0:
        r_clip = 1.0

    return lam * (float(r_clip) ** (float(p) - 1.0))

df_all["lambda_eff"] = df_all.apply(compute_lambda_eff, axis=1)
df_all = df_all.sort_values(["p", "lambda"])

# ---------------------------------------------------
# 4) Plot helpers
# ---------------------------------------------------
def plot_metric_vs_x(df, metric_name, xcol="lambda",
                     ylabel=None, title=None, filename=None,
                     xlog=True, save_pdf=True, save_png=False):

    fig, ax = plt.subplots(figsize=(7, 5))

    for p in sorted(df["p"].unique(), key=p_sort_key):
        sub = df[df["p"] == p].sort_values(xcol)
        ax.plot(sub[xcol], sub[metric_name], marker="o", label=p_label(p))

    if xlog:
        ax.set_xscale("symlog", linthresh=0.1)

    ax.set_xlabel(r"$\lambda$" if xcol=="lambda" else r"$\lambda_{\mathrm{eff}}$")
    ax.set_ylabel(ylabel if ylabel else metric_name)
    ax.set_title(title if title else f"{metric_name} vs {xcol}")
    ax.legend(loc="best")
    fig.tight_layout()

    if filename is None:
        filename = f"{metric_name}_vs_{xcol}".replace("/", "_")

    if save_pdf:
        fig.savefig(os.path.join(OUTDIR, f"{filename}.pdf"))
    if save_png:
        fig.savefig(os.path.join(OUTDIR, f"{filename}.png"), dpi=300)

    plt.close(fig)

def plot_tradeoff(df, x_metric, y_metric,
                  xlabel=None, ylabel=None, title=None, filename=None,
                  xlog=False, ylog=False, save_pdf=True):

    fig, ax = plt.subplots(figsize=(7, 5))

    for p in sorted(df["p"].unique(), key=p_sort_key):
        sub = df[df["p"] == p].sort_values("lambda")
        ax.plot(sub[x_metric], sub[y_metric], marker="o", label=p_label(p))

    ax.set_xlabel(xlabel if xlabel else x_metric)
    ax.set_ylabel(ylabel if ylabel else y_metric)
    ax.set_title(title if title else f"{y_metric} vs {x_metric}")
    ax.legend(loc="best")
    fig.tight_layout()

    if filename is None:
        filename = f"tradeoff_{y_metric}_vs_{x_metric}".replace("/", "_")

    if save_pdf:
        fig.savefig(os.path.join(OUTDIR, f"{filename}.pdf"))

    plt.close(fig)

# ---------------------------------------------------
# 5) Generate plots
# ---------------------------------------------------
plot_metric_vs_x(df_all, "test_RMSE_real", filename="rmse_compare_lambda")
plot_metric_vs_x(df_all, "test_MAE_real", filename="mae_compare_lambda")
plot_metric_vs_x(df_all, "IF_mean(URW_L1,log_pred)", filename="ifmean_compare_lambda")

plot_metric_vs_x(df_all, "test_MAE_real",
                 xcol="lambda_eff",
                 title=r"MAE vs $\lambda_{\mathrm{eff}}$",
                 filename="mae_compare_lambda_eff")

plot_metric_vs_x(df_all, "IF_mean(URW_L1,log_pred)",
                 xcol="lambda_eff",
                 title=r"IF Mean vs $\lambda_{\mathrm{eff}}$",
                 filename="ifmean_compare_lambda_eff")

plot_tradeoff(df_all,
              x_metric="IF_mean(URW_L1,log_pred)",
              y_metric="test_MAE_real",
              xlabel="IF Mean ↓",
              ylabel="MAE ↓",
              title="Trade-off: MAE vs IF Mean",
              filename="tradeoff_mae_vs_ifmean")

print("\nAll plots saved under results_pnorm/ (PDF).")