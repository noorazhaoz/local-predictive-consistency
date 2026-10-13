import os, re, glob
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# ----------------------------
# Style (your ML-paper style)
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

# ----------------------------
# 0) Settings
# ----------------------------
QNORMS = [1, 2, 4, "inf"]   # was PNORMS

ROOT = "results_pnorm/urw"
OUT_DIR = "results_pnorm/plots_binary_qnorm/urw"   # optional rename output folder
os.makedirs(OUT_DIR, exist_ok=True)

# ----------------------------
# helpers
# ----------------------------
_lambda_re = re.compile(r"lambda([0-9eE\.\+\-]+)")

def parse_lambda_from_filename(path: str) -> float:
    base = os.path.basename(path)
    m = _lambda_re.search(base)
    if not m:
        raise ValueError(f"Cannot parse lambda from filename: {base}")
    lam_str = m.group(1).strip().rstrip(".")
    lam_str = re.sub(r"[^0-9eE\.\+\-]", "", lam_str).rstrip(".")
    return float(lam_str)

def q_to_str(q):
    if isinstance(q, str) and q.lower() in ["inf", "infty", "infinity"]:
        return "inf"
    return str(int(q)) if float(q).is_integer() else str(q).replace(".", "p")

def q_label(q):
    if (isinstance(q, str) and q.lower() in ["inf", "infty", "infinity"]) or (isinstance(q, (float, int)) and np.isinf(q)):
        return r"q = $\infty$"
    return f"q = {q}"

def load_metrics_for_q(q):
    """
    NOTE (Mode A):
    We keep reading file patterns with 'p{...}' because your filenames are p-based.
    Only the *notation* in plots is changed to q.
    """
    qstr = q_to_str(q)

    # filenames still contain p{...}
    pattern_a = os.path.join(ROOT, f"metrics_urw_depth1_p{qstr}_0_lambda*.csv")
    pattern_b = os.path.join(ROOT, f"metrics_urw_depth1_p{qstr}_lambda*.csv")

    files = sorted(glob.glob(pattern_a) + glob.glob(pattern_b))
    if not files:
        print(f"[WARN] No files found for q={q} with patterns:\n  {pattern_a}\n  {pattern_b}")
        return None

    dfs = []
    for f in files:
        df = pd.read_csv(f)
        lam = parse_lambda_from_filename(f)
        df["lambda"] = lam

        # store q_norm numeric for sorting (inf -> np.inf)
        if qstr == "inf":
            df["q_norm"] = np.inf
        else:
            df["q_norm"] = float(q)

        dfs.append(df)

    return pd.concat(dfs, ignore_index=True).sort_values("lambda")

def plot_multi_q(metrics_by_q, metric_col, title, ylabel, filename,
                 xscale_log=True, y_margin_ratio=0.05, show_markers=True, ymax=None):

    all_values = []
    for _, df in metrics_by_q.items():
        if metric_col in df.columns:
            all_values.extend(df[metric_col].to_numpy())

    if len(all_values) == 0:
        print(f"[SKIP] Metric column not found in any df: {metric_col}")
        return

    all_values = np.asarray(all_values, dtype=float)
    y_min, y_max = float(np.min(all_values)), float(np.max(all_values))
    y_range = y_max - y_min
    y_pad = y_margin_ratio * y_range if y_range > 0 else 0.0

    plt.figure(figsize=(6.0, 4.0))

    def _q_sort_key(q):
        if isinstance(q, str) and q.lower() in ["inf", "infty", "infinity"]:
            return (1, np.inf)
        return (0, float(q))

    for q in sorted(metrics_by_q.keys(), key=_q_sort_key):
        df = metrics_by_q[q]
        if metric_col not in df.columns:
            continue
        x = df["lambda"].to_numpy()
        y = df[metric_col].to_numpy()

        if ymax is not None:
            plt.ylim(bottom=0.0, top=ymax)
        else:
            plt.ylim(y_min - y_pad, y_max + y_pad)

        if show_markers:
            plt.plot(x, y, marker="o", label=q_label(q))
        else:
            plt.plot(x, y, label=q_label(q))

    plt.xlabel(r"$\lambda$")
    plt.ylabel(ylabel)
    plt.title(title)

    if xscale_log:
        plt.xscale("symlog", linthresh=0.05)

    plt.ylim(y_min - y_pad, y_max + y_pad)
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(OUT_DIR, f"{filename}.pdf")
    plt.savefig(out_path)
    plt.close()
    print(f"[Saved] {out_path}")


# ----------------------------
# 1) Load all q
# ----------------------------
metrics_by_q = {}
for q in QNORMS:
    df_q = load_metrics_for_q(q)
    if df_q is not None:
        metrics_by_q[q] = df_q

if not metrics_by_q:
    raise RuntimeError("No metrics loaded for any q. Check ROOT / file names / QNORMS.")

print("\nLoaded q-norms:", list(metrics_by_q.keys()))
for q, df in metrics_by_q.items():
    print(f"\n=== q = {q} ===")
    print(df.head(3))

# ----------------------------
# 2) Plots (PDF)
# ----------------------------
plot_multi_q(metrics_by_q, "IF_mean",
             "IF Mean vs $\\lambda$", "IF Mean", "ifmean_compare_qmulti")
plot_multi_q(metrics_by_q, "test_bal_acc",  "Balanced Accuracy vs $\\lambda$", "Balanced Accuracy", "balacc_compare_pmulti")

print(f"\nDone. All PDF plots saved under: {OUT_DIR}")
