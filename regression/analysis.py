import pandas as pd
import numpy as np
import glob
import matplotlib as mpl
import matplotlib.pyplot as plt

def parse_info(fname):
    base = fname.split("/")[-1].replace(".csv", "")
    parts = base.split("_")
    # example: if_depth1_p4_lambda0.3
    p_token = [x for x in parts if x.startswith("p")][0]
    lam_token = [x for x in parts if x.startswith("lambda")][0]

    p = np.inf if p_token[1:] == "inf" else float(p_token[1:])
    lam = float(lam_token.replace("lambda",""))
    return p, lam

rows = []
for f in glob.glob("results_pnorm/urw_p*_lambda*.csv"):
    p, lam = parse_info(f)
    df = pd.read_csv(f)
    rows.append({
        "p": p,
        "lambda": lam,
        "IF_median": np.median(df["if_score_urw_log"]),
        "IF_q90": np.quantile(df["if_score_urw_log"], 0.9),
    })

df_if = pd.DataFrame(rows).sort_values(["p","lambda"])
print(df_if.head())

def plot_if_stat(df, col, ylabel, title, filename):
    fig, ax = plt.subplots(figsize=(7,5))
    for p in sorted(df["p"].unique(), key=lambda x: (np.isinf(x), x)):
        sub = df[df["p"] == p].sort_values("lambda")
        label = r"$p=\infty$" if np.isinf(p) else f"$p={int(p)}$"
        ax.plot(sub["lambda"], sub[col], marker="o", label=label)

    ax.set_xscale("log")
    ax.set_xlabel("lambda")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"results_pnorm/{filename}.pdf")
    plt.close(fig)

plot_if_stat(
    df_if, "IF_median",
    ylabel="IF median",
    title="IF Median vs Lambda",
    filename="if_median_vs_lambda"
)

plot_if_stat(
    df_if, "IF_q90",
    ylabel="IF 90% quantile",
    title="IF 90% Quantile vs Lambda",
    filename="if_q90_vs_lambda"
)
