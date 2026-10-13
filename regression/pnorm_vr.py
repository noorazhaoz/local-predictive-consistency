# plot_pnorm_vr.py  (p-norm version)

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


P_TAG_RE = re.compile(r"urw_(p1|p4|pinf)_lambda", re.IGNORECASE)


def _infer_p_tag(path_str):
    if not isinstance(path_str, str):
        return None
    m = P_TAG_RE.search(path_str)
    if not m:
        return None
    return m.group(1).lower()


def _p_label(ptag):
    # display p labels directly
    if ptag == "pinf":
        return r"$p=\infty$"
    return f"$p={ptag[1:]}$"


def plot_pnorm_vr_curves(
    vr_long_path,
    out_dir=None,
    eps_list=None,
    logx=True,
    include_zero=True,
    fmt="pdf",
):
    vr_long_path = os.path.abspath(vr_long_path)
    if not os.path.isfile(vr_long_path):
        raise FileNotFoundError(vr_long_path)

    out_dir = out_dir or os.path.dirname(vr_long_path)
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(vr_long_path)

    if "file" not in df.columns:
        raise ValueError(
            "vr_long.csv missing 'file' column. "
            "Recompute VR with include_file=True."
        )

    df["p_tag"] = df["file"].apply(_infer_p_tag)
    df = df[~df["p_tag"].isna()].copy()

    if df.empty:
        raise RuntimeError("No p-norm rows detected.")

    eps_vals = sorted(df["eps"].unique()) if eps_list is None else eps_list

    order = ["p1", "p4", "pinf"]
    df["p_tag"] = pd.Categorical(df["p_tag"], categories=order, ordered=True)

    saved = []

    for eps in eps_vals:
        sub = df[df["eps"] == eps]
        if sub.empty:
            continue

        g = (
            sub.groupby(["p_tag", "lambda"], as_index=False)
            .agg(vr=("vr", "mean"))
            .sort_values(["p_tag", "lambda"])
        )

        g["lambda_plot"] = g["lambda"]

        if logx:
            pos = g[g["lambda"] > 0]["lambda"]
            if not pos.empty:
                min_pos = pos.min()
                if include_zero:
                    g.loc[g["lambda"] == 0, "lambda_plot"] = min_pos / 10
                else:
                    g = g[g["lambda"] > 0]

        plt.figure()

        for ptag in order:
            gg = g[g["p_tag"] == ptag]
            if gg.empty:
                continue
            plt.plot(
                gg["lambda_plot"],
                gg["vr"],
                marker="o",
                label=_p_label(ptag),
            )

        if logx:
            plt.xscale("log")

        plt.xlabel("lambda")
        plt.ylabel(f"VR (eps={eps})")
        plt.title(f"Employee regression: p-norm VR (eps={eps})")
        plt.legend()

        out_path = os.path.join(out_dir, f"pnorm_vr_eps{eps}.{fmt}")
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()

        saved.append(out_path)

    return saved


saved = plot_pnorm_vr_curves(
    "results_pnorm/employee_regression_pnorm_vr_long.csv"
)