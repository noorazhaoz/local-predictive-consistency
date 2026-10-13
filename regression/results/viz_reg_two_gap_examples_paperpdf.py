#!/usr/bin/env python3
# viz_reg_examples_paperpdf.py
#
# Paper-friendly local neighbourhood viz for regression (log-pred):
#   - SAME neighbourhood N(i): tau-ball in your FAIR/metric space
#   - Left: Laplacian preds, Right: URW preds
#   - Edges from centre to neighbours encode |Δp| by thickness/alpha
#   - Mean marker shown as orange "x" (optional)
#   - No corner summary boxes: ALL numbers go into the title (2 lines).
#
# Two selection modes:
#   (A) avg_better:
#       Lap max|Δp| small AND URW |p-mean(N)| smaller
#   (B) pairworse_avgbetter:
#       URW pairwise stat WORSE than Lap by >= min_pair_worse
#       BUT URW |p-mean(N)| BETTER than Lap by >= min_avg_better
#
# NOTE on thresholds:
#   min_pair_worse: require (pair_stat_urw - pair_stat_lap) >= this
#   min_avg_better: require (|p-mean|_lap - |p-mean|_urw) >= this

import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


# ----------------------------
# Paper style
# ----------------------------
def set_paper_style():
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
        "legend.fontsize": 9,
        "axes.linewidth": 0.8,
        "grid.linewidth": 0.6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


# ----------------------------
# IO / transforms
# ----------------------------
def pick_raw_cols(df, prefix="raw_"):
    cols = [c for c in df.columns if c.startswith(prefix)]
    if not cols:
        raise ValueError(f"No raw feature columns found with prefix='{prefix}'.")
    return cols


def load_scaler(pkl_path):
    import joblib
    sc = joblib.load(pkl_path)
    if not hasattr(sc, "transform"):
        raise TypeError(f"Scaler at {pkl_path} has no .transform()")
    return sc


def build_space(df, raw_cols, scaler_pkl, proj_npy, metric_npy=None):
    """
    Neighbourhood space: raw -> scaler -> proj_compl -> (optional metric L)
    Must match the space used by your graph.
    """
    X_raw = df[raw_cols].to_numpy(dtype=np.float32)
    scaler = load_scaler(scaler_pkl)
    X_std = scaler.transform(X_raw).astype(np.float32)

    proj_compl = np.load(proj_npy).astype(np.float32)  # [d,d]
    X_fair = (X_std @ proj_compl).astype(np.float32)

    if metric_npy and metric_npy != "" and os.path.exists(metric_npy):
        L = np.load(metric_npy).astype(np.float32)      # [r,d]
        return (X_fair @ L.T).astype(np.float32)

    return X_fair


# ----------------------------
# Neighbours
# ----------------------------
def e_dist_sq_one_to_all(X, i):
    v = X[i:i+1]
    return np.maximum(((X - v) ** 2).sum(axis=1), 0.0)


def tau_neighbours(X, i, tau):
    d2 = e_dist_sq_one_to_all(X, i)
    nbr = np.where((d2 <= tau) & (np.arange(len(d2)) != i))[0]
    return nbr, d2[nbr]


# ----------------------------
# Local embedding
# ----------------------------
def local_tsne(X_sub, seed=0, perplexity=10.0, n_iter=1500):
    from sklearn.manifold import TSNE
    n = X_sub.shape[0]
    perp = float(perplexity)
    perp = min(perp, max(5.0, (n - 1) / 3.0))
    perp = min(perp, n - 1)
    tsne = TSNE(
        n_components=2, random_state=seed, init="pca",
        learning_rate="auto", perplexity=perp, n_iter=n_iter
    )
    return tsne.fit_transform(X_sub)


# ----------------------------
# Stats for titles
# ----------------------------
def panel_stats(vals, center_pos=0):
    """
    vals: [center + neighbours]
    returns:
      p0, meanN, gap_pm, mean_abs_dp, max_abs_dp
    """
    p0 = float(vals[center_pos])
    neigh = np.delete(vals, center_pos)
    meanN = float(np.mean(neigh)) if len(neigh) else float("nan")
    gap_pm = abs(p0 - meanN) if len(neigh) else float("nan")
    dp = np.abs(neigh - p0) if len(neigh) else np.array([0.0])
    mean_abs_dp = float(np.mean(dp)) if len(dp) else 0.0
    max_abs_dp = float(np.max(dp)) if len(dp) else 0.0
    return p0, meanN, gap_pm, mean_abs_dp, max_abs_dp


# ----------------------------
# Plot helpers
# ----------------------------
def pad_axis(ax, Z, pad=0.18):
    x_min, x_max = float(Z[:, 0].min()), float(Z[:, 0].max())
    y_min, y_max = float(Z[:, 1].min()), float(Z[:, 1].max())
    dx = max(x_max - x_min, 1e-9)
    dy = max(y_max - y_min, 1e-9)
    ax.set_xlim(x_min - pad * dx, x_max + pad * dx)
    ax.set_ylim(y_min - pad * dy, y_max + pad * dy)


def annotate_inward(ax, x, y, text, fs=9, weight="bold", xytext=(8, 8), zorder=20):
    """
    Push annotation towards panel interior to avoid clipping.
    """
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    xc = 0.5 * (x0 + x1)
    yc = 0.5 * (y0 + y1)

    dx = abs(xytext[0]) if x < xc else -abs(xytext[0])
    dy = abs(xytext[1]) if y < yc else -abs(xytext[1])

    ha = "left" if dx > 0 else "right"
    va = "bottom" if dy > 0 else "top"

    ax.annotate(
        text, (x, y),
        textcoords="offset points", xytext=(dx, dy),
        ha=ha, va=va,
        fontsize=fs, fontweight=weight,
        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=1.0),
        clip_on=True, zorder=zorder
    )


def plot_panel(
    ax, Z, vals, center_pos, title,
    ann_far_pos=None,
    point_size=30, alpha=0.95, vmin=None, vmax=None,
    show_mean=True, show_arrow=True,
    edge_alpha_base=0.08, edge_alpha_scale=5.0,
    edge_lw_base=0.55, edge_lw_scale=7.0,
    show_mean_marker=True,
):
    sc = ax.scatter(
        Z[:, 0], Z[:, 1],
        c=vals, s=point_size, alpha=alpha, linewidths=0,
        vmin=vmin, vmax=vmax, zorder=3
    )

    # IMPORTANT: set limits before inward annotations
    pad_axis(ax, Z, pad=0.18)

    # edges encode |Δp|
    p0 = float(vals[center_pos])
    for j in range(Z.shape[0]):
        if j == center_pos:
            continue
        dij = abs(float(vals[j]) - p0)
        a = float(np.clip(edge_alpha_base + edge_alpha_scale * dij, 0.05, 0.75))
        lw = float(np.clip(edge_lw_base + edge_lw_scale * dij, 0.45, 2.35))
        ax.plot(
            [Z[center_pos, 0], Z[j, 0]],
            [Z[center_pos, 1], Z[j, 1]],
            linewidth=lw, alpha=a, zorder=1
        )

    # centre star only
    ax.scatter(
        Z[center_pos, 0], Z[center_pos, 1],
        s=230, marker="*", edgecolor="black", linewidth=1.2, zorder=8
    )

    # mean marker + arrow (mean is computed in EMBEDDING space)
    if show_mean and Z.shape[0] >= 2:
        Z_neigh = np.delete(Z, center_pos, axis=0)
        z_mean = Z_neigh.mean(axis=0)

        if show_mean_marker:
            ax.scatter(
                z_mean[0], z_mean[1],
                s=180, marker="x", linewidths=2.4, zorder=30
            )

        if show_arrow:
            ax.annotate(
                "",
                xy=(z_mean[0], z_mean[1]),
                xytext=(Z[center_pos, 0], Z[center_pos, 1]),
                arrowprops=dict(arrowstyle="->", lw=1.25),
                zorder=25
            )

    # label a few far points (embedding-far), with inward offset to avoid clipping
    if ann_far_pos is not None:
        for pos in ann_far_pos:
            if pos == center_pos:
                continue
            annotate_inward(
                ax, Z[pos, 0], Z[pos, 1],
                f"p={float(vals[pos]):.3f}",
                fs=10, weight="bold", xytext=(10, 10), zorder=40
            )

    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])
    ax.grid(alpha=0.18)
    return sc


# ----------------------------
# Main callable
# ----------------------------
def run_viz(
    lap_csv,
    urw_csv,
    scaler_pkl,
    proj_compl_npy,
    metric_npy="",

    lap_pred_col="pred_log",
    urw_pred_col="pred_log",

    feat_prefix="raw_",
    tau=5.0,
    max_neigh=40,

    # candidate selection controls
    select_mode="pairworse_avgbetter",   # "avg_better" OR "pairworse_avgbetter"
    min_neigh=8,

    # avg_better mode thresholds
    delta_pair_lap_max=0.08,    # require Lap max|Δp| <= this
    eps_avg_urw=0.02,           # require URW |p-mean(N)| <= this

    # pairworse_avgbetter mode thresholds
    min_pair_worse=0.001,       # require URW pairwise stat - Lap pairwise stat >= this
    min_avg_better=0.001,       # require Lap |p-mean| - URW |p-mean| >= this
    pair_stat="mean",           # "max" or "mean" (which pairwise stat to compare)

    seed=0,
    perplexity=10.0,
    tsne_iter=1500,

    out_dir="results/inspect_reg_examples",
    title_prefix="Employee (log-salary)",
    lam_show=30.0,              # shown in title as λ
    save_png=True,
):
    set_paper_style()
    os.makedirs(out_dir, exist_ok=True)

    lap = pd.read_csv(lap_csv)
    urw = pd.read_csv(urw_csv)

    for df, name in [(lap, "lap"), (urw, "urw")]:
        if "orig_index" not in df.columns:
            raise KeyError(f"[{name}] missing 'orig_index'")
    if lap_pred_col not in lap.columns:
        raise KeyError(f"[lap] missing pred col {lap_pred_col}")
    if urw_pred_col not in urw.columns:
        raise KeyError(f"[urw] missing pred col {urw_pred_col}")

    raw_cols = pick_raw_cols(lap, prefix=feat_prefix)

    # merge + robust pred col resolve
    m = lap[["orig_index", lap_pred_col] + raw_cols].merge(
        urw[["orig_index", urw_pred_col]],
        on="orig_index", how="inner",
        suffixes=("_lap", "_urw")
    )

    lap_pred_merge = f"{lap_pred_col}_lap" if f"{lap_pred_col}_lap" in m.columns else lap_pred_col
    urw_pred_merge = f"{urw_pred_col}_urw" if f"{urw_pred_col}_urw" in m.columns else urw_pred_col
    if lap_pred_merge not in m.columns:
        raise KeyError(f"Cannot find lap pred col after merge. Tried '{lap_pred_merge}' and '{lap_pred_col}'.")
    if urw_pred_merge not in m.columns:
        raise KeyError(f"Cannot find urw pred col after merge. Tried '{urw_pred_merge}' and '{urw_pred_col}'.")

    m = m.rename(columns={lap_pred_merge: "pred_lap", urw_pred_merge: "pred_urw"})
    m = m.sort_values("orig_index").reset_index(drop=True)

    # build neighbourhood space
    X_space = build_space(m, raw_cols, scaler_pkl, proj_compl_npy, metric_npy)

    pred_lap = m["pred_lap"].to_numpy(dtype=float)
    pred_urw = m["pred_urw"].to_numpy(dtype=float)
    orig_ids = m["orig_index"].to_numpy(dtype=int)

    # -------- selection --------
    candidates = []
    for i in range(len(m)):
        nbr, d2 = tau_neighbours(X_space, i, tau)
        if len(nbr) < min_neigh:
            continue

        if len(nbr) > max_neigh:
            keep = np.argsort(d2)[:max_neigh]
            nbr = nbr[keep]
            d2 = d2[keep]

        d_lap = np.abs(pred_lap[i] - pred_lap[nbr])
        d_urw = np.abs(pred_urw[i] - pred_urw[nbr])

        pair_max_lap, pair_mean_lap = float(np.max(d_lap)), float(np.mean(d_lap))
        pair_max_urw, pair_mean_urw = float(np.max(d_urw)), float(np.mean(d_urw))

        mean_lap = float(np.mean(pred_lap[nbr]))
        mean_urw = float(np.mean(pred_urw[nbr]))
        avg_diff_lap = abs(float(pred_lap[i]) - mean_lap)
        avg_diff_urw = abs(float(pred_urw[i]) - mean_urw)

        if select_mode == "avg_better":
            if pair_max_lap > delta_pair_lap_max:
                continue
            if avg_diff_urw > eps_avg_urw:
                continue
            improve = avg_diff_lap - avg_diff_urw
            if improve <= 0:
                continue
            score = improve

        elif select_mode == "pairworse_avgbetter":
            pair_lap = pair_max_lap if pair_stat == "max" else pair_mean_lap
            pair_urw = pair_max_urw if pair_stat == "max" else pair_mean_urw

            worsen = pair_urw - pair_lap
            improve = avg_diff_lap - avg_diff_urw

            if worsen < min_pair_worse:
                continue
            if improve < min_avg_better:
                continue

            score = improve
        else:
            raise ValueError(f"Unknown select_mode: {select_mode}")

        candidates.append((
            score, i, len(nbr),
            avg_diff_lap, avg_diff_urw,
            pair_max_lap, pair_mean_lap,
            pair_max_urw, pair_mean_urw
        ))

    if len(candidates) == 0:
        raise RuntimeError(
            f"No candidates found for select_mode='{select_mode}'. "
            f"Try loosening min_pair_worse/min_avg_better, or increasing tau/max_neigh."
        )

    candidates.sort(reverse=True, key=lambda x: x[0])
    top2 = candidates[:2]

    # save chosen summary
    out_chosen = []
    for (score, i, nnb, adl, adu, pmax_l, pmean_l, pmax_u, pmean_u) in top2:
        out_chosen.append({
            "orig_index": int(orig_ids[i]),
            "select_mode": select_mode,
            "pair_stat": pair_stat,
            "score(avg_improve)": float(score),
            "avg_diff_lap": float(adl),
            "avg_diff_urw": float(adu),
            "avg_improve(lap-urw)": float(adl - adu),
            "pair_max_lap": float(pmax_l),
            "pair_mean_lap": float(pmean_l),
            "pair_max_urw": float(pmax_u),
            "pair_mean_urw": float(pmean_u),
            "pair_worsen(max)": float(pmax_u - pmax_l),
            "pair_worsen(mean)": float(pmean_u - pmean_l),
            "pred_lap_center": float(pred_lap[i]),
            "pred_urw_center": float(pred_urw[i]),
        })
    chosen_path = os.path.join(out_dir, "chosen_two_examples.csv")
    pd.DataFrame(out_chosen).to_csv(chosen_path, index=False)
    print(f"[OK] saved chosen -> {chosen_path}")

    # -------- plotting --------
    for t, (score, i, nnb, adl, adu, pmax_l, pmean_l, pmax_u, pmean_u) in enumerate(top2, start=1):
        nbr, d2 = tau_neighbours(X_space, i, tau)
        if len(nbr) > max_neigh:
            keep = np.argsort(d2)[:max_neigh]
            nbr = nbr[keep]
            d2 = d2[keep]

        sub_rows = np.concatenate([[i], nbr])
        X_sub = X_space[sub_rows]
        Z = local_tsne(X_sub, seed=seed, perplexity=perplexity, n_iter=tsne_iter)

        lap_sub = pred_lap[sub_rows]
        urw_sub = pred_urw[sub_rows]

        center_orig = int(orig_ids[i])

        # label a few far points by EMBEDDING distance (clean)
        k_far = 3
        dZ = np.sqrt(((Z - Z[0]) ** 2).sum(axis=1))
        far_order = np.argsort(dZ)[::-1]
        far_pos = [int(p) for p in far_order if p != 0][:k_far]

        # shared color scale
        vmin = float(min(lap_sub.min(), urw_sub.min()))
        vmax = float(max(lap_sub.max(), urw_sub.max()))

        # panel stats for titles
        pL, meanL, gapL, meanDpL, maxDpL = panel_stats(lap_sub, center_pos=0)
        pU, meanU, gapU, meanDpU, maxDpU = panel_stats(urw_sub, center_pos=0)

        # Titles: 2 lines (as you requested)
        supt1 = rf"{title_prefix} (example {t})"
        supt2 = rf"$\lambda={lam_show:g}$, $\tau={tau:g}$"
        title_lap = rf"Lap: $p_i={pL:.3f}$, $|p_i-\overline{{p}}_N|={gapL:.4f}$, $\mathrm{{mean}}|\Delta p|={meanDpL:.4f}$, $\max|\Delta p|={maxDpL:.4f}$"
        title_urw = rf"URW: $p_i={pU:.3f}$, $|p_i-\overline{{p}}_N|={gapU:.4f}$, $\mathrm{{mean}}|\Delta p|={meanDpU:.4f}$, $\max|\Delta p|={maxDpU:.4f}$"

        fig, axes = plt.subplots(1, 2, figsize=(14, 8))
        axL, axR = axes

        fig.suptitle(supt1, fontsize=12, y=0.985)
        fig.text(0.5, 0.945, supt2, ha="center", va="center", fontsize=10.3)
        fig.text(0.5, 0.913, title_lap, ha="center", va="center", fontsize=10.3)
        fig.text(0.5, 0.883, title_urw, ha="center", va="center", fontsize=10.3)

        sc1 = plot_panel(
            axL, Z, lap_sub, 0, r"Std. Graph Laplacian ($p_{\log}$)",
            ann_far_pos=far_pos,
            vmin=vmin, vmax=vmax,
            show_mean=True, show_arrow=True,
            show_mean_marker=True,
        )
        sc2 = plot_panel(
            axR, Z, urw_sub, 0, r"URW ($p_{\log}$)",
            ann_far_pos=far_pos,
            vmin=vmin, vmax=vmax,
            show_mean=True, show_arrow=True,
            show_mean_marker=True,
        )

        cb1 = fig.colorbar(sc1, ax=axL, fraction=0.046, pad=0.04)
        cb1.set_label(r"$p_{\log}$")
        cb2 = fig.colorbar(sc2, ax=axR, fraction=0.046, pad=0.04)
        cb2.set_label(r"$p_{\log}$")

        # more top margin to accommodate 3 title rows
        fig.subplots_adjust(left=0.055, right=0.975, bottom=0.10, top=0.82, wspace=0.18)

        out_pdf = os.path.join(out_dir, f"example_{t}_idx{center_orig}.pdf")
        fig.savefig(out_pdf)
        print(f"[OK] saved -> {out_pdf}")

        if save_png:
            out_png = os.path.join(out_dir, f"example_{t}_idx{center_orig}.png")
            fig.savefig(out_png, dpi=300)
            print(f"[OK] saved -> {out_png}")

        plt.close(fig)

    print("[DONE]")


# ----------------------------
# Your call (lambda=30)
# ----------------------------
run_viz(
    lap_csv="lap_depth1_lambda30.csv",
    urw_csv="urw_depth1_lambda30.csv",
    scaler_pkl="reg_urw_depth1_lambda30.0_scaler.pkl",
    proj_compl_npy="reg_urw_depth1_lambda30.0_proj_compl.npy",
    metric_npy="reg_urw_depth1_lambda30.0_L_metric.npy",
    out_dir="results/inspect_reg_examples_lambda30",
    save_png=True,

    lam_show=30.0,

    select_mode="pairworse_avgbetter",
    pair_stat="mean",
    min_pair_worse=0.02,
    min_avg_better=0.001,

    tau=5.0,
    max_neigh=40,
    min_neigh=8,
)
