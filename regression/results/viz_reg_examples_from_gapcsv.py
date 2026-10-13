#!/usr/bin/env python3
# viz_reg_examples_graphfaithful_no_main.py
#
# Graph-faithful local neighborhood visualization for regression (log-pred):
# - Neighbours are method-specific (Lap uses Lap P_te row, URW uses URW P_te row)
# - No union, no renormalization trick that changes q.
# - True weighted avg: q_full = (P @ p)[i]  (computed on full row)
# - For plotting, keep only top-mass neighbours (e.g. cumulative weight >= 0.90),
#   and report mass_covered and a visual q_vis = sum_{kept} w p  (NO renorm).
#
# One shared t-SNE embedding is built on the union of points that will appear in either panel
# ONLY for consistent layout. But each panel draws edges only to its own neighbours.
#
# Inputs:
# - lap_pred_csv, urw_pred_csv: must contain orig_index, pred_log, raw_*
# - lap_P_te.npz, urw_P_te.npz: CSR row-stochastic matrices (or nonneg weights)
# - lap_orig_index_te.npy, urw_orig_index_te.npy: graph-row -> orig_index mapping
# - scaler.pkl + proj_compl.npy (+ optional L_metric.npy): to build embedding space from raw_*
#
# Output:
# - results/.../example_idx<orig_index>.pdf/.png

import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import sparse


# ----------------------------
# Style
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


def build_space(df_merged, raw_cols, scaler_pkl, proj_compl_npy, metric_npy=""):
    """
    Embedding space: raw -> scaler -> proj_compl -> (optional metric L)
    """
    X_raw = df_merged[raw_cols].to_numpy(dtype=np.float32)
    scaler = load_scaler(scaler_pkl)
    X_std = scaler.transform(X_raw).astype(np.float32)

    proj_compl = np.load(proj_compl_npy).astype(np.float32)
    X_fair = (X_std @ proj_compl).astype(np.float32)

    if metric_npy and metric_npy != "" and os.path.exists(metric_npy):
        L = np.load(metric_npy).astype(np.float32)  # [r, d]
        return (X_fair @ L.T).astype(np.float32)

    return X_fair


def load_graph(P_npz, orig_index_npy):
    P = sparse.load_npz(P_npz).tocsr()
    orig = np.load(orig_index_npy).astype(int)
    if P.shape[0] != P.shape[1]:
        raise ValueError("P must be square")
    if P.shape[0] != len(orig):
        raise ValueError(f"Mismatch: P n={P.shape[0]} vs orig_index len={len(orig)}")
    return P, orig


def inv_map(orig_index_arr):
    return {int(o): i for i, o in enumerate(orig_index_arr)}


# ----------------------------
# t-SNE (local)
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
# Neighbour selection (graph-faithful)
# ----------------------------
def row_neighbors_topmass(P, row_i, mass_target=0.90, max_neigh=60, drop_self=True):
    """
    Select neighbors from CSR row by descending weight until cumulative mass >= mass_target,
    capped by max_neigh. Returns (idx, w, mass_covered) without renormalization.
    """
    row = P.getrow(row_i)
    idx = row.indices.astype(int)
    w = row.data.astype(np.float64)

    if drop_self:
        mask = idx != row_i
        idx, w = idx[mask], w[mask]

    if len(idx) == 0:
        return np.array([], dtype=int), np.array([], dtype=np.float64), 0.0

    # sort by weight desc
    order = np.argsort(w)[::-1]
    idx = idx[order]
    w = w[order]

    # cumulative mass (row-stochastic => sum ~1, else use sum as "mass")
    total = float(np.sum(w))
    if total <= 0:
        return np.array([], dtype=int), np.array([], dtype=np.float64), 0.0

    cum = 0.0
    keep_idx = []
    keep_w = []
    for j, wij in zip(idx, w):
        keep_idx.append(int(j))
        keep_w.append(float(wij))
        cum += float(wij)
        if cum / total >= mass_target:
            break
        if len(keep_idx) >= max_neigh:
            break

    keep_idx = np.array(keep_idx, dtype=int)
    keep_w = np.array(keep_w, dtype=np.float64)
    mass_covered = float(np.sum(keep_w) / (total + 1e-12))
    return keep_idx, keep_w, mass_covered


def compute_q_full(P, p):
    return P.dot(p)


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
        ha=ha, va=va, fontsize=fs, fontweight=weight,
        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=1.0),
        clip_on=True, zorder=zorder
    )


def plot_panel_graphfaithful(
    ax, Z, p_vals, center_pos, neigh_pos, neigh_w,
    title, vmin=None, vmax=None,
    point_size=30, alpha=0.95,
    edge_alpha_base=0.03, edge_alpha_scale=2.5,
    edge_lw_base=0.45, edge_lw_scale=7.0,
    label_topk=3,
):
    """
    neigh_pos: positions in Z/p_vals for the neighbors that belong to this method
    neigh_w:   raw weights corresponding to neigh_pos (NOT renormalized)
    """
    sc = ax.scatter(Z[:, 0], Z[:, 1], c=p_vals, s=point_size, alpha=alpha,
                    linewidths=0, vmin=vmin, vmax=vmax, zorder=3)
    pad_axis(ax, Z, pad=0.18)

    p0 = float(p_vals[center_pos])

    # edges: alpha ~ relative weight among shown neighbors, lw ~ |Δp|
    if len(neigh_pos) > 0:
        w = np.array(neigh_w, dtype=float)
        w_rel = w / (np.max(w) + 1e-12)  # purely visual scaling, not changing q
        for pos, wij_rel, wij in zip(neigh_pos, w_rel, w):
            dij = abs(float(p_vals[pos]) - p0)
            a = float(np.clip(edge_alpha_base + edge_alpha_scale * wij_rel, 0.05, 0.80))
            lw = float(np.clip(edge_lw_base + edge_lw_scale * dij, 0.45, 2.6))
            ax.plot([Z[center_pos, 0], Z[pos, 0]],
                    [Z[center_pos, 1], Z[pos, 1]],
                    linewidth=lw, alpha=a, zorder=1)

        # label top-weight neighbours (true weights)
        if label_topk > 0:
            order = np.argsort(w)[::-1]
            picks = order[:min(label_topk, len(order))]
            for k in picks:
                pos = int(neigh_pos[int(k)])
                annotate_inward(ax, Z[pos, 0], Z[pos, 1],
                                f"w={float(w[int(k)]):.2f}\np={float(p_vals[pos]):.3f}",
                                fs=9)

    # center star
    ax.scatter(Z[center_pos, 0], Z[center_pos, 1],
               s=230, marker="*", edgecolor="black", linewidth=1.2, zorder=8)

    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])
    ax.grid(alpha=0.18)
    return sc


# ----------------------------
# Main runner (NO if __main__)
# ----------------------------
def run_viz_graphfaithful(
    lap_csv,
    urw_csv,
    lap_P_npz,
    lap_orig_npy,
    urw_P_npz,
    urw_orig_npy,

    scaler_pkl,
    proj_compl_npy,
    metric_npy="",

    choose_from_csv="",
    choose_top=2,
    orig_indices=None,   # list[int] overrides choose_from_csv

    pred_col="pred_log",
    feat_prefix="raw_",

    # graph-faithful neighbour display controls
    mass_target=0.90,
    max_neigh_show=60,

    seed=0,
    perplexity=10.0,
    tsne_iter=1500,

    title_prefix="Employee (log-salary)",
    lam_show=30.0,
    out_dir="results/inspect_reg_examples_graphfaithful",
    save_png=True,
):
    set_paper_style()
    os.makedirs(out_dir, exist_ok=True)

    # load per-instance csvs
    lap_df = pd.read_csv(lap_csv)
    urw_df = pd.read_csv(urw_csv)
    if "orig_index" not in lap_df.columns or "orig_index" not in urw_df.columns:
        raise KeyError("Both csv must have orig_index")
    if pred_col not in lap_df.columns or pred_col not in urw_df.columns:
        raise KeyError(f"Both csv must have {pred_col}")

    raw_cols = pick_raw_cols(lap_df, prefix=feat_prefix)

    # merge for embedding + both preds (by orig_index)
    m = lap_df[["orig_index", pred_col] + raw_cols].merge(
        urw_df[["orig_index", pred_col]],
        on="orig_index", how="inner", suffixes=("_lap", "_urw")
    ).rename(columns={f"{pred_col}_lap": "pred_lap", f"{pred_col}_urw": "pred_urw"})
    m = m.sort_values("orig_index").reset_index(drop=True)

    X_space = build_space(m, raw_cols, scaler_pkl, proj_compl_npy, metric_npy)

    # preds aligned to m
    pred_lap_all = m["pred_lap"].to_numpy(dtype=float)
    pred_urw_all = m["pred_urw"].to_numpy(dtype=float)
    orig_all = m["orig_index"].to_numpy(dtype=int)
    mpos = {int(o): i for i, o in enumerate(orig_all)}

    # graphs
    P_lap, orig_lap = load_graph(lap_P_npz, lap_orig_npy)
    P_urw, orig_urw = load_graph(urw_P_npz, urw_orig_npy)
    map_lap = inv_map(orig_lap)
    map_urw = inv_map(orig_urw)

    # q_full on graph order (strict)
    # align p to graph order using orig_index mapping
    def _align_p_to_graph(orig_graph, pred_map):
        # pred_map: orig_index -> pred value (from m)
        p = np.empty(len(orig_graph), dtype=float)
        for i, oid in enumerate(orig_graph):
            if int(oid) not in pred_map:
                p[i] = np.nan
            else:
                p[i] = float(pred_map[int(oid)])
        if np.isnan(p).any():
            raise ValueError("Some graph nodes missing in merged predictions m. Ensure CSVs are TEST-only and matched.")
        return p

    pred_map_lap = {int(o): float(pred_lap_all[mpos[int(o)]]) for o in orig_all}
    pred_map_urw = {int(o): float(pred_urw_all[mpos[int(o)]]) for o in orig_all}
    p_lap_graph = _align_p_to_graph(orig_lap, pred_map_lap)
    p_urw_graph = _align_p_to_graph(orig_urw, pred_map_urw)

    q_full_lap = compute_q_full(P_lap, p_lap_graph)
    q_full_urw = compute_q_full(P_urw, p_urw_graph)

    # choose instances
    chosen = []
    if orig_indices is not None and len(orig_indices) > 0:
        chosen = [int(x) for x in orig_indices]
    elif choose_from_csv:
        cdf = pd.read_csv(choose_from_csv)
        if "orig_index" not in cdf.columns:
            raise KeyError(f"{choose_from_csv} missing orig_index")
        chosen = [int(x) for x in cdf["orig_index"].head(choose_top).tolist()]
    else:
        raise ValueError("Provide orig_indices or choose_from_csv")

    for t, center_orig in enumerate(chosen, start=1):
        if center_orig not in mpos:
            print(f"[skip] orig_index={center_orig} not in merged csv")
            continue
        if center_orig not in map_lap or center_orig not in map_urw:
            print(f"[skip] orig_index={center_orig} not in lap/urw graph mapping")
            continue

        i_lap = map_lap[center_orig]
        i_urw = map_urw[center_orig]

        # neighbors (graph-row indices) chosen per method
        nbr_lap_row, w_lap, mass_lap = row_neighbors_topmass(
            P_lap, i_lap, mass_target=mass_target, max_neigh=max_neigh_show, drop_self=True
        )
        nbr_urw_row, w_urw, mass_urw = row_neighbors_topmass(
            P_urw, i_urw, mass_target=mass_target, max_neigh=max_neigh_show, drop_self=True
        )

        # convert to orig_index space
        nbr_lap_orig = [int(orig_lap[j]) for j in nbr_lap_row]
        nbr_urw_orig = [int(orig_urw[j]) for j in nbr_urw_row]

        # union for embedding only (layout)
        S_orig = [center_orig] + sorted(list((set(nbr_lap_orig) | set(nbr_urw_orig)) - {center_orig}))
        S_orig = [o for o in S_orig if o in mpos]

        if len(S_orig) <= 2:
            print(f"[skip] too few nodes to embed for orig_index={center_orig}")
            continue

        sub_rows = np.array([mpos[o] for o in S_orig], dtype=int)
        X_sub = X_space[sub_rows]
        Z = local_tsne(X_sub, seed=seed, perplexity=perplexity, n_iter=tsne_iter)

        # values aligned to S_orig
        p_lap = pred_lap_all[sub_rows]
        p_urw = pred_urw_all[sub_rows]

        # positions of each method's neighbours inside S_orig
        pos_map = {int(o): k for k, o in enumerate(S_orig)}
        lap_neigh_pos = [pos_map[o] for o in nbr_lap_orig if o in pos_map]
        urw_neigh_pos = [pos_map[o] for o in nbr_urw_orig if o in pos_map]

        # compute VIS q (not renorm) using shown neighbors only, in each method
        # q_vis = sum_{kept} w p_j
        p0L = float(pred_map_lap[center_orig])
        p0U = float(pred_map_urw[center_orig])

        qfullL = float(q_full_lap[i_lap])
        qfullU = float(q_full_urw[i_urw])
        gapfullL = abs(p0L - qfullL)
        gapfullU = abs(p0U - qfullU)

        qvisL = float(np.sum(w_lap * np.array([pred_map_lap[o] for o in nbr_lap_orig], dtype=float))) if len(w_lap) else np.nan
        qvisU = float(np.sum(w_urw * np.array([pred_map_urw[o] for o in nbr_urw_orig], dtype=float))) if len(w_urw) else np.nan
        gapvisL = abs(p0L - qvisL) if np.isfinite(qvisL) else np.nan
        gapvisU = abs(p0U - qvisU) if np.isfinite(qvisU) else np.nan

        # pairwise stats over shown neighbors (unweighted mean/max)
        if len(lap_neigh_pos) > 0:
            dpL = np.abs(np.array([pred_map_lap[o] for o in nbr_lap_orig], dtype=float) - p0L)
            meanDpL = float(dpL.mean()); maxDpL = float(dpL.max())
        else:
            meanDpL = np.nan; maxDpL = np.nan

        if len(urw_neigh_pos) > 0:
            dpU = np.abs(np.array([pred_map_urw[o] for o in nbr_urw_orig], dtype=float) - p0U)
            meanDpU = float(dpU.mean()); maxDpU = float(dpU.max())
        else:
            meanDpU = np.nan; maxDpU = np.nan

        # shared color scale for visual comparability
        vmin = float(min(np.min(p_lap), np.min(p_urw)))
        vmax = float(max(np.max(p_lap), np.max(p_urw)))

        # Titles
        supt1 = rf"{title_prefix} (example {t}) — orig_index={center_orig}"
        supt2 = rf"$\lambda={lam_show:g}$, mass_target={mass_target:.2f}, max_show={max_neigh_show}"

        title_lap = (
            rf"Lap: $|N(i)|={len(nbr_lap_orig)}$, mass={mass_lap:.2f}, "
            rf"$p_i={p0L:.3f}$, $q_i(full)={qfullL:.3f}$, $|p-q|_{{full}}={gapfullL:.4f}$, "
            rf"$|p-q|_{{vis}}={gapvisL:.4f}$, mean$|\Delta p|={meanDpL:.4f}$, max$|\Delta p|={maxDpL:.4f}$"
        )
        title_urw = (
            rf"URW: $|N(i)|={len(nbr_urw_orig)}$, mass={mass_urw:.2f}, "
            rf"$p_i={p0U:.3f}$, $q_i(full)={qfullU:.3f}$, $|p-q|_{{full}}={gapfullU:.4f}$, "
            rf"$|p-q|_{{vis}}={gapvisU:.4f}$, mean$|\Delta p|={meanDpU:.4f}$, max$|\Delta p|={maxDpU:.4f}$"
        )

        fig, axes = plt.subplots(1, 2, figsize=(14, 8))
        axL, axR = axes

        fig.suptitle(supt1, fontsize=12, y=0.985)
        fig.text(0.5, 0.945, supt2, ha="center", va="center", fontsize=10.0)
        fig.text(0.5, 0.913, title_lap, ha="center", va="center", fontsize=9.7)
        fig.text(0.5, 0.883, title_urw, ha="center", va="center", fontsize=9.7)

        sc1 = plot_panel_graphfaithful(
            axL, Z, p_lap, center_pos=0,
            neigh_pos=lap_neigh_pos, neigh_w=w_lap[:len(lap_neigh_pos)],
            title=r"Laplacian ($p_{\log}$)",
            vmin=vmin, vmax=vmax, label_topk=3
        )

        sc2 = plot_panel_graphfaithful(
            axR, Z, p_urw, center_pos=0,
            neigh_pos=urw_neigh_pos, neigh_w=w_urw[:len(urw_neigh_pos)],
            title=r"URW ($p_{\log}$)",
            vmin=vmin, vmax=vmax, label_topk=3
        )

        cb1 = fig.colorbar(sc1, ax=axL, fraction=0.046, pad=0.04); cb1.set_label(r"$p_{\log}$")
        cb2 = fig.colorbar(sc2, ax=axR, fraction=0.046, pad=0.04); cb2.set_label(r"$p_{\log}$")

        fig.subplots_adjust(left=0.055, right=0.975, bottom=0.10, top=0.82, wspace=0.18)

        out_pdf = os.path.join(out_dir, f"example_idx{center_orig}.pdf")
        fig.savefig(out_pdf)
        print(f"[OK] saved -> {out_pdf}")

        if save_png:
            out_png = os.path.join(out_dir, f"example_idx{center_orig}.png")
            fig.savefig(out_png, dpi=300)
            print(f"[OK] saved -> {out_png}")

        plt.close(fig)


# ----------------------------
# EDIT HERE (NO if __main__)
# ----------------------------
run_viz_graphfaithful(
    lap_csv="lap_depth1_lambda30.csv",
    urw_csv="urw_depth1_lambda30.csv",

    lap_P_npz="reg_lap_depth1_lambda30.0_tau5.0_theta0.001_P_te.npz",
    lap_orig_npy="reg_lap_depth1_lambda30.0_tau5.0_theta0.001_orig_index_te.npy",

    urw_P_npz="reg_urw_depth1_lambda30.0_tau5.0_theta0.001_P_te.npz",
    urw_orig_npy="reg_urw_depth1_lambda30.0_tau5.0_theta0.001_orig_index_te.npy",

    scaler_pkl="reg_lap_depth1_lambda30.0_scaler.pkl",
    proj_compl_npy="reg_lap_depth1_lambda30.0_proj_compl.npy",
    metric_npy="reg_lap_depth1_lambda30.0_L_metric.npy",

    # pick from your new full-join csv
    choose_from_csv="inspect_reg_instances_lambda30/best_gap_improve_full.csv",
    choose_top=2,

    mass_target=0.90,
    max_neigh_show=60,

    seed=0,
    perplexity=10.0,
    tsne_iter=1500,

    lam_show=30.0,
    out_dir="inspect_reg_examples_lambda30_graphfaithful",
    save_png=True,
)
