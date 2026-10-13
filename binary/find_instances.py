#!/usr/bin/env python3
# viz_graph_neigh_two_lap_only_paperpdf.py
#
# Paper-ready visualization (PDF export):
# 1) Pick TWO "lap_only" instances using VR definition:
#       IF_lap > eps AND IF_ours <= eps
# 2) Define neighbourhood by tau-ball in a chosen space:
#       N(i) = { j : ||x_i - x_j||^2 <= tau, j != i }
#    - neigh_space=common (recommended) fixes a common neighbourhood for fair comparison.
# 3) Local t-SNE on {center + neighbours} to avoid overlap.
# 4) Paper-ready formatting:
#    - padding so points/labels don't go out of frame
#    - smart label offsets (push labels inward)
#    - white bbox for center label
# 5) Save both PDF and PNG (you can disable png via --no_png).
#
# Example:
# python viz_graph_neigh_two_lap_only_paperpdf.py \
#   --lap_csv results/lap_depth1_lambda10.csv \
#   --ours_csv results/urw_depth1_lambda10.csv \
#   --lap_if_col if_score_lapl_monitor_forwardKL_prob \
#   --ours_if_col if_score_forwardKL_prob \
#   --eps 0.01 \
#   --tau 5.0 \
#   --lap_scaler_pkl results/lap_scaler_depth1_lambda10.pkl \
#   --ours_scaler_pkl results/urw_scaler_depth1_lambda10.pkl \
#   --neigh_space common --common_scaler ours \
#   --max_neigh 40 --ann_k_close 2 --ann_k_diff 2 \
#   --out_dir results/inspect_two_graphneigh_paper

import os
import argparse
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

        # editable text in pdf
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


# ----------------------------
# IO / checks
# ----------------------------
def require_cols(df, cols, name):
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise KeyError(f"[{name}] missing columns: {miss}")

def load_df(path):
    df = pd.read_csv(path)
    if "orig_index" not in df.columns:
        raise KeyError(f"[csv] missing 'orig_index' in {path}")
    return df

def pick_raw_feature_cols(df, prefix="raw_"):
    cols = [c for c in df.columns if c.startswith(prefix)]
    if not cols:
        raise ValueError(f"No feature columns found with prefix '{prefix}'.")
    return cols

def drop_unwanted_raw(feat_cols):
    # match training: exclude y + orig_index, and drop sex_* dummies from model input
    out = []
    for c in feat_cols:
        if c in ("raw_y", "raw_orig_index"):
            continue
        if c.startswith("raw_sex_"):
            continue
        out.append(c)
    return out

def load_scaler(pkl_path):
    if pkl_path is None:
        return None
    import joblib
    sc = joblib.load(pkl_path)
    if not hasattr(sc, "transform"):
        raise TypeError(f"Scaler at {pkl_path} has no .transform()")
    return sc

def apply_scaler(X, scaler, name="scaler"):
    if scaler is None:
        return X
    if hasattr(scaler, "n_features_in_") and scaler.n_features_in_ != X.shape[1]:
        raise ValueError(
            f"{name} expects {scaler.n_features_in_} features but X has {X.shape[1]}. "
            f"Likely column mismatch (did you drop raw_sex_*?) or wrong scaler."
        )
    return scaler.transform(X)


# ----------------------------
# Selection (VR)
# ----------------------------
def select_two_lap_only(m, eps, pick="topgap", seed=0):
    lap_only = m[(m["if_lap"] > eps) & (m["if_ours"] <= eps)].copy()
    if lap_only.empty:
        raise RuntimeError(f"No lap_only instances found at eps={eps}.")
    lap_only["gap"] = lap_only["if_lap"] - lap_only["if_ours"]
    if pick == "topgap":
        chosen = lap_only.sort_values("gap", ascending=False).head(2)
    else:
        chosen = lap_only.sample(n=min(2, len(lap_only)), random_state=seed)
    if len(chosen) < 2:
        print(f"[WARN] only {len(chosen)} lap_only instance(s) exist at eps={eps}.")
    return chosen, lap_only


# ----------------------------
# Neighbours by tau-ball
# ----------------------------
def e_dist_sq_one_to_all(X, i):
    v = X[i:i+1]
    return np.maximum(((X - v) ** 2).sum(axis=1), 0.0)

def tau_neighbours(X, i, tau):
    d2 = e_dist_sq_one_to_all(X, i)
    nbr = np.where((d2 <= tau) & (np.arange(len(d2)) != i))[0]
    return nbr, d2[nbr]


# ----------------------------
# Local embedding (avoid overlap)
# ----------------------------
def local_tsne(X_sub, seed=0, perplexity=10.0, n_iter=1500):
    from sklearn.manifold import TSNE
    n = X_sub.shape[0]
    perp = float(perplexity)
    perp = min(perp, max(5.0, (n - 1) / 3.0))
    perp = min(perp, n - 1)
    tsne = TSNE(
        n_components=2,
        random_state=seed,
        init="pca",
        learning_rate="auto",
        perplexity=perp,
        n_iter=n_iter
    )
    return tsne.fit_transform(X_sub)

def uniq_keep(seq):
    out, seen = [], set()
    for x in seq:
        if x not in seen:
            out.append(x); seen.add(x)
    return out


# ----------------------------
# Paper-ready plotting helpers
# ----------------------------
def pad_axis(ax, Z, pad=0.12):
    x_min, x_max = float(Z[:, 0].min()), float(Z[:, 0].max())
    y_min, y_max = float(Z[:, 1].min()), float(Z[:, 1].max())
    dx = max(x_max - x_min, 1e-9)
    dy = max(y_max - y_min, 1e-9)
    ax.set_xlim(x_min - pad * dx, x_max + pad * dx)
    ax.set_ylim(y_min - pad * dy, y_max + pad * dy)

def smart_annotate(ax, x, y, text, fs=8):
    # push label toward the interior depending on point location
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    xc = 0.5 * (x0 + x1)
    yc = 0.5 * (y0 + y1)

    dx = 8 if x < xc else -8
    dy = 8 if y < yc else -8
    ha = "left" if dx > 0 else "right"
    va = "bottom" if dy > 0 else "top"

    ax.annotate(
        text,
        (x, y),
        textcoords="offset points",
        xytext=(dx, dy),
        ha=ha,
        va=va,
        fontsize=fs,
        clip_on=True,
    )

def plot_panel(ax, Z, vals, center_pos, title, orig_ids, ann_pos, center_text,
               point_size=28, alpha=0.9):
    sc = ax.scatter(Z[:, 0], Z[:, 1], c=vals, s=point_size, alpha=alpha, linewidths=0)
    pad_axis(ax, Z, pad=0.14)

    # center star
    ax.scatter(
        Z[center_pos, 0], Z[center_pos, 1],
        s=200, marker="*", edgecolor="black", linewidth=1.0, zorder=5
    )

    # center label with white background
    ax.annotate(
        center_text,
        (Z[center_pos, 0], Z[center_pos, 1]),
        textcoords="offset points",
        xytext=(10, 10),
        fontsize=9,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.75),
        clip_on=True,
        zorder=6
    )

    for p in ann_pos:
        if p == center_pos:
            continue
        smart_annotate(ax, Z[p, 0], Z[p, 1], f"{int(orig_ids[p])}\n{float(vals[p]):.3f}", fs=8)

    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(alpha=0.18)
    return sc


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--lap_csv", required=True)
    ap.add_argument("--ours_csv", required=True)

    ap.add_argument("--lap_if_col", default="if_score_lapl_monitor_forwardKL_prob")
    ap.add_argument("--ours_if_col", default="if_score_forwardKL_prob")

    ap.add_argument("--lap_pred_col", default="pred_prob")
    ap.add_argument("--ours_pred_col", default="pred_prob")

    ap.add_argument("--feat_prefix", default="raw_")
    ap.add_argument("--eps", type=float, default=0.01)
    ap.add_argument("--tau", type=float, default=5.0)

    # scalers
    ap.add_argument("--lap_scaler_pkl", type=str, default=None)
    ap.add_argument("--ours_scaler_pkl", type=str, default=None)

    # neighbour-space mode
    ap.add_argument("--neigh_space", choices=["common", "lap", "ours"], default="common")
    ap.add_argument("--common_scaler", choices=["lap", "ours", "none"], default="ours")

    ap.add_argument("--max_neigh", type=int, default=40)

    ap.add_argument("--ann_k_close", type=int, default=2)
    ap.add_argument("--ann_k_diff", type=int, default=2)

    ap.add_argument("--pick", choices=["topgap", "random"], default="topgap")
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--perplexity", type=float, default=10.0)
    ap.add_argument("--tsne_iter", type=int, default=1500)

    ap.add_argument("--out_dir", default="results/inspect_two_graphneigh_paper")
    ap.add_argument("--title_prefix", default="Adult")

    ap.add_argument("--save_png", action="store_true", help="also save PNG (default: PDF only)")
    args = ap.parse_args()

    set_paper_style()
    os.makedirs(args.out_dir, exist_ok=True)

    lap_df = load_df(args.lap_csv)
    ours_df = load_df(args.ours_csv)

    require_cols(lap_df, ["orig_index", args.lap_if_col, args.lap_pred_col], "lap_csv")
    require_cols(ours_df, ["orig_index", args.ours_if_col, args.ours_pred_col], "ours_csv")

    # Merge IF+preds with suffixes
    m = lap_df[["orig_index", args.lap_if_col, args.lap_pred_col]].merge(
        ours_df[["orig_index", args.ours_if_col, args.ours_pred_col]],
        on="orig_index", how="inner", suffixes=("_lap", "_ours")
    )
    if m.empty:
        raise RuntimeError("Merged dataframe is empty. Check orig_index overlap.")

    # resolve IF columns after merge
    lap_if_merged = f"{args.lap_if_col}_lap" if f"{args.lap_if_col}_lap" in m.columns else args.lap_if_col
    ours_if_merged = f"{args.ours_if_col}_ours" if f"{args.ours_if_col}_ours" in m.columns else args.ours_if_col
    m = m.rename(columns={lap_if_merged: "if_lap", ours_if_merged: "if_ours"})

    # resolve pred columns after merge
    pred_lap_col = f"{args.lap_pred_col}_lap" if f"{args.lap_pred_col}_lap" in m.columns else args.lap_pred_col
    pred_ours_col = f"{args.ours_pred_col}_ours" if f"{args.ours_pred_col}_ours" in m.columns else args.ours_pred_col
    require_cols(m, ["if_lap", "if_ours", pred_lap_col, pred_ours_col], "merged")

    # attach raw features from lap_df
    feat_cols_all = pick_raw_feature_cols(lap_df, prefix=args.feat_prefix)
    feat_cols = drop_unwanted_raw(feat_cols_all)
    feat_block = lap_df[["orig_index"] + feat_cols].copy()
    m = m.merge(feat_block, on="orig_index", how="inner")
    if m.empty:
        raise RuntimeError("Empty after joining raw features.")

    # stable ordering
    m = m.sort_values("orig_index").reset_index(drop=True)

    # load scalers
    lap_scaler = load_scaler(args.lap_scaler_pkl) if args.lap_scaler_pkl else None
    ours_scaler = load_scaler(args.ours_scaler_pkl) if args.ours_scaler_pkl else None

    # build raw X
    X_raw = m[feat_cols].to_numpy(dtype=float)

    # choose neighbourhood space
    if args.neigh_space == "lap":
        if lap_scaler is None:
            raise ValueError("--neigh_space lap requires --lap_scaler_pkl")
        X_neigh = apply_scaler(X_raw, lap_scaler, name="lap_scaler")
        neigh_note = "lap-scaler space"
    elif args.neigh_space == "ours":
        if ours_scaler is None:
            raise ValueError("--neigh_space ours requires --ours_scaler_pkl")
        X_neigh = apply_scaler(X_raw, ours_scaler, name="ours_scaler")
        neigh_note = "ours-scaler space"
    else:
        # common
        if args.common_scaler == "lap":
            if lap_scaler is None:
                raise ValueError("--common_scaler lap requires --lap_scaler_pkl")
            X_neigh = apply_scaler(X_raw, lap_scaler, name="lap_scaler")
            neigh_note = "common (lap-scaler) space"
        elif args.common_scaler == "ours":
            if ours_scaler is None:
                raise ValueError("--common_scaler ours requires --ours_scaler_pkl")
            X_neigh = apply_scaler(X_raw, ours_scaler, name="ours_scaler")
            neigh_note = "common (ours-scaler) space"
        else:
            X_neigh = X_raw
            neigh_note = "common (raw) space"

    # choose two lap_only points
    chosen2, lap_only_all = select_two_lap_only(m, eps=args.eps, pick=args.pick, seed=args.seed)
    chosen_path = os.path.join(args.out_dir, f"chosen_two_lap_only_eps{args.eps}.csv")
    chosen2.to_csv(chosen_path, index=False)
    print(f"[OK] saved chosen instances -> {chosen_path}")

    # index map
    orig2row = {int(o): i for i, o in enumerate(m["orig_index"].values)}

    for t, row in enumerate(chosen2.itertuples(index=False), start=1):
        center_orig = int(row.orig_index)
        i = orig2row[center_orig]

        nbr_rows, nbr_d2 = tau_neighbours(X_neigh, i, tau=args.tau)

        # cap neighbours
        if len(nbr_rows) > args.max_neigh:
            order = np.argsort(nbr_d2)[:args.max_neigh]
            nbr_rows = nbr_rows[order]
            nbr_d2 = nbr_d2[order]

        # subset
        sub_rows = np.concatenate([[i], nbr_rows])
        X_sub = X_neigh[sub_rows]

        # local t-SNE (subset only)
        Z = local_tsne(X_sub, seed=args.seed, perplexity=args.perplexity, n_iter=args.tsne_iter)

        orig_sub = m.loc[sub_rows, "orig_index"].astype(int).to_numpy()
        pred_lap_sub = m.loc[sub_rows, pred_lap_col].astype(float).to_numpy()
        pred_ours_sub = m.loc[sub_rows, pred_ours_col].astype(float).to_numpy()

        # annotations: closest + most different (per panel)
        k_close = min(args.ann_k_close, len(nbr_d2))
        ann_close = list(1 + np.argsort(nbr_d2)[:k_close])  # positions in sub-array

        diff_lap = np.abs(pred_lap_sub - pred_lap_sub[0])
        diff_ours = np.abs(pred_ours_sub - pred_ours_sub[0])
        ann_diff_lap = [p for p in np.argsort(diff_lap)[::-1] if p != 0][:args.ann_k_diff]
        ann_diff_ours = [p for p in np.argsort(diff_ours)[::-1] if p != 0][:args.ann_k_diff]

        ann_pos_lap = uniq_keep(ann_close + ann_diff_lap)
        ann_pos_ours = uniq_keep(ann_close + ann_diff_ours)

        # center label
        center_text_lap = f"idx={center_orig}\np={pred_lap_sub[0]:.3f}"
        center_text_ours = f"idx={center_orig}\np={pred_ours_sub[0]:.3f}"

        example_id = t
        if_lap = float(row.if_lap)  # from chosen2 (already merged)
        if_ours = float(row.if_ours)  # from chosen2 (already merged)
        eps = float(args.eps)
        tau = float(args.tau)
        n_neigh = int(len(sub_rows) - 1)

        # compact suptitle (shorter)
        supt = (
            f"Adult example {example_id}  "
            f"(IF_lap={if_lap:.4f}, IF_URW={if_ours:.6f}, eps={eps})\n"
            f"Neighbourhood: tau={tau} in common space; |N(i)|={n_neigh}"
        )
        fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.4))  # 更宽一点，避免瘦高
        axL, axR = axes

        # --- title (2 lines), control spacing manually ---
        supt1 = (f"{args.title_prefix} example {t}  "
                 f"(IF_lap={if_lap:.4f}, IF_URW={if_ours:.6f}, eps={eps})")
        supt2 = (f"Neighbourhood: tau={tau}; |N(i)|={n_neigh}")

        fig.suptitle(supt1, fontsize=12, y=0.985)
        fig.text(0.5, 0.93, supt2, ha="center", va="center", fontsize=11)

        # --- plot panels ---
        sc1 = plot_panel(
            axL, Z, pred_lap_sub, 0,
            title="Laplacian (predicted probability)",
            orig_ids=orig_sub, ann_pos=ann_pos_lap, center_text=center_text_lap,
            point_size=30, alpha=0.95
        )
        sc2 = plot_panel(
            axR, Z, pred_ours_sub, 0,
            title="URW (predicted probability)",
            orig_ids=orig_sub, ann_pos=ann_pos_ours, center_text=center_text_ours,
            point_size=30, alpha=0.95
        )

        # --- colorbars attached to each subplot (DO NOT allocate new columns) ---
        cb1 = fig.colorbar(sc1, ax=axL, fraction=0.046, pad=0.04)
        cb1.set_label("pred_prob")
        cb2 = fig.colorbar(sc2, ax=axR, fraction=0.046, pad=0.04)
        cb2.set_label("pred_prob")

        # --- manual canvas control: fixes your "top too much / bottom too tight" ---
        # left, bottom, right, top
        fig.subplots_adjust(left=0.055, right=0.975, bottom=0.10, top=0.86, wspace=0.18)

        out_pdf = os.path.join(args.out_dir, f"lap_only_{t}_idx{center_orig}_eps{args.eps}.pdf")
        fig.savefig(out_pdf)
        print(f"[OK] saved -> {out_pdf}")

        if args.save_png:
            out_png = os.path.join(args.out_dir, f"lap_only_{t}_idx{center_orig}_eps{args.eps}.png")
            fig.savefig(out_png, dpi=300)
            print(f"[OK] saved -> {out_png}")

        plt.close(fig)

    print("[DONE]")


if __name__ == "__main__":
    main()
