#!/usr/bin/env python3
# run_ifair_preproc_adult_fixed_cached_dual.py
#
# Runs TWO baselines in ONE run:
#   (1) Identity / No-iFair: Xhat = X_ifair (no transform)
#   (2) iFair: Xhat = ifair.transform(X_ifair)
#
# Both share:
#   - same split
#   - same MLP training (mini-batch + num_iterations)
#   - same evaluation (DP/Gap/IF/LIF/VR/GR-Con/S-Con)
# iFair is cached (so you don't refit repeatedly).

import sys, os, time
sys.path.append(os.path.join(os.path.dirname(__file__), "iFair"))
os.environ["NUMBA_DISABLE_JIT"] = "1"

import argparse, random
import numpy as np
import joblib
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange
from pprint import pprint

from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix

# -------------------------
# Import iFair implementation
# -------------------------
try:
    from iFair import iFair as IFairModel
    _HAS_IFAIR = True
except Exception as e:
    _HAS_IFAIR = False
    _IFAIR_IMPORT_ERR = e

# -------------------------
# Args
# -------------------------
parser = argparse.ArgumentParser(description="iFair pre-processing baseline (Adult binary) - dual run")

parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--depth", type=int, default=1)
parser.add_argument("--hidden", type=int, default=256)

# classifier training (URW-style)
parser.add_argument("--num_iterations", type=int, default=4000)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--nloglr", type=float, default=3.0)  # lr = 10^{-nloglr}
parser.add_argument("--select_best", type=int, default=1, choices=[0, 1])

# Graph locality for IF/LIF evaluation
parser.add_argument("--theta", type=float, default=1e-3)
parser.add_argument("--tau", type=float, default=5.0)
parser.add_argument("--svd_rank", type=int, default=1)

# iFair params
parser.add_argument("--ifair_k", type=int, default=8)
parser.add_argument("--ifair_Ax", type=float, default=1e-2)
parser.add_argument("--ifair_Az", type=float, default=1.0)
parser.add_argument("--ifair_max_iter", type=int, default=300)
parser.add_argument("--ifair_restarts", type=int, default=1)
parser.add_argument("--ifair_fit_n", type=int, default=2000)
parser.add_argument("--ifair_seed", type=int, default=0)

# caching + outputs
parser.add_argument("--cache_dir", type=str, default="results/ifair_cache")
parser.add_argument("--results_csv", type=str, default="results/metrics_ifair_dual.csv")
parser.add_argument("--save_if_csv", type=str, default="", help="optional: saves FULL per-instance table for iFair run only")

# data
parser.add_argument("--data_dir", type=str, default="../data/adult")

args = parser.parse_args()
print(vars(args))

if not _HAS_IFAIR:
    raise ImportError(
        f"Failed to import iFair baseline implementation. Error: {_IFAIR_IMPORT_ERR}\n"
        f"Fix: ensure iFair/ is importable and iFair class is available."
    )

os.makedirs("results", exist_ok=True)
os.makedirs(args.cache_dir, exist_ok=True)

torch.set_num_threads(min(8, torch.get_num_threads()))
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
device = torch.device("cpu")


# -------------------------
# Utils
# -------------------------
def manual_logloss(y_true, y_prob, eps=1e-7):
    y_true = np.asarray(y_true).astype(float).ravel()
    y_prob = np.asarray(y_prob).astype(float).ravel()
    p = np.clip(y_prob, eps, 1.0 - eps)
    return float(-np.mean(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p)))

def e_dist_sq(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A2 = (A**2).sum(axis=1, keepdims=True)
    B2 = (B**2).sum(axis=1, keepdims=True).T
    return np.maximum(A2 - 2.0 * (A @ B.T) + B2, 0.0)

def build_W_tau_only(X_for_metric: np.ndarray, theta: float, tau: float) -> np.ndarray:
    D2 = e_dist_sq(X_for_metric, X_for_metric)
    W = np.exp(-theta * D2, dtype=np.float64) * (D2 <= tau).astype(np.float64)
    np.fill_diagonal(W, 0.0)
    W = 0.5 * (W + W.T)
    n = W.shape[0]
    W[np.arange(n), np.arange(n)] = 1.0
    return W

def make_row_stochastic_P(W: np.ndarray) -> csr_matrix:
    rowsum = W.sum(axis=1, keepdims=True) + 1e-12
    return csr_matrix(W / rowsum)

def IF_cons_vector_forward_kl(P: csr_matrix, yhat: np.ndarray) -> np.ndarray:
    nbr = P.dot(yhat)
    eps = 1e-7
    p = np.clip(yhat, eps, 1.0 - eps)
    q = np.clip(nbr,  eps, 1.0 - eps)
    return p * np.log(p / q) + (1.0 - p) * np.log((1.0 - p) / (1.0 - q))

def lif_energy(W: np.ndarray, f: np.ndarray, normalize: str = "per_edge") -> float:
    d = W.sum(axis=1)
    Lf = d * f - (W @ f)
    E = float(f @ Lf)
    if normalize == "per_node":
        E /= len(f)
    elif normalize == "per_edge":
        m = int((W > 0).sum())
        E /= max(m, 1)
    return E

def demographic_parity_prob(y_prob: np.ndarray, sens_bin: np.ndarray) -> float:
    y_prob = np.asarray(y_prob).ravel()
    sens_bin = np.asarray(sens_bin).ravel()
    m1 = y_prob[sens_bin == 1].mean()
    m0 = y_prob[sens_bin == 0].mean()
    return float(abs(m1 - m0))

def group_gap_stats_prob(y_prob: np.ndarray, group: np.ndarray):
    y_prob = np.asarray(y_prob).ravel()
    group  = np.asarray(group).ravel()
    mu = float(y_prob.mean())
    uniq = pd.unique(group)
    mus = []
    for g in uniq:
        v = y_prob[group == g]
        if len(v) > 0:
            mus.append(float(v.mean()))
    mus = np.array(mus, dtype=float)
    gap_rms = float(np.sqrt(np.mean((mus - mu) ** 2)))
    gap_max = float(np.max(np.abs(mus - mu)))
    return gap_rms, gap_max


# -------------------------
# Adult loader (URW-aligned)
# -------------------------
def load_adult_clean(data_dir):
    headers = [
        "age", "workclass", "fnlwgt", "education", "education-num",
        "marital-status", "occupation", "relationship", "race", "sex",
        "capital-gain", "capital-loss", "hours-per-week",
        "native-country", "y"
    ]
    train_path = os.path.join(data_dir, "adult.data")
    test_path  = os.path.join(data_dir, "adult.test")
    train = pd.read_csv(train_path, header=None, sep=",", skipinitialspace=True, na_values="?")
    test  = pd.read_csv(test_path,  header=None, sep=",", skipinitialspace=True, na_values="?", comment="|")

    df = pd.concat([train, test], axis=0, ignore_index=True)
    df.columns = headers

    df["y"] = df["y"].astype(str).str.replace(".", "", regex=False).str.strip()
    df["y"] = df["y"].map({"<=50K": 0, ">50K": 1}).astype(int)

    df = df.dropna()
    df = pd.get_dummies(df, drop_first=False)

    native_cols = [c for c in df.columns if c.startswith("native-country_")]
    df = df.drop(columns=native_cols, errors="ignore")

    drop_cols = [c for c in df.columns if c.startswith("fnlwgt") or c.startswith("education_")]
    df = df.drop(columns=drop_cols, errors="ignore")

    df["orig_index"] = np.arange(len(df))
    return df


# -------------------------
# Consistency helper (generic transform)
# -------------------------
def get_consistency_generic(df_out, transform_fn, scaler_z, model, thr, device="cpu", raw_prefix="raw_"):
    """
    df_out contains raw_* one-hot features.
    transform_fn(Xi)->Xhat (can be identity or ifair.transform)
    scaler_z: fitted on TRAIN in that run
    model: classifier trained on Z
    thr: classification threshold
    """
    feat_cols = [
        c for c in df_out.columns
        if c.startswith(raw_prefix) and c not in (f"{raw_prefix}y", f"{raw_prefix}orig_index")
    ]
    base_X = df_out[feat_cols].to_numpy().astype(np.float32)
    names = [c[len(raw_prefix):] for c in feat_cols]

    sex_idxs  = [i for i,n in enumerate(names) if n.startswith("sex_")]
    race_idxs = [i for i,n in enumerate(names) if n.startswith("race_")]
    rel_idxs  = [i for i,n in enumerate(names) if n.startswith("relationship_")]

    female_idx = names.index("sex_Female") if "sex_Female" in names else None
    male_idx = names.index("sex_Male") if "sex_Male" in names else None
    if female_idx is None and male_idx is None:
        raise ValueError("Cannot find sex_Female/sex_Male in raw features.")

    white_idx = names.index("race_White") if "race_White" in names else race_idxs[0]
    nonwhite_idx = [i for i in race_idxs if i != white_idx][0]

    if "relationship_Husband" not in names or "relationship_Wife" not in names:
        raise ValueError("Need relationship_Husband and relationship_Wife for S-Con.")
    husband_idx = names.index("relationship_Husband")
    wife_idx    = names.index("relationship_Wife")

    # Build X_ifair input from raw: [nonsens (drop sex_*), sens(last)]
    nonsens_keep = [i for i,n in enumerate(names) if not n.startswith("sex_")]

    def to_ifair_input(X_full):
        X_nons = X_full[:, nonsens_keep]
        if female_idx is not None:
            sens = X_full[:, female_idx].reshape(-1,1)
        else:
            sens = (1.0 - X_full[:, male_idx].reshape(-1,1))
        return np.hstack([X_nons, sens]).astype(np.float32)

    def predict_labels(X_full):
        Xi = to_ifair_input(X_full)
        Xhat = transform_fn(Xi).astype(np.float32)
        Zs = scaler_z.transform(Xhat).astype(np.float32)
        z_t = torch.tensor(Zs, dtype=torch.float32, device=device)
        with torch.no_grad():
            logits = model(z_t).detach().cpu().numpy().ravel()
        probs = 1.0 / (1.0 + np.exp(-logits))
        return (probs >= thr).astype(int)

    def set_onehot(X, group_idxs, on_idx):
        X[:, group_idxs] = 0.0
        X[:, on_idx] = 1.0
        return X

    # GR-Con: 2x2 flips on (gender, race_bin)
    X00 = base_X.copy(); X01 = base_X.copy()
    X10 = base_X.copy(); X11 = base_X.copy()

    if female_idx is not None and male_idx is not None:
        set_onehot(X00, sex_idxs, male_idx)
        set_onehot(X10, sex_idxs, male_idx)
        set_onehot(X01, sex_idxs, female_idx)
        set_onehot(X11, sex_idxs, female_idx)

    set_onehot(X00, race_idxs, nonwhite_idx)
    set_onehot(X01, race_idxs, nonwhite_idx)
    set_onehot(X10, race_idxs, white_idx)
    set_onehot(X11, race_idxs, white_idx)

    y00 = predict_labels(X00); y01 = predict_labels(X01)
    y10 = predict_labels(X10); y11 = predict_labels(X11)
    gr_con = float(np.mean((y00 == y01) & (y00 == y10) & (y00 == y11)))

    # S-Con: Husband ↔ Wife
    X_h = base_X.copy(); X_w = base_X.copy()
    set_onehot(X_h, rel_idxs, husband_idx)
    set_onehot(X_w, rel_idxs, wife_idx)
    y_h = predict_labels(X_h); y_w = predict_labels(X_w)
    s_con = float(np.mean(y_h == y_w))
    return gr_con, s_con


# ============================================================
# Load data
# ============================================================
df = load_adult_clean(args.data_dir)

# protected attribute: sex_Female (female=1)
if "sex_Female" in df.columns:
    sens_full = df["sex_Female"].astype(int).values
elif "sex_Male" in df.columns:
    sens_full = 1 - df["sex_Male"].astype(int).values
else:
    raise ValueError("Cannot find sex dummy columns.")

y_full = df["y"].astype(int).values
rid_full = df["orig_index"].values

# Build X_ifair input: [nonsensitive features, protected as last col]
feature_cols_all = [c for c in df.columns if c not in ("y", "orig_index")]
nonsens_cols = [c for c in feature_cols_all if not c.startswith("sex_")]
X_nonsens = df[nonsens_cols].astype(np.float32).values
sens_col  = sens_full.reshape(-1, 1).astype(np.float32)
X_ifair_full = np.hstack([X_nonsens, sens_col])  # protected last

# Split (same for both runs)
X_tr, X_tmp, y_tr, y_tmp, sens_tr, sens_tmp, rid_tr, rid_tmp = train_test_split(
    X_ifair_full, y_full, sens_full, rid_full,
    test_size=0.4, random_state=args.seed, stratify=y_full
)
X_va, X_te, y_va, y_te, sens_va, sens_te, rid_va, rid_te = train_test_split(
    X_tmp, y_tmp, sens_tmp, rid_tmp,
    test_size=0.5, random_state=args.seed, stratify=y_tmp
)

print(f"train/val/test: {len(X_tr)}/{len(X_va)}/{len(X_te)}; X_ifair_dim={X_tr.shape[1]}")

# Build URW-aligned evaluation graph on ORIGINAL eval-space (sex_* dropped)
drop_for_Xorig = [c for c in df.columns if c.startswith("sex_")] + ["y", "orig_index"]
X_orig_full = df.drop(columns=drop_for_Xorig).astype(np.float32).values
pos = pd.Series(np.arange(len(df)), index=df["orig_index"].values)

idx_tr = pos.loc[rid_tr].to_numpy()
idx_te = pos.loc[rid_te].to_numpy()
Xo_tr = X_orig_full[idx_tr]
Xo_te = X_orig_full[idx_te]
so_tr = sens_tr
so_te = sens_te

scaler_orig = StandardScaler().fit(Xo_tr)
Xo_tr = scaler_orig.transform(Xo_tr).astype(np.float32)
Xo_te = scaler_orig.transform(Xo_te).astype(np.float32)

mu0 = Xo_tr[so_tr == 0].mean(axis=0, keepdims=True)
mu1 = Xo_tr[so_tr == 1].mean(axis=0, keepdims=True)
prot = np.vstack([mu0, mu1]).astype(np.float32)

svd_rank_used = min(args.svd_rank, prot.shape[0], Xo_tr.shape[1])
svd = TruncatedSVD(n_components=svd_rank_used, random_state=args.seed).fit(prot)
basis = svd.components_.T
proj = basis @ np.linalg.pinv(basis.T @ basis) @ basis.T
proj_compl = np.eye(Xo_tr.shape[1], dtype=np.float32) - proj

Xo_te_fair = Xo_te @ proj_compl
W_te = build_W_tau_only(Xo_te_fair, args.theta, args.tau)
P_te = make_row_stochastic_P(W_te)

# Prepare raw test table once (for gaps + consistency)
df_test_raw = df[df["orig_index"].isin(rid_te)].copy()
df_test_raw = df_test_raw.set_index("orig_index").loc[rid_te].reset_index()
df_test_raw = df_test_raw.add_prefix("raw_")

race_cols_raw = [c for c in df_test_raw.columns if c.startswith("raw_race_")]
race_onehot = df_test_raw[race_cols_raw].to_numpy().astype(int)
race_group = race_onehot.argmax(axis=1)


# ============================================================
# Cache / load iFair model once
# ============================================================
ifair = IFairModel(
    k=args.ifair_k,
    A_x=args.ifair_Ax,
    A_z=args.ifair_Az,
    max_iter=args.ifair_max_iter,
    nb_restarts=args.ifair_restarts,
)

rng = np.random.RandomState(args.ifair_seed)
n_tr = X_tr.shape[0]
if args.ifair_fit_n > 0 and args.ifair_fit_n < n_tr:
    idx_fit = rng.choice(n_tr, size=args.ifair_fit_n, replace=False)
    X_tr_fit = X_tr[idx_fit]
else:
    X_tr_fit = X_tr

cache_path = os.path.join(
    args.cache_dir,
    f"ifair_k{args.ifair_k}_Ax{args.ifair_Ax}_Az{args.ifair_Az}"
    f"_fitn{len(X_tr_fit)}_maxit{args.ifair_max_iter}_r{args.ifair_restarts}"
    f"_seed{args.ifair_seed}.pkl"
)

if os.path.exists(cache_path):
    print("[iFair] loading cache:", cache_path)
    ifair = joblib.load(cache_path)
else:
    n_fit = len(X_tr_fit)
    print(f"[iFair] fitting on n_fit={n_fit} (n_fit^2={n_fit*n_fit:,} scale indicator)")
    t0 = time.time()
    ifair.fit(X_tr_fit)
    t1 = time.time()
    print(f"[iFair] fit done in {(t1 - t0):.1f}s ({(t1 - t0)/60:.2f} min)")
    joblib.dump(ifair, cache_path)
    print("[iFair] saved cache ->", cache_path)


# ============================================================
# Model
# ============================================================
class MLPBin(nn.Module):
    def __init__(self, d_in, hidden=256, depth=1):
        super().__init__()
        layers = []
        in_dim = d_in
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, hidden))
            in_dim = hidden
        self.hidden_layers = nn.ModuleList(layers)
        self.out = nn.Linear(in_dim, 1)

    def forward(self, x):
        h = x
        for layer in self.hidden_layers:
            h = torch.relu(layer(h))
        return self.out(h).squeeze(1)


def run_one(name: str, transform_fn):
    """
    name: label for this run
    transform_fn: function(Xi)->Xhat (np.ndarray)
    """
    # Transform
    Xhat_tr = transform_fn(X_tr).astype(np.float32)
    Xhat_va = transform_fn(X_va).astype(np.float32)
    Xhat_te = transform_fn(X_te).astype(np.float32)

    print(f"\n[{name}] Xhat_tr shape = {Xhat_tr.shape}")
    print(f"[{name}] std min/median/max:",
          float(Xhat_tr.std(axis=0).min()),
          float(np.median(Xhat_tr.std(axis=0))),
          float(Xhat_tr.std(axis=0).max()))
    print(f"[{name}] nan/inf:", bool(np.isnan(Xhat_tr).any()), bool(np.isinf(Xhat_tr).any()))

    # Standardize using TRAIN only
    scaler_z = StandardScaler().fit(Xhat_tr)
    Z_tr = scaler_z.transform(Xhat_tr).astype(np.float32)
    Z_va = scaler_z.transform(Xhat_va).astype(np.float32)
    Z_te = scaler_z.transform(Xhat_te).astype(np.float32)

    # Torch tensors
    Ztr_t = torch.tensor(Z_tr, dtype=torch.float32, device=device)
    Zva_t = torch.tensor(Z_va, dtype=torch.float32, device=device)
    Zte_t = torch.tensor(Z_te, dtype=torch.float32, device=device)
    ytr_t = torch.tensor(y_tr, dtype=torch.float32, device=device)

    # Model + opt
    model = MLPBin(d_in=Z_tr.shape[1], hidden=args.hidden, depth=args.depth).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=10 ** (-args.nloglr))

    train_loader = DataLoader(
        TensorDataset(Ztr_t, ytr_t),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True
    )
    train_iter = iter(train_loader)

    # Train (URW-style) + best by val_bal_acc
    best_val_bal = -1.0
    best_state = None

    for it in trange(args.num_iterations, desc=f"{name} train", leave=False):
        model.train()
        try:
            zb, yb = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            zb, yb = next(train_iter)

        logits = model(zb)
        loss = F.binary_cross_entropy_with_logits(logits, yb)

        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()

        if (it + 1) % 200 == 0 or it == 0:
            model.eval()
            with torch.no_grad():
                p_va = torch.sigmoid(model(Zva_t)).detach().cpu().numpy().ravel()
            pred_va = (p_va >= 0.5).astype(int)
            bal_acc = float(balanced_accuracy_score(y_va, pred_va))

            if args.select_best == 1 and bal_acc > best_val_bal:
                best_val_bal = bal_acc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if args.select_best == 1 and best_state is not None:
        model.load_state_dict(best_state)

    # Best threshold on VAL for balanced accuracy
    model.eval()
    with torch.no_grad():
        p_va = torch.sigmoid(model(Zva_t)).detach().cpu().numpy().ravel()

    thr_grid = np.linspace(0.05, 0.95, 91)
    best_thr, best_ba = 0.5, -1.0
    for thr in thr_grid:
        ba = balanced_accuracy_score(y_va, (p_va >= thr).astype(int))
        if ba > best_ba:
            best_ba, best_thr = float(ba), float(thr)

    # TEST eval
    with torch.no_grad():
        logits_te = model(Zte_t).detach().cpu().numpy().ravel()
    pt_prob = (1.0 / (1.0 + np.exp(-logits_te))).ravel()
    pred_label = (pt_prob >= best_thr).astype(int)

    acc_te = float(accuracy_score(y_te, pred_label))
    bal_acc_te = float(balanced_accuracy_score(y_te, pred_label))
    auc_te = float(roc_auc_score(y_te, pt_prob))
    ce_te = float(manual_logloss(y_te, pt_prob, eps=1e-7))

    # IF/LIF/VR
    inst_if = IF_cons_vector_forward_kl(P_te, pt_prob)
    IF_mean = float(inst_if.mean())
    IF_median = float(np.median(inst_if))
    lif = float(lif_energy(W_te, pt_prob, normalize="per_edge"))
    vr_dict = {f"VR({eps})": float(np.mean(inst_if > eps)) for eps in [0.001, 0.005, 0.02]}

    # DP + gaps
    dp = float(demographic_parity_prob(pt_prob, sens_te))
    gap_g_rms, gap_g_max = group_gap_stats_prob(pt_prob, sens_te)
    gap_r_rms, gap_r_max = group_gap_stats_prob(pt_prob, race_group)

    # Consistency (generic transform)
    df_pred = pd.DataFrame({
        "orig_index": rid_te,
        "y_true": y_te,
        "pred_prob": pt_prob,
        "pred_label": pred_label,
        "sens": sens_te,
    })
    df_out_full = pd.concat([df_pred, df_test_raw], axis=1)

    gr_con, s_con = get_consistency_generic(
        df_out_full,
        transform_fn=transform_fn,
        scaler_z=scaler_z,
        model=model,
        thr=best_thr,
        device=device,
        raw_prefix="raw_",
    )

    out = {
        "METHOD": name,
        "seed": int(args.seed),
        "depth": int(args.depth),
        "hidden": int(args.hidden),
        "num_iterations": int(args.num_iterations),
        "batch_size": int(args.batch_size),
        "lr": float(10 ** (-args.nloglr)),
        "best_thr(val_bal_acc)": float(best_thr),

        "test_acc": acc_te,
        "test_bal_acc": bal_acc_te,
        "test_auc": auc_te,
        "test_logloss": ce_te,

        "DP(pred_prob; gender)": dp,
        "Gap_G_RMS": float(gap_g_rms),
        "Gap_G_max": float(gap_g_max),
        "Gap_R_RMS": float(gap_r_rms),
        "Gap_R_max": float(gap_r_max),

        "IF_mean(forwardKL,prob)": IF_mean,
        "IF_median(forwardKL,prob)": IF_median,
        **vr_dict,

        "LIF_energy(per_edge,prob)": lif,
        "GR_Consistency": float(gr_con),
        "S_Consistency": float(s_con),

        "theta": float(args.theta),
        "tau": float(args.tau),
        "svd_rank": int(svd_rank_used),

        # iFair params recorded for reproducibility
        "ifair_k": int(args.ifair_k),
        "ifair_Ax": float(args.ifair_Ax),
        "ifair_Az": float(args.ifair_Az),
        "ifair_max_iter": int(args.ifair_max_iter),
        "ifair_restarts": int(args.ifair_restarts),
        "ifair_fit_n": int(len(X_tr_fit)),
        "cache_path": cache_path,
    }
    return out, df_out_full


# ============================================================
# Run BOTH
# ============================================================
def identity_transform(Xi: np.ndarray) -> np.ndarray:
    return Xi

def ifair_transform(Xi: np.ndarray) -> np.ndarray:
    return ifair.transform(Xi)

out_id, _ = run_one("Identity(No-iFair) + ERM", identity_transform)
pprint(out_id)

out_if, df_if_full = run_one(f"iFair pre-proc (k={args.ifair_k}, Ax={args.ifair_Ax}, Az={args.ifair_Az}) + ERM", ifair_transform)
pprint(out_if)

# Save metrics (two rows)
df_metrics = pd.DataFrame([out_id, out_if])
df_metrics.to_csv(args.results_csv, index=False)
print(f"\n[Saved] Metrics CSV (2 rows) -> {args.results_csv}")

# Optional: save per-instance FULL table for iFair run only
if args.save_if_csv:
    out_dir = os.path.dirname(args.save_if_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df_if_full.to_csv(args.save_if_csv, index=False)
    print(f"[Saved] Full per-instance table (iFair run) -> {args.save_if_csv}")

print("\nDONE.")