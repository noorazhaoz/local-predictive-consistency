#!/usr/bin/env python3
# run_survival_inproc_deepsurv_urw_train_test_only.py
# ------------------------------------------------------------
# DeepSurv (Cox PH) + symmetric normalized-Laplacian regularization
# TRAIN/TEST ONLY (no validation, no early stopping, no model selection).
#
# For each (dataset, lambda):
#   1) split train/test
#   2) train for fixed num_iterations
#   3) evaluate on TEST: C-index + IBS + Brier + IF/LIF + group gap
#   4) save ONE metrics file + ONE IF file (no seed in filename)
#
# Notes on IBS:
#   sksurv's brier_score/integrated_brier_score fits censoring model on TRAIN,
#   and requires eval_times to be within follow-up of TEST and < max TRAIN time.
#   We therefore:
#     - filter TEST samples with t_test < max(t_train)
#     - choose eval_times strictly inside [min(t_test_filtered), max(t_test_filtered)) and also < max(t_train)
# ------------------------------------------------------------

import argparse, random, os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import trange
from pprint import pprint

from lifelines.datasets import load_rossi
from sksurv.datasets import load_whas500, load_flchain
from lifelines.utils import concordance_index

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.impute import SimpleImputer
from scipy.sparse import csr_matrix

from sksurv.metrics import brier_score, integrated_brier_score
from sksurv.util import Surv


# -------------------------
# Args
# -------------------------
parser = argparse.ArgumentParser(description="DeepSurv + symmetric normalized-Laplacian regularization (survival) | TRAIN/TEST ONLY")

parser.add_argument("--dataset", type=str, default="whas",
                    choices=["flchain", "whas", "rossi", "employee"])
parser.add_argument("--depth", type=int, default=1)
parser.add_argument("--seed", type=int, default=0)

parser.add_argument("--num_iterations", type=int, default=4000)
parser.add_argument("--nloglr", type=float, default=3.0)          # lr = 10^{-nloglr}
parser.add_argument("--lambda_cons", type=float, default=0.3)

# split
parser.add_argument("--test_size", type=float, default=0.2)

# Graph locality (tau-only)
parser.add_argument("--theta", type=float, default=1e-3)
parser.add_argument("--tau", type=float, default=5.0)
parser.add_argument("--max_graph_n", type=int, default=5000)

# Fair subspace
parser.add_argument("--svd_rank", type=int, default=1)

# Optional metric learning (linear)
parser.add_argument("--use_metric", action="store_true")
parser.add_argument("--metric_rank", type=int, default=64)

# IBS eval grid
parser.add_argument("--ibs_num_times", type=int, default=20)

# Output control (defaults create exactly ONE metrics + ONE IF per dataset+lambda)
parser.add_argument("--out_root", type=str, default="results/nrw")
parser.add_argument("--save_if", action="store_true")
args = parser.parse_args()

print(vars(args))

torch.set_num_threads(min(8, torch.get_num_threads()))
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
device = torch.device("cpu")


# -------------------------
# Utils (graph)
# -------------------------
def e_dist_sq(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A2 = (A ** 2).sum(axis=1, keepdims=True)
    B2 = (B ** 2).sum(axis=1, keepdims=True).T
    return np.maximum(A2 - 2.0 * (A @ B.T) + B2, 0.0)

def build_W_tau_only(X_for_metric: np.ndarray, theta: float, tau: float) -> np.ndarray:
    """Dense symmetric RBF W with tau threshold; keep self-loop W_ii=1."""
    D2 = e_dist_sq(X_for_metric, X_for_metric).astype(np.float64)
    mask = (D2 <= float(tau)).astype(np.float64)
    W = (np.exp(-float(theta) * D2) * mask).astype(np.float64)

    np.fill_diagonal(W, 0.0)
    W = 0.5 * (W + W.T)
    n = W.shape[0]
    W[np.arange(n), np.arange(n)] = 1.0
    return W

def make_sym_normalized_adjacency(W: np.ndarray, eps: float = 1e-12) -> csr_matrix:
    """Return S = D^{-1/2} W D^{-1/2}; L_sym = I - S."""
    d = W.sum(axis=1)
    inv_sqrt_d = 1.0 / np.sqrt(np.maximum(d, eps))
    S = inv_sqrt_d[:, None] * W * inv_sqrt_d[None, :]
    S = 0.5 * (S + S.T)
    return csr_matrix(S)

def lif_energy(W: np.ndarray, f: np.ndarray, normalize: str = "per_edge") -> float:
    d = W.sum(axis=1)
    Lf = d * f - (W @ f)
    E = float(f @ Lf)
    if normalize == "per_node":
        E /= max(len(f), 1)
    elif normalize == "per_edge":
        m = int((W > 0).sum())
        E /= max(m, 1)
    return E

def group_gap(y_score: np.ndarray, sens_bin: np.ndarray):
    m1 = float(y_score[sens_bin == 1].mean()) if np.any(sens_bin == 1) else 0.0
    m0 = float(y_score[sens_bin == 0].mean()) if np.any(sens_bin == 0) else 0.0
    gap = m1 - m0
    return gap, abs(gap)


# -------------------------
# Cox partial likelihood (full-batch)
# -------------------------
def cox_ph_loss(eta: torch.Tensor, t: np.ndarray, e: np.ndarray) -> torch.Tensor:
    """Mean negative log partial likelihood over events."""
    t_t = torch.tensor(t, dtype=torch.float32, device=eta.device)
    e_t = torch.tensor(e.astype(np.float32), dtype=torch.float32, device=eta.device)

    order = torch.argsort(t_t, descending=True)
    eta_s = eta[order]
    e_s = e_t[order]

    log_cum_sum_exp = torch.logcumsumexp(eta_s, dim=0)
    loss_vec = -(eta_s - log_cum_sum_exp) * e_s
    return loss_vec.sum() / (e_s.sum() + 1e-12)


# -------------------------
# Breslow baseline for survival curves (for IBS/Brier)
# -------------------------
def breslow_baseline(t_train: np.ndarray, e_train: np.ndarray, eta_train: np.ndarray):
    """Returns (event_times_sorted, H0_cum_at_event_times)."""
    t = np.asarray(t_train, dtype=float)
    e = np.asarray(e_train, dtype=int)
    eta = np.asarray(eta_train, dtype=float)
    risk = np.exp(eta)

    event_times = np.sort(np.unique(t[e == 1]))
    H0 = []
    cum = 0.0
    for ti in event_times:
        d_i = np.sum((t == ti) & (e == 1))
        at_risk = (t >= ti)
        denom = np.sum(risk[at_risk]) + 1e-12
        cum += float(d_i) / float(denom)
        H0.append(cum)
    return event_times, np.asarray(H0, dtype=float)

def predict_survival_from_breslow(eta: np.ndarray, event_times: np.ndarray, H0: np.ndarray, eval_times: np.ndarray):
    """eta: (n,) -> S: (n,T)"""
    eta = np.asarray(eta, dtype=float)
    hr = np.exp(eta)[:, None]  # (n,1)

    idx = np.searchsorted(event_times, eval_times, side="right") - 1
    idx = np.clip(idx, -1, len(H0) - 1)

    H_eval = np.zeros_like(eval_times, dtype=float)
    valid = idx >= 0
    H_eval[valid] = H0[idx[valid]]

    S = np.exp(-hr * H_eval[None, :])
    return S


import numpy as np

from sksurv.metrics import brier_score, integrated_brier_score
from sksurv.util import Surv

def make_eval_times_for_ibs(
    t_train: np.ndarray,
    t_test: np.ndarray,
    num: int = 20,
    q_low: float = 0.0,
    q_high: float = 0.95,   # 建议 0.95 更稳；要“完全一致”就两边用同一个值
):
    t_train = np.asarray(t_train, float)
    t_test  = np.asarray(t_test, float)

    t_train = t_train[np.isfinite(t_train)]
    t_test  = t_test[np.isfinite(t_test)]

    if t_train.size < 2 or t_test.size < 2:
        raise ValueError("Need at least 2 finite times in both train and test.")
    if num < 2:
        raise ValueError("num must be >= 2")

    tr_lo = float(np.quantile(t_train, q_low))
    tr_hi = float(np.quantile(t_train, q_high))
    te_lo = float(np.quantile(t_test,  q_low))
    te_hi = float(np.quantile(t_test,  q_high))

    tr_max = float(np.max(t_train))
    te_max = float(np.max(t_test))

    lower = max(tr_lo, te_lo)
    upper = min(tr_hi, te_hi, tr_max, te_max)

    lower_in = np.nextafter(lower, lower + 1.0)
    upper_in = np.nextafter(upper, upper - 1.0)

    upper_in = min(upper_in, np.nextafter(tr_max, tr_max - 1.0))
    upper_in = min(upper_in, np.nextafter(te_max, te_max - 1.0))

    if not (lower_in < upper_in):
        raise RuntimeError(
            f"No valid overlap for eval_times: lower={lower}, upper={upper}, "
            f"tr_max={tr_max}, te_max={te_max}"
        )

    return np.linspace(lower_in, upper_in, int(num)).astype(float)


def compute_ibs_and_brier(
    *,
    t_tr, e_tr, eta_tr_np,
    t_te, e_te, eta_te_np,
    event_times0, H0,
    predict_survival_from_breslow_fn,
    num_times: int = 20,
    q_high: float = 1.0,         # 两边脚本必须一致（例如都设 0.95）
    filter_test_by_train_max: bool = True,
):
    t_tr = np.asarray(t_tr, float)
    e_tr = np.asarray(e_tr, int)
    t_te = np.asarray(t_te, float)
    e_te = np.asarray(e_te, int)

    # ---- Step A: filter TEST to satisfy censoring model support (fit on TRAIN) ----
    if filter_test_by_train_max:
        tr_max = float(np.max(t_tr))
        eps = 1e-12
        mask = t_te < (tr_max - eps)
        if mask.sum() < len(t_te):
            print("[Warn] IBS: filtering TEST because some test times exceed train max time.")
            print(f"       train_max_time={tr_max:.6f}, kept {int(mask.sum())}/{len(t_te)} test samples")
        t_te_b = t_te[mask]
        e_te_b = e_te[mask]
        eta_te_b = eta_te_np[mask]
    else:
        mask = np.ones_like(t_te, dtype=bool)
        t_te_b, e_te_b, eta_te_b = t_te, e_te, eta_te_np

    if t_te_b.size < 2:
        raise ValueError(f"Too few test samples left for IBS: n={t_te_b.size}")

    # ---- Step B: eval_times (same rule for both scripts) ----
    eval_times = make_eval_times_for_ibs(t_tr, t_te_b, num=num_times, q_low=0.0, q_high=q_high)

    # ---- Step C: survival predictions on filtered TEST only ----
    S_te = predict_survival_from_breslow_fn(eta_te_b, event_times0, H0, eval_times)

    y_train = Surv.from_arrays(event=e_tr.astype(bool), time=t_tr.astype(float))
    y_test  = Surv.from_arrays(event=e_te_b.astype(bool), time=t_te_b.astype(float))

    _, brier_scores = brier_score(y_train, y_test, S_te, eval_times)
    ibs = integrated_brier_score(y_train, y_test, S_te, eval_times)

    return {
        "brier_mean": float(np.mean(brier_scores)),
        "ibs": float(ibs),
        "eval_times_num": int(len(eval_times)),
        "eval_times_min": float(np.min(eval_times)),
        "eval_times_max": float(np.max(eval_times)),
        "ibs_test_n_used": int(t_te_b.size),
        "ibs_test_n_total": int(t_te.size),
        "ibs_q_high": float(q_high),
        "filter_test_by_train_max": bool(filter_test_by_train_max),
    }




# -------------------------
# Dataset loaders
# -------------------------
def load_dataset(dataset: str):
    if dataset == "rossi":
        time_col = "week"
        event_col = "arrest"
        sens_col = "race"
        df = load_rossi().copy()
        df = df.dropna(subset=[time_col, event_col, sens_col]).copy()
        return df, time_col, event_col, sens_col

    if dataset == "whas":
        time_col = "lenfol"
        event_col = "fstat"
        sens_col = "gender"
        X_raw, y_struct = load_whas500()
        df = X_raw.copy()
        df[time_col] = y_struct[time_col].astype(np.float32)
        df[event_col] = y_struct[event_col].astype(int)
        df = df.dropna(subset=[time_col, event_col, sens_col]).copy()
        return df, time_col, event_col, sens_col

    if dataset == "employee":
        time_col = "stag"
        event_col = "event"
        sens_col = "gender"

        df = pd.read_csv("../data/employee.csv", engine="python", encoding="latin1")
        g = df[sens_col].astype(str).str.strip().str.lower()
        df[sens_col] = g.map({"f": 1, "female": 1, "m": 0, "male": 0})

        df[time_col] = df[time_col].astype(np.float32)
        df[event_col] = df[event_col].astype(int)
        df[sens_col] = df[sens_col].astype(int)
        df = df.dropna(subset=[time_col, event_col, sens_col]).copy()
        return df, time_col, event_col, sens_col

    # flchain
    time_col = "futime"
    event_col = "death"
    sens_col = "sex"
    X_raw, y_struct = load_flchain()
    df = X_raw.copy()
    df[time_col] = y_struct[time_col].astype(np.float32)
    df[event_col] = y_struct[event_col].astype(int)
    df[sens_col] = (df[sens_col].astype(str).str.upper().str[0] == "F").astype(int)
    df = df.dropna(subset=[time_col, event_col, sens_col]).copy()
    return df, time_col, event_col, sens_col


# -------------------------
# Load & preprocess
# -------------------------
df, time_col, event_col, sens_col = load_dataset(args.dataset)

t_full = df[time_col].astype(np.float32).values
e_full = df[event_col].astype(int).values
sens_full = df[sens_col].astype(int).values  # 0/1

# Features: drop time/event/sensitive
X_df = df.drop(columns=[time_col, event_col, sens_col], errors="ignore")
X_df = pd.get_dummies(X_df, drop_first=True)

# Impute
imputer = SimpleImputer(strategy="median")
X_df[:] = imputer.fit_transform(X_df)

X = X_df.astype(np.float32).values
idx_all = np.arange(len(df))

# Split train/test (stratify on event)
X_tr, X_te, t_tr, t_te, e_tr, e_te, s_tr, s_te, idx_tr, idx_te = train_test_split(
    X, t_full, e_full, sens_full, idx_all,
    test_size=float(args.test_size),
    random_state=args.seed
)

# Standardize on TRAIN only
scaler = StandardScaler()
scaler.fit(X_tr)
X_tr_std = scaler.transform(X_tr).astype(np.float32)
X_te_std = scaler.transform(X_te).astype(np.float32)

n_tr, d = X_tr_std.shape
print(f"num_train={n_tr}, num_test={X_te_std.shape[0]}, d={d}")
print(f"[Sens] train sens=1 count={int((s_tr==1).sum())}, sens=0 count={int((s_tr==0).sum())}")


# -------------------------
# Fair subspace projector (train-only prototypes, standardized space)
# -------------------------
# after standardization (X_tr_std, X_te_std)
if float(args.lambda_cons) == 0.0:
    # PURE baseline features (no fair projection, no metric)
    X_tr_fair = X_tr_std.astype(np.float32)
    X_te_fair = X_te_std.astype(np.float32)
    L_metric = None
else:
    # ---- your current fair subspace projector ----
    mu0 = X_tr_std[s_tr == 0].mean(axis=0, keepdims=True).astype(np.float32)
    mu1 = X_tr_std[s_tr == 1].mean(axis=0, keepdims=True).astype(np.float32)
    prot = np.vstack([mu0, mu1]).astype(np.float32)

    r_sens = min(args.svd_rank, prot.shape[0], d)
    svd = TruncatedSVD(n_components=r_sens, random_state=args.seed)
    svd.fit(prot)
    basis = svd.components_.T.astype(np.float32)

    proj = (basis @ np.linalg.pinv(basis.T @ basis) @ basis.T).astype(np.float32)
    proj_compl = (np.eye(d, dtype=np.float32) - proj).astype(np.float32)

    X_tr_fair = (X_tr_std @ proj_compl).astype(np.float32)
    X_te_fair = (X_te_std @ proj_compl).astype(np.float32)

    # Optional metric learning (only meaningful when lambda>0 anyway)
    L_metric = None
    if args.use_metric:
        max_rank = min(args.metric_rank, X_tr_fair.shape[1], X_tr_fair.shape[0] - 1)
        if max_rank < 1:
            raise ValueError(f"Invalid metric_rank={args.metric_rank} for train size={X_tr_fair.shape[0]}")
        svd_metric = TruncatedSVD(n_components=max_rank, random_state=args.seed)
        svd_metric.fit(X_tr_fair)
        L_metric = svd_metric.components_.astype(np.float32)

def apply_metric(X_fair_std: np.ndarray) -> np.ndarray:
    if L_metric is None:
        return X_fair_std.astype(np.float32)
    return (X_fair_std @ L_metric.T).astype(np.float32)


# -------------------------
# Model: DeepSurv (scalar eta)
# -------------------------
class DeepSurv(nn.Module):
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

model = DeepSurv(d_in=d, hidden=256, depth=args.depth).to(device)
optim = torch.optim.Adam(model.parameters(), lr=10 ** (-args.nloglr))

Xtr_t = torch.tensor(X_tr_fair, dtype=torch.float32, device=device)
Xte_t = torch.tensor(X_te_fair, dtype=torch.float32, device=device)


# -------------------------
# Graph on TRAIN subset (symmetric normalized Laplacian)
# -------------------------
if n_tr > args.max_graph_n:
    idx_graph = np.random.RandomState(args.seed).choice(n_tr, args.max_graph_n, replace=False)
else:
    idx_graph = np.arange(n_tr)

idx_graph_t = torch.tensor(idx_graph, dtype=torch.long, device=device)

X_tr_g_fair = X_tr_fair[idx_graph]
X_tr_g_metric = apply_metric(X_tr_g_fair)

W_tr = build_W_tau_only(X_tr_g_metric, args.theta, args.tau)
S_tr = make_sym_normalized_adjacency(W_tr).tocoo()

idx = torch.tensor(np.vstack([S_tr.row, S_tr.col]), dtype=torch.long, device=device)
val = torch.tensor(S_tr.data, dtype=torch.float32, device=device)
S_tr_sp = torch.sparse_coo_tensor(idx, val, torch.Size(S_tr.shape), device=device).coalesce()


# -------------------------
# Training loop (fixed iters, no selection)
# -------------------------
print("\n[Training] TRAIN/TEST only. loss = CoxPH + lambda * eta_g^T L_sym eta_g / n_g")
for it in trange(args.num_iterations):
    model.train()
    eta_tr = model(Xtr_t)

    task_loss = cox_ph_loss(eta_tr, t_tr, e_tr)

    eta_g = eta_tr[idx_graph_t]
    Seta = torch.sparse.mm(S_tr_sp, eta_g.unsqueeze(1)).squeeze(1)
    Leta = eta_g - Seta
    cons_loss = torch.sum(eta_g * Leta) / float(len(idx_graph))
    cons_loss = torch.clamp_min(cons_loss, 0.0)

    loss = task_loss + args.lambda_cons * cons_loss

    if not torch.isfinite(loss):
        raise ValueError(f"Non-finite loss at iter {it+1}: loss={loss.item()}")

    optim.zero_grad()
    loss.backward()
    optim.step()


# -------------------------
# Evaluation on TEST (final model)
# -------------------------
print("\n[Evaluation] TEST (final model)")
model.eval()
with torch.no_grad():
    eta_tr_np = model(Xtr_t).cpu().numpy().ravel()
    eta_te_np = model(Xte_t).cpu().numpy().ravel()

# C-index
c_index = concordance_index(event_times=t_te, predicted_scores=-eta_te_np, event_observed=e_te)

# TEST graph for normalized-Laplacian diagnostics / common LIF diagnostic
X_te_metric = apply_metric(X_te_fair)
W_te = build_W_tau_only(X_te_metric, args.theta, args.tau)
S_te = make_sym_normalized_adjacency(W_te)

Leta_te = eta_te_np - S_te.dot(eta_te_np)
IF_vec = np.abs(Leta_te)
IF_mean = float(IF_vec.mean())
IF_median = float(np.median(IF_vec))
NLAP_energy = float(eta_te_np @ Leta_te) / max(len(eta_te_np), 1)
NLAP_energy = max(NLAP_energy, 0.0)

lif = lif_energy(W_te, eta_te_np, normalize="per_edge")
gap_signed, gap_abs = group_gap(eta_te_np, s_te)

# IBS / Brier (ALIGNED with Lap): use the SAME helper
event_times0, H0 = breslow_baseline(t_tr, e_tr, eta_tr_np)

ibs_pack = compute_ibs_and_brier(
    t_tr=t_tr, e_tr=e_tr, eta_tr_np=eta_tr_np,
    t_te=t_te, e_te=e_te, eta_te_np=eta_te_np,
    event_times0=event_times0, H0=H0,
    predict_survival_from_breslow_fn=predict_survival_from_breslow,
    num_times=int(args.ibs_num_times),
    q_high=0.95,                 # 关键：URW/Lap 必须同一个值（建议 0.95 或 1.0）
    filter_test_by_train_max=True
)

brier_mean = ibs_pack["brier_mean"]
ibs = ibs_pack["ibs"]

ibs_test_n_used  = ibs_pack["ibs_test_n_used"]
ibs_test_n_total = ibs_pack["ibs_test_n_total"]
eval_times_num   = ibs_pack["eval_times_num"]
eval_time_min    = ibs_pack["eval_times_min"]
eval_time_max    = ibs_pack["eval_times_max"]

out = {
    "METHOD": "DeepSurv(CoxPH) + symmetric normalized-Laplacian on eta | TRAIN/TEST only",
    "dataset": args.dataset,
    "depth": int(args.depth),
    "seed": int(args.seed),
    "test_size": float(args.test_size),

    "num_iterations": int(args.num_iterations),
    "lr": float(10 ** (-args.nloglr)),
    "lambda_cons": float(args.lambda_cons),

    "test_c_index": float(c_index),
    "brier_mean": float(brier_mean),
    "ibs": float(ibs),

    "IF_mean(|L_sym eta|,test)": float(IF_mean),
    "IF_median(|L_sym eta|,test)": float(IF_median),
    "NLAP_energy(eta,test)": float(NLAP_energy),
    "LIF_energy(eta,test)": float(lif),

    "group_gap_eta_sens1_minus_0": float(gap_signed),
    "group_gap_abs_eta": float(gap_abs),

    "theta": float(args.theta),
    "tau_used": float(args.tau),
    "graph_train_subset": int(len(idx_graph)),
    "use_metric": bool(args.use_metric),
    "metric_rank": int(L_metric.shape[0]) if (args.use_metric and L_metric is not None) else 0,
}

pprint(out)

# -------------------------
# Save ONE metrics + ONE IF (no seed in filename)
# -------------------------
out_dir = os.path.join(args.out_root, args.dataset)
os.makedirs(out_dir, exist_ok=True)

# user-requested naming:
#   if_deepsurv_depth1_lambda{}.csv
#   metrics_deepsurv_nlap_depth1_lambda{}.csv
lam_str = f"{args.lambda_cons:g}"  # keeps 1e-6 style, and 0.3 as 0.3

metrics_path = os.path.join(
    out_dir,
    f"metrics_deepsurv_nlap_depth{args.depth}_lambda{lam_str}.csv"
)
pd.DataFrame([out]).to_csv(metrics_path, index=False)
print(f"[Saved] Metrics CSV → {metrics_path}")

if args.save_if:
    if_path = os.path.join(
        out_dir,
        f"if_deepsurv_depth{args.depth}_lambda{lam_str}.csv"
    )
    df_if = pd.DataFrame({
        "idx": idx_te,
        "time": t_te,
        "event": e_te,
        "sens": s_te,
        "eta": eta_te_np,
        "if_score_nlap_residual_eta": IF_vec,
    })
    df_if.to_csv(if_path, index=False)
    print(f"[Saved] per-instance IF → {if_path}")
