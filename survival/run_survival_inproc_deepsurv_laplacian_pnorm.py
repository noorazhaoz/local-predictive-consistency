#!/usr/bin/env python3
# run_survival_inproc_deepsurv_laplacian_pnorm.py
# ------------------------------------------------------------
# DeepSurv (Cox PH) + Ordinary Graph Laplacian p-norm consistency (UNNORMALIZED)
#
# Laplacian operator:
#   L = D - W   (unnormalized)
#
# p-norm consistency (NO ROOT):
#   r = |(L eta_g)_i|
#   Cons_p = mean_i r_i^p
#   Cons_inf ~= max_i r_i via logsumexp(alpha * r)/alpha - log(n)/alpha
# ------------------------------------------------------------

import argparse, random, os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import trange
from pprint import pprint

from sksurv.datasets import load_whas500, load_flchain
from lifelines.datasets import load_rossi
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
parser = argparse.ArgumentParser(description="DeepSurv + Laplacian p-norm consistency (survival)")

parser.add_argument("--dataset", type=str, default="flchain", choices=["flchain", "whas", "rossi"])

parser.add_argument("--p_norm", type=str, default="2.0")      # "1", "2", "4", "inf"
parser.add_argument("--alpha_max", type=float, default=50.0)  # for inf soft-max approx

parser.add_argument("--depth", type=int, default=1)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--num_iterations", type=int, default=4000)
parser.add_argument("--nloglr", type=float, default=3.0)          # lr = 10^{-nloglr}
parser.add_argument("--lambda_cons", type=float, default=0.3)

# Graph locality (tau-only)
parser.add_argument("--theta", type=float, default=1e-3)
parser.add_argument("--tau", type=float, default=5.0)
parser.add_argument("--max_graph_n", type=int, default=5000)

# Fair subspace
parser.add_argument("--svd_rank", type=int, default=1)

# Optional metric learning (linear)
parser.add_argument("--use_metric", action="store_true")
parser.add_argument("--metric_rank", type=int, default=64)

# Export per-instance IF on TEST
parser.add_argument("--save_if_csv", type=str, default="")

args = parser.parse_args()
print(vars(args))

# format p string for filenames
if str(args.p_norm).lower() in ["inf", "infty", "infinity"]:
    p_str = "inf"
else:
    p_val = float(args.p_norm)
    if float(p_val).is_integer():
        p_str = str(int(p_val))
    else:
        p_str = str(args.p_norm).replace(".", "p")

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
    """
    Dense W with RBF weights + tau threshold, symmetric, with self-loop W_ii=1.
    """
    D2 = e_dist_sq(X_for_metric, X_for_metric).astype(np.float64)
    mask = (D2 <= float(tau)).astype(np.float64)
    W = (np.exp(-float(theta) * D2) * mask).astype(np.float64)

    np.fill_diagonal(W, 0.0)
    W = 0.5 * (W + W.T)
    n = W.shape[0]
    W[np.arange(n), np.arange(n)] = 1.0
    return W

def make_row_stochastic_P(W: np.ndarray) -> csr_matrix:
    rowsum = W.sum(axis=1, keepdims=True) + 1e-12
    return csr_matrix(W / rowsum)

def IF_cons_vector(P: csr_matrix, yhat: np.ndarray) -> np.ndarray:
    """Monitor IF: |(I - P) yhat| (URW-style residual) on TEST."""
    nbr = P.dot(yhat)
    return np.abs(yhat - nbr)

def lif_energy(W: np.ndarray, f: np.ndarray, normalize: str = "per_edge") -> float:
    """Unnormalized Dirichlet energy f^T (D-W) f (optional normalization)."""
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
    m1 = float(y_score[sens_bin == 1].mean())
    m0 = float(y_score[sens_bin == 0].mean())
    gap = m1 - m0
    return gap, abs(gap)


# -------------------------
# Cox partial likelihood (full-batch)
# -------------------------
def cox_ph_loss(eta: torch.Tensor, t: np.ndarray, e: np.ndarray) -> torch.Tensor:
    device = eta.device
    t_t = torch.tensor(t, dtype=torch.float32, device=device)
    e_t = torch.tensor(e.astype(np.float32), dtype=torch.float32, device=device)

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
    eta = np.asarray(eta, dtype=float)
    hr = np.exp(eta)[:, None]
    idx = np.searchsorted(event_times, eval_times, side="right") - 1
    idx = np.clip(idx, -1, len(H0) - 1)
    H_eval = np.zeros_like(eval_times, dtype=float)
    valid = idx >= 0
    H_eval[valid] = H0[idx[valid]]
    S = np.exp(-hr * H_eval[None, :])
    return S


# -------------------------
# Data loaders
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

    time_col  = "futime"
    event_col = "death"
    sens_col  = "sex"
    X_raw, y_struct = load_flchain()
    df = X_raw.copy()
    df[time_col]  = y_struct[time_col].astype(np.float32)
    df[event_col] = y_struct[event_col].astype(int)
    df[sens_col] = (df[sens_col].astype(str).str.upper().str[0] == "F").astype(int)
    df = df.dropna(subset=[time_col, event_col, sens_col]).copy()
    return df, time_col, event_col, sens_col


# -------------------------
# Load data
# -------------------------
df, time_col, event_col, sens_col = load_dataset(args.dataset)

t_full = df[time_col].astype(np.float32).values
e_full = df[event_col].astype(int).values
sens_full = df[sens_col].astype(int).values

X_df = df.drop(columns=[time_col, event_col, sens_col], errors="ignore")
X_df = pd.get_dummies(X_df, drop_first=True)

imputer = SimpleImputer(strategy="median")
X_df[:] = imputer.fit_transform(X_df)

X = X_df.astype(np.float32).values
idx_all = np.arange(len(df))

X_tr, X_tmp, t_tr, t_tmp, e_tr, e_tmp, s_tr, s_tmp, idx_tr, idx_tmp = train_test_split(
    X, t_full, e_full, sens_full, idx_all,
    test_size=0.4, random_state=args.seed, stratify=e_full
)
X_va, X_te, t_va, t_te, e_va, e_te, s_va, s_te, idx_va, idx_te = train_test_split(
    X_tmp, t_tmp, e_tmp, s_tmp, idx_tmp,
    test_size=0.5, random_state=args.seed, stratify=e_tmp
)

scaler = StandardScaler()
scaler.fit(X_tr)
X_tr_std = scaler.transform(X_tr).astype(np.float32)
X_va_std = scaler.transform(X_va).astype(np.float32)
X_te_std = scaler.transform(X_te).astype(np.float32)

n_tr, d = X_tr_std.shape
print(f"dataset={args.dataset} | num_train={n_tr}, num_val={X_va_std.shape[0]}, num_test={X_te_std.shape[0]}, d={d}")


# -------------------------
# Fair subspace projector (train-only)
# -------------------------
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
X_va_fair = (X_va_std @ proj_compl).astype(np.float32)
X_te_fair = (X_te_std @ proj_compl).astype(np.float32)

# Optional metric learning
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
# Model
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
Xva_t = torch.tensor(X_va_fair, dtype=torch.float32, device=device)
Xte_t = torch.tensor(X_te_fair, dtype=torch.float32, device=device)


# -------------------------
# Graph on TRAIN subset (idx_graph order)
# -------------------------
if n_tr > args.max_graph_n:
    idx_graph = np.random.RandomState(args.seed).choice(n_tr, args.max_graph_n, replace=False)
else:
    idx_graph = np.arange(n_tr)

idx_graph_t = torch.tensor(idx_graph, dtype=torch.long, device=device)

X_tr_g_fair = X_tr_fair[idx_graph]
X_tr_g_metric = apply_metric(X_tr_g_fair)

W_tr = build_W_tau_only(X_tr_g_metric, args.theta, args.tau)
n_g = W_tr.shape[0]

deg_tr = torch.tensor(W_tr.sum(axis=1), dtype=torch.float32, device=device)

W_tr_coo = csr_matrix(W_tr).tocoo()
idx = torch.tensor(np.vstack([W_tr_coo.row, W_tr_coo.col]), dtype=torch.long, device=device)
val = torch.tensor(W_tr_coo.data, dtype=torch.float32, device=device)
W_tr_sp = torch.sparse_coo_tensor(idx, val, torch.Size(W_tr_coo.shape), device=device).coalesce()


# -------------------------
# Training loop
# -------------------------
print("\n[Training] DeepSurv + Laplacian-pnorm(no-root): loss = CoxPH + lambda * mean(|(D-W)eta_g|^p)")

for it in trange(args.num_iterations):
    model.train()
    eta_tr = model(Xtr_t)

    task_loss = cox_ph_loss(eta_tr, t_tr, e_tr)

    eta_g = eta_tr[idx_graph_t]  # [n_g] aligned with W_tr

    # Laplacian residual: L eta = (D-W) eta
    Weta = torch.sparse.mm(W_tr_sp, eta_g.unsqueeze(1)).squeeze(1)
    Leta = deg_tr * eta_g - Weta
    r = torch.abs(Leta)

    p_raw = str(args.p_norm).lower()
    if p_raw in ["inf", "infty", "infinity"]:
        alpha = float(args.alpha_max)
        cons_loss = torch.logsumexp(alpha * r, dim=0) / alpha
        cons_loss = cons_loss - (np.log(r.numel()) / alpha)
    else:
        p = float(p_raw)
        cons_loss = r.pow(p).mean()   # NO ROOT

    loss = task_loss + args.lambda_cons * cons_loss
    optim.zero_grad()
    loss.backward()
    optim.step()

    if (it + 1) % 200 == 0 or it == 0:
        model.eval()
        with torch.no_grad():
            eta_va = model(Xva_t).cpu().numpy().ravel()

        c_val = concordance_index(event_times=t_va, predicted_scores=-eta_va, event_observed=e_va)
        gap_signed, gap_abs = group_gap(eta_va, s_va)

        print({
            "iter": it + 1,
            "train_cox": float(task_loss.item()),
            "train_lap_pnorm_no_root": float(cons_loss.item()),
            "val_cindex": float(c_val),
            "val_gap_eta": float(gap_signed),
            "val_gap_abs_eta": float(gap_abs),
        })


# -------------------------
# Evaluation on TEST
# -------------------------
print("\n[Evaluation] TEST")
model.eval()
with torch.no_grad():
    eta_tr_np = model(Xtr_t).cpu().numpy().ravel()
    eta_te_np = model(Xte_t).cpu().numpy().ravel()

c_index = concordance_index(event_times=t_te, predicted_scores=-eta_te_np, event_observed=e_te)

# TEST graph for monitors (same as your other scripts)
X_te_metric = apply_metric(X_te_fair)
W_te = build_W_tau_only(X_te_metric, args.theta, args.tau)
P_te = make_row_stochastic_P(W_te)

IF_vec = IF_cons_vector(P_te, eta_te_np)
IF_mean = float(IF_vec.mean())
IF_median = float(np.median(IF_vec))

lif = lif_energy(W_te, eta_te_np, normalize="per_edge")
gap_signed, gap_abs = group_gap(eta_te_np, s_te)

event_times0, H0 = breslow_baseline(t_tr, e_tr, eta_tr_np)

t_min = float(np.min(t_te))
t_max = float(np.max(t_te))
t_min_in = np.nextafter(t_min, t_min + 1.0)
t_max_in = np.nextafter(t_max, t_max - 1.0)
if not (t_min_in < t_max_in):
    t_min_in = t_min + 1e-8
    t_max_in = t_max - 1e-8

eval_times = np.linspace(t_min_in, t_max_in, 20)
S_te_eval = predict_survival_from_breslow(eta_te_np, event_times0, H0, eval_times)

y_train_sksurv = Surv.from_arrays(event=e_tr.astype(bool), time=t_tr.astype(float))
y_test_sksurv  = Surv.from_arrays(event=e_te.astype(bool), time=t_te.astype(float))

_, brier_scores = brier_score(y_train_sksurv, y_test_sksurv, S_te_eval, eval_times)
ibs = integrated_brier_score(y_train_sksurv, y_test_sksurv, S_te_eval, eval_times)

out = {
    "METHOD": "DeepSurv(CoxPH) + Laplacian p-norm(no-root) on (D-W)eta",
    "dataset": str(args.dataset),
    "p_norm": str(args.p_norm),
    "alpha_max": float(args.alpha_max),

    "depth": int(args.depth),
    "seed": int(args.seed),
    "lambda_cons": float(args.lambda_cons),

    "test_c_index": float(c_index),
    "brier_mean": float(np.mean(brier_scores)),
    "ibs": float(ibs),

    "IF_mean": float(IF_mean),
    "IF_median": float(IF_median),
    "LIF_energy(eta,test)": float(lif),

    "theta": float(args.theta),
    "tau_used": float(args.tau),
    "graph_train_subset": int(len(idx_graph)),
    "svd_rank": int(r_sens),
    "use_metric": bool(args.use_metric),
    "metric_rank": int(L_metric.shape[0]) if (args.use_metric and L_metric is not None) else 0,
    "num_eval_times": int(len(eval_times)),
}

pprint(out)

# -------------------------
# Save metrics
# -------------------------
os.makedirs("results_pnorm", exist_ok=True)
out_dir = f"results_pnorm/lap/{args.dataset}"
os.makedirs(out_dir, exist_ok=True)
metrics_path = f"{out_dir}/metrics_deepsurv_lap_pnorm_depth{args.depth}_p{p_str}_lambda{args.lambda_cons}.csv"
pd.DataFrame([out]).to_csv(metrics_path, index=False)
print(f"[Saved] Metrics CSV → {metrics_path}")

# -------------------------
# Optional: save per-instance IF on TEST
# -------------------------
if args.save_if_csv:
    try:
        out_dir2 = os.path.dirname(args.save_if_csv)
        if out_dir2:
            os.makedirs(out_dir2, exist_ok=True)

        df_if = pd.DataFrame({
            "idx": idx_te,
            "time": t_te,
            "event": e_te,
            "sens": s_te,
            "eta": eta_te_np,
            "if_score_urw_eta_L1": IF_vec,  # monitor: URW residual on test graph
        })
        df_if.to_csv(args.save_if_csv, index=False)
        print(f"[Saved] per-instance IF → {args.save_if_csv}")
    except Exception as e:
        print(f"[Warn] Failed to save IF CSV: {e}")
