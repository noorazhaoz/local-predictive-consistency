#!/usr/bin/env python3
# run_survival_inproc_curve_laplacian_train_test_only.py
# ------------------------------------------------------------
# Curve-first survival (discrete-time hazard net; DeepHit-like)
# + Ordinary Graph Laplacian consistency (UNNORMALIZED) on SURVIVAL CURVES.
# TRAIN/TEST only (no VAL, no model selection).
#
# Loss:
#   L = discrete_time_NLL + lambda * (1/(n_g*T)) * tr(S_g^T (D - W) S_g)
#
# where S_g is the predicted survival curve matrix on the TRAIN graph subset.
#
# Eval on TEST:
#   - IBS + Brier via sksurv (IPCW censoring model fit on TRAIN)
#   - optional C-index via scalar risk = -log S(t*)
#   - curve IF: sqrt(mean_t (S_i(t) - (P S)(t))^2 )
#   - curve LIF: (1/(m*T)) * tr(S^T (D-W) S) on TEST graph
#   - group gap on scalar risk
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
parser = argparse.ArgumentParser(description="Curve-first survival + Laplacian on curves (TRAIN/TEST only)")

parser.add_argument("--dataset", type=str, default="whas",
                    choices=["flchain", "whas", "rossi", "employee"])
parser.add_argument("--depth", type=int, default=1)
parser.add_argument("--seed", type=int, default=0)

parser.add_argument("--num_iterations", type=int, default=4000)
parser.add_argument("--nloglr", type=float, default=3.0)   # lr = 10^{-nloglr}
parser.add_argument("--lambda_cons", type=float, default=0.3)

# train/test split
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

# Discrete-time bins
parser.add_argument("--num_bins", type=int, default=50)
parser.add_argument("--bin_strategy", type=str, default="quantile", choices=["quantile", "linspace"])

# IBS grid
parser.add_argument("--ibs_num_times", type=int, default=20)
parser.add_argument("--ibs_q_high", type=float, default=0.95)

# outputs
parser.add_argument("--out_root", type=str, default="results/lap_curve")
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
    """Dense W with RBF weights + tau threshold, symmetric, with self-loop W_ii=1."""
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

def group_gap(y_score: np.ndarray, sens_bin: np.ndarray):
    m1 = float(y_score[sens_bin == 1].mean()) if np.any(sens_bin == 1) else 0.0
    m0 = float(y_score[sens_bin == 0].mean()) if np.any(sens_bin == 0) else 0.0
    gap = m1 - m0
    return gap, abs(gap)

def lif_energy_matrix(W: np.ndarray, F: np.ndarray, normalize: str = "per_edge") -> float:
    """Curve Dirichlet energy: tr(F^T (D-W) F). F is (n,T)."""
    W = np.asarray(W, float)
    F = np.asarray(F, float)
    d = W.sum(axis=1)
    L_F = d[:, None] * F - (W @ F)
    E = float(np.sum(F * L_F))  # trace
    if normalize == "per_node":
        E /= max(F.shape[0], 1)
    elif normalize == "per_edge":
        m = int((W > 0).sum())
        E /= max(m, 1)
    return E


# -------------------------
# IBS helpers
# -------------------------
def make_eval_times_for_ibs(
    t_train: np.ndarray,
    t_test: np.ndarray,
    num: int = 20,
    q_low: float = 0.0,
    q_high: float = 0.95,
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


# -------------------------
# Discrete-time survival bins
# -------------------------
def make_time_bins(t_train: np.ndarray, num_bins: int = 50, strategy: str = "quantile") -> np.ndarray:
    t_train = np.asarray(t_train, float)
    t_train = t_train[np.isfinite(t_train)]
    if t_train.size < 2:
        raise ValueError("Need at least 2 train times.")
    if num_bins < 2:
        raise ValueError("num_bins must be >= 2")

    if strategy == "quantile":
        edges = np.quantile(t_train, np.linspace(0.0, 1.0, num_bins + 1))
    else:
        edges = np.linspace(float(t_train.min()), float(t_train.max()), num_bins + 1)

    edges = np.unique(edges)
    if edges.size < 3:
        raise ValueError("Too few unique bin edges; try fewer bins.")
    return edges.astype(float)

def time_to_bin(t: np.ndarray, edges: np.ndarray) -> np.ndarray:
    t = np.asarray(t, float)
    k = np.searchsorted(edges[1:], t, side="right")
    return np.clip(k, 0, edges.size - 2).astype(int)


# -------------------------
# Model: discrete-time hazard net (curve-first)
# -------------------------
class DeepHazardNet(nn.Module):
    def __init__(self, d_in: int, K: int, hidden: int = 256, depth: int = 1):
        super().__init__()
        layers = []
        in_dim = d_in
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, hidden))
            in_dim = hidden
        self.hidden_layers = nn.ModuleList(layers)
        self.out = nn.Linear(in_dim, K)  # logits per bin

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in self.hidden_layers:
            h = torch.relu(layer(h))
        return self.out(h)  # (n,K)

def logits_to_survival_torch(z_logits: torch.Tensor) -> torch.Tensor:
    h = torch.sigmoid(z_logits).clamp(1e-7, 1 - 1e-7)
    return torch.cumprod(1.0 - h, dim=1)  # (n,K)

def logits_to_survival_np(z_logits: np.ndarray) -> np.ndarray:
    h = 1.0 / (1.0 + np.exp(-z_logits))
    h = np.clip(h, 1e-7, 1 - 1e-7)
    return np.cumprod(1.0 - h, axis=1)

def discrete_time_nll_torch(z_logits: torch.Tensor, t_np: np.ndarray, e_np: np.ndarray, bin_edges: np.ndarray) -> torch.Tensor:
    device_ = z_logits.device
    n, K = z_logits.shape
    k = time_to_bin(t_np, bin_edges)
    k_t = torch.tensor(k, dtype=torch.long, device=device_)
    e_t = torch.tensor(e_np.astype(np.float32), dtype=torch.float32, device=device_)

    h = torch.sigmoid(z_logits).clamp(1e-7, 1 - 1e-7)
    log1m = torch.log1p(-h)           # log(1-h)
    # prefix: (n, K)  prefix[i,k] = sum_{j<=k} log(1-h_ij)
    prefix = torch.cumsum(log1m, dim=1)

    # safe gather for (k-1), then zero-out when k==0
    km1 = torch.clamp(k_t - 1, min=0)  # (n,)
    before = prefix.gather(1, km1.unsqueeze(1)).squeeze(1)  # (n,)
    before = before * (k_t > 0).to(before.dtype)  # if k==0 -> 0

    logh_at = torch.log(h.gather(1, k_t.unsqueeze(1)).squeeze(1))
    upto = prefix.gather(1, k_t.unsqueeze(1)).squeeze(1)

    loglik = e_t * (before + logh_at) + (1.0 - e_t) * (upto)
    return -loglik.mean()

def laplacian_curve_loss(S_all: torch.Tensor, idx_graph_t: torch.Tensor,
                         deg_g: torch.Tensor, W_sp: torch.Tensor) -> torch.Tensor:
    """
    Unnormalized Laplacian penalty on curves:
        (1/(n_g*T)) * tr(S_g^T (D-W) S_g)
      = (1/(n_g*T)) * sum_{i,k} S_{ik} * ((D-W)S)_{ik}
    """
    S_g = S_all[idx_graph_t]                         # (n_g,K)
    WS = torch.sparse.mm(W_sp, S_g)                  # (n_g,K)
    LS = deg_g.unsqueeze(1) * S_g - WS               # (n_g,K)
    dirich = torch.sum(S_g * LS)                     # scalar
    n_g = S_g.shape[0]
    T = S_g.shape[1]
    return dirich / max(n_g * T, 1)


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
sens_full = df[sens_col].astype(int).values

X_df = df.drop(columns=[time_col, event_col, sens_col], errors="ignore")
X_df = pd.get_dummies(X_df, drop_first=True)

imputer = SimpleImputer(strategy="median")
X_df[:] = imputer.fit_transform(X_df)

X = X_df.astype(np.float32).values
idx_all = np.arange(len(df))

# TRAIN/TEST split
X_tr, X_te, t_tr, t_te, e_tr, e_te, s_tr, s_te, idx_tr, idx_te = train_test_split(
    X, t_full, e_full, sens_full, idx_all,
    test_size=args.test_size, random_state=args.seed
)

# Standardize on TRAIN only
scaler = StandardScaler()
scaler.fit(X_tr)
X_tr_std = scaler.transform(X_tr).astype(np.float32)
X_te_std = scaler.transform(X_te).astype(np.float32)

n_tr, d = X_tr_std.shape
print(f"dataset={args.dataset} | num_train={n_tr}, num_test={X_te_std.shape[0]}, d={d}")
print(f"[Sens] train sens=1 count={int((s_tr==1).sum())}, sens=0 count={int((s_tr==0).sum())}")

# Bins from TRAIN only
bin_edges = make_time_bins(t_tr, num_bins=int(args.num_bins), strategy=args.bin_strategy)
K = int(bin_edges.size - 1)
print(f"[Bins] K={K}, edges_min={bin_edges[0]:.6f}, edges_max={bin_edges[-1]:.6f}")


# -------------------------
# Fair subspace projector
# -------------------------
if float(args.lambda_cons) == 0.0:
    X_tr_fair = X_tr_std.astype(np.float32)
    X_te_fair = X_te_std.astype(np.float32)
    L_metric = None
else:
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

    L_metric = None
    if args.use_metric:
        max_rank = min(args.metric_rank, X_tr_fair.shape[1], X_tr_fair.shape[0] - 1)
        if max_rank < 1:
            raise ValueError(f"Invalid metric_rank={args.metric_rank} for train size={X_tr_fair.shape[0]}")
        svd_metric = TruncatedSVD(n_components=max_rank, random_state=args.seed)
        svd_metric.fit(X_tr_fair)
        L_metric = svd_metric.components_.astype(np.float32)

def apply_metric(X_fair: np.ndarray) -> np.ndarray:
    if L_metric is None:
        return X_fair.astype(np.float32)
    return (X_fair @ L_metric.T).astype(np.float32)


# -------------------------
# Model
# -------------------------
model = DeepHazardNet(d_in=d, K=K, hidden=256, depth=args.depth).to(device)
optim = torch.optim.Adam(model.parameters(), lr=10 ** (-args.nloglr))

Xtr_t = torch.tensor(X_tr_fair, dtype=torch.float32, device=device)
Xte_t = torch.tensor(X_te_fair, dtype=torch.float32, device=device)


# -------------------------
# Graph on TRAIN subset (idx_graph order)
# -------------------------
if n_tr > args.max_graph_n:
    idx_graph = np.random.RandomState(args.seed).choice(n_tr, args.max_graph_n, replace=False)
else:
    idx_graph = np.arange(n_tr)

idx_graph_t = torch.tensor(idx_graph, dtype=torch.long, device=device)

X_tr_g_metric = apply_metric(X_tr_fair[idx_graph])
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
print("\n[Training] Curve-first + Laplacian on curves (TRAIN/TEST only)")
print("loss = discrete_time_NLL + lambda * (1/(n_g*T)) * tr(S_g^T (D-W) S_g)")

for it in trange(args.num_iterations):
    model.train()
    z_tr = model(Xtr_t)  # (n_train, K)

    pred_loss = discrete_time_nll_torch(z_tr, t_tr, e_tr, bin_edges)

    S_tr = logits_to_survival_torch(z_tr)  # (n_train, K)
    cons_loss = laplacian_curve_loss(S_tr, idx_graph_t, deg_tr, W_tr_sp)

    loss = pred_loss + args.lambda_cons * cons_loss
    if not torch.isfinite(loss):
        raise ValueError(f"Non-finite loss at iter {it+1}: loss={loss.item()}")

    optim.zero_grad()
    loss.backward()
    optim.step()


# -------------------------
# Evaluation on TEST
# -------------------------
print("\n[Evaluation] TEST (final model)")
model.eval()
with torch.no_grad():
    z_tr_np = model(Xtr_t).cpu().numpy()
    z_te_np = model(Xte_t).cpu().numpy()

S_tr_bins = logits_to_survival_np(z_tr_np)
S_te_bins = logits_to_survival_np(z_te_np)

# ---- IBS/Brier: filter TEST to be within TRAIN max time for censoring model support ----
tr_max = float(np.max(t_tr))
eps = 1e-12
mask_ibs = t_te < (tr_max - eps)
if mask_ibs.sum() < len(t_te):
    print("[Warn] IBS: filtering TEST because some test times exceed train max time.")
    print(f"       train_max_time={tr_max:.6f}, kept {int(mask_ibs.sum())}/{len(t_te)} test samples")

t_te_b = t_te[mask_ibs]
e_te_b = e_te[mask_ibs]
s_te_b = s_te[mask_ibs]
idx_te_b = idx_te[mask_ibs]
S_te_bins_b = S_te_bins[mask_ibs]

if t_te_b.size < 2:
    raise ValueError(f"Too few test samples left for IBS: n={t_te_b.size}")

eval_times = make_eval_times_for_ibs(
    t_tr, t_te_b, num=int(args.ibs_num_times), q_low=0.0, q_high=float(args.ibs_q_high)
)
k_eval = time_to_bin(eval_times, bin_edges)
S_te = S_te_bins_b[:, k_eval]  # (n_test_used, T)

y_train = Surv.from_arrays(event=e_tr.astype(bool), time=t_tr.astype(float))
y_test  = Surv.from_arrays(event=e_te_b.astype(bool), time=t_te_b.astype(float))

_, brier_scores = brier_score(y_train, y_test, S_te, eval_times)
ibs = integrated_brier_score(y_train, y_test, S_te, eval_times)
brier_mean = float(np.mean(brier_scores))

# ---- C-index via scalar risk at t* ----
# j = len(eval_times)//2
# risk_te = -np.log(np.clip(S_te[:, j], 1e-8, 1.0))
# c_index = concordance_index(event_times=t_te_b, predicted_scores=-risk_te, event_observed=e_te_b)
dt = np.diff(eval_times)
rmst = (S_te[:, :-1] * dt[None, :]).sum(axis=1)

risk_te = -rmst
c_index = concordance_index(t_te_b, rmst, e_te_b)

# ---- TEST graph monitoring: curve IF + curve LIF ----
X_te_metric_b = apply_metric(X_te_fair[mask_ibs])
W_te = build_W_tau_only(X_te_metric_b, args.theta, args.tau)
P_te = make_row_stochastic_P(W_te)

S_bar = P_te.dot(S_te)  # neighborhood-averaged curve on eval grid
IF_vec = np.sqrt(np.mean((S_te - S_bar) ** 2, axis=1))
IF_mean = float(IF_vec.mean())
IF_median = float(np.median(IF_vec))

lif_curve = lif_energy_matrix(W_te, S_te, normalize="per_edge")

out = {
    "METHOD": "Discrete-time hazard net (DeepHit-like) + Laplacian on curves (D-W) [TRAIN/TEST only]",
    "dataset": str(args.dataset),
    "depth": int(args.depth),
    "seed": int(args.seed),
    "test_size": float(args.test_size),

    "num_iterations": int(args.num_iterations),
    "lr": float(10 ** (-args.nloglr)),
    "lambda_cons": float(args.lambda_cons),

    "num_bins": int(K),
    "bin_strategy": str(args.bin_strategy),

    "test_c_index_from_risk=-logS(t*)": float(c_index),
    "brier_mean": float(brier_mean),
    "ibs": float(ibs),

    "curve_IF_mean(L2 over eval grid)": float(IF_mean),
    "curve_IF_median(L2 over eval grid)": float(IF_median),
    "curve_LIF_energy(S,test)": float(lif_curve),

    "theta": float(args.theta),
    "tau_used": float(args.tau),
    "graph_train_subset": int(len(idx_graph)),
    "use_metric": bool(args.use_metric),
    "metric_rank": int(L_metric.shape[0]) if (args.use_metric and L_metric is not None) else 0,

    "ibs_test_n_used": int(t_te_b.size),
    "ibs_test_n_total": int(t_te.size),
    "eval_times_num": int(len(eval_times)),
    "eval_times_min": float(np.min(eval_times)),
    "eval_times_max": float(np.max(eval_times)),
    "ibs_q_high": float(args.ibs_q_high),
    "filter_test_by_train_max": True,
}
pprint(out)

# -------------------------
# Save outputs
# -------------------------
out_dir = os.path.join(args.out_root, args.dataset)
os.makedirs(out_dir, exist_ok=True)

lam_str = f"{args.lambda_cons:g}"

metrics_path = os.path.join(
    out_dir,
    f"metrics_curve_lap_depth{args.depth}_lambda{lam_str}.csv"
)
pd.DataFrame([out]).to_csv(metrics_path, index=False)
print(f"[Saved] Metrics CSV → {metrics_path}")

if args.save_if:
    if_path = os.path.join(
        out_dir,
        f"if_curve_depth{args.depth}_lambda{lam_str}.csv"
    )
    df_if = pd.DataFrame({
        "idx": idx_te_b,
        "time": t_te_b,
        "event": e_te_b,
        "sens": s_te_b,
        "risk_-logS(t*)": risk_te,
        "if_curve_L2_over_eval_grid": IF_vec,
    })
    df_if.to_csv(if_path, index=False)
    print(f"[Saved] per-instance IF → {if_path}")
