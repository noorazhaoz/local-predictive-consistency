#!/usr/bin/env python3
# run_survival_inproc_deepsurv_laplacian_train_test_only.py
# ------------------------------------------------------------
# DeepSurv (Cox PH) + Ordinary Graph Laplacian consistency (UNNORMALIZED)
# TRAIN/TEST only (no VAL, no model selection).
#
# Loss:
#   L = CoxPH + lambda * (1/n_g) * eta_g^T (D - W) eta_g
#
# Eval on TEST:
#   - C-index (using -eta)
#   - Brier score + IBS (Breslow baseline + IPCW via sksurv)
#   - IF_mean/median: |(I-P_te) eta_te| using row-stochastic P on TEST (monitor)
#   - LIF_energy: Dirichlet energy on TEST using (D-W) (monitor)
#   - group gap on eta_te
#
# Robust IBS/Brier:
#   - eval_times within BOTH train and test follow-up ranges (right-open)
#   - filter test samples with t_test >= max(t_train) to satisfy censoring model support
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
parser = argparse.ArgumentParser(description="DeepSurv + Laplacian (TRAIN/TEST only)")

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

# IBS grid
parser.add_argument("--ibs_num_times", type=int, default=20)

# outputs
parser.add_argument("--out_root", type=str, default="results/lap")
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
    """Instance-level IF monitor: |(I - P) yhat|."""
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
    m1 = float(y_score[sens_bin == 1].mean()) if np.any(sens_bin == 1) else 0.0
    m0 = float(y_score[sens_bin == 0].mean()) if np.any(sens_bin == 0) else 0.0
    gap = m1 - m0
    return gap, abs(gap)


# -------------------------
# Cox partial likelihood (full-batch)
# -------------------------
def cox_ph_loss(eta: torch.Tensor, t: np.ndarray, e: np.ndarray) -> torch.Tensor:
    device_ = eta.device
    t_t = torch.tensor(t, dtype=torch.float32, device=device_)
    e_t = torch.tensor(e.astype(np.float32), dtype=torch.float32, device=device_)

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
    hr = np.exp(eta)[:, None]  # (n,1)

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

# TRAIN/TEST split only
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

def apply_metric(X_fair: np.ndarray) -> np.ndarray:
    if L_metric is None:
        return X_fair.astype(np.float32)
    return (X_fair @ L_metric.T).astype(np.float32)

import numpy as np
from sksurv.metrics import brier_score, integrated_brier_score
from sksurv.util import Surv

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
print("\n[Training] DeepSurv + Laplacian (TRAIN/TEST only)")
print("loss = CoxPH + lambda * (1/n_g) * eta_g^T (D-W) eta_g")

for it in trange(args.num_iterations):
    model.train()
    eta_tr = model(Xtr_t)

    task_loss = cox_ph_loss(eta_tr, t_tr, e_tr)

    eta_g = eta_tr[idx_graph_t]
    Weta = torch.sparse.mm(W_tr_sp, eta_g.unsqueeze(1)).squeeze(1)
    Leta = deg_tr * eta_g - Weta
    dirich = torch.sum(eta_g * Leta)
    cons_loss = dirich / max(n_g, 1)

    loss = task_loss + args.lambda_cons * cons_loss
    optim.zero_grad()
    loss.backward()
    optim.step()


# -------------------------
# Evaluation on TEST
# -------------------------
print("\n[Evaluation] TEST")
model.eval()
with torch.no_grad():
    eta_tr_np = model(Xtr_t).cpu().numpy().ravel()
    eta_te_np = model(Xte_t).cpu().numpy().ravel()

c_index = concordance_index(event_times=t_te, predicted_scores=-eta_te_np, event_observed=e_te)

# TEST graph for monitoring
X_te_metric = apply_metric(X_te_fair)
W_te = build_W_tau_only(X_te_metric, args.theta, args.tau)
P_te = make_row_stochastic_P(W_te)

IF_vec = IF_cons_vector(P_te, eta_te_np)
IF_mean = float(IF_vec.mean())
IF_median = float(np.median(IF_vec))

lif = lif_energy(W_te, eta_te_np, normalize="per_edge")
gap_signed, gap_abs = group_gap(eta_te_np, s_te)


# IBS / Brier using Breslow baseline from TRAIN
event_times0, H0 = breslow_baseline(t_tr, e_tr, eta_tr_np)

ibs_pack = compute_ibs_and_brier(
    t_tr=t_tr, e_tr=e_tr, eta_tr_np=eta_tr_np,
    t_te=t_te, e_te=e_te, eta_te_np=eta_te_np,
    event_times0=event_times0, H0=H0,
    predict_survival_from_breslow_fn=predict_survival_from_breslow,
    num_times=args.ibs_num_times,
    q_high=1.0,                    # 或 0.95，但 URW 必须同样
    filter_test_by_train_max=True
)

brier_mean = ibs_pack["brier_mean"]
ibs = ibs_pack["ibs"]


out = {
    "METHOD": "DeepSurv(CoxPH) + Laplacian consistency on eta (D-W) [TRAIN/TEST only]",
    "dataset": str(args.dataset),
    "depth": int(args.depth),
    "seed": int(args.seed),
    "lambda_cons": float(args.lambda_cons),

    "test_c_index": float(c_index),
    "brier_mean": float(brier_mean),
    "ibs": float(ibs),

    "IF_mean(|(I-P)eta|,test)": float(IF_mean),
    "IF_median(|(I-P)eta|,test)": float(IF_median),
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
# Save outputs (ONE file per lambda)
# -------------------------
out_dir = os.path.join(args.out_root, args.dataset)
os.makedirs(out_dir, exist_ok=True)

metrics_path = os.path.join(
    out_dir,
    f"metrics_deepsurv_lap_depth{args.depth}_lambda{args.lambda_cons}.csv"
)
pd.DataFrame([out]).to_csv(metrics_path, index=False)
print(f"[Saved] Metrics CSV → {metrics_path}")

if args.save_if:
    if_path = os.path.join(
        out_dir,
        f"if_deepsurv_depth{args.depth}_lambda{args.lambda_cons}.csv"
    )
    df_if = pd.DataFrame({
        "idx": idx_te,
        "time": t_te,
        "event": e_te,
        "sens": s_te,
        "eta": eta_te_np,
        "if_score_urw_eta": IF_vec,  # monitor: URW residual on TEST graph
    })
    df_if.to_csv(if_path, index=False)
    print(f"[Saved] per-instance IF → {if_path}")
