#!/usr/bin/env python3
# run_regression_inproc_urw.py  (pnorm version, monotone + stable)
# ------------------------------------------------------------
# URW consistency (REGRESSION)
#
# Goal (your requirement):
#   p larger => stronger penalty => worse performance, better fairness (more smoothing)
#
# Key changes vs your current version:
#   (A) finite p uses true p-norm WITH ROOT:
#         Cons_p = ( mean(r^p) )^(1/p)
#       This is monotone in p (for nonnegative r) and is much more stable than raw moments.
#   (B) add gradient clipping to stabilize p=4 (and larger p).
# ------------------------------------------------------------

import argparse, random, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange
from pprint import pprint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
import pandas as pd


# -------------------------
# Args
# -------------------------
parser = argparse.ArgumentParser(description="URW consistency in-processing (regression)")

parser.add_argument("--p_norm", type=str, default="1")           # "1", "4", "8", "inf"
parser.add_argument("--alpha_max", type=float, default=200.0)    # for inf logsumexp (if enabled)

parser.add_argument("--depth", type=int, default=1, help="number of hidden layers in MLP (1,2,3...)")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--num_iterations", type=int, default=4000)
parser.add_argument("--nloglr", type=float, default=3.0, help="lr = 10^{-nloglr}")
parser.add_argument("--lambda_cons", type=float, default=0.3)

# Graph locality (tau-only)
parser.add_argument("--theta", type=float, default=1e-3, help="RBF scale")
parser.add_argument("--tau", type=float, default=5.0, help="distance^2 threshold")
parser.add_argument("--max_graph_n", type=int, default=5000, help="Max #train pts to build W on (subset)")

# Fair subspace
parser.add_argument("--svd_rank", type=int, default=1, help="#components for sensitive subspace SVD (1~3 OK)")

# OPTIONAL: learn a metric from data on fair space (linear projection)
parser.add_argument("--use_metric", action="store_true")
parser.add_argument("--metric_rank", type=int, default=64)

# Export per-instance IF on TEST (path, if provided)
parser.add_argument("--save_if_csv", type=str, default="")

# Stability knobs


parser.add_argument("--r_clip", type=float, default=10.0,
                    help="clip r_norm to <= r_clip before p-norm/topk (10~50 recommended for p=4 stability)")

# p=inf options

parser.add_argument("--inf_use_logsumexp", action="store_true",
                    help="If set, use logsumexp(alpha*r_use) max-approx for p=inf (harsher)")

# Gradient clipping (important for p=4 stability)
parser.add_argument("--grad_clip", type=float, default=1.0,
                    help="clip grad norm (0 disables). 0.5~1.0 recommended for p=4")
# Guarantee: larger p => stronger penalty (scale the regularizer)


# Data path
parser.add_argument("--data_path", type=str, default="../data/Employers_data.csv")

args = parser.parse_args()
print(vars(args))

# format p string for filenames
if str(args.p_norm).lower() in ["inf", "infty", "infinity"]:
    p_str = "inf"
else:
    try:
        pv = float(args.p_norm)
        p_str = str(int(pv)) if pv.is_integer() else str(args.p_norm).replace(".", "p")
    except Exception:
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
    A2 = (A**2).sum(axis=1, keepdims=True)
    B2 = (B**2).sum(axis=1, keepdims=True).T
    return np.maximum(A2 - 2.0 * (A @ B.T) + B2, 0.0)

def build_W_tau_only(X_for_metric: np.ndarray, theta: float, tau: float) -> np.ndarray:
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
    nbr = P.dot(yhat)
    return np.abs(yhat - nbr)

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

def group_gap_reg(y_pred: np.ndarray, sens_bin: np.ndarray):
    m1 = float(y_pred[sens_bin == 1].mean())
    m0 = float(y_pred[sens_bin == 0].mean())
    gap = m1 - m0
    return gap, abs(gap)


# -------------------------
# Gender Counterfactual Consistency (Regression)
# -------------------------
def gender_consistency_reg(
    model: nn.Module,
    scaler: StandardScaler,
    X_full: np.ndarray,
    gender_col_idx_full: int,
    model_col_idx: list,
    proj_compl: np.ndarray,
    device="cpu",
):
    model.eval()

    def _predict_from_full(X_full_in: np.ndarray):
        X_model_raw = X_full_in[:, model_col_idx].astype(np.float32)
        X_std = scaler.transform(X_model_raw).astype(np.float32)
        X_fair = (X_std @ proj_compl).astype(np.float32)
        with torch.no_grad():
            pred_log = model(torch.tensor(X_fair, dtype=torch.float32, device=device)).cpu().numpy().ravel()
        pred_real = np.exp(pred_log)
        return pred_real, pred_log

    pred_orig_real, _ = _predict_from_full(X_full)

    X_flip = X_full.copy()
    X_flip[:, gender_col_idx_full] = 1.0 - X_flip[:, gender_col_idx_full]
    pred_flip_real, _ = _predict_from_full(X_flip)

    delta = np.abs(pred_orig_real - pred_flip_real)
    mean_delta = float(delta.mean())
    median_delta = float(np.median(delta))

    eps = 0.05 * float(pred_orig_real.mean())
    consistency_ratio = float((delta < eps).mean())

    return {
        "gender_consistency_ratio": consistency_ratio,
        "gender_sensitivity_mean": mean_delta,
        "gender_sensitivity_median": median_delta,
    }


# -------------------------
# Data
# -------------------------
df = pd.read_csv(args.data_path)
df = df.drop(columns=["Employee_ID", "Name"], errors="ignore")

sens_full = (df["Gender"].astype(str).str.lower() == "female").astype(int).values
df["Gender_bin"] = sens_full

salary_raw = df["Salary"].astype(np.float32).values
salary_raw = np.clip(salary_raw, 1e-6, None).astype(np.float32)
y_log = np.log(salary_raw).astype(np.float32)

features_full = df.drop(columns=["Gender", "Salary"], errors="ignore")  # includes Gender_bin
X_full_df = pd.get_dummies(features_full, drop_first=True).astype(np.float32)
if "Gender_bin" not in X_full_df.columns:
    raise ValueError("Gender_bin missing after get_dummies; check df['Gender_bin'].")

X_model_df = X_full_df.drop(columns=["Gender_bin"]).copy()

X = X_model_df.values.astype(np.float32)
model_feature_names = list(X_model_df.columns)
full_feature_names = list(X_full_df.columns)

indices = np.arange(X.shape[0])

# Split train/val/test (stratify on true gender)
X_tr, X_tmp, y_tr, y_tmp, sens_tr, sens_tmp, sal_tr, sal_tmp, idx_tr, idx_tmp = train_test_split(
    X, y_log, sens_full, salary_raw, indices,
    test_size=0.4, random_state=args.seed, stratify=sens_full
)
X_va, X_te, y_va, y_te, sens_va, sens_te, sal_va, sal_te, idx_va, idx_te = train_test_split(
    X_tmp, y_tmp, sens_tmp, sal_tmp, idx_tmp,
    test_size=0.5, random_state=args.seed, stratify=sens_tmp
)

# -------------------------
# Standardize on TRAIN only
# -------------------------
scaler = StandardScaler()
scaler.fit(X_tr)

X_tr_std = scaler.transform(X_tr).astype(np.float32)
X_va_std = scaler.transform(X_va).astype(np.float32)
X_te_std = scaler.transform(X_te).astype(np.float32)

n_tr, d = X_tr_std.shape
print(f"num_train={n_tr}, num_val={X_va_std.shape[0]}, num_test={X_te_std.shape[0]}, d={d}")

# -------------------------
# Fair projector
# -------------------------
mu0 = X_tr_std[sens_tr == 0].mean(axis=0, keepdims=True).astype(np.float32)
mu1 = X_tr_std[sens_tr == 1].mean(axis=0, keepdims=True).astype(np.float32)
prot = np.vstack([mu0, mu1]).astype(np.float32)

r_sens = min(args.svd_rank, prot.shape[0], d)
svd = TruncatedSVD(n_components=r_sens, random_state=args.seed)
svd.fit(prot)
basis = svd.components_.T.astype(np.float32)  # [d, r]

proj = (basis @ np.linalg.pinv(basis.T @ basis) @ basis.T).astype(np.float32)
proj_compl = (np.eye(d, dtype=np.float32) - proj).astype(np.float32)

X_tr_fair = (X_tr_std @ proj_compl).astype(np.float32)
X_va_fair = (X_va_std @ proj_compl).astype(np.float32)
X_te_fair = (X_te_std @ proj_compl).astype(np.float32)

# -------------------------
# Optional metric learning on FAIR space
# -------------------------
L_metric = None
if args.use_metric:
    max_rank = min(args.metric_rank, X_tr_fair.shape[1], X_tr_fair.shape[0] - 1)
    if max_rank < 1:
        raise ValueError(f"Invalid metric_rank={args.metric_rank} for train size={X_tr_fair.shape[0]}")
    svd_metric = TruncatedSVD(n_components=max_rank, random_state=args.seed)
    svd_metric.fit(X_tr_fair)
    L_metric = svd_metric.components_.astype(np.float32)  # [r_metric, d]

def apply_metric(X_fair_std: np.ndarray) -> np.ndarray:
    if L_metric is None:
        return X_fair_std.astype(np.float32)
    return (X_fair_std @ L_metric.T).astype(np.float32)

# -------------------------
# Model
# -------------------------
class MLPReg(nn.Module):
    def __init__(self, d_in: int, hidden: int = 256, depth: int = 1):
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

model = MLPReg(d_in=d, hidden=256, depth=args.depth).to(device)
optim = torch.optim.Adam(model.parameters(), lr=10 ** (-args.nloglr))

Xtr_t = torch.tensor(X_tr_fair, dtype=torch.float32, device=device)
ytr_t = torch.tensor(y_tr,      dtype=torch.float32, device=device)
Xva_t = torch.tensor(X_va_fair, dtype=torch.float32, device=device)
yva_t = torch.tensor(y_va,      dtype=torch.float32, device=device)
Xte_t = torch.tensor(X_te_fair, dtype=torch.float32, device=device)
yte_t = torch.tensor(y_te,      dtype=torch.float32, device=device)

# -------------------------
# Graph on TRAIN subset
# -------------------------
if n_tr > args.max_graph_n:
    idx_graph = np.random.RandomState(args.seed).choice(n_tr, args.max_graph_n, replace=False)
else:
    idx_graph = np.arange(n_tr)

idx_graph_t = torch.tensor(idx_graph, dtype=torch.long, device=device)

X_tr_g_fair = X_tr_fair[idx_graph]
X_tr_g_metric = apply_metric(X_tr_g_fair)

W_tr = build_W_tau_only(X_tr_g_metric, args.theta, args.tau)
P_tr = make_row_stochastic_P(W_tr).tocoo()

idx = torch.tensor(np.vstack([P_tr.row, P_tr.col]), dtype=torch.long, device=device)
val = torch.tensor(P_tr.data, dtype=torch.float32, device=device)
P_tr_sp = torch.sparse_coo_tensor(idx, val, torch.Size(P_tr.shape), device=device).coalesce()

# -------------------------
# Consistency loss helper (MONOTONE + STABLE)
# -------------------------
def compute_cons_loss_from_r(r: torch.Tensor) -> torch.Tensor:
    """
    Tail-sensitive local predictive-consistency regularizer.

    For finite p:
        Omega_p = ( mean_i r_i^p )^(1/p)

    For p = infinity:
        smooth approximation to max_i r_i using normalized log-sum-exp

        Omega_inf_tilde
          = [ logsumexp(alpha * r) - log(n) ] / alpha

    where r_i >= 0 is the absolute local predictive-consistency residual.

    No p-dependent gain is used: differences across p arise only from the
    aggregation itself.
    """

    if r.ndim != 1:
        r = r.reshape(-1)

    if r.numel() == 0:
        raise ValueError("Residual vector is empty.")

    # Optional common numerical clipping.
    # Use the same rule for every p.
    r_use = r
    if args.r_clip is not None and float(args.r_clip) > 0:
        r_use = torch.clamp(r_use, max=float(args.r_clip))

    p_raw = str(args.p_norm).lower()

    # ---------------------------------------------------------
    # p = infinity: smooth approximation to maximum residual
    # ---------------------------------------------------------
    if p_raw in ["inf", "infty", "infinity"]:
        alpha = float(args.alpha_max)

        if alpha <= 0:
            raise ValueError("--alpha_max must be positive for p=inf.")

        n = r_use.numel()

        cons = (
            torch.logsumexp(alpha * r_use, dim=0)
            - np.log(float(n))
        ) / alpha

        return cons

    # ---------------------------------------------------------
    # finite p: empirical Lp norm
    # ---------------------------------------------------------
    p = float(p_raw)

    if p < 1:
        raise ValueError("p_norm must satisfy p >= 1.")

    cons = (
        r_use.pow(p).mean() + 1e-12
    ).pow(1.0 / p)

    return cons


# -------------------------
# Train
# -------------------------
print("\n[Training] URW: loss = MSE(y_log) + lambda * Cons(L_URW f_g)")
for it in trange(args.num_iterations):
    model.train()
    pred_tr_log = model(Xtr_t)
    task_loss = F.mse_loss(pred_tr_log, ytr_t)

    f_g = pred_tr_log[idx_graph_t]
    Pf = torch.sparse.mm(P_tr_sp, f_g.unsqueeze(1)).squeeze(1)
    Lf = f_g - Pf
    r = torch.abs(Lf)

    cons_loss = compute_cons_loss_from_r(r)

    loss = task_loss + args.lambda_cons * cons_loss
    optim.zero_grad()
    loss.backward()

    # gradient clipping to stabilize p=4 / inf
    if args.grad_clip is not None and float(args.grad_clip) > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(args.grad_clip))

    optim.step()

    if (it + 1) % 200 == 0 or it == 0:
        model.eval()
        with torch.no_grad():
            pv_log = model(Xva_t).cpu().numpy().ravel()
        pv_real = np.exp(pv_log)

        rmse = mean_squared_error(sal_va, pv_real, squared=False)
        mae = mean_absolute_error(sal_va, pv_real)
        r2 = r2_score(sal_va, pv_real)
        print({
            "iter": it + 1,
            "train_mse_log": float(task_loss.item()),
            "train_cons_urw_pnorm": float(cons_loss.item()),
            "val_RMSE_real": rmse,
            "val_MAE_real": mae,
            "val_R2_real": r2,
        })

# -------------------------
# Evaluation on TEST
# -------------------------
print("\n[Evaluation] TEST")
model.eval()
with torch.no_grad():
    pt_log = model(Xte_t).cpu().numpy().ravel()
pt_real = np.exp(pt_log)

rmse = mean_squared_error(sal_te, pt_real, squared=False)
mae  = mean_absolute_error(sal_te, pt_real)
r2   = r2_score(sal_te, pt_real)
mse  = mean_squared_error(sal_te, pt_real, squared=True)

X_te_metric = apply_metric(X_te_fair)
W_te = build_W_tau_only(X_te_metric, args.theta, args.tau)
P_te = make_row_stochastic_P(W_te)

inst_if = IF_cons_vector(P_te, pt_log)
IF_mean = float(inst_if.mean())
IF_median = float(np.median(inst_if))
lif = lif_energy(W_te, pt_log, normalize="per_edge")

gap_real, gap_abs_real = group_gap_reg(pt_real, sens_te)
gap_log,  gap_abs_log  = group_gap_reg(pt_log,  sens_te)

pred_std_real = float(pt_real.std())
pred_std_log  = float(pt_log.std())

tradeoff_real = gap_abs_real / (pred_std_real + 1e-12)
tradeoff_log  = gap_abs_log  / (pred_std_log  + 1e-12)

X_full_all = X_full_df.values.astype(np.float32)
X_full_te  = X_full_all[idx_te]

gender_col_idx_full = full_feature_names.index("Gender_bin")
model_col_idx = [full_feature_names.index(c) for c in model_feature_names]

cons_gender = gender_consistency_reg(
    model=model,
    scaler=scaler,
    X_full=X_full_te,
    gender_col_idx_full=gender_col_idx_full,
    model_col_idx=model_col_idx,
    proj_compl=proj_compl,
    device=device,
)

out = {
    "METHOD": "URW tail-sensitive empirical Lp regularization",
    "p_norm": str(args.p_norm),
    "alpha_max": float(args.alpha_max),

    "r_clip": float(args.r_clip),

    "inf_use_logsumexp": bool(args.inf_use_logsumexp),
    "grad_clip": float(args.grad_clip),
    "test_RMSE_real": rmse,
    "test_MAE_real": mae,
    "test_R2_real": r2,
    "test_MSE_real": mse,

    "group_gap_real_female_minus_male": gap_real,
    "group_gap_abs_real_female_minus_male": gap_abs_real,
    "group_gap_log_female_minus_male": gap_log,
    "group_gap_abs_log_female_minus_male": gap_abs_log,

    "pred_std_real": pred_std_real,
    "pred_std_log": pred_std_log,
    "tradeoff_abs_gap_over_std_real": tradeoff_real,
    "tradeoff_abs_gap_over_std_log": tradeoff_log,

    "IF_mean(URW_L1,log_pred)": IF_mean,
    "IF_median(URW_L1,log_pred)": IF_median,
    "LIF_energy(unorm,log_pred)": lif,

    "lambda_cons": float(args.lambda_cons),
    "tau_used": float(args.tau),
    "theta": float(args.theta),
    "graph_train_subset": int(len(idx_graph)),
    "svd_rank": int(r_sens),
    "use_metric": bool(args.use_metric),
    "metric_rank": int(L_metric.shape[0]) if (args.use_metric and L_metric is not None) else 0,
}
out.update(cons_gender)

pprint(out)

# -------------------------
# Save metrics
# -------------------------
os.makedirs("results_pnorm", exist_ok=True)
metrics_path = f"results_pnorm/metrics_urw_depth{args.depth}_p{p_str}_lambda{args.lambda_cons}.csv"
pd.DataFrame([out]).to_csv(metrics_path, index=False)
print(f"[Saved] metrics → {metrics_path}")

# -------------------------
# Optional: save per-instance IF on TEST
# -------------------------
if args.save_if_csv:
    try:
        out_dir = os.path.dirname(args.save_if_csv)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        df_if = pd.DataFrame({
            "index": np.arange(len(pt_log)),
            "y_real": sal_te,
            "y_log": y_te,
            "pred_real": pt_real,
            "pred_log": pt_log,
            "abs_error_real": np.abs(sal_te - pt_real),
            "abs_error_log": np.abs(y_te - pt_log),
            "if_score_urw_log": inst_if,
        })
        df_if.to_csv(args.save_if_csv, index=False)
        print(f"[Saved] per-instance IF → {args.save_if_csv}")
    except Exception as e:
        print(f"[Warn] Failed to save IF CSV: {e}")
