# run_regression_inproc_nlap.py
# ------------------------------------------------------------
# Symmetric Normalized-Laplacian regularization for REGRESSION
#
# Task:
#   y = log(Salary)
#   minimize:  MSE(y_log, f(x)) + lambda * (1/n_g) * f_g^T L_sym f_g
# where:
#   L_sym = I - D^{-1/2} W D^{-1/2},   D_ii = sum_j W_ij
#
# This is the standard symmetric normalized-Laplacian quadratic regularizer.
# Importantly, there is NO second row-normalization after D^{-1/2} W D^{-1/2}.
#
# Key correctness points (matched to the same data/graph pipeline as URW):
#   1) Model ALWAYS sees FAIR-projected standardized inputs
#   2) TRAIN graph is built on a TRAIN subset (idx_graph) IN THE SAME SPACE
#   3) Consistency loss uses f_g in EXACTLY the SAME ORDER as the graph nodes (idx_graph order)
#   4) Optional metric rank is clipped to <= min(n_samples-1, n_features)
# ------------------------------------------------------------

import argparse, random, os
import numpy as np
import pandas as pd

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


# -------------------------
# Args
# -------------------------
def build_args():
    parser = argparse.ArgumentParser(description="Symmetric normalized-Laplacian in-processing (regression)")

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

    # Optional metric learning on FAIR space
    parser.add_argument("--use_metric", action="store_true")
    parser.add_argument("--metric_rank", type=int, default=64)

    # Export per-instance IF on TEST
    parser.add_argument("--save_if_csv", type=str, default="")

    # Data path
    parser.add_argument("--data_path", type=str, default="../data/Employers_data.csv")

    return parser.parse_args()


# -------------------------
# Utils (graph)
# -------------------------
def e_dist_sq(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A2 = (A**2).sum(axis=1, keepdims=True)
    B2 = (B**2).sum(axis=1, keepdims=True).T
    return np.maximum(A2 - 2.0 * (A @ B.T) + B2, 0.0)

def build_W_tau_only(X_for_metric: np.ndarray, theta: float, tau: float) -> np.ndarray:
    """
    Dense W with RBF weights and tau threshold (UNnormalized), symmetric,
    with explicit self-loop W_ii = 1.
    """
    D2 = e_dist_sq(X_for_metric, X_for_metric).astype(np.float64)
    mask = (D2 <= float(tau)).astype(np.float64)
    W = (np.exp(-float(theta) * D2) * mask).astype(np.float64)

    np.fill_diagonal(W, 0.0)
    W = 0.5 * (W + W.T)
    n = W.shape[0]
    W[np.arange(n), np.arange(n)] = 1.0
    return W

def make_S_norm(W: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Return the symmetric normalized adjacency

        S = D^{-1/2} W D^{-1/2},

    so that the normalized Laplacian is L_sym = I - S.
    W is assumed symmetric and nonnegative.
    """
    d = W.sum(axis=1)
    inv_sqrt_d = 1.0 / np.sqrt(np.maximum(d, eps))
    S = inv_sqrt_d[:, None] * W * inv_sqrt_d[None, :]
    # Numerical symmetrization only; analytically S is symmetric when W is symmetric.
    S = 0.5 * (S + S.T)
    return S

def nlap_residual_vector(S: np.ndarray, yhat: np.ndarray) -> np.ndarray:
    """Per-node magnitude |(I - S) yhat|, used only as a diagnostic."""
    return np.abs(yhat - S @ yhat)

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

def group_gap_reg(y_pred: np.ndarray, sens_bin: np.ndarray):
    """gap = mean(pred|sens=1) - mean(pred|sens=0), returns (signed_gap, abs_gap)."""
    m1 = float(y_pred[sens_bin == 1].mean())
    m0 = float(y_pred[sens_bin == 0].mean())
    gap = m1 - m0
    return gap, abs(gap)

def demographic_parity_reg(y_pred: np.ndarray, sens_bin: np.ndarray) -> float:
    """DP for regression here = abs difference of group means."""
    return float(abs(y_pred[sens_bin == 1].mean() - y_pred[sens_bin == 0].mean()))


# -------------------------
# Gender Counterfactual Consistency (Regression) -- URW pipeline
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
    """
    Counterfactual consistency: flip Gender_bin (0<->1) in X_full,
    then follow the SAME pipeline as training:
      X_full -> select model cols -> scaler.transform -> proj_compl -> model -> exp
    """
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


# -------------------------
# Main (no if __name__ == "__main__")
# -------------------------
args = build_args()
print(vars(args))

torch.set_num_threads(min(8, torch.get_num_threads()))
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
device = torch.device("cpu")

# -------------------------
# Load data
# -------------------------
df = pd.read_csv(args.data_path)
df = df.drop(columns=["Employee_ID", "Name"], errors="ignore")

# Sensitive: 1=female, 0=male
sens_full = (df["Gender"].astype(str).str.lower() == "female").astype(int).values
df["Gender_bin"] = sens_full

# Target: Salary -> log(Salary)
salary_raw = df["Salary"].astype(np.float32).values
salary_raw = np.clip(salary_raw, 1e-6, None).astype(np.float32)
y_log = np.log(salary_raw).astype(np.float32)

# Build FULL (for counterfactual) and MODEL (no Gender_bin) design matrices
features_full = df.drop(columns=["Gender", "Salary"], errors="ignore")  # includes Gender_bin
X_full_df = pd.get_dummies(features_full, drop_first=True).astype(np.float32)

if "Gender_bin" not in X_full_df.columns:
    raise ValueError("Gender_bin missing after get_dummies; check df['Gender_bin'].")

X_model_df = X_full_df.drop(columns=["Gender_bin"]).copy()

X = X_model_df.values.astype(np.float32)
model_feature_names = list(X_model_df.columns)
full_feature_names = list(X_full_df.columns)

indices = np.arange(X.shape[0])

# -------------------------
# Split train/val/test
# -------------------------
X_tr, X_tmp, y_tr, y_tmp, sens_tr, sens_tmp, sal_tr, sal_tmp, idx_tr, idx_tmp = train_test_split(
    X, y_log, sens_full, salary_raw, indices,
    test_size=0.4, random_state=args.seed, stratify=sens_full
)
X_va, X_te, y_va, y_te, sens_va, sens_te, sal_va, sal_te, idx_va, idx_te = train_test_split(
    X_tmp, y_tmp, sens_tmp, sal_tmp, idx_tmp,
    test_size=0.5, random_state=args.seed, stratify=sens_tmp
)

# -------------------------
# Standardize on TRAIN only (RAW model space)
# -------------------------
scaler = StandardScaler()
scaler.fit(X_tr)

X_tr_std = scaler.transform(X_tr).astype(np.float32)
X_va_std = scaler.transform(X_va).astype(np.float32)
X_te_std = scaler.transform(X_te).astype(np.float32)

n_tr, d = X_tr_std.shape
print(f"num_train={n_tr}, num_val={X_va_std.shape[0]}, num_test={X_te_std.shape[0]}, d={d}")

# -------------------------
# Fair projector proj_compl (learned in STANDARDIZED space)
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

# MODEL ALWAYS sees FAIR-projected standardized inputs
X_tr_fair = (X_tr_std @ proj_compl).astype(np.float32)
X_va_fair = (X_va_std @ proj_compl).astype(np.float32)
X_te_fair = (X_te_std @ proj_compl).astype(np.float32)

# -------------------------
# Optional metric learning on FAIR space (rank clipped)
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
# Torch tensors (MODEL ALWAYS sees FAIR inputs)
# -------------------------
Xtr_t = torch.tensor(X_tr_fair, dtype=torch.float32, device=device)
ytr_t = torch.tensor(y_tr,      dtype=torch.float32, device=device)
Xva_t = torch.tensor(X_va_fair, dtype=torch.float32, device=device)
yva_t = torch.tensor(y_va,      dtype=torch.float32, device=device)
Xte_t = torch.tensor(X_te_fair, dtype=torch.float32, device=device)
yte_t = torch.tensor(y_te,      dtype=torch.float32, device=device)

# -------------------------
# Build TRAIN graph on subset (IMPORTANT: keep idx_graph ORDER)
# -------------------------
if n_tr > args.max_graph_n:
    idx_graph = np.random.RandomState(args.seed).choice(n_tr, args.max_graph_n, replace=False)
else:
    idx_graph = np.arange(n_tr)

idx_graph_t = torch.tensor(idx_graph, dtype=torch.long, device=device)

X_tr_g_fair = X_tr_fair[idx_graph]            # [n_g, d] in idx_graph order
X_tr_g_metric = apply_metric(X_tr_g_fair)

W_tr = build_W_tau_only(X_tr_g_metric, args.theta, args.tau)   # [n_g, n_g]
n_g = W_tr.shape[0]

# Symmetric normalized adjacency S = D^{-1/2} W D^{-1/2}.
# The normalized-Laplacian penalty is f^T (I-S) f / n_g.
S_tr = make_S_norm(W_tr)
S_coo = csr_matrix(S_tr).tocoo()
idx = torch.tensor(np.vstack([S_coo.row, S_coo.col]), dtype=torch.long, device=device)
val = torch.tensor(S_coo.data, dtype=torch.float32, device=device)
S_tr_sp = torch.sparse_coo_tensor(idx, val, torch.Size(S_coo.shape), device=device).coalesce()

# -------------------------
# Model + Optim
# -------------------------
model = MLPReg(d_in=d, hidden=256, depth=args.depth).to(device)
optim = torch.optim.Adam(model.parameters(), lr=10 ** (-args.nloglr))

# -------------------------
# Train
# -------------------------
print("\n[Training] N-Lap: loss = MSE(y_log) + lambda * (1/n_g) * f_g^T (I - D^{-1/2} W D^{-1/2}) f_g")
for it in trange(args.num_iterations):
    model.train()
    pred_tr_log = model(Xtr_t)  # [n_tr]
    task_loss = F.mse_loss(pred_tr_log, ytr_t)

    # IMPORTANT: f_g must match graph node order (idx_graph order)
    f_g = pred_tr_log[idx_graph_t]  # [n_g]

    Sf = torch.sparse.mm(S_tr_sp, f_g.unsqueeze(1)).squeeze(1)  # S f
    Lf = f_g - Sf                                                 # L_sym f = (I-S)f
    # Standard normalized-Laplacian quadratic form: f^T L_sym f / n_g.
    # clamp_min protects against tiny negative values from floating-point error.
    cons_loss = torch.sum(f_g * Lf) / float(n_g)
    cons_loss = torch.clamp_min(cons_loss, 0.0)

    loss = task_loss + args.lambda_cons * cons_loss
    optim.zero_grad()
    loss.backward()
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
            "train_cons_nlap_quad": float(cons_loss.item()),
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

# DP on REAL scale (mean gap abs)
dp_real = demographic_parity_reg(pt_real, sens_te)

# TEST graph for normalized-Laplacian diagnostics (FAIR space + optional metric)
X_te_metric = apply_metric(X_te_fair)
W_te = build_W_tau_only(X_te_metric, args.theta, args.tau)
S_te = make_S_norm(W_te)

# Per-node normalized-Laplacian residual diagnostic on LOG preds: |(I-S)f|
inst_if = nlap_residual_vector(S_te, pt_log)
IF_mean = float(inst_if.mean())
IF_median = float(np.median(inst_if))
Lf_te = pt_log - (S_te @ pt_log)
NLAP_energy = float(pt_log @ Lf_te) / max(len(pt_log), 1)
NLAP_energy = max(NLAP_energy, 0.0)

# LIF energy (unnormalized Laplacian energy) on LOG preds
lif = lif_energy(W_te, pt_log, normalize="per_edge")

# group gap + tradeoff
gap_real, gap_abs_real = group_gap_reg(pt_real, sens_te)
gap_log,  gap_abs_log  = group_gap_reg(pt_log,  sens_te)

pred_std_real = float(pt_real.std())
pred_std_log  = float(pt_log.std())

tradeoff_real = gap_abs_real / (pred_std_real + 1e-12)
tradeoff_log  = gap_abs_log  / (pred_std_log  + 1e-12)

# -------------------------
# Gender counterfactual consistency on TEST
# -------------------------
X_full_all = X_full_df.values.astype(np.float32)
X_full_te = X_full_all[idx_te]

if "Gender_bin" not in full_feature_names:
    raise ValueError(f"'Gender_bin' not found in full_feature_names={full_feature_names}")
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
    "METHOD": "N-Lap (L_sym = I - D^{-1/2} W D^{-1/2})",
    # performance (real scale)
    "test_RMSE_real": rmse,
    "test_MAE_real": mae,
    "test_R2_real": r2,
    "test_MSE_real": mse,
    # DP on real scale
    "DP_abs_mean_gap_real": dp_real,
    # group gap + tradeoff
    "group_gap_real_female_minus_male": gap_real,
    "group_gap_abs_real_female_minus_male": gap_abs_real,
    "group_gap_log_female_minus_male": gap_log,
    "group_gap_abs_log_female_minus_male": gap_abs_log,
    "pred_std_real": pred_std_real,
    "pred_std_log": pred_std_log,
    "tradeoff_abs_gap_over_std_real": tradeoff_real,
    "tradeoff_abs_gap_over_std_log": tradeoff_log,
    # individual fairness (log scale)
    "IF_mean(NLAP_residual,log_pred)": IF_mean,
    "IF_median(NLAP_residual,log_pred)": IF_median,
    "NLAP_energy(sym,log_pred)": NLAP_energy,
    "LIF_energy(unorm,log_pred)": lif,
    # hyper / graph info
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
os.makedirs("results", exist_ok=True)
out_path = f"results/nrw/metrics_nlap_depth{args.depth}_lambda{args.lambda_cons}.csv"
pd.DataFrame([out]).to_csv(out_path, index=False)
print(f"[Saved] metrics → {out_path}")

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
            "if_score_nlap_residual_log": inst_if,
        })
        df_if.to_csv(args.save_if_csv, index=False)
        print(f"[Saved] per-instance IF → {args.save_if_csv}")
    except Exception as e:
        print(f"[Warn] Failed to save IF CSV: {e}")
