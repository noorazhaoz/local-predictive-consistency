# run_binary_inproc_urw_bin_unified.py
# Unified URW for binary classification (Adult / COMPAS)
#
# Paper-aligned training:
#   Loss = BCE(y, p) + lambda * mean_i (p_i - q_i)^2   (probability-space squared deviation)
#
# Reference q_i (choose by --agg):
#   - forward: q = P p                 (Forward-KL barycenter, prob averaging)
#   - reverse: q = sigmoid(P logit(p)) (Reverse-KL barycenter, logit averaging)
#
# Instance-level IF on TEST (kept KL, per your preference):
#   - forward: IF_i = KL( Ber(p_i) || Ber(q_i) )
#   - reverse: IF_i = KL( Ber(q_i) || Ber(p_i) )
#
# One script can run one mode (--agg forward/reverse) or both (--run_both).

import argparse, random, os
import numpy as np
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange
from pprint import pprint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, log_loss
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
import pandas as pd


# -------------------------
# Args
# -------------------------
parser = argparse.ArgumentParser(
    description='URW consistency in-processing (binary, Adult/COMPAS, unified forward/reverse reference)'
)
parser.add_argument('--dataset', type=str, default='adult', choices=['adult', 'compas'])
parser.add_argument('--data_dir', type=str, default='../data')

# reference choice
parser.add_argument('--agg', type=str, default='forward', choices=['forward', 'reverse'],
                    help='Neighborhood reference: forward=q=Pp, reverse=q=sigmoid(P logit(p))')
parser.add_argument('--run_both', action='store_true',
                    help='Run both agg modes sequentially (forward then reverse) in one invocation')

parser.add_argument('--depth', type=int, default=1)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--num_iterations', type=int, default=4000)
parser.add_argument('--nloglr', type=float, default=3.0)          # lr = 10^{-nloglr}
parser.add_argument('--lambda_cons', type=float, default=0.3)

# Graph locality (tau-only)
parser.add_argument('--theta', type=float, default=1e-3)
parser.add_argument('--tau', type=float, default=5.0)
parser.add_argument('--max_graph_n', type=int, default=5000)

# Threshold sensitivity analysis
parser.add_argument('--sensitivity_tau', '--sensitivity-tau', action='store_true',
                    dest='sensitivity_tau',
                    help='Run sensitivity analysis over tau values selected from empirical distance quantiles.')
parser.add_argument('--tau_quantiles', type=str, default='5,10,15,20,25',
                    help='Comma-separated distance quantiles used to construct tau grid, e.g. 5,10,15,20,25.')
parser.add_argument('--diagnostic_sample_n', type=int, default=5000,
                    help='Maximum number of points used to estimate distance quantiles and graph diagnostics.')
parser.add_argument('--sensitivity_out', type=str, default='',
                    help='Output CSV path for threshold sensitivity results.')

# Fair subspace
parser.add_argument('--svd_rank', type=int, default=1)

# OPTIONAL: learn a metric from data on fair space (linear Mahalanobis)
parser.add_argument('--use_metric', action='store_true')
parser.add_argument('--metric_rank', type=int, default=64)

# Export IF on TEST
parser.add_argument('--save_if_csv', type=str, default='',
                    help='If set, saves per-instance table for the run. When --run_both, will append suffix.')

args = parser.parse_args()
print(vars(args))

torch.set_num_threads(min(8, torch.get_num_threads()))
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
device = torch.device('cpu')


# -------------------------
# Utils
# -------------------------
def e_dist_sq(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A2 = (A**2).sum(axis=1, keepdims=True)
    B2 = (B**2).sum(axis=1, keepdims=True).T
    return np.maximum(A2 - 2.0 * (A @ B.T) + B2, 0.0)

def build_W_tau_only(X_for_metric: np.ndarray, theta: float, tau: float) -> np.ndarray:
    """
    Dense W with RBF weights and Euclidean-distance threshold tau.
    Self-loop W_ii=1 is added after sparsification.

    Note:
        e_dist_sq returns squared Euclidean distances, so the threshold is tau^2.
    """
    D2 = e_dist_sq(X_for_metric, X_for_metric)
    mask = D2 <= (tau ** 2)

    W = np.exp(-theta * D2, dtype=np.float64) * mask.astype(np.float64)

    # Remove diagonal before symmetrization for graph diagnostics consistency.
    np.fill_diagonal(W, 0.0)
    W = 0.5 * (W + W.T)

    # Add self-loops for row-stochastic P.
    n = W.shape[0]
    W[np.arange(n), np.arange(n)] = 1.0
    return W


def graph_diagnostics_from_W(W: np.ndarray) -> dict:
    """
    Compute graph-coverage diagnostics excluding self-loops.
    This is important because self-loops would otherwise make the isolated-node rate always zero.
    """
    W_no_self = W.copy()
    np.fill_diagonal(W_no_self, 0.0)

    A = (W_no_self > 0).astype(np.int32)
    n = A.shape[0]
    degrees = A.sum(axis=1)

    return {
        "graph_n": int(n),
        "avg_degree": float(np.mean(degrees)),
        "median_degree": float(np.median(degrees)),
        "min_degree": int(np.min(degrees)),
        "max_degree": int(np.max(degrees)),
        "isolated_rate": float(np.mean(degrees == 0)),
        "edge_density": float(A.sum() / max(n * (n - 1), 1)),
    }


def estimate_tau_grid_from_quantiles(
    X_for_metric: np.ndarray,
    quantiles=(5, 10, 15, 20, 25),
    sample_n=5000,
    seed=0,
) -> list:
    """
    Estimate tau values from empirical Euclidean distance quantiles.
    Uses a subsample if the training set is large.
    """
    rng = np.random.RandomState(seed)
    n = X_for_metric.shape[0]

    if n > sample_n:
        idx = rng.choice(n, sample_n, replace=False)
        X_sample = X_for_metric[idx]
    else:
        X_sample = X_for_metric

    D2 = e_dist_sq(X_sample, X_sample)

    # Upper triangular distances, excluding diagonal.
    tri = np.triu_indices_from(D2, k=1)
    dists = np.sqrt(D2[tri])
    dists = dists[np.isfinite(dists)]

    tau_values = np.percentile(dists, quantiles)

    tau_grid = []
    for q, tau in zip(quantiles, tau_values):
        tau_grid.append({
            "tau_quantile": float(q),
            "tau": float(tau),
        })
    return tau_grid

def make_row_stochastic_P(W: np.ndarray) -> csr_matrix:
    rowsum = W.sum(axis=1, keepdims=True) + 1e-12
    return csr_matrix(W / rowsum)

def kl_bernoulli(p: np.ndarray, q: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """Vector KL(Ber(p) || Ber(q)) for p,q in (0,1)."""
    p = np.clip(np.asarray(p).ravel(), eps, 1.0 - eps)
    q = np.clip(np.asarray(q).ravel(), eps, 1.0 - eps)
    return p * np.log(p / q) + (1.0 - p) * np.log((1.0 - p) / (1.0 - q))

def lif_energy(W: np.ndarray, f: np.ndarray, normalize: str = 'per_edge') -> float:
    """Unnormalized Dirichlet energy f^T (D-W) f; here f is probability."""
    d = W.sum(axis=1)
    Lf = d * f - (W @ f)
    E = float(f @ Lf)
    if normalize == 'per_node':
        E /= max(len(f), 1)
    elif normalize == 'per_edge':
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


# -------- IF-based prediction consistency (GR-Con, S-Con) ----------
def get_consistency_pytorch(df_if, model, scaler, device="cpu", raw_prefix="raw_"):
    feat_cols = [
        c for c in df_if.columns
        if c.startswith(raw_prefix)
           and c not in (f"{raw_prefix}y", f"{raw_prefix}orig_index")
    ]
    base_X = df_if[feat_cols].to_numpy().astype(np.float32)
    names = [c[len(raw_prefix):] for c in feat_cols]

    gender_candidates = [n for n in names if n.startswith("sex_")]
    if not gender_candidates:
        raise ValueError(f"No sex_ dummy columns found among names={names}")
    gender_name = "sex_Female" if "sex_Female" in gender_candidates else gender_candidates[0]
    gender_idx = names.index(gender_name)

    race_candidates = [n for n in names if n.startswith("race_")]
    if not race_candidates:
        raise ValueError(f"No race_ dummy columns found among names={names}")
    race_name = "race_White" if "race_White" in race_candidates else race_candidates[0]
    race_idx = names.index(race_name)

    non_sex_indices = [i for i, n in enumerate(names) if not n.startswith("sex_")]

    if hasattr(scaler, "n_features_in_"):
        if len(non_sex_indices) != scaler.n_features_in_:
            raise ValueError(
                f"Feature mismatch: scaler expects {scaler.n_features_in_} features, "
                f"but consistency module produced {len(non_sex_indices)} after dropping sex_*."
            )

    def predict_labels(X_full):
        X_model = X_full[:, non_sex_indices]
        X_std   = scaler.transform(X_model)
        x_t     = torch.tensor(X_std, dtype=torch.float32, device=device)
        with torch.no_grad():
            logits = model(x_t).cpu().numpy().ravel()
        probs = 1 / (1 + np.exp(-logits))
        return (probs >= 0.5).astype(int)

    # GR-Con: flip (gender, race)
    X00 = base_X.copy()
    X00[:, [gender_idx, race_idx]] = 0

    X01 = base_X.copy()
    X01[:, [gender_idx, race_idx]] = 0
    X01[:, gender_idx] = 1

    X10 = base_X.copy()
    X10[:, [gender_idx, race_idx]] = 0
    X10[:, race_idx] = 1

    X11 = base_X.copy()
    X11[:, [gender_idx, race_idx]] = 0
    X11[:, gender_idx] = 1
    X11[:, race_idx] = 1

    y00 = predict_labels(X00)
    y01 = predict_labels(X01)
    y10 = predict_labels(X10)
    y11 = predict_labels(X11)

    gr_consistency = float(np.mean((y00 == y01) & (y00 == y10) & (y00 == y11)))

    # S-Con: Husband ↔ Wife (Adult only; COMPAS -> NaN)
    rel_indices = [i for i, n in enumerate(names) if n.startswith("relationship_")]
    husband_name = "relationship_Husband"
    wife_name    = "relationship_Wife"

    s_consistency = np.nan
    if rel_indices and (husband_name in names) and (wife_name in names):
        husband_idx = names.index(husband_name)
        wife_idx    = names.index(wife_name)

        X_h = base_X.copy()
        X_h[:, rel_indices] = 0
        X_h[:, husband_idx] = 1

        X_w = base_X.copy()
        X_w[:, rel_indices] = 0
        X_w[:, wife_idx] = 1

        y_h = predict_labels(X_h)
        y_w = predict_labels(X_w)
        s_consistency = float(np.mean(y_h == y_w))

    return gr_consistency, s_consistency


# -------------------------
# Data loaders
# -------------------------
def load_adult_clean(data_dir="../data/adult"):
    headers = [
        'age', 'workclass', 'fnlwgt', 'education', 'education-num',
        'marital-status', 'occupation', 'relationship', 'race', 'sex',
        'capital-gain', 'capital-loss', 'hours-per-week',
        'native-country', 'y'
    ]
    train_path = os.path.join(data_dir, "adult.data")
    test_path  = os.path.join(data_dir, "adult.test")
    train = pd.read_csv(train_path, header=None, sep=",", skipinitialspace=True, na_values="?")
    test  = pd.read_csv(test_path,  header=None, sep=",", skipinitialspace=True, na_values="?", comment="|")
    df = pd.concat([train, test], axis=0, ignore_index=True)
    df.columns = headers

    df["y"] = df["y"].astype(str)
    df["y"] = df["y"].str.replace(".", "", regex=False).str.strip()
    df["y"] = df["y"].map({"<=50K": 0, ">50K": 1})
    df = df.dropna(subset=["y"])
    df["y"] = df["y"].astype(int)

    df = df.dropna()
    df = pd.get_dummies(df, drop_first=False)

    native_cols = [c for c in df.columns if c.startswith("native-country_")]
    df = df.drop(columns=native_cols, errors="ignore")

    drop_cols = [c for c in df.columns if c.startswith("fnlwgt") or c.startswith("education_")]
    df = df.drop(columns=drop_cols, errors="ignore")

    df["orig_index"] = np.arange(len(df))
    return df

def load_compas_clean(data_dir="../data/compas"):
    path = os.path.join(data_dir, "compas-scores-two-years.csv")
    df = pd.read_csv(path)
    if "two_year_recid" not in df.columns:
        raise ValueError("COMPAS file missing 'two_year_recid' column.")
    keep_cols = [
        "age", "priors_count", "juv_fel_count", "juv_misd_count", "juv_other_count",
        "c_charge_degree", "sex", "race", "two_year_recid"
    ]
    missing = [c for c in keep_cols if c not in df.columns]
    if missing:
        raise ValueError(f"COMPAS missing columns: {missing}")
    df = df[keep_cols].copy().dropna()

    df["y"] = df["two_year_recid"].astype(int)
    df = df.drop(columns=["two_year_recid"])

    for c in ["age", "priors_count", "juv_fel_count", "juv_misd_count", "juv_other_count"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna()

    cat_cols = ["c_charge_degree", "sex", "race"]
    df = pd.get_dummies(df, columns=cat_cols, drop_first=False)
    df["orig_index"] = np.arange(len(df))
    return df


# -------------------------
# Model
# -------------------------
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
        logit = self.out(h).squeeze(1)
        return logit


# -------------------------
# Core runner for one mode
# -------------------------
def run_one_mode(
    agg_mode: str,
    save_if_csv_path: str,
    tau_override: float = None,
    tau_quantile: float = None,
    sensitivity_mode: bool = False,
):
    assert agg_mode in ("forward", "reverse")
    tau_used = float(args.tau if tau_override is None else tau_override)

    # Re-seed per run for comparability when --run_both
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Load data
    if args.dataset == "adult":
        df = load_adult_clean(os.path.join(args.data_dir, "adult"))
    else:
        df = load_compas_clean(os.path.join(args.data_dir, "compas"))

    y_full = df["y"].astype(int).values
    orig_index_full = df["orig_index"].values

    sex_cols = [c for c in df.columns if c.startswith("sex_")]
    print("\nsex dummy columns:", sex_cols)

    if "sex_Female" in df.columns:
        sens_full = df["sex_Female"].astype(int).values
    elif "sex_Male" in df.columns:
        sens_full = 1 - df["sex_Male"].astype(int).values
    else:
        raise ValueError(f"Cannot find sex dummy column among: {sex_cols}")

    drop_for_X = [c for c in df.columns if c.startswith("sex_")] + ["y", "orig_index"]
    X = df.drop(columns=drop_for_X).astype(np.float32).values

    X_tr, X_tmp, y_tr, y_tmp, sens_tr, sens_tmp, rid_tr, rid_tmp = train_test_split(
        X, y_full, sens_full, orig_index_full,
        test_size=0.4, random_state=args.seed, stratify=y_full
    )
    X_va, X_te, y_va, y_te, sens_va, sens_te, rid_va, rid_te = train_test_split(
        X_tmp, y_tmp, sens_tmp, rid_tmp,
        test_size=0.5, random_state=args.seed, stratify=y_tmp
    )

    scaler = StandardScaler()
    scaler.fit(X_tr)
    X_tr = scaler.transform(X_tr).astype(np.float32)
    X_va = scaler.transform(X_va).astype(np.float32)
    X_te = scaler.transform(X_te).astype(np.float32)

    n_tr, d = X_tr.shape
    print(f"\n[Setup] dataset={args.dataset} | agg={agg_mode} | num_train={n_tr}, num_val={X_va.shape[0]}, num_test={X_te.shape[0]}, d={d}")

    # Fair subspace (train-only prototypes)
    mu0 = X_tr[sens_tr == 0].mean(axis=0, keepdims=True)
    mu1 = X_tr[sens_tr == 1].mean(axis=0, keepdims=True)
    prot = np.vstack([mu0, mu1]).astype(np.float32)

    svd_rank_used = min(args.svd_rank, prot.shape[0], d)
    svd = TruncatedSVD(n_components=svd_rank_used, random_state=args.seed)
    svd.fit(prot)
    basis = svd.components_.T

    proj = basis @ np.linalg.pinv(basis.T @ basis) @ basis.T
    proj_compl = np.eye(d, dtype=np.float32) - proj

    # Optional metric learning in fair space
    X_tr_fair_full = X_tr @ proj_compl
    fair_dim = X_tr_fair_full.shape[1]
    L_metric = None
    if args.use_metric:
        r_metric = min(args.metric_rank, fair_dim, max(X_tr_fair_full.shape[0] - 1, 1))
        svd_metric = TruncatedSVD(n_components=r_metric, random_state=args.seed)
        svd_metric.fit(X_tr_fair_full)
        L_metric = svd_metric.components_

    def apply_metric(X_fair: np.ndarray) -> np.ndarray:
        if L_metric is None:
            return X_fair
        return X_fair @ L_metric.T

    # Build TRAIN graph on subset
    if n_tr > args.max_graph_n:
        idx_graph = np.random.RandomState(args.seed).choice(n_tr, args.max_graph_n, replace=False)
    else:
        idx_graph = np.arange(n_tr)
    idx_graph_t = torch.tensor(idx_graph, dtype=torch.long, device=device)

    X_tr_g_fair = (X_tr @ proj_compl)[idx_graph]
    X_tr_g_metric = apply_metric(X_tr_g_fair)

    W_tr = build_W_tau_only(X_tr_g_metric, args.theta, tau_used)
    graph_diag_train = graph_diagnostics_from_W(W_tr)
    print({
        "tau_quantile": tau_quantile,
        "tau_used": tau_used,
        "graph_train_avg_degree": graph_diag_train["avg_degree"],
        "graph_train_isolated_rate": graph_diag_train["isolated_rate"],
        "graph_train_edge_density": graph_diag_train["edge_density"],
    })

    P_tr = make_row_stochastic_P(W_tr).tocoo()
    n_g = P_tr.shape[0]

    idx = torch.tensor(np.vstack([P_tr.row, P_tr.col]), dtype=torch.long, device=device)
    val = torch.tensor(P_tr.data, dtype=torch.float32, device=device)
    P_tr_sp = torch.sparse_coo_tensor(idx, val, torch.Size(P_tr.shape), device=device)

    # Model
    model = MLPBin(d_in=d, hidden=256, depth=args.depth).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=10 ** (-args.nloglr))

    Xtr_t = torch.tensor(X_tr, dtype=torch.float32, device=device)
    ytr_t = torch.tensor(y_tr, dtype=torch.float32, device=device)
    Xva_t = torch.tensor(X_va, dtype=torch.float32, device=device)
    yva_t = torch.tensor(y_va, dtype=torch.float32, device=device)
    Xte_t = torch.tensor(X_te, dtype=torch.float32, device=device)

    eps = 1e-7

    if agg_mode == "forward":
        print("\n[Training] Forward-KL reference: q = Pp")
    else:
        print("\n[Training] Reverse-KL reference: q = sigmoid(P logit(p))")
    print("[Training] Loss = BCE(y,p) + lambda * mean_i (p_i - q_i)^2   (paper-aligned)")

    # Train
    for it in trange(args.num_iterations):
        model.train()
        logits_tr = model(Xtr_t)
        p_tr = torch.sigmoid(logits_tr)

        task_loss = F.binary_cross_entropy_with_logits(logits_tr, ytr_t)

        # compute q on graph subset
        p_g = p_tr[idx_graph_t]

        if agg_mode == "forward":
            q_g = torch.sparse.mm(P_tr_sp, p_g.unsqueeze(1)).squeeze(1)
        else:
            p_g_cl = p_g.clamp(eps, 1.0 - eps)
            logit_p_g = torch.log(p_g_cl / (1.0 - p_g_cl))
            logit_q_g = torch.sparse.mm(P_tr_sp, logit_p_g.unsqueeze(1)).squeeze(1)
            q_g = torch.sigmoid(logit_q_g)

        # Paper penalty: squared deviation in probability space
        cons_loss = torch.mean((p_g - q_g) ** 2)

        loss = task_loss + args.lambda_cons * cons_loss
        optim.zero_grad()
        loss.backward()
        optim.step()

        if (it + 1) % 200 == 0 or it == 0:
            model.eval()
            with torch.no_grad():
                logits_va = model(Xva_t)
                p_va = torch.sigmoid(logits_va).cpu().numpy()

            acc = accuracy_score(y_va, (p_va >= 0.5).astype(int))
            bal_acc = balanced_accuracy_score(y_va, (p_va >= 0.5).astype(int))
            auc = roc_auc_score(y_va, p_va)
            ce = log_loss(y_va, p_va, eps=1e-7)

            print({
                'iter': it + 1,
                'train_CE': float(task_loss.item()),
                'train_cons_sqdev_prob': float(cons_loss.item()),
                'val_acc': acc,
                'val_bal_acc': bal_acc,
                'val_auc': auc,
                'val_logloss': ce,
            })

    # TEST eval
    print("\n[Evaluation] TEST")
    model.eval()
    with torch.no_grad():
        logits_te = model(Xte_t).cpu().numpy()
    pt_prob = 1.0 / (1.0 + np.exp(-logits_te))

    pred_label = (pt_prob >= 0.5).astype(int)
    acc_te = accuracy_score(y_te, pred_label)
    auc_te = roc_auc_score(y_te, pt_prob)
    bal_acc_te = balanced_accuracy_score(y_te, pred_label)
    ce_te = log_loss(y_te, pt_prob, eps=1e-7)

    # TEST graph for IF/LIF (graph built in same fair space)
    X_te_fair = X_te @ proj_compl
    X_te_metric = apply_metric(X_te_fair)
    W_te = build_W_tau_only(X_te_metric, args.theta, tau_used)
    graph_diag_test = graph_diagnostics_from_W(W_te)
    P_te = make_row_stochastic_P(W_te)

    p_np = pt_prob.ravel()

    # (A) Unified URW-monitor IF for ALL modes: KL(p || Pp)
    q_f = P_te.dot(p_np)
    inst_if_urw = kl_bernoulli(p_np, q_f, eps=eps)  # KL(p||Pp)

    if agg_mode == "forward":
        # forward's method-specific IF is the same as URW-monitor
        inst_if = inst_if_urw
        if_name = "if_score_forwardKL_node||ref"
        method_name = "URW Forward-KL ref (q=Pp) + sqdev penalty; IF=KL(p||Pp)"
    else:
        # (B) Reverse-ref method-specific IF: KL(q_rev || p)
        p_cl = np.clip(p_np, eps, 1.0 - eps)
        logit_p = np.log(p_cl / (1.0 - p_cl))
        logit_q = P_te.dot(logit_p)
        q_rev = 1.0 / (1.0 + np.exp(-logit_q))
        q_rev = np.clip(q_rev, eps, 1.0 - eps)

        inst_if = kl_bernoulli(q_rev, p_np, eps=eps)  # KL(q_rev||p)
        if_name = "if_score_reverseKL_ref||node"
        method_name = "URW Reverse-KL ref (q=sigmoid(Plogit(p))) + sqdev penalty; IF=KL(q_rev||p)"

    IF_mean = float(inst_if.mean())
    IF_median = float(np.median(inst_if))

    IF_mean_urw = float(inst_if_urw.mean())
    IF_median_urw = float(np.median(inst_if_urw))
    lif = lif_energy(W_te, p_np, normalize='per_edge')

    dp = demographic_parity_prob(pt_prob, sens_te)

    # Reconstruct TEST raw rows
    df_test_raw = df[df["orig_index"].isin(rid_te)].copy()
    df_test_raw = df_test_raw.set_index("orig_index").loc[rid_te].reset_index()
    df_test_raw = df_test_raw.add_prefix("raw_")

    df_pred_dict = {
        'orig_index': rid_te,
        'y_true': y_te,
        'pred_prob': p_np,
        'pred_label': pred_label,
        'abs_error_0_1': np.abs(y_te - pred_label),
        'if_score_urw_monitor_KL_p||Pp': inst_if_urw,  # <-- NEW (always)
        'sens': sens_te,
    }
    # method-specific IF column (forward: same as monitor; reverse: different)
    df_pred_dict[if_name] = inst_if

    df_pred = pd.DataFrame(df_pred_dict)
    df_out_full = pd.concat([df_pred, df_test_raw], axis=1)

    # Gaps
    gap_g_rms, gap_g_max = group_gap_stats_prob(pt_prob, sens_te)

    race_cols = [c for c in df_out_full.columns if c.startswith("raw_race_")]
    if len(race_cols) == 0:
        raise ValueError("No raw_race_* columns found in df_out_full, cannot compute race gaps.")
    race_onehot = df_out_full[race_cols].to_numpy().astype(int)

    row_sum = race_onehot.sum(axis=1)
    valid = (row_sum == 1)
    if not np.any(valid):
        raise ValueError("All rows have invalid race one-hot (no valid rows).")
    if not np.all(valid):
        race_group_valid = race_onehot[valid].argmax(axis=1)
        gap_r_rms, gap_r_max = group_gap_stats_prob(p_np[valid], race_group_valid)
    else:
        race_group = race_onehot.argmax(axis=1)
        gap_r_rms, gap_r_max = group_gap_stats_prob(p_np, race_group)

    print(f"DP(pred_prob; gender): {dp:.6f}")
    print(f"Gap_G_RMS: {gap_g_rms:.6f}, Gap_G_max: {gap_g_max:.6f}")
    print(f"Gap_R_RMS: {gap_r_rms:.6f}, Gap_R_max: {gap_r_max:.6f}")

    # GR/S consistency (counterfactual label consistency)
    gr_con, s_con = get_consistency_pytorch(
        df_out_full,
        model=model,
        scaler=scaler,
        device=device,
        raw_prefix="raw_",
    )
    print(f"GR-Con (gender+race): {gr_con:.4f}")
    print(f"S-Con  (Husband vs Wife): {s_con if not np.isnan(s_con) else 'NaN'}")

    out = {
        'DATASET': args.dataset,
        'METHOD': method_name,
        'agg_mode': agg_mode,
        'depth': int(args.depth),

        'test_acc': float(acc_te),
        'test_bal_acc': float(bal_acc_te),
        'test_auc': float(auc_te),
        'test_logloss': float(ce_te),

        'DP(pred_prob; gender)': float(dp),

        'Gap_G_RMS': float(gap_g_rms),
        'Gap_G_max': float(gap_g_max),
        'Gap_R_RMS': float(gap_r_rms),
        'Gap_R_max': float(gap_r_max),

        'IF_mean(method,KL,prob)': float(IF_mean),
        'IF_median(method,KL,prob)': float(IF_median),

        'IF_mean(URW_monitor,KL_p||Pp)': float(IF_mean_urw),
        'IF_median(URW_monitor,KL_p||Pp)': float(IF_median_urw),

        'LIF_energy(unorm,prob)': float(lif),
        'GR_Consistency': float(gr_con),
        'S_Consistency': float(s_con) if not np.isnan(s_con) else np.nan,

        'lambda_cons': float(args.lambda_cons),
        'tau_used': float(tau_used),
        'tau_quantile': float(tau_quantile) if tau_quantile is not None else np.nan,
        'theta': float(args.theta),
        'graph_train_subset': int(n_g),

        'graph_train_avg_degree': graph_diag_train["avg_degree"],
        'graph_train_median_degree': graph_diag_train["median_degree"],
        'graph_train_min_degree': graph_diag_train["min_degree"],
        'graph_train_max_degree': graph_diag_train["max_degree"],
        'graph_train_isolated_rate': graph_diag_train["isolated_rate"],
        'graph_train_edge_density': graph_diag_train["edge_density"],

        'graph_test_avg_degree': graph_diag_test["avg_degree"],
        'graph_test_median_degree': graph_diag_test["median_degree"],
        'graph_test_min_degree': graph_diag_test["min_degree"],
        'graph_test_max_degree': graph_diag_test["max_degree"],
        'graph_test_isolated_rate': graph_diag_test["isolated_rate"],
        'graph_test_edge_density': graph_diag_test["edge_density"],
        'svd_rank': int(svd_rank_used),
        'use_metric': bool(args.use_metric),
        'metric_rank': int(L_metric.shape[0]) if (args.use_metric and L_metric is not None) else 0,
    }
    pprint(out)

    # Save results (separate per mode)
    os.makedirs("results_unified", exist_ok=True)
    tau_tag = f"_tauq{int(tau_quantile)}" if tau_quantile is not None else ""
    metrics_path = f"results_unified/metrics_urw_{args.dataset}_agg{agg_mode}_depth{args.depth}_lambda{args.lambda_cons}{tau_tag}.csv"
    pd.DataFrame([out]).to_csv(metrics_path, index=False)
    print(f"[Saved] Metrics CSV → {metrics_path}")

    model_path = f"results_unified/urw_model_{args.dataset}_agg{agg_mode}_depth{args.depth}_lambda{args.lambda_cons}{tau_tag}.pt"
    torch.save({"state_dict": model.state_dict(), "input_dim": d, "hidden": 256, "depth": args.depth}, model_path)
    print(f"[Saved] Model → {model_path}")

    scaler_path = f"results_unified/urw_scaler_{args.dataset}_agg{agg_mode}_depth{args.depth}_lambda{args.lambda_cons}{tau_tag}.pkl"
    joblib.dump(scaler, scaler_path)
    print(f"[Saved] Scaler → {scaler_path}")

    if save_if_csv_path:
        try:
            out_dir = os.path.dirname(save_if_csv_path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            df_out_full.to_csv(save_if_csv_path, index=False)
            print(f"[Saved] Full IF CSV with raw data → {save_if_csv_path}")
        except Exception as e:
            print(f"[Warn] Failed to save IF CSV: {e}")

    return out


def prepare_metric_space_for_tau_grid():
    """
    Reconstruct the projected metric space used for graph construction,
    only for estimating tau quantiles before running sensitivity experiments.

    This mirrors the preprocessing and fair-projection steps in run_one_mode.
    """
    if args.dataset == "adult":
        df = load_adult_clean(os.path.join(args.data_dir, "adult"))
    else:
        df = load_compas_clean(os.path.join(args.data_dir, "compas"))

    y_full = df["y"].astype(int).values
    orig_index_full = df["orig_index"].values

    if "sex_Female" in df.columns:
        sens_full = df["sex_Female"].astype(int).values
    elif "sex_Male" in df.columns:
        sens_full = 1 - df["sex_Male"].astype(int).values
    else:
        sex_cols = [c for c in df.columns if c.startswith("sex_")]
        raise ValueError(f"Cannot find sex dummy column among: {sex_cols}")

    drop_for_X = [c for c in df.columns if c.startswith("sex_")] + ["y", "orig_index"]
    X = df.drop(columns=drop_for_X).astype(np.float32).values

    X_tr, X_tmp, y_tr, y_tmp, sens_tr, sens_tmp, rid_tr, rid_tmp = train_test_split(
        X, y_full, sens_full, orig_index_full,
        test_size=0.4, random_state=args.seed, stratify=y_full
    )

    scaler = StandardScaler()
    scaler.fit(X_tr)
    X_tr = scaler.transform(X_tr).astype(np.float32)

    n_tr, d = X_tr.shape

    # Fair subspace, same as run_one_mode.
    mu0 = X_tr[sens_tr == 0].mean(axis=0, keepdims=True)
    mu1 = X_tr[sens_tr == 1].mean(axis=0, keepdims=True)
    prot = np.vstack([mu0, mu1]).astype(np.float32)

    svd_rank_used = min(args.svd_rank, prot.shape[0], d)
    svd = TruncatedSVD(n_components=svd_rank_used, random_state=args.seed)
    svd.fit(prot)
    basis = svd.components_.T

    proj = basis @ np.linalg.pinv(basis.T @ basis) @ basis.T
    proj_compl = np.eye(d, dtype=np.float32) - proj

    X_tr_fair_full = X_tr @ proj_compl

    # Optional metric learning in fair space, same as run_one_mode.
    if args.use_metric:
        fair_dim = X_tr_fair_full.shape[1]
        r_metric = min(args.metric_rank, fair_dim, max(X_tr_fair_full.shape[0] - 1, 1))
        svd_metric = TruncatedSVD(n_components=r_metric, random_state=args.seed)
        svd_metric.fit(X_tr_fair_full)
        L_metric = svd_metric.components_
        X_for_metric = X_tr_fair_full @ L_metric.T
    else:
        X_for_metric = X_tr_fair_full

    return X_for_metric


def run_tau_sensitivity():
    quantiles = [float(q.strip()) for q in args.tau_quantiles.split(",") if q.strip()]

    X_for_metric = prepare_metric_space_for_tau_grid()

    tau_grid = estimate_tau_grid_from_quantiles(
        X_for_metric,
        quantiles=quantiles,
        sample_n=args.diagnostic_sample_n,
        seed=args.seed,
    )

    print("\n[Tau sensitivity] Candidate thresholds:")
    pprint(tau_grid)

    all_results = []
    modes = ["forward", "reverse"] if args.run_both else [args.agg]

    for mode in modes:
        for item in tau_grid:
            q = item["tau_quantile"]
            tau = item["tau"]

            print("\n" + "=" * 80)
            print(f"[Tau sensitivity] dataset={args.dataset}, agg={mode}, q={q}, tau={tau:.6f}")
            print("=" * 80)

            if args.save_if_csv:
                root, ext = os.path.splitext(args.save_if_csv)
                save_path = f"{root}_agg{mode}_tauq{int(q)}{ext if ext else '.csv'}"
            else:
                save_path = ""

            out = run_one_mode(
                agg_mode=mode,
                save_if_csv_path=save_path,
                tau_override=tau,
                tau_quantile=q,
                sensitivity_mode=True,
            )
            all_results.append(out)

    df_sens = pd.DataFrame(all_results)

    os.makedirs("results_unified", exist_ok=True)
    if args.sensitivity_out:
        out_path = args.sensitivity_out
    else:
        mode_tag = "both" if args.run_both else args.agg
        out_path = (
            f"results_unified/tau_sensitivity_urw_{args.dataset}_"
            f"agg{mode_tag}_depth{args.depth}_lambda{args.lambda_cons}.csv"
        )

    df_sens.to_csv(out_path, index=False)
    print(f"\n[Saved] Tau sensitivity results → {out_path}")

    return df_sens


# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    print("DEBUG final sensitivity_tau =", args.sensitivity_tau)
    print("DEBUG final use_metric =", args.use_metric)

    if args.sensitivity_tau:
        print("DEBUG: entering tau sensitivity mode")
        run_tau_sensitivity()
    else:
        print("DEBUG: entering normal single-tau mode")
        if args.run_both:
            # run forward then reverse
            for mode in ["forward", "reverse"]:
                if args.save_if_csv:
                    root, ext = os.path.splitext(args.save_if_csv)
                    save_path = f"{root}_agg{mode}{ext if ext else '.csv'}"
                else:
                    save_path = ""
                run_one_mode(mode, save_path)
        else:
            run_one_mode(args.agg, args.save_if_csv)
