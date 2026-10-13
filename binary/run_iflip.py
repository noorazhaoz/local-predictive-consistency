#!/usr/bin/env python3
# run_binary_inproc_urw_adult.py
#
# Two modes in ONE script:
#   (A) URW in-processing (your original): CE + lambda * forward-KL consistency
#   (B) iFlipper pre-processing baseline: rectify TRAIN labels on the SAME train graph,
#       then train with CE ONLY (no consistency term)
#
# Usage examples:
#   1) URW (original):
#      python run_binary_inproc_urw_adult.py --lambda_cons 0.3
#
#   2) iFlipper baseline (pre-processing):
#      python run_binary_inproc_urw_adult.py --use_iflipper --rho_if 0.9
#
# Notes:
# - iFlipper is applied on the TRAIN graph subset (idx_graph) used to build W_tr.
# - It modifies ONLY y_tr on that subset, then trains the same MLP with BCE only.
# - TEST evaluation (DP/Gap/IF/LIF/GR-Con/S-Con) is identical across modes.
#
# Requirements for iFlipper:
# - Your iFlipper repo folder (with iFlipper/iflipper.py etc.) must be in PYTHONPATH
#   or the same directory as this script.
# - MOSEK license if iFlipper uses mosek_solver by default.

from __future__ import absolute_import, division, print_function

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
# OPTIONAL: iFlipper imports
# -------------------------
# If import fails, you can still run URW mode.
try:
    from iFlipper.iflipper import iFlipper as IFlipper
    from iFlipper.utils import measure_error
    _HAS_IFLIPPER = True
except Exception as e:
    _HAS_IFLIPPER = False
    _IFLIPPER_IMPORT_ERR = e


# -------------------------
# Args
# -------------------------
parser = argparse.ArgumentParser(
    description='URW consistency in-processing (binary, Adult, forward KL) + iFlipper pre-processing baseline'
)
parser.add_argument('--depth', type=int, default=1,
                    help='number of hidden layers in MLP (e.g., 1, 2, 3)')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--num_iterations', type=int, default=4000)
parser.add_argument('--nloglr', type=float, default=3.0)          # lr = 10^{-nloglr}
parser.add_argument('--lambda_cons', type=float, default=0.3)

# Graph locality (tau-only)
parser.add_argument('--theta', type=float, default=1e-3)          # RBF scale
parser.add_argument('--tau', type=float, default=5.0)             # distance^2 threshold
parser.add_argument('--max_graph_n', type=int, default=5000,
                    help='Max #train pts to build W on (subset)')

# Pseudo sensitive attribute (kept for consistency, but we use gender here)
parser.add_argument('--sens_col', type=int, default=0,
                    help='Which standardized feature column to median-split (unused)')
parser.add_argument('--svd_rank', type=int, default=1,
                    help='#components for sensitive subspace SVD (1~3 OK)')

# OPTIONAL: learn a metric from data on fair space (linear Mahalanobis)
parser.add_argument('--use_metric', action='store_true')
parser.add_argument('--metric_rank', type=int, default=64)

# Export IF on TEST
parser.add_argument('--save_if_csv', type=str, default='')

# iFlipper baseline switches
parser.add_argument('--use_iflipper', action='store_true',
                    help='Run iFlipper pre-processing baseline: rectify TRAIN labels on train graph, train with BCE only.')
parser.add_argument('--rho_if', type=float, default=0.9,
                    help='iFlipper budget ratio: m = rho * err0 on train-graph labels.')

parser.add_argument('--dataset', type=str, default='adult', choices=['adult', 'compas'],
                    help='Which dataset to run: adult or compas')
parser.add_argument('--data_dir', type=str, default='../data/adult',
                    help='Dataset directory. For compas expects compas-scores-two-years.csv inside.')

args = parser.parse_args()
print(vars(args))

if args.use_iflipper and not _HAS_IFLIPPER:
    raise ImportError(
        f"Failed to import iFlipper modules. Error: {_IFLIPPER_IMPORT_ERR}\n"
        f"Fix by ensuring the iFlipper repo folder is on PYTHONPATH, e.g.\n"
        f"  export PYTHONPATH=/path/to/repo:$PYTHONPATH\n"
        f"or run this script from the parent directory that contains iFlipper/."
    )

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
    Dense W with RBF weights and tau threshold (UNnormalized),
    with explicit self-loop W_ii = 1.
    """
    D2 = e_dist_sq(X_for_metric, X_for_metric)
    W = np.exp(-theta * D2, dtype=np.float64) * (D2 <= tau).astype(np.float64)

    np.fill_diagonal(W, 0.0)
    W = 0.5 * (W + W.T)       # symmetrize
    n = W.shape[0]
    W[np.arange(n), np.arange(n)] = 1.0  # self-loop
    return W

def make_row_stochastic_P(W: np.ndarray) -> csr_matrix:
    rowsum = W.sum(axis=1, keepdims=True) + 1e-12
    return csr_matrix(W / rowsum)

def IF_cons_vector_forward_kl(P: csr_matrix, yhat: np.ndarray) -> np.ndarray:
    """
    Instance-level IF for URW with forward KL:
    IF_i = KL( Ber(p_i) || Ber(q_i) ),
    where q_i = sum_j P_ij p_j is barycenter of neighbours.
    """
    nbr = P.dot(yhat)   # q_i
    eps = 1e-7
    p = np.clip(yhat, eps, 1.0 - eps)
    q = np.clip(nbr,  eps, 1.0 - eps)
    return p * np.log(p / q) + (1.0 - p) * np.log((1.0 - p) / (1.0 - q))

def violation_rates_from_if(inst_if: np.ndarray, eps_list):
    """
    VR(eps) = mean(IF_i > eps)
    """
    inst_if = np.asarray(inst_if).ravel()
    return {f"VR({eps})": float(np.mean(inst_if > eps)) for eps in eps_list}

def lif_energy(W: np.ndarray, f: np.ndarray, normalize: str = 'per_edge') -> float:
    """Unnormalized Dirichlet energy f^T (D-W) f; here f is probability."""
    d = W.sum(axis=1)
    Lf = d * f - (W @ f)
    E = float(f @ Lf)  # f^T (D-W) f
    if normalize == 'per_node':
        E /= len(f)
    elif normalize == 'per_edge':
        m = int((W > 0).sum())
        E /= max(m, 1)
    return E

def demographic_parity_prob(y_prob: np.ndarray, sens_bin: np.ndarray) -> float:
    """
    Demographic parity difference on predicted probabilities:
    DP = | E[p | sens=1] - E[p | sens=0] |.
    """
    m1 = y_prob[sens_bin == 1].mean()
    m0 = y_prob[sens_bin == 0].mean()
    return float(abs(m1 - m0))

def group_gap_stats_prob(y_prob: np.ndarray, group: np.ndarray):
    """
    Group-mean gaps w.r.t global mean:
      gap_rms = sqrt(mean_k (mu_k - mu)^2)
      gap_max = max_k |mu_k - mu|
    Works for binary or multi-group attributes.
    """
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

def W_to_edge_wedge(W: np.ndarray, thr: float = 0.0):
    """
    Convert dense symmetric W into:
      edge: list of (i,j) with i<j and W_ij>thr
      w_edge: np.ndarray aligned with edge
    (Diagonal is ignored automatically.)
    """
    n = W.shape[0]
    edge = []
    w_edge = []
    for i in range(n):
        js = np.where(W[i, i+1:] > thr)[0]
        for off in js:
            j = i + 1 + int(off)
            edge.append((i, j))
            w_edge.append(float(W[i, j]))
    return edge, np.array(w_edge, dtype=np.float64)


# -------- IF-based prediction consistency (GR-Con, S-Con) ----------
def get_consistency_pytorch(
    df_if: pd.DataFrame,
    model: torch.nn.Module,
    scaler,
    device="cpu",
    raw_prefix="raw_",
    threshold: float = 0.5,
    prefer_race_names=("race_White", "race_Caucasian"),
    prefer_sex_names=("sex_Female", "sex_Male"),
):
    """
    Compute:
      - GR-Con: counterfactual consistency under (sex, race) changes.
      - S-Con : counterfactual consistency under (relationship Husband <-> Wife) if available, else NaN.

    Assumptions:
      - df_if contains raw one-hot columns prefixed by raw_prefix, plus raw_y/raw_orig_index etc.
      - model was trained on features that DROPPED sex_* columns (as in your script).
      - scaler was fit on that same model input feature space.
    """

    # -------- 1) collect raw feature columns --------
    feat_cols = [
        c for c in df_if.columns
        if c.startswith(raw_prefix)
        and c not in (f"{raw_prefix}y", f"{raw_prefix}orig_index")
    ]
    if len(feat_cols) == 0:
        raise ValueError(f"No raw feature columns found with prefix='{raw_prefix}'")

    base_X = df_if[feat_cols].to_numpy().astype(np.float32)
    names = [c[len(raw_prefix):] for c in feat_cols]  # strip prefix

    # indices by group
    sex_idxs  = [i for i, n in enumerate(names) if n.startswith("sex_")]
    race_idxs = [i for i, n in enumerate(names) if n.startswith("race_")]

    if len(sex_idxs) == 0:
        raise ValueError("No sex_* one-hot columns found in raw features. Cannot compute GR-Con.")
    if len(race_idxs) == 0:
        raise ValueError("No race_* one-hot columns found in raw features. Cannot compute GR-Con.")

    # model input: drop sex_* (consistent with your training)
    non_sex_indices = [i for i in range(len(names)) if i not in sex_idxs]

    # sanity check scaler dimension
    if hasattr(scaler, "n_features_in_"):
        if len(non_sex_indices) != scaler.n_features_in_:
            raise ValueError(
                f"Feature mismatch: scaler expects {scaler.n_features_in_} features, "
                f"but after dropping sex_* we have {len(non_sex_indices)}."
            )

    # -------- 2) helper: predict labels for full raw feature matrix --------
    def predict_labels(X_full: np.ndarray) -> np.ndarray:
        X_model = X_full[:, non_sex_indices]
        X_std = scaler.transform(X_model)
        x_t = torch.tensor(X_std, dtype=torch.float32, device=device)
        model.eval()
        with torch.no_grad():
            logits = model(x_t).detach().cpu().numpy().ravel()
        probs = 1.0 / (1.0 + np.exp(-logits))
        return (probs >= threshold).astype(int)

    # -------- 3) helper: choose a "target" index to set within a group --------
    def pick_preferred_index(candidates, preferred_names):
        # choose by name if possible
        for pn in preferred_names:
            if pn in candidates:
                return candidates.index(pn)
        return 0  # fallback

    sex_names  = [names[i] for i in sex_idxs]
    race_names = [names[i] for i in race_idxs]

    # preferred indices inside the *group list*
    sex_pref_pos  = pick_preferred_index(sex_names,  prefer_sex_names)
    race_pref_pos = pick_preferred_index(race_names, prefer_race_names)

    sex_pref_idx  = sex_idxs[sex_pref_pos]
    race_pref_idx = race_idxs[race_pref_pos]

    # Another sex idx (for binary flip); if only one dummy, we'll just toggle 0/1 on that column
    if len(sex_idxs) >= 2:
        # choose an alternative sex column different from preferred
        sex_alt_idx = sex_idxs[0] if sex_idxs[0] != sex_pref_idx else sex_idxs[1]
    else:
        sex_alt_idx = sex_pref_idx

    # For race: if multi-class, we can flip to preferred; also optionally flip to an alternative
    if len(race_idxs) >= 2:
        race_alt_idx = race_idxs[0] if race_idxs[0] != race_pref_idx else race_idxs[1]
    else:
        race_alt_idx = race_pref_idx

    # -------- 4) build counterfactual variants for GR-Con --------
    # base: zero out sex and race, then set desired ones
    def make_variant(set_sex: str, set_race: str) -> np.ndarray:
        """
        set_sex:  'as_is' | 'pref' | 'alt' | 'flip'
        set_race: 'as_is' | 'pref' | 'alt'
        """
        Xv = base_X.copy()

        # store original argmax (if one-hot valid)
        # if rows are all-zero (rare), argmax returns 0; we handle gracefully
        sex_orig_pos  = np.argmax(Xv[:, sex_idxs], axis=1) if len(sex_idxs) > 0 else None
        race_orig_pos = np.argmax(Xv[:, race_idxs], axis=1) if len(race_idxs) > 0 else None

        # clear group one-hots
        Xv[:, sex_idxs] = 0
        Xv[:, race_idxs] = 0

        # set sex
        if set_sex == "as_is":
            # restore original sex position
            Xv[np.arange(len(Xv)), np.array(sex_idxs)[sex_orig_pos]] = 1
        elif set_sex == "pref":
            Xv[:, sex_pref_idx] = 1
        elif set_sex == "alt":
            Xv[:, sex_alt_idx] = 1
        elif set_sex == "flip":
            if len(sex_idxs) >= 2:
                # flip to the other column: if orig was pref -> alt, else -> pref
                orig_global = np.array(sex_idxs)[sex_orig_pos]
                target = np.where(orig_global == sex_pref_idx, sex_alt_idx, sex_pref_idx)
                Xv[np.arange(len(Xv)), target] = 1
            else:
                # only one dummy column: toggle 0/1 on that column
                # (since we cleared it to 0 above, "flip" -> set to 1 always;
                # but to be meaningful, we instead restore orig then invert)
                Xv[:, sex_pref_idx] = 1 - base_X[:, sex_pref_idx]
        else:
            raise ValueError(f"Unknown set_sex={set_sex}")

        # set race
        if set_race == "as_is":
            Xv[np.arange(len(Xv)), np.array(race_idxs)[race_orig_pos]] = 1
        elif set_race == "pref":
            Xv[:, race_pref_idx] = 1
        elif set_race == "alt":
            Xv[:, race_alt_idx] = 1
        else:
            raise ValueError(f"Unknown set_race={set_race}")

        return Xv

    # We evaluate 4 combos similar to your old code, but robust:
    #   (sex as-is, race as-is)
    #   (sex flipped, race as-is)
    #   (sex as-is, race pref)
    #   (sex flipped, race pref)
    # This works for binary/multi race; you can add more combos if you want.
    X00 = make_variant(set_sex="as_is", set_race="as_is")
    X01 = make_variant(set_sex="flip",  set_race="as_is")
    X10 = make_variant(set_sex="as_is", set_race="pref")
    X11 = make_variant(set_sex="flip",  set_race="pref")

    y00 = predict_labels(X00)
    y01 = predict_labels(X01)
    y10 = predict_labels(X10)
    y11 = predict_labels(X11)

    gr_consistency = float(np.mean((y00 == y01) & (y00 == y10) & (y00 == y11)))

    # -------- 5) S-Con: Husband <-> Wife if present (Adult only) --------
    rel_idxs = [i for i, n in enumerate(names) if n.startswith("relationship_")]
    husband_name = "relationship_Husband"
    wife_name = "relationship_Wife"

    if (husband_name in names) and (wife_name in names) and (len(rel_idxs) > 0):
        husband_idx = names.index(husband_name)
        wife_idx = names.index(wife_name)

        X_h = base_X.copy()
        X_h[:, rel_idxs] = 0
        X_h[:, husband_idx] = 1

        X_w = base_X.copy()
        X_w[:, rel_idxs] = 0
        X_w[:, wife_idx] = 1

        y_h = predict_labels(X_h)
        y_w = predict_labels(X_w)
        s_consistency = float(np.mean(y_h == y_w))
    else:
        s_consistency = float("nan")

    return gr_consistency, s_consistency


# -------------------------
# Data: Adult loader
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
# Load + split
# -------------------------
if args.dataset == "adult":
    df = load_adult_clean(args.data_dir)
elif args.dataset == "compas":
    df = load_compas_clean(args.data_dir)
else:
    raise ValueError(f"Unknown dataset={args.dataset}")

y_full = df["y"].astype(int).values
orig_index_full = df["orig_index"].values

sex_cols = [c for c in df.columns if c.startswith("sex_")]
print("sex dummy columns:", sex_cols)

if "sex_Female" in df.columns:
    sens_full = df["sex_Female"].astype(int).values
elif "sex_Male" in df.columns:
    sens_full = 1 - df["sex_Male"].astype(int).values
else:
    raise ValueError(f"Cannot find sex dummy column among: {sex_cols}")

# ---- build X (drop sex_* for training as you already do) ----
drop_for_X = []
drop_for_X += [c for c in df.columns if c.startswith("sex_")]  # drop gender from model input
drop_for_X += ["y", "orig_index"]

X = df.drop(columns=drop_for_X, errors="ignore").astype(np.float32).values

# Split train/val/test with stratify on labels
X_tr, X_tmp, y_tr, y_tmp, sens_tr, sens_tmp, rid_tr, rid_tmp = train_test_split(
    X, y_full, sens_full, orig_index_full,
    test_size=0.4, random_state=args.seed, stratify=y_full
)

X_va, X_te, y_va, y_te, sens_va, sens_te, rid_va, rid_te = train_test_split(
    X_tmp, y_tmp, sens_tmp, rid_tmp,
    test_size=0.5, random_state=args.seed, stratify=y_tmp
)

# Standardize using TRAIN only
scaler = StandardScaler()
scaler.fit(X_tr)

X_tr = scaler.transform(X_tr).astype(np.float32)
X_va = scaler.transform(X_va).astype(np.float32)
X_te = scaler.transform(X_te).astype(np.float32)

n_tr, d = X_tr.shape
print(f"num_train={n_tr}, num_val={X_va.shape[0]}, num_test={X_te.shape[0]}, d={d}")

# -------------------------
# Fair subspace (train-only prototypes)
# -------------------------
mu0 = X_tr[sens_tr == 0].mean(axis=0, keepdims=True)
mu1 = X_tr[sens_tr == 1].mean(axis=0, keepdims=True)
prot = np.vstack([mu0, mu1]).astype(np.float32)

svd_rank_used = min(args.svd_rank, prot.shape[0], d)
svd = TruncatedSVD(n_components=svd_rank_used, random_state=args.seed)
svd.fit(prot)
basis = svd.components_.T

proj = basis @ np.linalg.pinv(basis.T @ basis) @ basis.T
proj_compl = np.eye(d, dtype=np.float32) - proj

# Metric learning (optional) on TRAIN fair space
X_tr_fair_full = X_tr @ proj_compl
fair_dim = X_tr_fair_full.shape[1]
L_metric = None

if args.use_metric:
    r_metric = min(args.metric_rank, fair_dim)
    svd_metric = TruncatedSVD(n_components=r_metric, random_state=args.seed)
    svd_metric.fit(X_tr_fair_full)
    L_metric = svd_metric.components_

def apply_metric(X_fair: np.ndarray) -> np.ndarray:
    if L_metric is None:
        return X_fair
    return X_fair @ L_metric.T


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


model = MLPBin(d_in=d, hidden=256, depth=args.depth).to(device)
optim = torch.optim.Adam(model.parameters(), lr=10 ** (-args.nloglr))

Xtr_t = torch.tensor(X_tr, dtype=torch.float32, device=device)
Xva_t = torch.tensor(X_va, dtype=torch.float32, device=device)
Xte_t = torch.tensor(X_te, dtype=torch.float32, device=device)

# -------------------------
# Graph on TRAIN subset (in FAIR space + optional metric)
# -------------------------
if n_tr > args.max_graph_n:
    idx_graph = np.random.RandomState(args.seed).choice(n_tr, args.max_graph_n, replace=False)
else:
    idx_graph = np.arange(n_tr)

idx_graph_t = torch.tensor(idx_graph, dtype=torch.long, device=device)

X_tr_g_fair = (X_tr @ proj_compl)[idx_graph]
X_tr_g_metric = apply_metric(X_tr_g_fair)

W_tr = build_W_tau_only(X_tr_g_metric, args.theta, args.tau)

# -------------------------
# OPTIONAL: iFlipper pre-processing on TRAIN graph subset
# -------------------------
y_tr_used = y_tr.copy()
iflipper_info = {
    "iflipper_used": bool(args.use_iflipper),
    "rho_if": float(args.rho_if) if args.use_iflipper else np.nan,
    "train_err0_graphsubset": np.nan,
    "train_err_after_graphsubset": np.nan,
    "train_flip_rate_graphsubset": np.nan,
    "train_flip_rate_full": np.nan,
}

if args.use_iflipper:
    y_g = y_tr[idx_graph].astype(int)
    edge_g, w_edge_g = W_to_edge_wedge(W_tr, thr=0.0)

    edge_g = np.asarray(edge_g, dtype=np.int64)
    w_edge_g = np.asarray(w_edge_g, dtype=np.float64)

    err0 = float(measure_error(y_g, edge_g, w_edge_g))

    m = float(args.rho_if * err0)

    iflip = IFlipper(label=y_g, w_sim=W_tr, edge=edge_g, w_edge=w_edge_g, ablation_option=None)
    y_g_flip = iflip.transform(m).astype(int)

    err_after = float(measure_error(y_g_flip, edge_g, w_edge_g))
    flip_rate_sub = float(np.mean(y_g_flip != y_g))

    y_tr_used[idx_graph] = y_g_flip
    flip_rate_full = float(np.mean(y_tr_used != y_tr))

    iflipper_info.update({
        "train_err0_graphsubset": err0,
        "train_err_after_graphsubset": err_after,
        "train_flip_rate_graphsubset": flip_rate_sub,
        "train_flip_rate_full": flip_rate_full,
    })

    print("\n[iFlipper] pre-processing DONE")
    print({
        "rho_if": args.rho_if,
        "m": m,
        "err0": err0,
        "err_after": err_after,
        "flip_rate_sub": flip_rate_sub,
        "flip_rate_full": flip_rate_full,
        "graph_n": int(len(idx_graph)),
        "edges_sub": int(len(edge_g)),
    })

# tensors for labels
ytr_t = torch.tensor(y_tr_used, dtype=torch.float32, device=device)
yva_t = torch.tensor(y_va, dtype=torch.float32, device=device)
yte_t = torch.tensor(y_te, dtype=torch.float32, device=device)

# -------------------------
# URW sparse P on TRAIN subset (for URW mode only)
# -------------------------
P_tr = make_row_stochastic_P(W_tr).tocoo()
n_g = P_tr.shape[0]

idx = torch.tensor(np.vstack([P_tr.row, P_tr.col]), dtype=torch.long, device=device)
val = torch.tensor(P_tr.data, dtype=torch.float32, device=device)
P_tr_sp = torch.sparse_coo_tensor(idx, val, torch.Size(P_tr.shape), device=device)

eps = 1e-7

if args.use_iflipper:
    print("\n[Training] iFlipper baseline: loss = CE(y_rectified, p)  (NO consistency term)")
else:
    print("\n[Training] URW (binary, forward KL): loss = CE(y,p) + lambda * mean_i KL(Ber(p_i)||Ber(q_i))")

for it in trange(args.num_iterations):
    model.train()
    logits_tr = model(Xtr_t)
    p_tr = torch.sigmoid(logits_tr)

    task_loss = F.binary_cross_entropy_with_logits(logits_tr, ytr_t)

    if args.use_iflipper:
        loss = task_loss
        cons_loss = torch.tensor(0.0, device=device)
    else:
        p_g = p_tr[idx_graph_t]
        q_g = torch.sparse.mm(P_tr_sp, p_g.unsqueeze(1)).squeeze(1)

        p_clamp = p_g.clamp(eps, 1.0 - eps)
        q_clamp = q_g.clamp(eps, 1.0 - eps)

        kl_vec = (
            p_clamp * torch.log(p_clamp / q_clamp)
            + (1.0 - p_clamp) * torch.log((1.0 - p_clamp) / (1.0 - q_clamp))
        )
        cons_loss = kl_vec.mean()
        loss = task_loss + args.lambda_cons * cons_loss

    optim.zero_grad()
    loss.backward()
    optim.step()

    if (it + 1) % 200 == 0 or it == 0:
        model.eval()
        with torch.no_grad():
            logits_va = model(Xva_t)
            p_va = torch.sigmoid(logits_va).cpu().numpy()
        y_va_np = y_va

        acc = accuracy_score(y_va_np, (p_va >= 0.5).astype(int))
        bal_acc = balanced_accuracy_score(y_va_np, (p_va >= 0.5).astype(int))
        auc = roc_auc_score(y_va_np, p_va)
        ce = log_loss(y_va_np, p_va, eps=1e-7)

        print({
            'iter': it + 1,
            'train_CE': float(task_loss.item()),
            'train_cons_forward_KL': float(cons_loss.item()),
            'val_acc': acc,
            'val_bal_acc': bal_acc,
            'val_auc': auc,
            'val_logloss': ce,
        })


# -------------------------
# Evaluation on TEST
# -------------------------
print("\n[Evaluation] TEST")
model.eval()
with torch.no_grad():
    logits_te = model(Xte_t).cpu().numpy()
    pt_prob = 1.0 / (1.0 + np.exp(-logits_te))

y_te_np = y_te
pred_label = (pt_prob >= 0.5).astype(int)

acc_te = accuracy_score(y_te_np, pred_label)
auc_te = roc_auc_score(y_te_np, pt_prob)
bal_acc_te = balanced_accuracy_score(y_te_np, pred_label)
ce_te = log_loss(y_te_np, pt_prob, eps=1e-7)

# TEST graph for IF/LIF
X_te_fair = X_te @ proj_compl
X_te_metric = apply_metric(X_te_fair)
W_te = build_W_tau_only(X_te_metric, args.theta, args.tau)
P_te = make_row_stochastic_P(W_te)

inst_if = IF_cons_vector_forward_kl(P_te, pt_prob.ravel())
IF_mean = float(inst_if.mean())
IF_median = float(np.median(inst_if))
lif = lif_energy(W_te, pt_prob.ravel(), normalize='per_edge')

eps_list = [0.005, 0.001, 0.01, 0.05, 0.1]
vr_dict = violation_rates_from_if(inst_if, eps_list)

print("\nViolation Rates:")
for k, v in vr_dict.items():
    print(f"{k}: {v:.6f}")

dp = demographic_parity_prob(pt_prob, sens_te)

# -------------------------------------------------
# Reconstruct TEST raw rows & build full IF table
# -------------------------------------------------
df_test_raw = df[df["orig_index"].isin(rid_te)].copy()
df_test_raw = df_test_raw.set_index("orig_index").loc[rid_te].reset_index()
df_test_raw = df_test_raw.add_prefix("raw_")

df_pred = pd.DataFrame({
    'orig_index': rid_te,
    'y_true': y_te_np,
    'pred_prob': pt_prob.ravel(),
    'pred_label': pred_label,
    'abs_error_0_1': np.abs(y_te_np - pred_label),
    'if_score_forwardKL_prob': inst_if,
    'sens': sens_te,
})

df_out_full = pd.concat([df_pred, df_test_raw], axis=1)

# NEW: compute Gender/Race gaps on predicted probabilities
gap_g_rms, gap_g_max = group_gap_stats_prob(pt_prob, sens_te)

race_cols = [c for c in df_out_full.columns if c.startswith("raw_race_")]
if len(race_cols) == 0:
    raise ValueError("No raw_race_* columns found in df_out_full, cannot compute race gaps.")
race_onehot = df_out_full[race_cols].to_numpy().astype(int)

row_sum = race_onehot.sum(axis=1)
if not np.all((row_sum == 1) | (row_sum == 0)):
    raise ValueError("Race one-hot rows are not valid (not exactly one active category).")

race_group = race_onehot.argmax(axis=1)  # 0..K-1 group id
gap_r_rms, gap_r_max = group_gap_stats_prob(pt_prob, race_group)

print(f"DP(pred_prob; gender): {dp:.6f}")
print(f"Gap_G_RMS: {gap_g_rms:.6f}, Gap_G_max: {gap_g_max:.6f}")
print(f"Gap_R_RMS: {gap_r_rms:.6f}, Gap_R_max: {gap_r_max:.6f}")

# -------------------------------------------------
# Consistency: GR-Con and S-Con
# -------------------------------------------------
gr_con, s_con = get_consistency_pytorch(
    df_out_full,
    model=model,
    scaler=scaler,
    device=device,
    raw_prefix="raw_",
)
print(f"GR-Con (gender+race): {gr_con:.4f}")
print(f"S-Con  (Husband vs Wife): {s_con:.4f}")

# -------------------------------------------------
# metrics
# -------------------------------------------------
if args.use_iflipper:
    method_name = f"iFlipper pre-proc (rho={args.rho_if}) + ERM"
else:
    method_name = "URW forward-KL (node || barycenter) on probs"

out = {
    'METHOD': method_name,
    'depth': args.depth,
    'test_acc': acc_te,
    'test_bal_acc': bal_acc_te,
    'test_auc': auc_te,
    'test_logloss': ce_te,

    'DP(pred_prob; gender)': dp,

    # group gaps
    'Gap_G_RMS': float(gap_g_rms),
    'Gap_G_max': float(gap_g_max),
    'Gap_R_RMS': float(gap_r_rms),
    'Gap_R_max': float(gap_r_max),

    'IF_mean(forwardKL,prob)': IF_mean,
    'IF_median(forwardKL,prob)': IF_median,
    'LIF_energy(unorm,prob)': lif,
    'GR_Consistency': float(gr_con),
    'S_Consistency': float(s_con),

    'lambda_cons': float(args.lambda_cons),
    'tau_used': float(args.tau),
    'theta': float(args.theta),
    'graph_train_subset': int(n_g),
    'svd_rank': int(svd_rank_used),
    'use_metric': bool(args.use_metric),
    'metric_rank': int(min(args.metric_rank, X_tr_fair_full.shape[1])) if args.use_metric else 0,
}

out.update(vr_dict)

# attach iFlipper diagnostics
out.update({
    'iflipper_used': iflipper_info["iflipper_used"],
    'rho_if': iflipper_info["rho_if"],
    'train_err0_graphsubset': iflipper_info["train_err0_graphsubset"],
    'train_err_after_graphsubset': iflipper_info["train_err_after_graphsubset"],
    'train_flip_rate_graphsubset': iflipper_info["train_flip_rate_graphsubset"],
    'train_flip_rate_full': iflipper_info["train_flip_rate_full"],
})

pprint(out)

os.makedirs("results_unified", exist_ok=True)

if args.use_iflipper:
    metrics_path = f"results_unified/metrics_compas_iflipper_depth{args.depth}_rho{args.rho_if}.csv"
    model_path   = f"results_unified/iflipper_compas_model_depth{args.depth}_rho{args.rho_if}.pt"
    scaler_path  = f"results_unified/iflipper_compas_scaler_depth{args.depth}_rho{args.rho_if}.pkl"
else:
    metrics_path = f"results_unified/metrics_compas_urw_depth{args.depth}_lambda{args.lambda_cons}.csv"
    model_path   = f"results_unified/urw_model_compas_depth{args.depth}_lambda{args.lambda_cons}.pt"
    scaler_path  = f"results_unified/urw_scaler_compas_depth{args.depth}_lambda{args.lambda_cons}.pkl"

pd.DataFrame([out]).to_csv(metrics_path, index=False)
print(f"[Saved] Metrics CSV → {metrics_path}")

torch.save({"state_dict": model.state_dict(), "input_dim": d, "hidden": 256, "depth": args.depth}, model_path)
print(f"[Saved] Model → {model_path}")

joblib.dump(scaler, scaler_path)
print(f"[Saved] Scaler → {scaler_path}")

# Optional: per-instance IF + raw features
if args.save_if_csv:
    try:
        out_dir = os.path.dirname(args.save_if_csv)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        df_out_full.to_csv(args.save_if_csv, index=False)
        print(f"[Saved] Full IF CSV with raw data → {args.save_if_csv}")
    except Exception as e:
        print(f"[Warn] Failed to save IF CSV: {e}")
