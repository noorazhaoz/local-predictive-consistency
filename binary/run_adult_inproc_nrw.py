# run_binary_inproc_nlap.py
# ------------------------------------------------------------
# Symmetric normalized-Laplacian baseline for binary classification
# (Adult / COMPAS), aligned to the existing binary URW pipeline.
#
# Training objective:
#   Loss = BCEWithLogits(y, z)
#        + lambda * (1/n_g) * p_g^T L_sym p_g
#
# where:
#   p_g   = sigmoid(z_g)
#   S     = D^{-1/2} W D^{-1/2}
#   L_sym = I - S
#
# IMPORTANT:
#   1) This is ONE normalized-Laplacian baseline. There is no FKL/RKL split.
#   2) There is NO second row-normalization after S = D^{-1/2} W D^{-1/2}.
#   3) Adult/COMPAS loading, train/val/test splitting, graph construction,
#      fair-subspace construction, and optional metric projection follow the
#      supplied binary URW script.
#   4) The model input follows the supplied binary URW script exactly:
#      standardized X is fed to the network; the fair projection is used to
#      construct the comparison-graph representation.
#   5) Cross-method fairness is reported using EdgeMeanAbs on positive-class
#      probabilities over unique non-self test edges.
# ------------------------------------------------------------

import argparse
import random
import os
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
    description="Symmetric normalized-Laplacian in-processing (binary, Adult/COMPAS)"
)

parser.add_argument("--dataset", type=str, default="adult", choices=["adult", "compas"])
parser.add_argument("--data_dir", type=str, default="../data")

parser.add_argument("--depth", type=int, default=1)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--num_iterations", type=int, default=4000)
parser.add_argument("--nloglr", type=float, default=3.0)  # lr = 10^{-nloglr}
parser.add_argument("--lambda_cons", type=float, default=0.3)

# Graph locality
parser.add_argument("--theta", type=float, default=1e-3)
parser.add_argument("--tau", type=float, default=5.0)
parser.add_argument("--max_graph_n", type=int, default=5000)

# Fair subspace
parser.add_argument("--svd_rank", type=int, default=1)

# Optional metric projection on fair space
parser.add_argument("--use_metric", action="store_true")
parser.add_argument("--metric_rank", type=int, default=64)

# Export per-instance table on test
parser.add_argument(
    "--save_if_csv",
    type=str,
    default="",
    help="If set, save per-instance test predictions, N-Lap residual, and raw rows."
)

# Output directory
parser.add_argument("--results_dir", type=str, default="results_nlap")

args = parser.parse_args()
print(vars(args))

torch.set_num_threads(min(8, torch.get_num_threads()))
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
device = torch.device("cpu")


# -------------------------
# Utils
# -------------------------
def e_dist_sq(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A2 = (A ** 2).sum(axis=1, keepdims=True)
    B2 = (B ** 2).sum(axis=1, keepdims=True).T
    return np.maximum(A2 - 2.0 * (A @ B.T) + B2, 0.0)


def build_W_tau_only(X_for_metric: np.ndarray, theta: float, tau: float) -> np.ndarray:
    """
    Dense W with RBF weights and Euclidean-distance threshold tau.
    Self-loop W_ii=1 is added after sparsification.

    e_dist_sq returns squared Euclidean distances, so thresholding uses tau^2.
    This matches the supplied binary URW implementation.
    """
    D2 = e_dist_sq(X_for_metric, X_for_metric)
    mask = D2 <= (tau ** 2)

    W = np.exp(-float(theta) * D2).astype(np.float64) * mask.astype(np.float64)

    np.fill_diagonal(W, 0.0)
    W = 0.5 * (W + W.T)

    n = W.shape[0]
    W[np.arange(n), np.arange(n)] = 1.0
    return W


def graph_diagnostics_from_W(W: np.ndarray) -> dict:
    """
    Diagnostics excluding self-loops.
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
        "coverage": float(np.mean(degrees > 0)),
        "isolated_rate": float(np.mean(degrees == 0)),
        "edge_density": float(A.sum() / max(n * (n - 1), 1)),
    }


def make_sym_normalized_S(W: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Symmetric normalized adjacency:
        S = D^{-1/2} W D^{-1/2}
    and normalized Laplacian:
        L_sym = I - S

    NO second row normalization.
    """
    d = W.sum(axis=1)
    inv_sqrt_d = 1.0 / np.sqrt(np.maximum(d, eps))
    S = inv_sqrt_d[:, None] * W * inv_sqrt_d[None, :]

    # Numerical symmetrization only.
    S = 0.5 * (S + S.T)
    return S.astype(np.float64)


def nlap_residual_vector(S: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Method-specific node diagnostic |(I-S)p|."""
    p = np.asarray(p).ravel()
    return np.abs(p - S.dot(p))


def edge_pairwise_stats(W: np.ndarray, p: np.ndarray) -> dict:
    """
    Independent pairwise diagnostics over unique non-self undirected edges.
    """
    p = np.asarray(p).ravel()

    mask = np.triu(W > 0, k=1)
    ii, jj = np.where(mask)

    if len(ii) == 0:
        return {
            "EdgeMeanAbs": np.nan,
            "EdgeMedianAbs": np.nan,
            "EdgeRMS": np.nan,
            "EdgeMaxAbs": np.nan,
            "EdgeCount": 0,
        }

    d = np.abs(p[ii] - p[jj])

    return {
        "EdgeMeanAbs": float(np.mean(d)),
        "EdgeMedianAbs": float(np.median(d)),
        "EdgeRMS": float(np.sqrt(np.mean(d ** 2))),
        "EdgeMaxAbs": float(np.max(d)),
        "EdgeCount": int(len(d)),
    }


def lif_energy(W: np.ndarray, f: np.ndarray, normalize: str = "per_edge") -> float:
    """
    Optional combinatorial Dirichlet-energy diagnostic:
        f^T (D-W) f
    """
    d = W.sum(axis=1)
    Lf = d * f - (W @ f)
    E = float(f @ Lf)

    if normalize == "per_node":
        E /= max(len(f), 1)
    elif normalize == "per_edge":
        W_no_self = W.copy()
        np.fill_diagonal(W_no_self, 0.0)
        m = int(np.triu(W_no_self > 0, k=1).sum())
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
    group = np.asarray(group).ravel()
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
# GR/S consistency
# -------------------------
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
        X_std = scaler.transform(X_model)

        x_t = torch.tensor(X_std, dtype=torch.float32, device=device)
        with torch.no_grad():
            logits = model(x_t).cpu().numpy().ravel()

        probs = 1.0 / (1.0 + np.exp(-logits))
        return (probs >= 0.5).astype(int)

    # GR-Con: flip gender and race combinations.
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

    gr_consistency = float(
        np.mean(
            (y00 == y01)
            & (y00 == y10)
            & (y00 == y11)
        )
    )

    # S-Con: Husband <-> Wife for Adult.
    rel_indices = [i for i, n in enumerate(names) if n.startswith("relationship_")]
    husband_name = "relationship_Husband"
    wife_name = "relationship_Wife"

    s_consistency = np.nan

    if rel_indices and (husband_name in names) and (wife_name in names):
        husband_idx = names.index(husband_name)
        wife_idx = names.index(wife_name)

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
        "age", "workclass", "fnlwgt", "education", "education-num",
        "marital-status", "occupation", "relationship", "race", "sex",
        "capital-gain", "capital-loss", "hours-per-week",
        "native-country", "y"
    ]

    train_path = os.path.join(data_dir, "adult.data")
    test_path = os.path.join(data_dir, "adult.test")

    train = pd.read_csv(
        train_path,
        header=None,
        sep=",",
        skipinitialspace=True,
        na_values="?",
    )

    test = pd.read_csv(
        test_path,
        header=None,
        sep=",",
        skipinitialspace=True,
        na_values="?",
        comment="|",
    )

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

    drop_cols = [
        c for c in df.columns
        if c.startswith("fnlwgt") or c.startswith("education_")
    ]
    df = df.drop(columns=drop_cols, errors="ignore")

    df["orig_index"] = np.arange(len(df))
    return df


def load_compas_clean(data_dir="../data/compas"):
    path = os.path.join(data_dir, "compas-scores-two-years.csv")
    df = pd.read_csv(path)

    if "two_year_recid" not in df.columns:
        raise ValueError("COMPAS file missing 'two_year_recid' column.")

    keep_cols = [
        "age", "priors_count", "juv_fel_count",
        "juv_misd_count", "juv_other_count",
        "c_charge_degree", "sex", "race",
        "two_year_recid",
    ]

    missing = [c for c in keep_cols if c not in df.columns]
    if missing:
        raise ValueError(f"COMPAS missing columns: {missing}")

    df = df[keep_cols].copy().dropna()

    df["y"] = df["two_year_recid"].astype(int)
    df = df.drop(columns=["two_year_recid"])

    for c in [
        "age",
        "priors_count",
        "juv_fel_count",
        "juv_misd_count",
        "juv_other_count",
    ]:
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

        return self.out(h).squeeze(1)


# -------------------------
# Main runner
# -------------------------
def run_nlap(save_if_csv_path: str):
    # Re-seed.
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Load data.
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

    # Same split as binary URW.
    X_tr, X_tmp, y_tr, y_tmp, sens_tr, sens_tmp, rid_tr, rid_tmp = train_test_split(
        X,
        y_full,
        sens_full,
        orig_index_full,
        test_size=0.4,
        random_state=args.seed,
        stratify=y_full,
    )

    X_va, X_te, y_va, y_te, sens_va, sens_te, rid_va, rid_te = train_test_split(
        X_tmp,
        y_tmp,
        sens_tmp,
        rid_tmp,
        test_size=0.5,
        random_state=args.seed,
        stratify=y_tmp,
    )

    # Standardization: fit TRAIN only.
    scaler = StandardScaler()
    scaler.fit(X_tr)

    X_tr = scaler.transform(X_tr).astype(np.float32)
    X_va = scaler.transform(X_va).astype(np.float32)
    X_te = scaler.transform(X_te).astype(np.float32)

    n_tr, d = X_tr.shape

    print(
        f"\n[Setup] dataset={args.dataset} | method=N-Lap | "
        f"num_train={n_tr}, num_val={X_va.shape[0]}, "
        f"num_test={X_te.shape[0]}, d={d}"
    )

    # Fair subspace, train-only prototypes.
    mu0 = X_tr[sens_tr == 0].mean(axis=0, keepdims=True)
    mu1 = X_tr[sens_tr == 1].mean(axis=0, keepdims=True)

    prot = np.vstack([mu0, mu1]).astype(np.float32)

    svd_rank_used = min(args.svd_rank, prot.shape[0], d)

    svd = TruncatedSVD(
        n_components=svd_rank_used,
        random_state=args.seed,
    )
    svd.fit(prot)

    basis = svd.components_.T

    proj = basis @ np.linalg.pinv(basis.T @ basis) @ basis.T
    proj_compl = np.eye(d, dtype=np.float32) - proj

    # Optional metric learning in the same fair space used by URW.
    X_tr_fair_full = X_tr @ proj_compl
    fair_dim = X_tr_fair_full.shape[1]

    L_metric = None

    if args.use_metric:
        r_metric = min(
            args.metric_rank,
            fair_dim,
            max(X_tr_fair_full.shape[0] - 1, 1),
        )

        svd_metric = TruncatedSVD(
            n_components=r_metric,
            random_state=args.seed,
        )
        svd_metric.fit(X_tr_fair_full)

        L_metric = svd_metric.components_

    def apply_metric(X_fair: np.ndarray) -> np.ndarray:
        if L_metric is None:
            return X_fair
        return X_fair @ L_metric.T

    # TRAIN graph subset.
    if n_tr > args.max_graph_n:
        idx_graph = np.random.RandomState(args.seed).choice(
            n_tr,
            args.max_graph_n,
            replace=False,
        )
    else:
        idx_graph = np.arange(n_tr)

    idx_graph_t = torch.tensor(
        idx_graph,
        dtype=torch.long,
        device=device,
    )

    X_tr_g_fair = (X_tr @ proj_compl)[idx_graph]
    X_tr_g_metric = apply_metric(X_tr_g_fair)

    W_tr = build_W_tau_only(
        X_tr_g_metric,
        args.theta,
        args.tau,
    )

    graph_diag_train = graph_diagnostics_from_W(W_tr)

    print({
        "tau_used": float(args.tau),
        "graph_train_avg_degree": graph_diag_train["avg_degree"],
        "graph_train_isolated_rate": graph_diag_train["isolated_rate"],
        "graph_train_edge_density": graph_diag_train["edge_density"],
    })

    # Symmetric normalized adjacency.
    S_tr = make_sym_normalized_S(W_tr)
    S_tr_coo = csr_matrix(S_tr).tocoo()

    n_g = S_tr_coo.shape[0]

    idx = torch.tensor(
        np.vstack([S_tr_coo.row, S_tr_coo.col]),
        dtype=torch.long,
        device=device,
    )

    val = torch.tensor(
        S_tr_coo.data,
        dtype=torch.float32,
        device=device,
    )

    S_tr_sp = torch.sparse_coo_tensor(
        idx,
        val,
        torch.Size(S_tr_coo.shape),
        device=device,
    ).coalesce()

    # Model.
    model = MLPBin(
        d_in=d,
        hidden=256,
        depth=args.depth,
    ).to(device)

    optim = torch.optim.Adam(
        model.parameters(),
        lr=10 ** (-args.nloglr),
    )

    # IMPORTANT:
    # Match the supplied binary URW model-input pipeline exactly.
    Xtr_t = torch.tensor(X_tr, dtype=torch.float32, device=device)
    ytr_t = torch.tensor(y_tr, dtype=torch.float32, device=device)
    Xva_t = torch.tensor(X_va, dtype=torch.float32, device=device)
    Xte_t = torch.tensor(X_te, dtype=torch.float32, device=device)

    print(
        "\n[Training] Symmetric normalized Laplacian\n"
        "Loss = BCEWithLogits(y,z) + lambda * (1/n_g) * "
        "p_g^T (I - D^{-1/2} W D^{-1/2}) p_g"
    )

    # Train.
    for it in trange(args.num_iterations):
        model.train()

        logits_tr = model(Xtr_t)
        p_tr = torch.sigmoid(logits_tr)

        task_loss = F.binary_cross_entropy_with_logits(
            logits_tr,
            ytr_t,
        )

        # p_g in EXACTLY the same order as graph nodes.
        p_g = p_tr[idx_graph_t]

        Sp = torch.sparse.mm(
            S_tr_sp,
            p_g.unsqueeze(1),
        ).squeeze(1)

        Lp = p_g - Sp

        # Standard symmetric normalized-Laplacian quadratic form.
        cons_loss = torch.sum(p_g * Lp) / float(n_g)

        # Numerical protection only.
        cons_loss = torch.clamp_min(cons_loss, 0.0)

        loss = task_loss + args.lambda_cons * cons_loss

        optim.zero_grad()
        loss.backward()
        optim.step()

        if (it + 1) % 200 == 0 or it == 0:
            model.eval()

            with torch.no_grad():
                logits_va = model(Xva_t)
                p_va = torch.sigmoid(logits_va).cpu().numpy().ravel()

            pred_va = (p_va >= 0.5).astype(int)

            acc = accuracy_score(y_va, pred_va)
            bal_acc = balanced_accuracy_score(y_va, pred_va)
            auc = roc_auc_score(y_va, p_va)
            ce = log_loss(y_va, p_va)

            print({
                "iter": it + 1,
                "train_CE": float(task_loss.item()),
                "train_cons_nlap_quad_prob": float(cons_loss.item()),
                "val_acc": float(acc),
                "val_bal_acc": float(bal_acc),
                "val_auc": float(auc),
                "val_logloss": float(ce),
            })

    # TEST evaluation.
    print("\n[Evaluation] TEST")

    model.eval()

    with torch.no_grad():
        logits_te = model(Xte_t).cpu().numpy().ravel()

    pt_prob = 1.0 / (1.0 + np.exp(-logits_te))
    pred_label = (pt_prob >= 0.5).astype(int)

    acc_te = accuracy_score(y_te, pred_label)
    auc_te = roc_auc_score(y_te, pt_prob)
    bal_acc_te = balanced_accuracy_score(y_te, pred_label)
    ce_te = log_loss(y_te, pt_prob)

    # TEST graph in the SAME fair/metric space.
    X_te_fair = X_te @ proj_compl
    X_te_metric = apply_metric(X_te_fair)

    W_te = build_W_tau_only(
        X_te_metric,
        args.theta,
        args.tau,
    )

    graph_diag_test = graph_diagnostics_from_W(W_te)

    # N-Lap test diagnostics.
    S_te = make_sym_normalized_S(W_te)

    p_np = pt_prob.ravel()

    inst_if_nlap = nlap_residual_vector(
        S_te,
        p_np,
    )

    IF_mean_nlap = float(inst_if_nlap.mean())
    IF_median_nlap = float(np.median(inst_if_nlap))

    Lp_te = p_np - S_te.dot(p_np)

    NLAP_energy = float(
        p_np @ Lp_te
    ) / max(len(p_np), 1)

    NLAP_energy = max(
        NLAP_energy,
        0.0,
    )

    # Independent pairwise diagnostic.
    pair_stats = edge_pairwise_stats(
        W_te,
        p_np,
    )

    # Optional combinatorial energy diagnostic.
    lif = lif_energy(
        W_te,
        p_np,
        normalize="per_edge",
    )

    # Group diagnostics.
    dp = demographic_parity_prob(
        p_np,
        sens_te,
    )

    gap_g_rms, gap_g_max = group_gap_stats_prob(
        p_np,
        sens_te,
    )

    # Reconstruct TEST raw rows.
    df_test_raw = (
        df[df["orig_index"].isin(rid_te)]
        .copy()
        .set_index("orig_index")
        .loc[rid_te]
        .reset_index()
        .add_prefix("raw_")
    )

    df_pred = pd.DataFrame({
        "orig_index": rid_te,
        "y_true": y_te,
        "pred_prob": p_np,
        "pred_label": pred_label,
        "abs_error_0_1": np.abs(y_te - pred_label),
        "if_score_nlap_abs_(I-S)p": inst_if_nlap,
        "sens": sens_te,
    })

    df_out_full = pd.concat(
        [df_pred, df_test_raw],
        axis=1,
    )

    # Race group gaps.
    race_cols = [
        c for c in df_out_full.columns
        if c.startswith("raw_race_")
    ]

    if len(race_cols) == 0:
        raise ValueError(
            "No raw_race_* columns found in df_out_full; "
            "cannot compute race gaps."
        )

    race_onehot = df_out_full[race_cols].to_numpy().astype(int)
    row_sum = race_onehot.sum(axis=1)
    valid = (row_sum == 1)

    if not np.any(valid):
        raise ValueError("All rows have invalid race one-hot.")

    if not np.all(valid):
        race_group_valid = race_onehot[valid].argmax(axis=1)
        gap_r_rms, gap_r_max = group_gap_stats_prob(
            p_np[valid],
            race_group_valid,
        )
    else:
        race_group = race_onehot.argmax(axis=1)
        gap_r_rms, gap_r_max = group_gap_stats_prob(
            p_np,
            race_group,
        )

    print(f"DP(pred_prob; gender): {dp:.6f}")
    print(f"Gap_G_RMS: {gap_g_rms:.6f}, Gap_G_max: {gap_g_max:.6f}")
    print(f"Gap_R_RMS: {gap_r_rms:.6f}, Gap_R_max: {gap_r_max:.6f}")
    print(f"EdgeMeanAbs: {pair_stats['EdgeMeanAbs']:.8f}")
    print(f"N-Lap energy: {NLAP_energy:.8f}")

    # GR/S consistency.
    gr_con, s_con = get_consistency_pytorch(
        df_out_full,
        model=model,
        scaler=scaler,
        device=device,
        raw_prefix="raw_",
    )

    print(f"GR-Con (gender+race): {gr_con:.4f}")
    print(
        "S-Con  (Husband vs Wife): "
        f"{s_con if not np.isnan(s_con) else 'NaN'}"
    )

    method_name = (
        "Normalized Laplacian "
        "(L_sym = I - D^{-1/2} W D^{-1/2})"
    )

    out = {
        "DATASET": args.dataset,
        "METHOD": method_name,
        "depth": int(args.depth),

        # Performance
        "test_acc": float(acc_te),
        "test_bal_acc": float(bal_acc_te),
        "test_auc": float(auc_te),
        "test_logloss": float(ce_te),

        # Independent pairwise fairness
        "EdgeMeanAbs": float(pair_stats["EdgeMeanAbs"]),
        "EdgeMedianAbs": float(pair_stats["EdgeMedianAbs"]),
        "EdgeRMS": float(pair_stats["EdgeRMS"]),
        "EdgeMaxAbs": float(pair_stats["EdgeMaxAbs"]),
        "EdgeCount": int(pair_stats["EdgeCount"]),

        # Method-specific N-Lap diagnostics
        "IF_mean(NLap_residual,prob)": float(IF_mean_nlap),
        "IF_median(NLap_residual,prob)": float(IF_median_nlap),
        "NLAP_energy(sym,prob)": float(NLAP_energy),

        # Additional fairness/group diagnostics
        "DP(pred_prob; gender)": float(dp),
        "Gap_G_RMS": float(gap_g_rms),
        "Gap_G_max": float(gap_g_max),
        "Gap_R_RMS": float(gap_r_rms),
        "Gap_R_max": float(gap_r_max),
        "LIF_energy(unorm,prob)": float(lif),
        "GR_Consistency": float(gr_con),
        "S_Consistency": (
            float(s_con)
            if not np.isnan(s_con)
            else np.nan
        ),

        # Hyperparameters
        "lambda_cons": float(args.lambda_cons),
        "tau_used": float(args.tau),
        "theta": float(args.theta),
        "graph_train_subset": int(n_g),

        # Graph diagnostics
        "graph_train_avg_degree": graph_diag_train["avg_degree"],
        "graph_train_median_degree": graph_diag_train["median_degree"],
        "graph_train_min_degree": graph_diag_train["min_degree"],
        "graph_train_max_degree": graph_diag_train["max_degree"],
        "graph_train_coverage": graph_diag_train["coverage"],
        "graph_train_isolated_rate": graph_diag_train["isolated_rate"],
        "graph_train_edge_density": graph_diag_train["edge_density"],

        "graph_test_avg_degree": graph_diag_test["avg_degree"],
        "graph_test_median_degree": graph_diag_test["median_degree"],
        "graph_test_min_degree": graph_diag_test["min_degree"],
        "graph_test_max_degree": graph_diag_test["max_degree"],
        "graph_test_coverage": graph_diag_test["coverage"],
        "graph_test_isolated_rate": graph_diag_test["isolated_rate"],
        "graph_test_edge_density": graph_diag_test["edge_density"],

        # Projection/metric information
        "svd_rank": int(svd_rank_used),
        "use_metric": bool(args.use_metric),
        "metric_rank": (
            int(L_metric.shape[0])
            if (
                args.use_metric
                and L_metric is not None
            )
            else 0
        ),
    }

    pprint(out)

    # --------------------------------------------------------
    # Save results + artifacts
    # --------------------------------------------------------
    os.makedirs(args.results_dir, exist_ok=True)

    tag = (
        f"nlap_{args.dataset}_"
        f"depth{args.depth}_"
        f"lambda{args.lambda_cons}"
    )

    metrics_path = os.path.join(
        args.results_dir,
        f"metrics_{tag}.csv",
    )

    pd.DataFrame([out]).to_csv(
        metrics_path,
        index=False,
    )

    print(f"[Saved] Metrics CSV -> {metrics_path}")

    # Model
    model_path = os.path.join(
        args.results_dir,
        f"model_{tag}.pt",
    )

    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": d,
            "hidden": 256,
            "depth": args.depth,
        },
        model_path,
    )

    print(f"[Saved] Model -> {model_path}")

    # Scaler
    scaler_path = os.path.join(
        args.results_dir,
        f"scaler_{tag}.pkl",
    )

    joblib.dump(
        scaler,
        scaler_path,
    )

    print(f"[Saved] Scaler -> {scaler_path}")

    # Fair projection
    proj_path = os.path.join(
        args.results_dir,
        f"proj_compl_{tag}.npy",
    )

    np.save(
        proj_path,
        proj_compl,
    )

    print(f"[Saved] Fair projection -> {proj_path}")

    # Metric projection
    if L_metric is not None:
        metric_path = os.path.join(
            args.results_dir,
            f"L_metric_{tag}.npy",
        )

        np.save(
            metric_path,
            L_metric,
        )

        print(f"[Saved] Metric projection -> {metric_path}")

    # Per-instance table
    if save_if_csv_path:
        try:
            out_dir = os.path.dirname(save_if_csv_path)

            if out_dir:
                os.makedirs(
                    out_dir,
                    exist_ok=True,
                )

            df_out_full.to_csv(
                save_if_csv_path,
                index=False,
            )

            print(
                "[Saved] Full per-instance CSV with raw data -> "
                f"{save_if_csv_path}"
            )

        except Exception as e:
            print(
                "[Warn] Failed to save per-instance CSV: "
                f"{e}"
            )

    return out


# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    print("DEBUG use_metric =", args.use_metric)
    run_nlap(args.save_if_csv)
