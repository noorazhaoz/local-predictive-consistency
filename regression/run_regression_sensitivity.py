# run_regression_inproc_urw_tau_sensitivity.py
# ------------------------------------------------------------
# URW (Unnormalized Random-Walk) consistency for REGRESSION
# with optional threshold-sensitivity analysis.
#
# Task:
#   y = log(target)
#   minimize: MSE(y_log, f(x)) + lambda * mean_i ((I - D^{-1}W) f_g)_i^2
#
# Supported datasets:
#   - employers: sensitive = Gender (female=1, male=0), target = Salary
#   - insurance: sensitive = sex    (female=1, male=0), target = expenses
#
# Notes:
#   - Model always sees fair-projected standardized inputs.
#   - Graphs are built in the same fair-projected space, optionally after metric projection.
#   - tau is interpreted as a Euclidean-distance threshold. Internally D2 <= tau^2.
#   - IF is computed on log predictions: |f_i - (P f)_i|.
# ------------------------------------------------------------

import argparse
import random
import os
from pprint import pprint

import joblib
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.decomposition import TruncatedSVD

from scipy.sparse import csr_matrix


# -------------------------
# Args
# -------------------------
def build_args():
    parser = argparse.ArgumentParser(description="URW consistency in-processing (regression) with tau sensitivity")

    parser.add_argument(
        "--dataset",
        type=str,
        default="employers",
        choices=["employers", "insurance"],
        help="which dataset pipeline to use",
    )

    parser.add_argument("--depth", type=int, default=1, help="number of hidden layers in MLP")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_iterations", type=int, default=4000)
    parser.add_argument("--nloglr", type=float, default=3.0, help="lr = 10^{-nloglr}")
    parser.add_argument("--lambda_cons", type=float, default=0.3)

    # Graph locality
    parser.add_argument("--theta", type=float, default=1e-3, help="RBF scale")
    parser.add_argument("--tau", type=float, default=5.0, help="Euclidean-distance threshold")
    parser.add_argument("--max_graph_n", type=int, default=5000, help="Max #train pts to build W on")

    # Threshold sensitivity analysis
    parser.add_argument("--sensitivity_tau", "--sensitivity-tau", action="store_true", dest="sensitivity_tau",
                        help="Run sensitivity analysis over tau values selected from empirical distance quantiles.")
    parser.add_argument("--tau_quantiles", type=str, default="5,10,15,20,25",
                        help="Comma-separated distance quantiles used to construct tau grid.")
    parser.add_argument("--diagnostic_sample_n", type=int, default=5000,
                        help="Maximum number of points used to estimate distance quantiles and graph diagnostics.")
    parser.add_argument("--sensitivity_out", type=str, default="",
                        help="Output CSV path for threshold sensitivity results.")

    # Fair subspace
    parser.add_argument("--svd_rank", type=int, default=1, help="#components for sensitive subspace SVD")

    # Optional metric learning on fair space
    parser.add_argument("--use_metric", action="store_true")
    parser.add_argument("--metric_rank", type=int, default=64)

    # Export per-instance IF on TEST
    parser.add_argument("--save_if_csv", type=str, default="")

    # Data path
    parser.add_argument("--data_path", type=str, default="../data/Employers_data.csv")

    return parser.parse_args()


# -------------------------
# Utils
# -------------------------
def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def e_dist_sq(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A2 = (A ** 2).sum(axis=1, keepdims=True)
    B2 = (B ** 2).sum(axis=1, keepdims=True).T
    return np.maximum(A2 - 2.0 * (A @ B.T) + B2, 0.0)


def build_W_tau_only(X_for_metric: np.ndarray, theta: float, tau: float) -> np.ndarray:
    """
    Dense W with RBF weights and Euclidean-distance threshold tau,
    symmetric, with explicit self-loop W_ii = 1.
    """
    D2 = e_dist_sq(X_for_metric, X_for_metric).astype(np.float64)
    mask = D2 <= (float(tau) ** 2)
    W = np.exp(-float(theta) * D2).astype(np.float64) * mask.astype(np.float64)

    # Remove diagonal before symmetrization so graph diagnostics can exclude self-loops.
    np.fill_diagonal(W, 0.0)
    W = 0.5 * (W + W.T)

    # Add self-loops for row-stochastic P.
    n = W.shape[0]
    W[np.arange(n), np.arange(n)] = 1.0
    return W


def make_row_stochastic_P(W: np.ndarray) -> csr_matrix:
    rowsum = W.sum(axis=1, keepdims=True) + 1e-12
    return csr_matrix(W / rowsum)


def graph_diagnostics_from_W(W: np.ndarray) -> dict:
    """Graph diagnostics excluding self-loops."""
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
    """Estimate tau values from empirical Euclidean-distance quantiles."""
    rng = np.random.RandomState(seed)
    n = X_for_metric.shape[0]
    if n > sample_n:
        idx = rng.choice(n, sample_n, replace=False)
        X_sample = X_for_metric[idx]
    else:
        X_sample = X_for_metric

    D2 = e_dist_sq(X_sample, X_sample)
    tri = np.triu_indices_from(D2, k=1)
    dists = np.sqrt(D2[tri])
    dists = dists[np.isfinite(dists)]
    tau_values = np.percentile(dists, quantiles)

    return [
        {"tau_quantile": float(q), "tau": float(tau)}
        for q, tau in zip(quantiles, tau_values)
    ]


def IF_cons_vector(P: csr_matrix, yhat: np.ndarray) -> np.ndarray:
    """Instance-level IF for URW on log predictions: |(I - P) yhat|."""
    nbr = P.dot(yhat)
    return np.abs(yhat - nbr)


def group_gap_reg(y_pred: np.ndarray, sens_bin: np.ndarray):
    """gap = mean(pred|sens=1) - mean(pred|sens=0), returns signed and absolute gap."""
    m1 = float(y_pred[sens_bin == 1].mean())
    m0 = float(y_pred[sens_bin == 0].mean())
    gap = m1 - m0
    return gap, abs(gap)


def rmse_score(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


# -------------------------
# Data loader
# -------------------------
def load_dataset_for_regression(args):
    df = pd.read_csv(args.data_path).copy()

    if args.dataset == "employers":
        df = df.drop(columns=["Employee_ID", "Name"], errors="ignore")
        if "Gender" not in df.columns or "Salary" not in df.columns:
            raise ValueError("Employers dataset must contain columns ['Gender', 'Salary'].")

        sens_full = (df["Gender"].astype(str).str.lower() == "female").astype(int).values
        df["Gender_bin"] = sens_full

        target_raw = df["Salary"].astype(np.float32).values
        target_raw = np.clip(target_raw, 1e-6, None).astype(np.float32)
        y_log = np.log(target_raw).astype(np.float32)

        features_full = df.drop(columns=["Gender", "Salary"], errors="ignore")
        X_full_df = pd.get_dummies(features_full, drop_first=True).astype(np.float32)
        if "Gender_bin" not in X_full_df.columns:
            raise ValueError("Gender_bin missing after get_dummies; check df['Gender_bin'].")
        X_model_df = X_full_df.drop(columns=["Gender_bin"]).copy()
        meta = {"dataset_name": "employers", "sensitive_name": "Gender_bin", "target_name": "Salary"}

    elif args.dataset == "insurance":
        required_cols = ["age", "sex", "bmi", "children", "smoker", "region", "expenses"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Insurance dataset missing columns: {missing}")

        sens_full = (df["sex"].astype(str).str.lower() == "female").astype(int).values
        df["sex_bin"] = sens_full

        target_raw = df["expenses"].astype(np.float32).values
        target_raw = np.clip(target_raw, 1e-6, None).astype(np.float32)
        y_log = np.log(target_raw).astype(np.float32)

        features_full = df.drop(columns=["sex", "expenses"], errors="ignore")
        X_full_df = pd.get_dummies(features_full, drop_first=True).astype(np.float32)
        if "sex_bin" not in X_full_df.columns:
            raise ValueError("sex_bin missing after get_dummies; check df['sex_bin'].")
        X_model_df = X_full_df.drop(columns=["sex_bin"]).copy()
        meta = {"dataset_name": "insurance", "sensitive_name": "sex_bin", "target_name": "expenses"}

    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    X = X_model_df.values.astype(np.float32)
    model_feature_names = list(X_model_df.columns)
    full_feature_names = list(X_full_df.columns)
    return X, y_log, sens_full, target_raw, X_full_df, X_model_df, model_feature_names, full_feature_names, df, meta


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
# Shared preprocessing helper
# -------------------------
def prepare_data_and_spaces(args):
    (
        X, y_log, sens_full, target_raw, X_full_df, X_model_df,
        model_feature_names, full_feature_names, df, data_meta,
    ) = load_dataset_for_regression(args)

    indices = np.arange(X.shape[0])

    X_tr, X_tmp, y_tr, y_tmp, sens_tr, sens_tmp, target_tr, target_tmp, idx_tr, idx_tmp = train_test_split(
        X, y_log, sens_full, target_raw, indices,
        test_size=0.4, random_state=args.seed, stratify=sens_full,
    )
    X_va, X_te, y_va, y_te, sens_va, sens_te, target_va, target_te, idx_va, idx_te = train_test_split(
        X_tmp, y_tmp, sens_tmp, target_tmp, idx_tmp,
        test_size=0.5, random_state=args.seed, stratify=sens_tmp,
    )

    scaler = StandardScaler()
    scaler.fit(X_tr)
    X_tr_std = scaler.transform(X_tr).astype(np.float32)
    X_va_std = scaler.transform(X_va).astype(np.float32)
    X_te_std = scaler.transform(X_te).astype(np.float32)

    n_tr, d = X_tr_std.shape

    # Fair projector learned in standardized space.
    mu0 = X_tr_std[sens_tr == 0].mean(axis=0, keepdims=True).astype(np.float32)
    mu1 = X_tr_std[sens_tr == 1].mean(axis=0, keepdims=True).astype(np.float32)
    prot = np.vstack([mu0, mu1]).astype(np.float32)

    r_sens = min(args.svd_rank, prot.shape[0], d)
    svd = TruncatedSVD(n_components=r_sens, random_state=args.seed)
    svd.fit(prot)
    basis = svd.components_.T.astype(np.float32)
    BtB_inv = np.linalg.pinv((basis.T @ basis).astype(np.float32)).astype(np.float32)
    proj = (basis @ BtB_inv @ basis.T).astype(np.float32)
    proj_compl = (np.eye(d, dtype=np.float32) - proj).astype(np.float32)

    X_tr_fair = (X_tr_std @ proj_compl).astype(np.float32)
    X_va_fair = (X_va_std @ proj_compl).astype(np.float32)
    X_te_fair = (X_te_std @ proj_compl).astype(np.float32)

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

    return {
        "X_tr_fair": X_tr_fair,
        "X_va_fair": X_va_fair,
        "X_te_fair": X_te_fair,
        "y_tr": y_tr,
        "y_va": y_va,
        "y_te": y_te,
        "sens_tr": sens_tr,
        "sens_va": sens_va,
        "sens_te": sens_te,
        "target_tr": target_tr,
        "target_va": target_va,
        "target_te": target_te,
        "idx_te": idx_te,
        "X_te_raw_model": X_te,
        "scaler": scaler,
        "proj_compl": proj_compl,
        "L_metric": L_metric,
        "apply_metric": apply_metric,
        "n_tr": n_tr,
        "d": d,
        "r_sens": r_sens,
        "model_feature_names": model_feature_names,
        "data_meta": data_meta,
    }


def prepare_metric_space_for_tau_grid(args):
    data = prepare_data_and_spaces(args)
    return data["apply_metric"](data["X_tr_fair"])


# -------------------------
# Core runner
# -------------------------
def run_one_tau(args, tau_override=None, tau_quantile=None):
    set_seed(args.seed)
    tau_used = float(args.tau if tau_override is None else tau_override)

    data = prepare_data_and_spaces(args)
    X_tr_fair = data["X_tr_fair"]
    X_va_fair = data["X_va_fair"]
    X_te_fair = data["X_te_fair"]
    y_tr = data["y_tr"]
    y_va = data["y_va"]
    y_te = data["y_te"]
    sens_te = data["sens_te"]
    target_va = data["target_va"]
    target_te = data["target_te"]
    idx_te = data["idx_te"]
    X_te_raw_model = data["X_te_raw_model"]
    scaler = data["scaler"]
    proj_compl = data["proj_compl"]
    L_metric = data["L_metric"]
    apply_metric = data["apply_metric"]
    n_tr = data["n_tr"]
    d = data["d"]
    r_sens = data["r_sens"]
    model_feature_names = data["model_feature_names"]
    data_meta = data["data_meta"]

    print(f"num_train={n_tr}, num_val={X_va_fair.shape[0]}, num_test={X_te_fair.shape[0]}, d={d}")
    print({"tau_quantile": tau_quantile, "tau_used": tau_used, "use_metric": args.use_metric})

    device = torch.device("cpu")

    Xtr_t = torch.tensor(X_tr_fair, dtype=torch.float32, device=device)
    ytr_t = torch.tensor(y_tr, dtype=torch.float32, device=device)
    Xva_t = torch.tensor(X_va_fair, dtype=torch.float32, device=device)
    Xte_t = torch.tensor(X_te_fair, dtype=torch.float32, device=device)

    # Build train graph on subset.
    if n_tr > args.max_graph_n:
        idx_graph = np.random.RandomState(args.seed).choice(n_tr, args.max_graph_n, replace=False)
    else:
        idx_graph = np.arange(n_tr)

    X_tr_g_metric = apply_metric(X_tr_fair[idx_graph])
    W_tr = build_W_tau_only(X_tr_g_metric, args.theta, tau_used)
    graph_diag_train = graph_diagnostics_from_W(W_tr)
    pprint({
        "graph_train_avg_degree": graph_diag_train["avg_degree"],
        "graph_train_isolated_rate": graph_diag_train["isolated_rate"],
        "graph_train_edge_density": graph_diag_train["edge_density"],
    })

    P_tr = make_row_stochastic_P(W_tr).tocoo()
    n_g = P_tr.shape[0]

    idx = torch.tensor(np.vstack([P_tr.row, P_tr.col]), dtype=torch.long, device=device)
    val = torch.tensor(P_tr.data, dtype=torch.float32, device=device)
    P_tr_sp = torch.sparse_coo_tensor(idx, val, torch.Size(P_tr.shape), device=device).coalesce()
    idx_graph_t = torch.tensor(idx_graph, dtype=torch.long, device=device)

    model = MLPReg(d_in=d, hidden=256, depth=args.depth).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=10 ** (-args.nloglr))

    print("\n[Training] URW regression: MSE(y_log) + lambda * mean((I-P)f)^2")
    for it in trange(args.num_iterations):
        model.train()
        pred_tr_log = model(Xtr_t)
        task_loss = F.mse_loss(pred_tr_log, ytr_t)

        f_g = pred_tr_log[idx_graph_t]
        Pf = torch.sparse.mm(P_tr_sp, f_g.unsqueeze(1)).squeeze(1)
        cons_loss = torch.mean((f_g - Pf) ** 2)

        loss = task_loss + args.lambda_cons * cons_loss
        optim.zero_grad()
        loss.backward()
        optim.step()

        if (it + 1) % 200 == 0 or it == 0:
            model.eval()
            with torch.no_grad():
                pv_log = model(Xva_t).cpu().numpy().ravel()
            pv_real = np.exp(pv_log)
            print({
                "iter": it + 1,
                "train_mse_log": float(task_loss.item()),
                "train_cons_urw_L2": float(cons_loss.item()),
                "val_RMSE_real": rmse_score(target_va, pv_real),
                "val_MAE_real": float(mean_absolute_error(target_va, pv_real)),
                "val_R2_real": float(r2_score(target_va, pv_real)),
            })

    # Evaluation.
    print("\n[Evaluation] TEST")
    model.eval()
    with torch.no_grad():
        pt_log = model(Xte_t).cpu().numpy().ravel()
    pt_real = np.exp(pt_log)

    rmse = rmse_score(target_te, pt_real)
    mae = float(mean_absolute_error(target_te, pt_real))
    r2 = float(r2_score(target_te, pt_real))
    mse = float(mean_squared_error(target_te, pt_real))

    X_te_metric = apply_metric(X_te_fair)
    W_te = build_W_tau_only(X_te_metric, args.theta, tau_used)
    graph_diag_test = graph_diagnostics_from_W(W_te)
    P_te = make_row_stochastic_P(W_te)

    inst_if = IF_cons_vector(P_te, pt_log)
    IF_mean = float(inst_if.mean())
    IF_median = float(np.median(inst_if))

    gap_real, gap_abs_real = group_gap_reg(pt_real, sens_te)
    gap_log, gap_abs_log = group_gap_reg(pt_log, sens_te)

    pred_std_real = float(pt_real.std())
    pred_std_log = float(pt_log.std())
    tradeoff_real = float(gap_abs_real / (pred_std_real + 1e-12))
    tradeoff_log = float(gap_abs_log / (pred_std_log + 1e-12))

    out = {
        "METHOD": "URW (I - D^{-1}W)",
        "dataset": data_meta["dataset_name"],
        "sensitive_attr": data_meta["sensitive_name"],
        "target_attr": data_meta["target_name"],

        "test_RMSE_real": rmse,
        "test_MAE_real": mae,
        "test_R2_real": r2,
        "test_MSE_real": mse,

        "group_gap_real_s1_minus_s0": float(gap_real),
        "group_gap_abs_real_s1_minus_s0": float(gap_abs_real),
        "group_gap_log_s1_minus_s0": float(gap_log),
        "group_gap_abs_log_s1_minus_s0": float(gap_abs_log),
        "pred_std_real": pred_std_real,
        "pred_std_log": pred_std_log,
        "tradeoff_abs_gap_over_std_real": tradeoff_real,
        "tradeoff_abs_gap_over_std_log": tradeoff_log,

        "IF_mean(URW_L1,log_pred)": IF_mean,
        "IF_median(URW_L1,log_pred)": IF_median,

        "lambda_cons": float(args.lambda_cons),
        "tau_used": float(tau_used),
        "tau_quantile": float(tau_quantile) if tau_quantile is not None else np.nan,
        "theta": float(args.theta),
        "graph_train_subset": int(len(idx_graph)),

        "graph_train_avg_degree": graph_diag_train["avg_degree"],
        "graph_train_median_degree": graph_diag_train["median_degree"],
        "graph_train_min_degree": graph_diag_train["min_degree"],
        "graph_train_max_degree": graph_diag_train["max_degree"],
        "graph_train_isolated_rate": graph_diag_train["isolated_rate"],
        "graph_train_edge_density": graph_diag_train["edge_density"],

        "graph_test_avg_degree": graph_diag_test["avg_degree"],
        "graph_test_median_degree": graph_diag_test["median_degree"],
        "graph_test_min_degree": graph_diag_test["min_degree"],
        "graph_test_max_degree": graph_diag_test["max_degree"],
        "graph_test_isolated_rate": graph_diag_test["isolated_rate"],
        "graph_test_edge_density": graph_diag_test["edge_density"],

        "svd_rank": int(r_sens),
        "use_metric": bool(args.use_metric),
        "metric_rank": int(L_metric.shape[0]) if (args.use_metric and L_metric is not None) else 0,
    }

    pprint(out)

    os.makedirs("results", exist_ok=True)
    tau_tag = f"_tauq{int(tau_quantile)}" if tau_quantile is not None else ""
    out_path = f"results/metrics_{args.dataset}_urw_depth{args.depth}_lambda{args.lambda_cons}{tau_tag}.csv"
    pd.DataFrame([out]).to_csv(out_path, index=False)
    print(f"[Saved] metrics -> {out_path}")

    artifact_tag = f"{args.dataset}_reg_urw_depth{args.depth}_lambda{args.lambda_cons}{tau_tag}"
    joblib.dump(scaler, f"results/{artifact_tag}_scaler.pkl")
    np.save(f"results/{artifact_tag}_proj_compl.npy", proj_compl)
    if L_metric is not None:
        np.save(f"results/{artifact_tag}_L_metric.npy", L_metric)
    torch.save({"state_dict": model.state_dict(), "input_dim": d, "hidden": 256, "depth": args.depth},
               f"results/{artifact_tag}_model.pt")

    if args.save_if_csv:
        try:
            out_dir = os.path.dirname(args.save_if_csv)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            raw_feat_cols = [f"raw_{c}" for c in model_feature_names]
            raw_feat_df = pd.DataFrame(X_te_raw_model, columns=raw_feat_cols)
            df_if = pd.DataFrame({
                "orig_index": idx_te.astype(int),
                "y_real": target_te,
                "y_log": y_te,
                "pred_real": pt_real,
                "pred_log": pt_log,
                "abs_error_real": np.abs(target_te - pt_real),
                "abs_error_log": np.abs(y_te - pt_log),
                "if_score_urw_log": inst_if,
            })
            df_if = pd.concat([df_if.reset_index(drop=True), raw_feat_df.reset_index(drop=True)], axis=1)
            df_if.to_csv(args.save_if_csv, index=False)
            print(f"[Saved] per-instance IF+pred+raw -> {args.save_if_csv}")
        except Exception as e:
            print(f"[Warn] Failed to save IF CSV: {e}")

    return out


def run_tau_sensitivity(args):
    quantiles = [float(q.strip()) for q in args.tau_quantiles.split(",") if q.strip()]
    X_for_metric = prepare_metric_space_for_tau_grid(args)
    tau_grid = estimate_tau_grid_from_quantiles(
        X_for_metric,
        quantiles=quantiles,
        sample_n=args.diagnostic_sample_n,
        seed=args.seed,
    )

    print("\n[Tau sensitivity] Candidate thresholds:")
    pprint(tau_grid)

    all_results = []
    for item in tau_grid:
        q = item["tau_quantile"]
        tau = item["tau"]
        print("\n" + "=" * 80)
        print(f"[Tau sensitivity] dataset={args.dataset}, q={q}, tau={tau:.6f}")
        print("=" * 80)
        out = run_one_tau(args, tau_override=tau, tau_quantile=q)
        all_results.append(out)

    df_sens = pd.DataFrame(all_results)
    os.makedirs("results", exist_ok=True)
    if args.sensitivity_out:
        out_path = args.sensitivity_out
    else:
        out_path = f"results/tau_sensitivity_urw_{args.dataset}_reg_depth{args.depth}_lambda{args.lambda_cons}.csv"
    df_sens.to_csv(out_path, index=False)
    print(f"\n[Saved] Tau sensitivity results -> {out_path}")
    return df_sens


# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    args = build_args()
    print(vars(args))

    torch.set_num_threads(min(8, torch.get_num_threads()))
    set_seed(args.seed)

    print("DEBUG final sensitivity_tau =", args.sensitivity_tau)
    print("DEBUG final use_metric =", args.use_metric)

    if args.sensitivity_tau:
        print("DEBUG: entering tau sensitivity mode")
        run_tau_sensitivity(args)
    else:
        print("DEBUG: entering normal single-tau mode")
        run_one_tau(args)
