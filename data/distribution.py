import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# -------------------------------------------------------------------
# Load dataset
# -------------------------------------------------------------------
df = pd.read_csv("Employers_data.csv")

# Salary distribution ------------------------------------------------
salary = df["Salary"].astype(float)

plt.figure(figsize=(7,5))
plt.hist(salary, bins=40)
plt.xlabel("Salary")
plt.ylabel("Count")
plt.title("Salary Distribution (RAW)")
plt.tight_layout()
plt.show()

# -------------------------------------------------------------------
# Standardize salary (what your model uses)
# -------------------------------------------------------------------
scaler = StandardScaler()
salary_std = scaler.fit_transform(salary.values.reshape(-1,1)).ravel()

plt.figure(figsize=(7,5))
plt.hist(salary_std, bins=40)
plt.xlabel("Standardized Salary")
plt.ylabel("Count")
plt.title("Salary Distribution (STANDARDIZED)")
plt.tight_layout()
plt.show()

# -------------------------------------------------------------------
# Compute fair-space graph and degree distribution
# -------------------------------------------------------------------
# Prepare features (same as your training pipeline)
df2 = df.drop(columns=["Employee_ID","Name"], errors="ignore")
sens = (df2["Gender"].astype(str).str.lower() == "female").astype(int).values
X = df2.drop(columns=["Gender","Salary"], errors="ignore")

X = pd.get_dummies(X, drop_first=True).astype(np.float32)
X = StandardScaler().fit_transform(X)

# Fair projector (same as your pipeline)
from sklearn.decomposition import TruncatedSVD
mu0 = X[sens==0].mean(axis=0,keepdims=True)
mu1 = X[sens==1].mean(axis=0,keepdims=True)
prot = np.vstack([mu0,mu1]).astype(np.float32)

svd = TruncatedSVD(n_components=min(1,prot.shape[0],X.shape[1]),random_state=0)
svd.fit(prot)
basis = svd.components_.T
proj = basis @ np.linalg.pinv(basis.T @ basis) @ basis.T
proj_compl = np.eye(X.shape[1]) - proj

# project into fair space
X_fair = X @ proj_compl

# -------------------------------------------------------------------
# Build graph W (same τ-only method)
# -------------------------------------------------------------------
def e_dist_sq(A,B):
    A2 = (A**2).sum(axis=1, keepdims=True)
    B2 = (B**2).sum(axis=1, keepdims=True).T
    return np.maximum(A2 - 2*(A @ B.T) + B2, 0.0)

def build_W_tau_only(X, theta=1e-3, tau=5.0):
    D2 = e_dist_sq(X,X)
    W = np.exp(-theta * D2) * (D2 <= tau)
    np.fill_diagonal(W, 0)
    return 0.5 * (W + W.T)

W = build_W_tau_only(X_fair)

degree = W.sum(axis=1)

plt.figure(figsize=(7,5))
plt.hist(degree, bins=40)
plt.xlabel("Degree")
plt.ylabel("Count")
plt.title("Degree Distribution of Employee Graph")
plt.tight_layout()
plt.show()

# -------------------------------------------------------------------
# Print key stats
# -------------------------------------------------------------------
print("=== Degree Distribution Statistics ===")
print("min:", degree.min())
print("max:", degree.max())
print("mean:", degree.mean())
print("median:", np.median(degree))
print("std:", degree.std())

print("\n=== Salary Statistics ===")
print("Raw min/max:", salary.min(), salary.max())
print("Standardized mean/std:", salary_std.mean(), salary_std.std())
