import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sksurv.util import Surv

# ========== Load and preprocess ==========
df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")

# Create survival_discrete columns
df["event"] = df["Attrition"].map({"Yes": 1, "No": 0})
df["duration"] = df["YearsAtCompany"]

# Drop identifier and redundant columns
drop_cols = ["Attrition", "EmployeeNumber", "Over18", "StandardHours", "EmployeeCount"]
df = df.drop(columns=drop_cols)

# Separate features and target
target_cols = ["duration", "event"]
X = df.drop(columns=target_cols)
y_structured = Surv.from_dataframe("event", "duration", df)

# Identify feature types
numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

# Preprocessing pipeline
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numerical_cols),
    ("cat", OneHotEncoder(sparse=False, handle_unknown="ignore"), categorical_cols)
])

X_processed = preprocessor.fit_transform(X)

# Save preprocessor and processed data
os.makedirs("data", exist_ok=True)
joblib.dump(preprocessor, "data/employee_preprocessor.pkl")
joblib.dump((X_processed, y_structured), "data/employee_processed.pkl")

print("✅ Preprocessing complete and saved.")
