import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib
import os

# ========== Step 1: Load Data ==========
df = pd.read_csv("student-mat.csv", sep=';')

# ========== Step 2: Define Features and Target ==========
target_col = "G3"
X = df.drop(columns=[target_col])
y = df[target_col]

# ========== Step 3: Identify Columns ==========
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

# ========== Step 4: Preprocessing Pipeline ==========
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numerical_cols),
    ("cat", OneHotEncoder(sparse=False, handle_unknown="ignore"), categorical_cols)
])

X_processed = preprocessor.fit_transform(X)
joblib.dump(preprocessor, "data/student_preprocessor.pkl")

# ========== Step 5: Split and Save ==========
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

os.makedirs("data", exist_ok=True)
joblib.dump((X_train, X_test, y_train, y_test), "data/student_split.pkl")

print("✅ Data preprocessing completed and saved.")
