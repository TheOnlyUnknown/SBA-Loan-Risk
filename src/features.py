import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression


DROP_FOR_MODELING = [
    "ApprovalDate", "BorrZip", "BankZip",
    "BankName", "BankCity", "ProjectCounty", "CongressionalDistrict",
]

def select_model_features(X: pd.DataFrame) -> pd.DataFrame:
    return X.drop(columns=DROP_FOR_MODELING)

def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
   
    numeric_cols = X.select_dtypes(include="number").columns.tolist()
    categorical_cols = X.select_dtypes(include="str").columns.tolist()

    return ColumnTransformer([
        ("numeric", StandardScaler(), numeric_cols),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ])

def build_baseline_model(X: pd.DataFrame) -> Pipeline:
   
    preprocessor = build_preprocessor(X)
    model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    return Pipeline([
        ("preprocess", preprocessor),
        ("model", model),
    ])
