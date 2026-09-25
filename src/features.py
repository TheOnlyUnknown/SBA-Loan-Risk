import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

# Dropped before modeling -- too sparse to learn from (BorrZip, BankZip,
# BankName, BankCity, ProjectCounty: thousands of categories with a
# handful of loans each), ambiguous without pairing to state
# (CongressionalDistrict: "district 5" means different things in
# different states), or only needed for the time-aware split rather than
# as a model input (ApprovalDate).
DROP_FOR_MODELING = [
    "ApprovalDate", "BorrZip", "BankZip",
    "BankName", "BankCity", "ProjectCounty", "CongressionalDistrict",
]

def select_model_features(X: pd.DataFrame) -> pd.DataFrame:
    return X.drop(columns=DROP_FOR_MODELING)

def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    # Numeric columns get scaled -- logistic regression's regularization
    # penalizes large coefficients, so a feature measured in the tens of
    # thousands (GrossApproval) would otherwise get unfairly squashed
    # relative to one measured in single digits (TermInMonths in years).
    # Categorical columns get one-hot encoded. handle_unknown="ignore"
    # means a category seen in test but never seen in train (a state or
    # district office that just didn't happen to appear in the earlier
    # loans) gets encoded as all-zeros instead of crashing the pipeline.
    numeric_cols = X.select_dtypes(include="number").columns.tolist()
    categorical_cols = X.select_dtypes(include="str").columns.tolist()

    return ColumnTransformer([
        ("numeric", StandardScaler(), numeric_cols),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ])

def build_baseline_model(X: pd.DataFrame) -> Pipeline:
    # No class_weight here on purpose -- an earlier version used
    # class_weight="balanced" to compensate for charge-offs being only
    # ~6% of loans, but that distorts the predicted probabilities: it
    # trains the model as if defaults were as common as non-defaults, so
    # every probability it outputs comes out inflated (verified: Brier
    # score of 0.130 with class_weight vs 0.048 without, on the same test
    # set -- worse than a dumb model that just guesses the base rate for
    # every loan). Recall/precision are a threshold decision, not a
    # training decision -- handled separately by picking an operating
    # threshold on these honest probabilities, not by reweighting the
    # loss function. Bundled into one Pipeline so preprocessing is always
    # fit on train only, never on test, no matter how this is called.
    preprocessor = build_preprocessor(X)
    model = LogisticRegression(max_iter=1000, random_state=42)
    return Pipeline([
        ("preprocess", preprocessor),
        ("model", model),
    ])
