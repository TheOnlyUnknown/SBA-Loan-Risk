import pandas as pd

DEFAULT_TRAIN_FRAC = 0.80

def time_aware_split(X: pd.DataFrame, y: pd.Series, train_frac: float = DEFAULT_TRAIN_FRAC):
    cutoff_date = X["ApprovalDate"].quantile(train_frac, interpolation="nearest")

    train_mask = X["ApprovalDate"] <= cutoff_date
    test_mask = ~train_mask

    X_train = X[train_mask].copy()
    X_test = X[test_mask].copy()
    y_train = y[train_mask].copy()
    y_test = y[test_mask].copy()

    return X_train, X_test, y_train, y_test, cutoff_date
