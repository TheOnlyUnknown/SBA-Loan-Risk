import pandas as pd

DEFAULT_TRAIN_FRAC = 0.80

def time_aware_split(X: pd.DataFrame, y: pd.Series, train_frac: float = DEFAULT_TRAIN_FRAC):
    # Chronological split, not a random shuffle -- earlier approval dates
    # go to train, later ones go to test. This mirrors how the model will
    # actually be used: trained on the past, scored on loans that haven't
    # happened yet. interpolation="nearest" keeps the cutoff an actual
    # observed ApprovalDate instead of an artificial timestamp sitting
    # between two real ones.
    cutoff_date = X["ApprovalDate"].quantile(train_frac, interpolation="nearest")

    train_mask = X["ApprovalDate"] <= cutoff_date
    test_mask = ~train_mask

    X_train = X[train_mask].copy()
    X_test = X[test_mask].copy()
    y_train = y[train_mask].copy()
    y_test = y[test_mask].copy()

    return X_train, X_test, y_train, y_test, cutoff_date
