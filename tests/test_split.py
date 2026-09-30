import pandas as pd

from src.split import time_aware_split

def test_time_aware_split_has_no_date_overlap():
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    X = pd.DataFrame({"ApprovalDate": dates})
    y = pd.Series(range(100))
    X_train, X_test, y_train, y_test, cutoff = time_aware_split(X, y, train_frac=0.8)
    assert X_train["ApprovalDate"].max() <= cutoff
    assert X_test["ApprovalDate"].min() > cutoff
    assert len(X_train) + len(X_test) == 100

def test_time_aware_split_respects_train_fraction():
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    X = pd.DataFrame({"ApprovalDate": dates})
    y = pd.Series(range(100))
    X_train, X_test, y_train, y_test, cutoff = time_aware_split(X, y, train_frac=0.8)
    assert 75 <= len(X_train) <= 85

def test_time_aware_split_keeps_labels_aligned_with_features():
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    X = pd.DataFrame({"ApprovalDate": dates})
    y = pd.Series(range(10))
    X_train, X_test, y_train, y_test, cutoff = time_aware_split(X, y, train_frac=0.7)
    assert X_train.index.equals(y_train.index)
    assert X_test.index.equals(y_test.index)
