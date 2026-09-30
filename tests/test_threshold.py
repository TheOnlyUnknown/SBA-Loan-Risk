import numpy as np

from src.threshold import threshold_sweep, best_f1_threshold, report_at_threshold

def test_report_at_threshold_computes_correct_confusion_counts():
    y_true = np.array([1, 1, 0, 0, 1])
    y_proba = np.array([0.9, 0.4, 0.8, 0.2, 0.6])
    report = report_at_threshold(y_true, y_proba, threshold=0.5)
    assert report["tp"] == 2
    assert report["fp"] == 1
    assert report["fn"] == 1
    assert report["tn"] == 1

def test_report_at_threshold_precision_recall_match_manual_calc():
    y_true = np.array([1, 1, 0, 0, 1])
    y_proba = np.array([0.9, 0.4, 0.8, 0.2, 0.6])
    report = report_at_threshold(y_true, y_proba, threshold=0.5)
    assert report["precision"] == 2 / 3
    assert report["recall"] == 2 / 3

def test_best_f1_threshold_picks_highest_f1_row():
    y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    y_proba = np.array([0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4])
    sweep = threshold_sweep(y_true, y_proba)
    best = best_f1_threshold(sweep)
    assert best["f1"] == sweep["f1"].max()

def test_threshold_sweep_lift_matches_precision_over_base_rate():
    y_true = np.array([1, 0, 0, 0])
    y_proba = np.array([0.9, 0.1, 0.2, 0.3])
    sweep = threshold_sweep(y_true, y_proba)
    base_rate = y_true.mean()
    row = sweep.iloc[0]
    assert abs(row["lift"] - row["precision"] / base_rate) < 1e-9

def test_report_at_threshold_handles_no_flagged_loans():
    y_true = np.array([1, 0, 1, 0])
    y_proba = np.array([0.1, 0.1, 0.2, 0.2])
    report = report_at_threshold(y_true, y_proba, threshold=0.9)
    assert report["tp"] == 0
    assert report["fp"] == 0
    assert report["precision"] == 0.0
