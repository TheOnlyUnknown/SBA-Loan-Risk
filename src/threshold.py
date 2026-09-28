import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, f1_score

def threshold_sweep(y_true, y_proba):
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    precision = precision[:-1]
    recall = recall[:-1]
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) != 0,
    )
    base_rate = y_true.mean()
    lift = np.divide(precision, base_rate, out=np.zeros_like(precision), where=base_rate != 0)
    return pd.DataFrame({
        "threshold": thresholds,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "lift": lift,
    })

def best_f1_threshold(sweep_df):
    return sweep_df.loc[sweep_df["f1"].idxmax()]

def threshold_for_recall(sweep_df, min_recall):
    candidates = sweep_df[sweep_df["recall"] >= min_recall]
    if candidates.empty:
        return None
    return candidates.loc[candidates["threshold"].idxmax()]

def report_at_threshold(y_true, y_proba, threshold):
    y_pred = (y_proba >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = f1_score(y_true, y_pred, zero_division=0)
    base_rate = y_true.mean()
    lift = precision / base_rate if base_rate > 0 else 0.0
    return {
        "threshold": threshold,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "flagged_share": (y_pred == 1).mean(),
        "lift": lift,
    }
