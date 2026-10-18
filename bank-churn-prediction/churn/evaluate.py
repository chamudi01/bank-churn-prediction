"""Metric helpers shared by training and the dashboard."""
import numpy as np
from sklearn.metrics import (
    accuracy_score, average_precision_score, f1_score, precision_score,
    recall_score, roc_auc_score,
)


def metrics_at(y_true, prob, threshold=0.5) -> dict:
    y_true = np.asarray(y_true)
    pred = (np.asarray(prob) >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, prob)),
        "average_precision": float(average_precision_score(y_true, prob)),
    }


def confusion_at(y_true, prob, threshold) -> dict:
    y_true = np.asarray(y_true).astype(bool)
    pred = np.asarray(prob) >= threshold
    return {
        "tn": int((~y_true & ~pred).sum()),
        "fp": int((~y_true & pred).sum()),
        "fn": int((y_true & ~pred).sum()),
        "tp": int((y_true & pred).sum()),
    }


def threshold_sweep(y_true, prob, thresholds):
    """Precision, recall, F1 and flagged share across thresholds."""
    y_true = np.asarray(y_true).astype(bool)
    prob = np.asarray(prob)
    rows = []
    for t in thresholds:
        pred = prob >= t
        tp = int((y_true & pred).sum())
        fp = int((~y_true & pred).sum())
        fn = int((y_true & ~pred).sum())
        precision = tp / (tp + fp) if (tp + fp) else np.nan
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (tp and precision + recall) else 0.0
        rows.append({"threshold": float(t), "precision": precision, "recall": recall,
                     "f1": f1, "flagged_share": float(pred.mean())})
    return rows
