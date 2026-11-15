import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def binary_metrics(labels, probabilities, threshold=0.5):
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    predictions = (probabilities >= threshold).astype(np.int64)

    auc = float("nan")
    if len(np.unique(labels)) == 2:
        auc = float(roc_auc_score(labels, probabilities))

    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0

    return {
        "auc": auc,
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "threshold": float(threshold),
    }


def find_best_threshold(labels, probabilities, strategy="youden"):
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)

    thresholds = np.unique(probabilities)
    if thresholds.size == 0:
        return 0.5

    best_threshold = 0.5
    best_score = -np.inf

    for threshold in thresholds:
        metrics = binary_metrics(labels, probabilities, threshold=threshold)
        if strategy == "f1":
            score = metrics["f1"]
        elif strategy == "balanced_accuracy":
            score = (metrics["sensitivity"] + metrics["specificity"]) / 2.0
        else:
            score = metrics["sensitivity"] + metrics["specificity"] - 1.0

        if score > best_score:
            best_score = score
            best_threshold = float(threshold)

    return best_threshold
