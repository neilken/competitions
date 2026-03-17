from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def macro_roc_auc_skip_empty(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true={y_true.shape}, y_pred={y_pred.shape}")

    per_class = []
    for c in range(y_true.shape[1]):
        y_t = y_true[:, c]
        y_p = y_pred[:, c]
        positives = int((y_t > 0.5).sum())
        negatives = int((y_t <= 0.5).sum())
        if positives == 0 or negatives == 0:
            continue
        try:
            per_class.append(float(roc_auc_score(y_t, y_p)))
        except ValueError:
            continue

    if not per_class:
        return float("nan")
    return float(np.mean(per_class))
