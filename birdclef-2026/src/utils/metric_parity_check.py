from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from src.utils.metrics import macro_roc_auc_skip_empty


def main() -> int:
    rng = np.random.default_rng(2026)
    n = 500
    c = 10
    y_true = rng.integers(0, 2, size=(n, c)).astype(np.float32)
    y_true[:, 0] = 0.0  # force an all-negative class that should be skipped
    y_pred = rng.random((n, c), dtype=np.float32)

    metric_val = macro_roc_auc_skip_empty(y_true, y_pred)
    ref_vals = []
    for i in range(c):
        if y_true[:, i].sum() == 0 or y_true[:, i].sum() == n:
            continue
        ref_vals.append(roc_auc_score(y_true[:, i], y_pred[:, i]))
    ref = float(np.mean(ref_vals))

    delta = abs(metric_val - ref)
    print(f"[INFO] metric={metric_val:.8f} reference={ref:.8f} delta={delta:.10f}")
    if delta > 1e-8:
        print("[FAIL] Metric parity check failed.")
        return 1
    print("[PASS] Metric parity check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
