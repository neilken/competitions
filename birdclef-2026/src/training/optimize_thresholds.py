from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from tqdm.auto import tqdm


def parse_labels_to_set(value: str) -> set[str]:
    text = str(value or "").strip()
    if not text:
        return set()
    return {tok.strip() for tok in text.split(";") if tok.strip()}


def build_targets(frame: pd.DataFrame, classes: List[str]) -> np.ndarray:
    label_sets = frame["primary_label"].astype(str).map(parse_labels_to_set).tolist()
    y = np.zeros((len(frame), len(classes)), dtype=np.int32)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    for i, labels in enumerate(label_sets):
        for label in labels:
            idx = class_to_idx.get(label)
            if idx is not None:
                y[i, idx] = 1
    return y


def main() -> int:
    parser = argparse.ArgumentParser(description="Optimize class-wise thresholds from OOF predictions.")
    parser.add_argument("--oof-csv", required=True, help="OOF prediction CSV path")
    parser.add_argument("--folds-csv", required=True, help="folds CSV path with primary_label")
    parser.add_argument("--output-csv", required=True, help="Output thresholds CSV path")
    parser.add_argument("--min-threshold", type=float, default=0.01)
    parser.add_argument("--max-threshold", type=float, default=0.99)
    parser.add_argument("--steps", type=int, default=25)
    args = parser.parse_args()

    print("[STEP 1/5] Loading OOF and fold label data...")
    oof_path = Path(args.oof_csv)
    folds_path = Path(args.folds_csv)
    if not oof_path.exists():
        print(f"[ERROR] oof_csv not found: {oof_path}")
        return 1
    if not folds_path.exists():
        print(f"[ERROR] folds_csv not found: {folds_path}")
        return 1

    oof = pd.read_csv(oof_path)
    folds = pd.read_csv(folds_path)
    if "row_id" not in oof.columns or "row_id" not in folds.columns:
        print("[ERROR] Both files must contain row_id.")
        return 1
    if "primary_label" not in folds.columns:
        print("[ERROR] folds_csv must contain primary_label for threshold optimization.")
        return 1

    print("[STEP 2/5] Joining OOF predictions with labels...")
    merged = oof.merge(folds[["row_id", "primary_label"]], on="row_id", how="inner")
    if len(merged) == 0:
        print("[ERROR] No overlapping rows between OOF and folds.")
        return 1

    class_cols = [c for c in oof.columns if c not in {"row_id", "fold"}]
    y_true = build_targets(merged, class_cols)
    y_pred = merged[class_cols].to_numpy(dtype=np.float32)

    print("[STEP 3/5] Sweeping thresholds per class...")
    thr_grid = np.linspace(args.min_threshold, args.max_threshold, num=args.steps, dtype=np.float32)
    rows: List[Dict[str, float]] = []
    for class_idx, class_name in tqdm(enumerate(class_cols), total=len(class_cols), desc="classes", unit="class"):
        yt = y_true[:, class_idx]
        yp = y_pred[:, class_idx]
        positives = int(yt.sum())
        if positives == 0:
            rows.append(
                {
                    "class_name": class_name,
                    "threshold": 0.5,
                    "f1_score": np.nan,
                    "positives": positives,
                }
            )
            continue

        best_thr = 0.5
        best_f1 = -1.0
        for thr in thr_grid:
            pred_bin = (yp >= float(thr)).astype(np.int32)
            score = f1_score(yt, pred_bin, zero_division=0)
            if score > best_f1:
                best_f1 = float(score)
                best_thr = float(thr)
        rows.append(
            {
                "class_name": class_name,
                "threshold": best_thr,
                "f1_score": best_f1,
                "positives": positives,
            }
        )

    print("[STEP 4/5] Writing threshold table...")
    out_df = pd.DataFrame(rows).sort_values("class_name").reset_index(drop=True)
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    print("[STEP 5/5] Reporting summary...")
    valid_scores = out_df["f1_score"].dropna()
    mean_f1 = float(valid_scores.mean()) if not valid_scores.empty else float("nan")
    print(f"[OK] Wrote thresholds: {out_path}")
    print(f"[INFO] classes={len(out_df)} mean_class_f1={mean_f1:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
