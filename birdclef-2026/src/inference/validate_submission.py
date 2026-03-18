from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate submission.csv against sample_submission.csv.")
    parser.add_argument("--sample-submission", required=True)
    parser.add_argument("--submission", required=True)
    args = parser.parse_args()

    print("[STEP 1/4] Loading sample and candidate submissions...")
    sample_path = Path(args.sample_submission)
    sub_path = Path(args.submission)
    if not sample_path.exists():
        print(f"[ERROR] sample submission not found: {sample_path}")
        return 1
    if not sub_path.exists():
        print(f"[ERROR] submission not found: {sub_path}")
        return 1

    sample_df = pd.read_csv(sample_path)
    sub_df = pd.read_csv(sub_path)

    print("[STEP 2/4] Checking schema and row identity...")
    if list(sample_df.columns) != list(sub_df.columns):
        print("[ERROR] Column mismatch versus sample_submission.csv")
        return 1
    if len(sample_df) != len(sub_df):
        print(f"[ERROR] Row count mismatch: expected={len(sample_df)} actual={len(sub_df)}")
        return 1
    if not sample_df["row_id"].astype(str).equals(sub_df["row_id"].astype(str)):
        print("[ERROR] row_id mismatch or ordering mismatch.")
        return 1

    print("[STEP 3/4] Checking probability values...")
    prob_cols = [c for c in sub_df.columns if c != "row_id"]
    probs = sub_df[prob_cols]
    if probs.isna().any().any():
        print("[ERROR] Found NaN in probabilities.")
        return 1

    numeric_probs = probs.apply(pd.to_numeric, errors="coerce")
    if numeric_probs.isna().any().any():
        print("[ERROR] Found non-numeric probability values.")
        return 1
    if not np.isfinite(numeric_probs.to_numpy(dtype=np.float64)).all():
        print("[ERROR] Found non-finite probability values.")
        return 1

    min_val = float(numeric_probs.min().min())
    max_val = float(numeric_probs.max().max())
    if min_val < 0.0 or max_val > 1.0:
        print(f"[ERROR] Probability out of [0,1] range: min={min_val}, max={max_val}")
        return 1

    print("[STEP 4/4] Validation complete.")
    print(f"[PASS] Submission is valid. rows={len(sub_df)} cols={len(sub_df.columns)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
