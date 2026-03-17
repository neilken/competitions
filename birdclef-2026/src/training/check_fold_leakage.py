from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm


def main() -> int:
    parser = argparse.ArgumentParser(description="Check fold leakage and fold integrity.")
    parser.add_argument("--folds-csv", required=True, help="Path to folds.csv")
    parser.add_argument("--group-col", default="group_id", help="Grouping column to enforce non-overlap")
    parser.add_argument("--fold-col", default="fold", help="Fold column name")
    args = parser.parse_args()

    folds_csv = Path(args.folds_csv)
    if not folds_csv.exists():
        print(f"[ERROR] folds_csv not found: {folds_csv}")
        return 1

    print("[STEP 1/4] Loading folds CSV...")
    df = pd.read_csv(folds_csv)
    required = {"row_id", args.fold_col, args.group_col}
    missing = required - set(df.columns)
    if missing:
        print(f"[ERROR] Missing required columns: {sorted(missing)}")
        return 1

    print("[STEP 2/4] Checking fold assignment completeness...")
    if df[args.fold_col].isna().any():
        print("[ERROR] Found rows with missing fold assignment.")
        return 1
    if (df[args.fold_col] < 0).any():
        print("[ERROR] Found rows with negative fold assignment.")
        return 1

    print("[STEP 3/4] Checking group leakage across folds...")
    leak_count = 0
    fold_values = sorted(df[args.fold_col].unique().tolist())
    group_sets = {
        fold: set(df.loc[df[args.fold_col] == fold, args.group_col].astype(str).tolist())
        for fold in fold_values
    }
    for i, f1 in tqdm(enumerate(fold_values), total=len(fold_values), desc="fold pairs", unit="fold"):
        for f2 in fold_values[i + 1 :]:
            overlap = group_sets[f1].intersection(group_sets[f2])
            if overlap:
                leak_count += len(overlap)
                print(
                    f"[ERROR] Leakage between fold {f1} and fold {f2}: "
                    f"{len(overlap)} shared groups"
                )

    print("[STEP 4/4] Reporting summary...")
    print(f"[INFO] rows={len(df)} unique_groups={df[args.group_col].nunique()} folds={len(fold_values)}")
    print(df[args.fold_col].value_counts().sort_index())
    if leak_count > 0:
        print(f"[FAIL] Leakage detected. shared_group_count={leak_count}")
        return 1
    print("[PASS] No leakage detected across folds.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
