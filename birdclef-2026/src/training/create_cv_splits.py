from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Tuple

import pandas as pd
from tqdm.auto import tqdm

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("PyYAML is required. Install with: pip install pyyaml") from exc

try:
    from sklearn.model_selection import GroupKFold
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("scikit-learn is required. Install with: pip install scikit-learn") from exc


def to_seconds(value: object) -> float:
    text = str(value).strip()
    if text == "":
        raise ValueError("Empty time value cannot be converted to seconds.")
    try:
        return float(text)
    except ValueError:
        pass
    if ":" in text:
        parts = text.split(":")
        try:
            if len(parts) == 3:
                h = float(parts[0])
                m = float(parts[1])
                s = float(parts[2])
                return h * 3600.0 + m * 60.0 + s
            if len(parts) == 2:
                m = float(parts[0])
                s = float(parts[1])
                return m * 60.0 + s
        except ValueError:
            pass
    td = pd.to_timedelta(text, errors="coerce")
    if pd.isna(td):
        raise ValueError(f"Could not convert time value to seconds: {value!r}")
    return float(td.total_seconds())


def parse_site_and_date(filename: str, pattern: re.Pattern[str]) -> Tuple[str, str]:
    match = pattern.match(filename)
    if not match:
        return "", ""
    return match.group(1), match.group(2)


def build_group_id(row: pd.Series, pattern: re.Pattern[str]) -> str:
    site, date = parse_site_and_date(str(row["filename"]), pattern)
    if site and date:
        return f"{site}_{date}"
    return f"fallback_{row['filename']}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Create leakage-resistant GroupKFold splits.")
    parser.add_argument(
        "--config",
        default=str(Path("configs") / "cv_policy.yaml"),
        help="Path to CV policy YAML.",
    )
    args = parser.parse_args()
    print("[STEP 1/6] Loading CV config...")

    cfg_path = Path(args.config)
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    labels_csv = Path(str(cfg["input"]["labels_csv"]))
    folds_csv = Path(str(cfg["output"]["folds_csv"]))
    n_splits = int(cfg["cv_policy"]["n_splits"])
    regex = cfg["cv_policy"]["parser_regex"]
    pattern = re.compile(regex)
    print(f"[INFO] config={cfg_path}")
    print(f"[INFO] labels_csv={labels_csv}")
    print(f"[INFO] folds_csv={folds_csv}")
    print(f"[INFO] n_splits={n_splits}")

    if not labels_csv.exists():
        raise FileNotFoundError(f"labels_csv not found: {labels_csv}")

    print("[STEP 2/6] Reading labels CSV...")
    df = pd.read_csv(labels_csv)
    required_cols = {"filename", "start", "end", "primary_label"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in labels CSV: {sorted(missing)}")

    print("[STEP 3/6] Normalizing time columns and building row_id/group_id...")
    df = df.copy()
    tqdm.pandas(desc="time_to_seconds")
    df["start"] = df["start"].progress_apply(to_seconds).astype(float)
    df["end"] = df["end"].progress_apply(to_seconds).astype(float)
    df["row_id"] = (
        df["filename"].astype(str)
        + "_"
        + df["start"].astype(str)
        + "_"
        + df["end"].astype(str)
    )
    tqdm.pandas(desc="group_id")
    # Progress apply makes long metadata transforms visible in Colab.
    df["group_id"] = df.progress_apply(lambda row: build_group_id(row, pattern), axis=1)
    df["fold"] = -1

    print("[STEP 4/6] Assigning GroupKFold splits...")
    gkf = GroupKFold(n_splits=n_splits)
    for fold, (_, valid_idx) in tqdm(
        enumerate(gkf.split(df, groups=df["group_id"])),
        total=n_splits,
        desc="fold assignment",
        unit="fold",
    ):
        df.loc[valid_idx, "fold"] = fold

    if (df["fold"] < 0).any():
        raise RuntimeError("Some rows did not receive a fold assignment.")

    print("[STEP 5/6] Writing folds CSV...")
    out_cols = ["row_id", "filename", "start", "end", "primary_label", "group_id", "fold"]
    folds_csv.parent.mkdir(parents=True, exist_ok=True)
    df[out_cols].to_csv(folds_csv, index=False)

    print("[STEP 6/6] Reporting split summary...")
    fallback_count = int(df["group_id"].astype(str).str.startswith("fallback_").sum())
    print(f"[OK] Wrote folds to: {folds_csv}")
    print(f"[INFO] Rows: {len(df)} | Groups: {df['group_id'].nunique()} | Fallback parse count: {fallback_count}")
    print("[INFO] Fold distribution:")
    print(df["fold"].value_counts().sort_index())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
