from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from tqdm.auto import tqdm

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("PyYAML is required. Install with: pip install pyyaml") from exc


REQUIRED_TRACKER_COLUMNS = [
    "resource_id",
    "source_url",
    "license",
    "publicly_accessible",
    "reasonably_accessible",
    "approved_for_use",
]


def normalize_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y"}


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def check_competition_attestations(cfg: dict) -> Tuple[List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []

    comp = cfg.get("competition", {})
    required_true = [
        "one_account_confirmed",
        "private_sharing_policy_confirmed",
        "hidden_test_not_used_for_training",
        "solo_mode_confirmed",
        "submission_notebook_mode_confirmed",
    ]

    for key in required_true:
        if not normalize_bool(comp.get(key, False)):
            errors.append(f"Competition attestation missing or false: {key}")

    if normalize_bool(comp.get("solo_mode_confirmed", False)) is False:
        warnings.append("Solo mode not confirmed. Verify team assumptions and sharing boundaries.")

    return errors, warnings


def check_submission_schema(cfg: dict) -> Tuple[List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []

    sub = cfg.get("submission", {})
    expected_filename = sub.get("expected_filename", "submission.csv")
    sample_path = Path(str(sub.get("sample_submission_path", "")))
    submission_path = Path(str(sub.get("submission_path", "")))
    if not normalize_bool(sub.get("cpu_only_confirmed", False)):
        errors.append("Submission config must confirm cpu_only_confirmed=true.")
    if not normalize_bool(sub.get("internet_disabled_confirmed", False)):
        errors.append("Submission config must confirm internet_disabled_confirmed=true.")

    if submission_path.name != expected_filename:
        errors.append(
            f"Submission filename must be '{expected_filename}', got '{submission_path.name}'."
        )

    if not sample_path.exists():
        errors.append(f"Sample submission not found: {sample_path}")
        return errors, warnings
    if not submission_path.exists():
        errors.append(f"Submission not found: {submission_path}")
        return errors, warnings

    sample_df = pd.read_csv(sample_path)
    pred_df = pd.read_csv(submission_path)

    sample_cols = list(sample_df.columns)
    pred_cols = list(pred_df.columns)

    if sample_cols != pred_cols:
        errors.append("Submission columns or order do not match sample_submission.csv exactly.")
        missing = sorted(set(sample_cols) - set(pred_cols))
        extra = sorted(set(pred_cols) - set(sample_cols))
        if missing:
            errors.append(f"Missing columns: {missing[:10]}{'...' if len(missing) > 10 else ''}")
        if extra:
            errors.append(f"Unexpected columns: {extra[:10]}{'...' if len(extra) > 10 else ''}")
        return errors, warnings

    if len(sample_df) != len(pred_df):
        errors.append(
            f"Submission row count mismatch. expected={len(sample_df)}, actual={len(pred_df)}"
        )

    if "row_id" in pred_df.columns:
        same_row_ids = sample_df["row_id"].astype(str).equals(pred_df["row_id"].astype(str))
        if not same_row_ids:
            errors.append("row_id values or ordering do not match sample submission.")

    prob_cols = [c for c in pred_df.columns if c != "row_id"]
    probs = pred_df[prob_cols]
    if probs.isna().any().any():
        errors.append("Submission contains NaN probability values.")
    if not probs.applymap(lambda x: isinstance(x, (int, float)) and math.isfinite(float(x))).all().all():
        errors.append("Submission contains non-numeric or non-finite probability values.")

    min_val = float(probs.min().min())
    max_val = float(probs.max().max())
    if min_val < 0.0 or max_val > 1.0:
        errors.append(f"Probabilities out of range [0, 1]. min={min_val:.6f}, max={max_val:.6f}")

    budget = sub.get("max_runtime_minutes_cpu", None)
    observed = sub.get("runtime_minutes_observed", None)
    if budget is not None and observed is not None:
        try:
            budget_f = float(budget)
            observed_f = float(observed)
            if observed_f > budget_f:
                errors.append(
                    f"Runtime budget exceeded. observed={observed_f:.2f} > allowed={budget_f:.2f} minutes."
                )
        except ValueError:
            warnings.append("Could not parse runtime budget or observed runtime as float.")
    elif budget is not None and observed is None:
        warnings.append("Runtime observed minutes are missing; budget check skipped.")

    return errors, warnings


def check_external_resources(cfg: dict) -> Tuple[List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []

    ext = cfg.get("external_resources", {})
    tracker_path = Path(str(ext.get("tracker_csv", "")))
    require_all_approved = normalize_bool(ext.get("require_all_rows_approved", True))

    if not tracker_path.exists():
        errors.append(f"External resource tracker not found: {tracker_path}")
        return errors, warnings

    tracker = pd.read_csv(tracker_path)
    missing_cols = [c for c in REQUIRED_TRACKER_COLUMNS if c not in tracker.columns]
    if missing_cols:
        errors.append(f"Tracker missing required columns: {missing_cols}")
        return errors, warnings

    active = tracker[tracker["resource_id"].astype(str).str.upper() != "BASE_NONE"].copy()
    if active.empty:
        warnings.append("No external resources listed in tracker yet.")
        return errors, warnings

    for _, row in active.iterrows():
        rid = str(row["resource_id"])
        if not normalize_bool(row["publicly_accessible"]):
            errors.append(f"Resource '{rid}' is not marked publicly accessible.")
        if not normalize_bool(row["reasonably_accessible"]):
            errors.append(f"Resource '{rid}' is not marked reasonably accessible.")
        if require_all_approved and not normalize_bool(row["approved_for_use"]):
            errors.append(f"Resource '{rid}' is not approved_for_use.")
        if str(row["source_url"]).strip() == "":
            errors.append(f"Resource '{rid}' missing source_url.")
        if str(row["license"]).strip() == "":
            errors.append(f"Resource '{rid}' missing license.")

    return errors, warnings


def run_gate(config_path: Path) -> int:
    print("[STEP 1/4] Loading rules gate config...")
    cfg = load_yaml(config_path)

    all_errors: List[str] = []
    all_warnings: List[str] = []

    print("[STEP 2/4] Running validation checks...")
    checks = (
        check_competition_attestations,
        check_external_resources,
        check_submission_schema,
    )
    for check in tqdm(checks, desc="rules checks", unit="check"):
        errors, warnings = check(cfg)
        all_errors.extend(errors)
        all_warnings.extend(warnings)

    print("[STEP 3/4] Building final report...")
    print("=== RULES GATE REPORT ===")
    print(f"Config: {config_path}")
    for w in all_warnings:
        print(f"[WARN] {w}")
    for e in all_errors:
        print(f"[ERROR] {e}")

    fail_on_warning = normalize_bool(cfg.get("reporting", {}).get("fail_on_warning", False))
    failed = len(all_errors) > 0 or (fail_on_warning and len(all_warnings) > 0)
    if failed:
        print("[STEP 4/4] Rules gate finished with failure state.")
        print("[FAIL] Rules gate did not pass.")
        return 1

    print("[STEP 4/4] Rules gate finished successfully.")
    print("[PASS] Rules gate passed.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run BirdCLEF rules gate checks.")
    p.add_argument(
        "--config",
        default=str(Path("configs") / "rules_gate.yaml"),
        help="Path to rules gate YAML config.",
    )
    return p


def main() -> int:
    args = build_parser().parse_args()
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"[ERROR] Config not found: {config_path}")
        return 1

    # Allow users to inject config path via environment in Colab scripts.
    os.environ["BIRDCLEF_RULES_GATE_CONFIG"] = str(config_path.resolve())
    return run_gate(config_path)


if __name__ == "__main__":
    raise SystemExit(main())
