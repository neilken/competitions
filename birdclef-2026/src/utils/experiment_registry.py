from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List


DEFAULT_COLUMNS: List[str] = [
    "run_id",
    "timestamp_utc",
    "phase",
    "config_id",
    "seed",
    "fold_strategy",
    "model_family",
    "feature_version",
    "checkpoint_id",
    "oof_metric_public",
    "oof_metric_private_proxy",
    "runtime_minutes",
    "status",
    "notes",
]


def ensure_registry(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=DEFAULT_COLUMNS)
        writer.writeheader()


def append_row(path: Path, values: Dict[str, str]) -> None:
    ensure_registry(path)
    row = {k: "" for k in DEFAULT_COLUMNS}
    row.update(values)
    if not row["timestamp_utc"]:
        row["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=DEFAULT_COLUMNS)
        writer.writerow(row)


def parse_key_values(items: List[str]) -> Dict[str, str]:
    values: Dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --set value '{item}'. Expected key=value format.")
        key, value = item.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Append one experiment record row.")
    p.add_argument("--registry", required=True, help="Path to experiment_registry.csv")
    p.add_argument(
        "--set",
        nargs="+",
        default=[],
        help="key=value pairs to set in row. Example: run_id=abc123 status=completed",
    )
    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    registry_path = Path(args.registry)
    values = parse_key_values(args.set)
    append_row(registry_path, values)
    print(f"[OK] Appended experiment row to {registry_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
