from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List


DEFAULT_COLUMNS: List[str] = [
    "submission_id",
    "timestamp_utc",
    "run_id",
    "config_id",
    "is_final_candidate",
    "public_lb_score",
    "runtime_minutes_cpu",
    "rules_gate_passed",
    "notes",
]


def ensure_log(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=DEFAULT_COLUMNS)
        writer.writeheader()


def parse_key_values(items: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --set item: '{item}'. Expected key=value.")
        key, val = item.split("=", 1)
        out[key.strip()] = val.strip()
    return out


def append_row(path: Path, values: Dict[str, str]) -> None:
    ensure_log(path)
    row = {k: "" for k in DEFAULT_COLUMNS}
    row.update(values)
    if not row["timestamp_utc"]:
        row["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=DEFAULT_COLUMNS)
        writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(description="Append a submission log row.")
    parser.add_argument("--log-csv", required=True, help="Path to submission_log.csv")
    parser.add_argument(
        "--set",
        nargs="+",
        default=[],
        help="key=value fields for row. Example: submission_id=123 run_id=abc rules_gate_passed=true",
    )
    args = parser.parse_args()

    values = parse_key_values(args.set)
    append_row(Path(args.log_csv), values)
    print(f"[OK] Appended submission log row to {args.log_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
