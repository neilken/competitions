from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from tqdm.auto import tqdm


EXPECTED_ENTRIES = [
    "train_audio",
    "train_soundscapes",
    "train.csv",
    "taxonomy.csv",
    "sample_submission.csv",
    "train_soundscapes_labels.csv",
    "recording_location.txt",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate BirdCLEF expected data artifacts.")
    parser.add_argument("--data-root", required=True, help="Path to BirdCLEF data directory")
    parser.add_argument(
        "--raw-archive",
        default="",
        help="Optional path to raw Kaggle zip archive for size sanity check.",
    )
    args = parser.parse_args()
    print("[STEP 1/4] Validating configured data root...")

    data_root = Path(args.data_root)
    if not data_root.exists():
        print(f"[ERROR] data_root not found: {data_root}")
        return 1

    print("[STEP 2/4] Checking required entries with progress bar...")
    missing: List[str] = []
    for entry in tqdm(EXPECTED_ENTRIES, desc="data entries", unit="entry"):
        p = data_root / entry
        if not p.exists():
            missing.append(str(p))
            continue
        if p.is_file():
            print(f"[OK] {entry}: {p.stat().st_size} bytes")
        else:
            direct_count = sum(1 for _ in p.iterdir())
            print(f"[OK] {entry}: {direct_count} direct entries")

    if missing:
        print("[STEP 3/4] Missing entries detected.")
        print("[ERROR] Missing expected artifacts:")
        for m in missing:
            print(f"  - {m}")
        return 1

    print("[STEP 3/4] Core dataset entries validated.")
    if args.raw_archive:
        raw_archive = Path(args.raw_archive)
        if not raw_archive.exists():
            print(f"[WARN] raw archive path does not exist: {raw_archive}")
        else:
            print(f"[OK] Raw archive found: {raw_archive} ({raw_archive.stat().st_size} bytes)")

    print("[STEP 4/4] Data integrity checks complete.")
    print("[PASS] Data integrity checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
