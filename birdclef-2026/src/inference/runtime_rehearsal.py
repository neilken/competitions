from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Runtime rehearsal wrapper for submission generation.")
    parser.add_argument("--config", default="configs/inference_cpu_submission.yaml")
    parser.add_argument("--max-minutes", type=float, default=90.0)
    args = parser.parse_args()

    print("[STEP 1/3] Starting runtime rehearsal...")
    cmd = ["python", "-m", "src.inference.run_submission_from_config", "--config", args.config]
    print("[RUN] " + " ".join(cmd))

    t0 = time.time()
    proc = subprocess.run(cmd, check=False)
    elapsed_min = (time.time() - t0) / 60.0

    print("[STEP 2/3] Rehearsal run completed.")
    print(f"[INFO] elapsed_minutes={elapsed_min:.2f}")
    if proc.returncode != 0:
        print(f"[FAIL] Submission generation failed with code {proc.returncode}.")
        return proc.returncode

    print("[STEP 3/3] Checking runtime budget...")
    if elapsed_min > args.max_minutes:
        print(
            f"[FAIL] Runtime budget exceeded. elapsed={elapsed_min:.2f} > "
            f"budget={args.max_minutes:.2f} minutes"
        )
        return 1

    print("[PASS] Runtime rehearsal within budget.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
