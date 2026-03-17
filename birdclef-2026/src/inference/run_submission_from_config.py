from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("PyYAML is required. Install with: pip install pyyaml") from exc


def main() -> int:
    parser = argparse.ArgumentParser(description="Run submission generation from YAML config.")
    parser.add_argument("--config", default="configs/inference_cpu_submission.yaml")
    args = parser.parse_args()
    print("[STEP 1/3] Loading inference config...")

    cfg_path = Path(args.config)
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    paths = cfg["paths"]
    print(f"[INFO] config={cfg_path}")
    print("[STEP 2/3] Building submission command from config...")
    cmd = [
        "python",
        "-m",
        "src.inference.generate_submission_cpu",
        "--sample-submission",
        str(paths["sample_submission_csv"]),
        "--output",
        str(paths["submission_csv"]),
    ]
    predictions_csv = str(paths.get("predictions_csv", "")).strip()
    if predictions_csv:
        cmd.extend(["--predictions", predictions_csv])
    checkpoint = str(paths.get("checkpoint_path", "")).strip()
    soundscape_dir = str(paths.get("soundscape_dir", "")).strip()
    if checkpoint and soundscape_dir:
        cmd.extend(["--checkpoint", checkpoint, "--soundscape-dir", soundscape_dir])

    print("[RUN] " + " ".join(cmd))
    print("[STEP 3/3] Executing submission generation...")
    proc = subprocess.run(cmd, check=False)
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
