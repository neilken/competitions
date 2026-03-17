from __future__ import annotations

import argparse
import copy
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("PyYAML is required. Install with: pip install pyyaml") from exc


def set_nested(config: dict[str, Any], key: str, value: Any) -> None:
    parts = key.split(".")
    node = config
    for p in parts[:-1]:
        if p not in node or not isinstance(node[p], dict):
            node[p] = {}
        node = node[p]
    node[parts[-1]] = value


def main() -> int:
    parser = argparse.ArgumentParser(description="Run ablation matrix by generating temporary configs.")
    parser.add_argument("--base-config", default="configs/baseline_colab.yaml")
    parser.add_argument("--ablation-config", default="configs/ablations.yaml")
    parser.add_argument(
        "--train-module",
        default="src.training.run_baseline",
        help="Python module used for training command.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    print("[STEP 1/4] Loading base and ablation configs...")

    with Path(args.base_config).open("r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)
    with Path(args.ablation_config).open("r", encoding="utf-8") as f:
        ablation_cfg = yaml.safe_load(f)

    entries = ablation_cfg.get("ablations", [])
    if not entries:
        print("[ERROR] No ablation entries found.")
        return 1

    print(f"[INFO] total_ablations={len(entries)}")
    print("[STEP 2/4] Building per-ablation configs and running training...")
    for entry in tqdm(entries, desc="ablations", unit="run"):
        run_cfg = copy.deepcopy(base_cfg)
        ablation_id = str(entry["id"])
        overrides = entry.get("overrides", {})
        for k, v in overrides.items():
            set_nested(run_cfg, k, v)

        run_cfg["project"]["run_name"] = f"{run_cfg['project']['run_name']}_{ablation_id}"
        run_cfg["reproducibility"]["config_id"] = f"{run_cfg['reproducibility']['config_id']}_{ablation_id}"
        run_cfg["reproducibility"]["checkpoint_id"] = (
            f"{run_cfg['reproducibility']['checkpoint_id']}_{ablation_id}"
        )

        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as tf:
            yaml.safe_dump(run_cfg, tf, sort_keys=False)
            temp_cfg = Path(tf.name)

        cmd = ["python", "-m", args.train_module, "--config", str(temp_cfg)]
        print(f"[ABLATION] {ablation_id}")
        print(f"[RUN] {' '.join(cmd)}")
        if args.dry_run:
            continue

        proc = subprocess.run(cmd, check=False)
        if proc.returncode != 0:
            print(f"[FAIL] Ablation {ablation_id} failed with exit code {proc.returncode}")
            return proc.returncode

    print("[STEP 3/4] Completed ablation command loop.")
    print("[STEP 4/4] Ablation run finished.")
    print("[PASS] Ablation run completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
