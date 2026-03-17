from __future__ import annotations

import argparse
import random
from pathlib import Path

from tqdm.auto import tqdm

try:
    import soundfile as sf
except ImportError:  # pragma: no cover
    sf = None


def sample_files(root: Path, limit: int) -> list[Path]:
    files = [p for p in root.rglob("*.ogg") if p.is_file()]
    if not files:
        return []
    random.shuffle(files)
    return files[:limit]


def main() -> int:
    parser = argparse.ArgumentParser(description="Decode random OGG files as an audio smoke test.")
    parser.add_argument("--train-audio-dir", required=True)
    parser.add_argument("--train-soundscapes-dir", required=True)
    parser.add_argument("--samples-per-dir", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    print("[STEP 1/4] Initializing audio smoke test...")

    random.seed(args.seed)
    if sf is None:
        print("[ERROR] soundfile is required. Install with: pip install soundfile")
        return 1
    dirs = [Path(args.train_audio_dir), Path(args.train_soundscapes_dir)]
    for d in dirs:
        if not d.exists():
            print(f"[ERROR] Directory not found: {d}")
            return 1

    print("[STEP 2/4] Sampling files from each directory...")
    failures = 0
    for d in dirs:
        picks = sample_files(d, args.samples_per_dir)
        print(f"[INFO] Testing {len(picks)} files from {d}")
        for path in tqdm(picks, desc=f"decode:{d.name}", unit="file"):
            try:
                wav, sr = sf.read(str(path), dtype="float32")
                frames = len(wav)
                if frames == 0:
                    raise ValueError("Decoded zero frames")
                print(f"[OK] {path.name} sr={sr} frames={frames}")
            except Exception as exc:
                failures += 1
                print(f"[FAIL] {path}: {exc}")

    print("[STEP 3/4] Decode loop complete.")
    if failures > 0:
        print(f"[FAIL] Audio smoke test failed with {failures} decode errors.")
        return 1

    print("[STEP 4/4] Audio smoke test complete.")
    print("[PASS] Audio smoke test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
