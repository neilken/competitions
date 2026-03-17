from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from src.models.cnn_baseline import BirdCLEFBaselineCNN


def main() -> int:
    parser = argparse.ArgumentParser(description="Check deterministic inference for baseline model.")
    parser.add_argument("--num-classes", type=int, default=234)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--n-mels", type=int, default=128)
    parser.add_argument("--time-steps", type=int, default=313)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.use_deterministic_algorithms(True)

    model = BirdCLEFBaselineCNN(num_classes=args.num_classes).eval()
    x = torch.randn(args.batch_size, 1, args.n_mels, args.time_steps)
    with torch.no_grad():
        y1 = model(x)
        y2 = model(x)

    max_abs_diff = float((y1 - y2).abs().max().item())
    print(f"[INFO] max_abs_diff={max_abs_diff:.12f}")
    if max_abs_diff > 1e-12:
        print("[FAIL] Determinism check failed.")
        return 1
    print("[PASS] Determinism check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
