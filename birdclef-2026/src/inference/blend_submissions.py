from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd
from tqdm.auto import tqdm


def main() -> int:
    parser = argparse.ArgumentParser(description="Blend multiple submission CSVs by weighted average.")
    parser.add_argument("--inputs", nargs="+", required=True, help="Input submission CSV paths")
    parser.add_argument("--weights", nargs="*", default=[], help="Optional weights matching --inputs")
    parser.add_argument("--output", required=True, help="Output blended submission path")
    parser.add_argument(
        "--clip-thresholds",
        default="",
        help="Optional CSV with columns: class_name,threshold; clips values below threshold to 0.",
    )
    args = parser.parse_args()
    print("[STEP 1/4] Loading input submissions...")

    inputs = [Path(p) for p in args.inputs]
    for p in inputs:
        if not p.exists():
            raise FileNotFoundError(f"Input submission not found: {p}")

    if args.weights:
        if len(args.weights) != len(inputs):
            raise ValueError("Number of --weights must match number of --inputs")
        weights = [float(w) for w in args.weights]
    else:
        weights = [1.0] * len(inputs)

    frames = []
    for p in tqdm(inputs, desc="read submissions", unit="file"):
        frames.append(pd.read_csv(p))
    base_cols = list(frames[0].columns)
    for idx, f in enumerate(frames[1:], start=1):
        if list(f.columns) != base_cols:
            raise ValueError(f"Schema mismatch in input index {idx}")
        if not frames[0]["row_id"].astype(str).equals(f["row_id"].astype(str)):
            raise ValueError(f"row_id mismatch in input index {idx}")

    prob_cols = [c for c in base_cols if c != "row_id"]
    total_w = sum(weights)
    out = frames[0][["row_id"]].copy()
    out[prob_cols] = 0.0
    print("[STEP 2/4] Blending probabilities...")
    for w, f in tqdm(list(zip(weights, frames)), desc="blend models", unit="model"):
        out[prob_cols] += f[prob_cols] * (w / total_w)

    if args.clip_thresholds:
        print("[STEP 3/4] Applying optional class thresholds...")
        thr_df = pd.read_csv(args.clip_thresholds)
        for _, row in thr_df.iterrows():
            cname = str(row["class_name"])
            thr = float(row["threshold"])
            if cname in out.columns:
                out.loc[out[cname] < thr, cname] = 0.0

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print("[STEP 4/4] Blend output written.")
    print(f"[OK] Blended submission written: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
