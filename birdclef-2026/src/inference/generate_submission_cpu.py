from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from src.features.audio_features import waveform_to_logmel
from src.models.model_factory import create_model

try:
    import soundfile as sf
except ImportError:  # pragma: no cover
    sf = None


ROW_ID_PATTERN = re.compile(r"^(?P<fname>.+)_(?P<end>\d+)$")


def parse_row_id(row_id: str) -> Tuple[str, int]:
    match = ROW_ID_PATTERN.match(row_id)
    if not match:
        raise ValueError(f"Invalid row_id format: {row_id}")
    return match.group("fname"), int(match.group("end"))


def load_full_waveform(path: Path) -> Tuple[np.ndarray, int]:
    if sf is None:
        raise RuntimeError("soundfile is required. Install with: pip install soundfile")
    wav, sr = sf.read(str(path), dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    return wav, sr


def slice_segment(waveform: np.ndarray, sample_rate: int, end_sec: int, clip_sec: int = 5) -> np.ndarray:
    end_frame = int(end_sec * sample_rate)
    start_frame = max(0, end_frame - int(clip_sec * sample_rate))
    segment = waveform[start_frame:end_frame]
    need = int(clip_sec * sample_rate)
    if len(segment) < need:
        padded = np.zeros(need, dtype=np.float32)
        padded[: len(segment)] = segment
        segment = padded
    elif len(segment) > need:
        segment = segment[:need]
    return segment.astype(np.float32)


def validate_submission(sample_df: pd.DataFrame, pred_df: pd.DataFrame) -> None:
    if list(sample_df.columns) != list(pred_df.columns):
        raise ValueError("Submission columns/order must exactly match sample_submission.csv")
    if len(sample_df) != len(pred_df):
        raise ValueError(f"Row count mismatch. expected={len(sample_df)} actual={len(pred_df)}")
    if not sample_df["row_id"].astype(str).equals(pred_df["row_id"].astype(str)):
        raise ValueError("row_id ordering/values must match sample_submission.csv")
    prob_cols = [c for c in pred_df.columns if c != "row_id"]
    if pred_df[prob_cols].isna().any().any():
        raise ValueError("Submission contains NaN values.")
    min_val = float(pred_df[prob_cols].min().min())
    max_val = float(pred_df[prob_cols].max().max())
    if min_val < 0.0 or max_val > 1.0:
        raise ValueError(f"Probability bounds violated. min={min_val}, max={max_val}")


def run_model_inference(
    sample_df: pd.DataFrame,
    soundscape_dir: Path,
    checkpoint_path: Path,
    sample_rate: int,
    n_mels: int,
    n_fft: int,
    hop_length: int,
    batch_size: int,
) -> pd.DataFrame:
    print("[STEP A] Loading checkpoint and preparing model...")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    classes: List[str] = list(ckpt["classes"])
    model_family = str(ckpt.get("model_family", "cnn_logmel_baseline"))
    model = create_model(model_family=model_family, num_classes=len(classes))
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"[INFO] checkpoint={checkpoint_path}")
    print(f"[INFO] model_family={model_family}")
    print(f"[INFO] num_classes={len(classes)}")

    # If checkpoint stored feature config, prefer it unless explicit CLI override differs.
    ckpt_feat = ckpt.get("feature_config", {}) or {}
    if ckpt_feat:
        sample_rate = int(ckpt_feat.get("sample_rate", sample_rate))
        n_mels = int(ckpt_feat.get("n_mels", n_mels))
        n_fft = int(ckpt_feat.get("fft_size", n_fft))
        hop_length = int(ckpt_feat.get("hop_length", hop_length))
        print(
            "[INFO] Using feature config from checkpoint: "
            f"sr={sample_rate}, n_mels={n_mels}, n_fft={n_fft}, hop={hop_length}"
        )

    expected_cols = ["row_id"] + classes
    if set(expected_cols) != set(sample_df.columns):
        missing = sorted(set(expected_cols) - set(sample_df.columns))
        if missing:
            raise ValueError(f"sample_submission missing class columns expected by checkpoint: {missing[:10]}")

    file_to_rows: Dict[str, List[Tuple[int, str]]] = defaultdict(list)
    print("[STEP B] Indexing row_id windows by source soundscape...")
    for row_id in sample_df["row_id"].astype(str):
        fname_no_ext, end_sec = parse_row_id(row_id)
        fname = f"{fname_no_ext}.ogg"
        file_to_rows[fname].append((end_sec, row_id))

    pred_map: Dict[str, np.ndarray] = {}
    print("[STEP C] Running soundscape inference with progress bars...")
    for fname, row_specs in tqdm(file_to_rows.items(), desc="soundscapes"):
        wav_path = soundscape_dir / fname
        waveform, sr = load_full_waveform(wav_path)
        if sr != sample_rate:
            x_old = np.linspace(0.0, 1.0, num=len(waveform), endpoint=False)
            x_new = np.linspace(0.0, 1.0, num=int(len(waveform) * sample_rate / sr), endpoint=False)
            waveform = np.interp(x_new, x_old, waveform).astype(np.float32)
            sr = sample_rate

        features = []
        row_order = []
        for end_sec, row_id in sorted(row_specs):
            seg = slice_segment(waveform, sample_rate=sr, end_sec=end_sec, clip_sec=5)
            feat = waveform_to_logmel(
                waveform=seg,
                sample_rate=sample_rate,
                n_mels=n_mels,
                n_fft=n_fft,
                hop_length=hop_length,
            )
            features.append(feat)
            row_order.append(row_id)

        for i in tqdm(
            range(0, len(features), batch_size),
            desc=f"segments:{fname}",
            leave=False,
            unit="batch",
        ):
            batch_feats = torch.stack(features[i : i + batch_size], dim=0)
            with torch.no_grad():
                probs = torch.sigmoid(model(batch_feats)).cpu().numpy()
            for row_id, pred in zip(row_order[i : i + batch_size], probs):
                pred_map[row_id] = pred

    out = sample_df.copy()
    print("[STEP D] Mapping probabilities back to submission rows...")
    for col_idx, col in enumerate(classes):
        out[col] = out["row_id"].map(lambda rid: float(pred_map[str(rid)][col_idx]))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate submission.csv from model checkpoint or prediction matrix.")
    parser.add_argument("--sample-submission", required=True, help="Path to sample_submission.csv")
    parser.add_argument("--output", required=True, help="Output path for submission CSV")
    parser.add_argument("--predictions", default="", help="Optional direct predictions CSV path")
    parser.add_argument("--checkpoint", default="", help="Model checkpoint for direct inference")
    parser.add_argument("--soundscape-dir", default="", help="Directory containing test soundscapes")
    parser.add_argument("--sample-rate", type=int, default=32000)
    parser.add_argument("--n-mels", type=int, default=128)
    parser.add_argument("--n-fft", type=int, default=2048)
    parser.add_argument("--hop-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    print("[STEP 1/5] Resolving paths and loading sample submission...")

    sample_path = Path(args.sample_submission)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not sample_path.exists():
        raise FileNotFoundError(f"sample_submission not found: {sample_path}")

    sample_df = pd.read_csv(sample_path)
    print(f"[INFO] sample_submission={sample_path}")
    print(f"[INFO] rows={len(sample_df)} cols={len(sample_df.columns)}")

    pred_df: pd.DataFrame
    pred_path = Path(args.predictions) if args.predictions else None
    ckpt_path = Path(args.checkpoint) if args.checkpoint else None
    soundscape_dir = Path(args.soundscape_dir) if args.soundscape_dir else None

    if pred_path and pred_path.exists():
        print("[STEP 2/5] Loading direct predictions file...")
        pred_df = pd.read_csv(pred_path)
        print(f"[INFO] Loaded predictions CSV: {pred_path}")
    elif ckpt_path and soundscape_dir:
        print("[STEP 2/5] Running checkpoint-based inference...")
        print(f"[INFO] checkpoint={ckpt_path}")
        print(f"[INFO] soundscape_dir={soundscape_dir}")
        pred_df = run_model_inference(
            sample_df=sample_df,
            soundscape_dir=soundscape_dir,
            checkpoint_path=ckpt_path,
            sample_rate=args.sample_rate,
            n_mels=args.n_mels,
            n_fft=args.n_fft,
            hop_length=args.hop_length,
            batch_size=args.batch_size,
        )
        print(f"[INFO] Generated predictions from checkpoint: {ckpt_path}")
    else:
        print("[STEP 2/5] No model outputs provided; creating zero baseline...")
        pred_df = sample_df.copy()
        prob_cols = [c for c in pred_df.columns if c != "row_id"]
        pred_df[prob_cols] = 0.0
        print("[WARN] No predictions or checkpoint provided. Writing all-zero baseline submission.")

    print("[STEP 3/5] Validating submission schema and probability range...")
    validate_submission(sample_df, pred_df)
    print("[STEP 4/5] Writing submission CSV...")
    pred_df.to_csv(output_path, index=False)
    print(f"[OK] Submission written: {output_path}")
    print("[STEP 5/5] Final checks complete.")
    if output_path.name != "submission.csv":
        print("[WARN] Output filename is not submission.csv. Rename for scored submission.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
