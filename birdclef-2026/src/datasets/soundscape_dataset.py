from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd
import torch
from torch.utils.data import Dataset

from src.features.audio_features import (
    build_multi_hot,
    load_audio_segment,
    parse_multilabel,
    waveform_to_logmel,
)


class SoundscapeSegmentDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        soundscape_dir: Path,
        class_to_idx: Dict[str, int],
        sample_rate: int,
        clip_seconds: float,
        n_mels: int,
        n_fft: int,
        hop_length: int,
    ) -> None:
        self.frame = frame.reset_index(drop=True).copy()
        self.soundscape_dir = soundscape_dir
        self.class_to_idx = class_to_idx
        self.sample_rate = sample_rate
        self.clip_seconds = clip_seconds
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int):
        row = self.frame.iloc[idx]
        audio_path = self.soundscape_dir / str(row["filename"])
        start_sec = float(row["start"])
        labels = parse_multilabel(str(row["primary_label"]))

        waveform = load_audio_segment(
            audio_path=audio_path,
            sample_rate=self.sample_rate,
            start_sec=start_sec,
            duration_sec=self.clip_seconds,
        )
        feature = waveform_to_logmel(
            waveform=waveform,
            sample_rate=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )
        target = torch.from_numpy(build_multi_hot(labels, self.class_to_idx))

        return {
            "row_id": str(row["row_id"]),
            "x": feature,
            "y": target,
        }
