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

    @staticmethod
    def _to_seconds(value: object) -> float:
        """
        Parse either numeric seconds or timestamp-like values into float seconds.
        Supported examples:
        - 10, 10.0, "10", "10.5"
        - "00:00:10", "00:10", "1:02:03.5"
        """
        text = str(value).strip()
        if text == "":
            raise ValueError("Empty timestamp value cannot be converted to seconds.")

        # Fast path for plain numeric values.
        try:
            return float(text)
        except ValueError:
            pass

        # Fallback for time-like strings.
        if ":" in text:
            parts = text.split(":")
            try:
                if len(parts) == 3:
                    hours = float(parts[0])
                    minutes = float(parts[1])
                    seconds = float(parts[2])
                    return hours * 3600.0 + minutes * 60.0 + seconds
                if len(parts) == 2:
                    minutes = float(parts[0])
                    seconds = float(parts[1])
                    return minutes * 60.0 + seconds
            except ValueError:
                pass

        td = pd.to_timedelta(text, errors="coerce")
        if pd.isna(td):
            raise ValueError(f"Could not convert timestamp value to seconds: {value!r}")
        return float(td.total_seconds())

    def __getitem__(self, idx: int):
        row = self.frame.iloc[idx]
        audio_path = self.soundscape_dir / str(row["filename"])
        start_sec = self._to_seconds(row["start"])
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
