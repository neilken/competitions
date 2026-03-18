from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
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
        fail_on_error: bool = False,
        max_error_logs: int = 20,
        cache_features: bool = False,
    ) -> None:
        self.frame = frame.reset_index(drop=True).copy()
        self.soundscape_dir = soundscape_dir
        self.class_to_idx = class_to_idx
        self.sample_rate = sample_rate
        self.clip_seconds = clip_seconds
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.fail_on_error = fail_on_error
        self.max_error_logs = max(0, int(max_error_logs))
        self._error_count = 0
        self.cache_features = bool(cache_features)
        self._feature_cache: Dict[int, tuple[str, torch.Tensor, torch.Tensor]] = {}

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
        if self.cache_features and idx in self._feature_cache:
            row_id, feature, target = self._feature_cache[idx]
            return {
                "row_id": row_id,
                "x": feature,
                "y": target,
            }

        row = self.frame.iloc[idx]
        row_id = str(row["row_id"])
        labels = parse_multilabel(str(row["primary_label"]))
        try:
            audio_path = self.soundscape_dir / str(row["filename"])
            start_sec = self._to_seconds(row["start"])
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
        except Exception as exc:
            if self.fail_on_error:
                raise
            self._error_count += 1
            if self._error_count <= self.max_error_logs:
                print(
                    "[WARN][dataset] "
                    f"idx={idx} filename={row.get('filename', '<unknown>')} "
                    f"start={row.get('start', '<unknown>')} error={exc}. "
                    "Using zero-waveform fallback.",
                    flush=True,
                )
            elif self._error_count == self.max_error_logs + 1:
                print(
                    "[WARN][dataset] Additional sample-load errors suppressed.",
                    flush=True,
                )

            fallback_samples = int(self.sample_rate * self.clip_seconds)
            waveform = np.zeros(fallback_samples, dtype=np.float32)
            feature = waveform_to_logmel(
                waveform=waveform,
                sample_rate=self.sample_rate,
                n_mels=self.n_mels,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
            )

        target = torch.from_numpy(build_multi_hot(labels, self.class_to_idx))

        sample = {
            "row_id": row_id,
            "x": feature,
            "y": target,
        }
        if self.cache_features:
            self._feature_cache[idx] = (row_id, feature, target)
        return sample
