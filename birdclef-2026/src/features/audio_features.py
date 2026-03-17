from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import numpy as np
import torch

try:
    import soundfile as sf
except ImportError:  # pragma: no cover
    sf = None


def parse_multilabel(value: str) -> List[str]:
    text = str(value or "").strip()
    if not text:
        return []
    return [token.strip() for token in text.split(";") if token.strip()]


def load_audio_segment(
    audio_path: Path,
    sample_rate: int,
    start_sec: float,
    duration_sec: float,
) -> np.ndarray:
    if sf is None:
        raise RuntimeError("soundfile is required. Install with: pip install soundfile")
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    start_frame = int(max(0, start_sec) * sample_rate)
    num_frames = int(duration_sec * sample_rate)
    stop_frame = start_frame + num_frames

    wav, sr = sf.read(
        str(audio_path),
        start=start_frame,
        stop=stop_frame,
        dtype="float32",
        always_2d=True,
    )
    if wav.size == 0:
        wav = np.zeros((num_frames, 1), dtype=np.float32)

    wav = wav.mean(axis=1)  # mono
    if sr != sample_rate:
        # Lightweight linear resample fallback.
        x_old = np.linspace(0.0, 1.0, num=len(wav), endpoint=False)
        x_new = np.linspace(0.0, 1.0, num=num_frames, endpoint=False)
        wav = np.interp(x_new, x_old, wav).astype(np.float32)

    if len(wav) < num_frames:
        padded = np.zeros(num_frames, dtype=np.float32)
        padded[: len(wav)] = wav
        wav = padded
    elif len(wav) > num_frames:
        wav = wav[:num_frames]

    return wav.astype(np.float32)


def build_multi_hot(labels: Iterable[str], class_to_idx: dict[str, int]) -> np.ndarray:
    target = np.zeros(len(class_to_idx), dtype=np.float32)
    for label in labels:
        idx = class_to_idx.get(label)
        if idx is not None:
            target[idx] = 1.0
    return target


def waveform_to_logmel(
    waveform: np.ndarray,
    sample_rate: int,
    n_mels: int,
    n_fft: int,
    hop_length: int,
    eps: float = 1e-6,
) -> torch.Tensor:
    wav_t = torch.from_numpy(waveform).float()
    if wav_t.ndim != 1:
        wav_t = wav_t.flatten()

    window = torch.hann_window(n_fft, device=wav_t.device)
    stft = torch.stft(
        wav_t,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        return_complex=True,
    )
    power = stft.abs().pow(2.0)  # [freq, time]

    mel_filter = _mel_filter_bank(
        sample_rate=sample_rate,
        n_fft=n_fft,
        n_mels=n_mels,
        fmin=0.0,
        fmax=sample_rate / 2.0,
    ).to(power.device)
    mel = mel_filter @ power
    logmel = torch.log(mel + eps)
    return logmel.unsqueeze(0)  # [1, n_mels, time]


def _hz_to_mel(hz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10 ** (mel / 2595.0) - 1.0)


def _mel_filter_bank(
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    fmin: float,
    fmax: float,
) -> torch.Tensor:
    num_spectrogram_bins = n_fft // 2 + 1
    fft_freqs = np.linspace(0.0, sample_rate / 2.0, num_spectrogram_bins)

    mel_min = _hz_to_mel(np.array([fmin]))[0]
    mel_max = _hz_to_mel(np.array([fmax]))[0]
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = _mel_to_hz(mel_points)

    filter_bank = np.zeros((n_mels, num_spectrogram_bins), dtype=np.float32)
    for i in range(n_mels):
        left, center, right = hz_points[i], hz_points[i + 1], hz_points[i + 2]
        left_slope = (fft_freqs - left) / (center - left + 1e-12)
        right_slope = (right - fft_freqs) / (right - center + 1e-12)
        filter_bank[i] = np.maximum(0.0, np.minimum(left_slope, right_slope))

    return torch.from_numpy(filter_bank)
