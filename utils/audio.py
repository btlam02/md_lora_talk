"""Các tiện ích trích xuất đặc trưng audio."""

from __future__ import annotations

from pathlib import Path
import numpy as np


def extract_f0_from_wav(
    wav_path: Path,
    mel_len: int,
    sr: int = 16000,
    frame_length: int = 1024,
    hop_length: int = 200,
) -> np.ndarray:
    """
    Trích xuất log-F0 chuẩn hóa bằng YIN và căn chỉnh theo độ dài mel.

    Kết quả có dạng ``(mel_len, 3)``: ``[log_f0, voiced_flag, delta_f0]``.
    """
    import librosa
    wav, sr = librosa.load(str(wav_path), sr=sr, mono=True)
    f0 = librosa.yin(
        wav,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=sr,
        frame_length=frame_length,
        hop_length=hop_length,
    ).astype(np.float32)

    if len(f0) < mel_len:
        f0 = np.pad(f0, (0, mel_len - len(f0)), mode="edge")
    elif len(f0) > mel_len:
        f0 = f0[:mel_len]

    voiced = np.isfinite(f0) & (f0 > 0)
    log_f0 = np.zeros_like(f0, dtype=np.float32)
    if voiced.any():
        raw = np.log(f0[voiced]).astype(np.float32)
        mean = raw.mean()
        std = raw.std()
        if std < 1e-4:
            std = 1.0
        log_f0[voiced] = (raw - mean) / std

    delta = np.zeros_like(log_f0, dtype=np.float32)
    if len(log_f0) > 1:
        delta[1:] = log_f0[1:] - log_f0[:-1]

    return np.stack([log_f0, voiced.astype(np.float32), delta], axis=1).astype(np.float32)
