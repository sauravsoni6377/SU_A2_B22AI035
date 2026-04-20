"""Audio I/O and framing helpers."""
from __future__ import annotations
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import soundfile as sf
import torch
import torchaudio


def load_wav(path: str | os.PathLike, sr: int = 16000, mono: bool = True) -> Tuple[torch.Tensor, int]:
    """Load via soundfile (no torchcodec dependency), then resample via torchaudio."""
    data, native_sr = sf.read(str(path), always_2d=True, dtype="float32")
    wav = torch.from_numpy(data.T)   # [C, T]
    if mono and wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if native_sr != sr:
        wav = torchaudio.functional.resample(wav, native_sr, sr)
    return wav, sr


def save_wav(path: str | os.PathLike, wav: torch.Tensor | np.ndarray, sr: int) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if isinstance(wav, torch.Tensor):
        wav = wav.detach().cpu().numpy()
    if wav.ndim == 2 and wav.shape[0] in (1, 2):
        wav = wav.T
    sf.write(str(path), wav.astype(np.float32), sr, subtype="PCM_16")


def peak_normalize(wav: torch.Tensor, peak: float = 0.97) -> torch.Tensor:
    m = wav.abs().max().clamp(min=1e-9)
    return wav * (peak / m)


def rms_normalize(wav: torch.Tensor, target_dbfs: float = -23.0) -> torch.Tensor:
    rms = wav.pow(2).mean().sqrt().clamp(min=1e-9)
    target = 10 ** (target_dbfs / 20.0)
    return wav * (target / rms)


def frame_signal(wav: torch.Tensor, sr: int, frame_ms: float, hop_ms: float) -> torch.Tensor:
    """Return [n_frames, frame_len] float tensor (mono)."""
    if wav.dim() == 2:
        wav = wav.squeeze(0)
    fl = int(sr * frame_ms / 1000)
    hl = int(sr * hop_ms / 1000)
    if wav.numel() < fl:
        pad = fl - wav.numel()
        wav = torch.nn.functional.pad(wav, (0, pad))
    frames = wav.unfold(0, fl, hl)
    return frames.contiguous()


def snr_db(clean: torch.Tensor, noisy: torch.Tensor) -> float:
    noise = noisy - clean
    p_s = clean.pow(2).mean().clamp(min=1e-12)
    p_n = noise.pow(2).mean().clamp(min=1e-12)
    return 10.0 * torch.log10(p_s / p_n).item()


def add_noise_for_target_snr(clean: torch.Tensor, noise: torch.Tensor, snr_db_target: float) -> torch.Tensor:
    p_s = clean.pow(2).mean().clamp(min=1e-12)
    p_n = noise.pow(2).mean().clamp(min=1e-12)
    target_p_n = p_s / (10 ** (snr_db_target / 10))
    scale = torch.sqrt(target_p_n / p_n)
    return clean + scale * noise
