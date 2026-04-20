"""Denoising + normalization.

Two backends:
  - 'deepfilternet' : deep learning SE (preferred)
  - 'spectral_sub'  : Boll 1979 spectral subtraction (fallback, no deps)

Spectral subtraction:  |X_clean(k,t)|^2 = max(|Y(k,t)|^2 - alpha * |N(k)|^2, beta * |Y(k,t)|^2)
with N estimated from the first 500 ms assumed non-speech.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
import torchaudio

from src.utils.audio import load_wav, save_wav, rms_normalize


def spectral_subtraction(wav: torch.Tensor, sr: int, alpha: float = 2.0, beta: float = 0.02,
                         noise_ms: int = 500, n_fft: int = 512, hop: int = 128) -> torch.Tensor:
    if wav.dim() == 2:
        wav = wav.squeeze(0)
    spec = torch.stft(wav, n_fft=n_fft, hop_length=hop, return_complex=True, window=torch.hann_window(n_fft))
    mag, phase = spec.abs(), torch.angle(spec)
    n_noise_frames = max(1, int(noise_ms / 1000 * sr / hop))
    noise_mag = mag[:, :n_noise_frames].mean(dim=1, keepdim=True)
    power = mag.pow(2)
    noise_power = noise_mag.pow(2)
    clean_power = torch.clamp(power - alpha * noise_power, min=beta * power)
    clean_mag = clean_power.sqrt()
    clean_spec = clean_mag * torch.exp(1j * phase)
    out = torch.istft(clean_spec, n_fft=n_fft, hop_length=hop, window=torch.hann_window(n_fft),
                      length=wav.numel())
    return out.unsqueeze(0)


def deepfilter_denoise(wav: torch.Tensor, sr: int) -> torch.Tensor:
    try:
        from df.enhance import enhance, init_df
    except Exception as e:
        raise RuntimeError("DeepFilterNet not installed; use spectral_sub backend") from e
    model, df_state, _ = init_df()
    if sr != df_state.sr():
        wav = torchaudio.functional.resample(wav, sr, df_state.sr())
    enhanced = enhance(model, df_state, wav)
    if sr != df_state.sr():
        enhanced = torchaudio.functional.resample(enhanced, df_state.sr(), sr)
    return enhanced


def run(in_wav: str, out_wav: str, sr: int = 16000, backend: str = "deepfilternet",
        target_dbfs: float = -23.0, high_pass_hz: int = 80) -> str:
    wav, _ = load_wav(in_wav, sr=sr)
    # Highpass to cut HVAC rumble typical of classrooms
    if high_pass_hz:
        wav = torchaudio.functional.highpass_biquad(wav, sr, high_pass_hz)
    if backend == "deepfilternet":
        try:
            wav = deepfilter_denoise(wav, sr)
        except RuntimeError:
            wav = spectral_subtraction(wav, sr)
    elif backend == "spectral_sub":
        wav = spectral_subtraction(wav, sr)
    else:
        raise ValueError(f"unknown denoiser backend: {backend}")
    wav = rms_normalize(wav, target_dbfs=target_dbfs).clamp(-1.0, 1.0)
    save_wav(out_wav, wav, sr)
    return out_wav


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_wav", required=True)
    ap.add_argument("--out_wav", required=True)
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--backend", default="deepfilternet")
    args = ap.parse_args()
    print(run(args.in_wav, args.out_wav, args.sr, args.backend))
