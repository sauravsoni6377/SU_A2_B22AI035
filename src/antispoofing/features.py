"""LFCC and CQCC feature extractors for anti-spoofing — scipy/numpy only.

LFCC (Linear-Frequency Cepstral Coefficients): like MFCC but with linearly
spaced filterbank, which retains more high-frequency information — crucial
for spoof detection because synthetic-speech artefacts concentrate in
higher bands.

CQCC (Constant-Q Cepstral Coefficients): uses a logarithmic-frequency
filterbank (Constant-Q-like), then log + DCT. We use an FFT-based
approximation of CQT so the code has no librosa dependency.

Δ and ΔΔ are appended if include_delta=True.
"""
from __future__ import annotations
import numpy as np
from scipy.fftpack import dct
from scipy.signal import stft as _stft


def _frame_spec(y: np.ndarray, sr: int, n_fft: int = 512, hop: int = 160):
    f, t, Z = _stft(y, fs=sr, nperseg=n_fft, noverlap=n_fft - hop,
                    window="hann", padded=True, boundary="zeros")
    return np.abs(Z)  # [n_fft//2+1, T]


def _linear_filterbank(n_filters: int, n_fft: int, sr: int,
                       fmin: float = 0, fmax: float | None = None) -> np.ndarray:
    if fmax is None:
        fmax = sr / 2
    freqs = np.linspace(fmin, fmax, n_filters + 2)
    bins = np.floor((n_fft + 1) * freqs / sr).astype(int)
    fb = np.zeros((n_filters, n_fft // 2 + 1), dtype=np.float32)
    for m in range(1, n_filters + 1):
        lo, mid, hi = bins[m - 1], bins[m], bins[m + 1]
        for k in range(lo, mid):
            fb[m - 1, k] = (k - lo) / max(mid - lo, 1)
        for k in range(mid, hi):
            fb[m - 1, k] = (hi - k) / max(hi - mid, 1)
    return fb


def _log_filterbank(n_filters: int, n_fft: int, sr: int,
                    fmin: float = 30.0, fmax: float | None = None) -> np.ndarray:
    """Log-spaced triangular filterbank (CQT-ish approximation)."""
    if fmax is None:
        fmax = sr / 2
    freqs = np.geomspace(fmin, fmax, n_filters + 2)
    bins = np.floor((n_fft + 1) * freqs / sr).astype(int)
    fb = np.zeros((n_filters, n_fft // 2 + 1), dtype=np.float32)
    for m in range(1, n_filters + 1):
        lo, mid, hi = bins[m - 1], bins[m], bins[m + 1]
        for k in range(lo, mid):
            fb[m - 1, k] = (k - lo) / max(mid - lo, 1)
        for k in range(mid, hi):
            fb[m - 1, k] = (hi - k) / max(hi - mid, 1)
    return fb


def _delta(x: np.ndarray, axis: int = 0) -> np.ndarray:
    """Simple first-order delta. x: [T, F]."""
    padded = np.pad(x, ((1, 1), (0, 0)), mode="edge") if axis == 0 else \
             np.pad(x, ((0, 0), (1, 1)), mode="edge")
    if axis == 0:
        return (padded[2:] - padded[:-2]) / 2.0
    return (padded[:, 2:] - padded[:, :-2]) / 2.0


def lfcc(y: np.ndarray, sr: int = 16000, n_fft: int = 512, hop: int = 160,
         n_filters: int = 40, n_coeffs: int = 20, include_delta: bool = True) -> np.ndarray:
    spec = _frame_spec(y, sr, n_fft, hop)
    fb = _linear_filterbank(n_filters, n_fft, sr)
    energy = fb @ (spec ** 2)
    log_energy = np.log(np.maximum(energy, 1e-10))
    cepstra = dct(log_energy, type=2, axis=0, norm="ortho")[:n_coeffs]
    out = cepstra.T
    if include_delta:
        d = _delta(out); dd = _delta(d)
        out = np.concatenate([out, d, dd], axis=1)
    return out.astype(np.float32)


def cqcc(y: np.ndarray, sr: int = 16000, n_fft: int = 2048, hop: int = 160,
         n_filters: int = 96, n_coeffs: int = 20, include_delta: bool = True) -> np.ndarray:
    """Constant-Q-like cepstra via log-spaced filterbank + DCT."""
    spec = _frame_spec(y, sr, n_fft, hop)
    fb = _log_filterbank(n_filters, n_fft, sr)
    energy = fb @ (spec ** 2)
    log_energy = np.log(np.maximum(energy, 1e-10))
    cepstra = dct(log_energy, type=2, axis=0, norm="ortho")[:n_coeffs]
    out = cepstra.T
    if include_delta:
        d = _delta(out); dd = _delta(d)
        out = np.concatenate([out, d, dd], axis=1)
    return out.astype(np.float32)
