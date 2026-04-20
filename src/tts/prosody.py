"""Prosody extraction + Dynamic Time Warping transfer.

Pipeline:
  1. Extract F0 (pyworld DIO+StoneMask) and RMS energy from professor's lecture.
  2. Extract the same from the flat-synthesised TTS output (before transfer).
  3. Run DTW on a joint feature (log-F0 ⊕ log-energy) to align src→tgt frames.
  4. Warp the target contours: replace F0 and energy of target with *scaled*
     source values so that the *style* transfers but the target speaker
     identity (controlled by x-vector) is preserved.
  5. Resynthesise with pyworld CheapTrick/D4C + Synthesize using target
     spectral envelope and aperiodicity, but warped F0/energy.

Why this preserves "teaching style":
  - F0 contour encodes intonation (emphasis, questioning rises)
  - Energy envelope encodes emphasis/pacing
  The cloned voice (same spectrum) now tracks the professor's prosody.

Input:  reference_prof.wav, cloned_flat.wav  (same 22.05 kHz)
Output: cloned_warped.wav
"""
from __future__ import annotations
from pathlib import Path
import math
import numpy as np
import torch
import torchaudio

from src.utils.audio import load_wav, save_wav


def _safe_pyworld():
    try:
        import pyworld as pw
        return pw
    except ImportError as e:
        raise RuntimeError("pyworld is required for prosody transfer") from e


def extract_prosody(wav: np.ndarray, sr: int):
    pw = _safe_pyworld()
    wav = wav.astype(np.float64)
    f0, t = pw.dio(wav, sr, frame_period=5.0)
    f0 = pw.stonemask(wav, f0, t, sr)
    sp = pw.cheaptrick(wav, f0, t, sr)
    ap = pw.d4c(wav, f0, t, sr)
    energy = np.sqrt(np.mean(sp, axis=1) + 1e-9)
    return f0, sp, ap, energy, t


def dtw_path(src: np.ndarray, tgt: np.ndarray, sub: int | None = None):
    """Classical DTW with diagonal-preferred transitions on 1-D inputs.

    For long sequences we subsample before DTW to keep the cost O(N/sub × M/sub).
    The returned path is upsampled back to original-frame indices.
    """
    n0, m0 = len(src), len(tgt)
    if sub is None:
        # Auto-pick subsample so the accumulator fits in ~2M cells (< 100 ms on CPU)
        target_cells = 2_000_000
        sub = max(1, int(math.ceil(math.sqrt(n0 * m0 / target_cells))))
    src_s = src[::sub]; tgt_s = tgt[::sub]
    n, m = len(src_s), len(tgt_s)
    D = np.abs(src_s[:, None] - tgt_s[None, :]).astype(np.float32)
    acc = np.full((n + 1, m + 1), np.inf, dtype=np.float32)
    acc[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            acc[i, j] = D[i - 1, j - 1] + min(acc[i - 1, j - 1],
                                              acc[i - 1, j],
                                              acc[i, j - 1])
    i, j, path_s = n, m, []
    while i > 0 and j > 0:
        path_s.append((i - 1, j - 1))
        step = int(np.argmin([acc[i - 1, j - 1], acc[i - 1, j], acc[i, j - 1]]))
        if step == 0: i, j = i - 1, j - 1
        elif step == 1: i -= 1
        else: j -= 1
    path_s.reverse()
    # Upsample back to original indices (linearly interpolate between anchors)
    if sub == 1:
        return path_s
    path = []
    for a in range(len(path_s) - 1):
        (i0, j0), (i1, j1) = path_s[a], path_s[a + 1]
        for k in range(sub):
            ii = min(i0 * sub + k, n0 - 1)
            jj = min(j0 * sub + k, m0 - 1)
            path.append((ii, jj))
    last_i = min(path_s[-1][0] * sub, n0 - 1)
    last_j = min(path_s[-1][1] * sub, m0 - 1)
    path.append((last_i, last_j))
    return path


def warp_contours(src_f0: np.ndarray, src_e: np.ndarray,
                  tgt_f0: np.ndarray, tgt_e: np.ndarray):
    """Project source (prof) onto target (cloned) timeline via DTW; return
    new (warped) F0 and energy aligned to target frames."""
    # Use voiced-only log-F0 for alignment (avoid zeros)
    s = np.log(np.maximum(src_f0, 1.0))
    t = np.log(np.maximum(tgt_f0, 1.0))
    path = dtw_path(s, t)
    new_f0 = np.zeros_like(tgt_f0)
    new_e = np.zeros_like(tgt_e)
    counts = np.zeros(len(tgt_f0))
    # Normalise source F0 range to target's to preserve speaker identity
    s_med, t_med = np.median(src_f0[src_f0 > 0]), np.median(tgt_f0[tgt_f0 > 0])
    scale = (t_med / s_med) if s_med > 1 else 1.0
    for si, tj in path:
        new_f0[tj] += src_f0[si] * scale
        new_e[tj] += src_e[si] * (np.median(tgt_e) / (np.median(src_e) + 1e-9))
        counts[tj] += 1
    new_f0 /= np.maximum(counts, 1)
    new_e /= np.maximum(counts, 1)
    new_f0[tgt_f0 == 0] = 0   # keep unvoiced frames unvoiced
    return new_f0, new_e


def resynthesise(f0: np.ndarray, sp: np.ndarray, ap: np.ndarray,
                 energy_scale: np.ndarray, sr: int) -> np.ndarray:
    pw = _safe_pyworld()
    # Scale spectral envelope by energy ratio so warped loudness takes effect
    orig_e = np.sqrt(np.mean(sp, axis=1) + 1e-9)
    gain = energy_scale / (orig_e + 1e-9)
    sp_scaled = sp * (gain[:, None] ** 2)
    y = pw.synthesize(f0, sp_scaled.astype(np.float64),
                      ap.astype(np.float64), sr, 5.0)
    return y.astype(np.float32)


def transfer_prosody(src_wav: str, tgt_wav: str, out_wav: str,
                     sr: int = 22050) -> str:
    src, _ = load_wav(src_wav, sr=sr)
    tgt, _ = load_wav(tgt_wav, sr=sr)
    src = src.squeeze(0).numpy()
    tgt = tgt.squeeze(0).numpy()
    s_f0, _, _, s_e, _ = extract_prosody(src, sr)
    t_f0, t_sp, t_ap, t_e, _ = extract_prosody(tgt, sr)
    new_f0, new_e = warp_contours(s_f0, s_e, t_f0, t_e)
    y = resynthesise(new_f0, t_sp, t_ap, new_e, sr)
    save_wav(out_wav, torch.from_numpy(y).unsqueeze(0), sr)
    return out_wav


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="professor lecture wav")
    ap.add_argument("--tgt", required=True, help="flat cloned wav")
    ap.add_argument("--out", default="outputs/output_LRL_cloned.wav")
    ap.add_argument("--sr", type=int, default=22050)
    a = ap.parse_args()
    transfer_prosody(a.src, a.tgt, a.out, a.sr)
    print(f"prosody-warped wav → {a.out}")
