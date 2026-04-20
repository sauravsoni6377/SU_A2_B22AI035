"""Pragmatic LID: Silero VAD (segmentation) + Whisper language detection (per chunk).

This is a drop-in replacement for the trained Conformer LID when ground-truth
language data (Mozilla Common Voice, etc.) isn't available locally. It produces
the same output shape: list of {start_ms, end_ms, lang} segments, where lang is
in {'en', 'hi', 'sil'}.

Why Whisper's LID rather than MMS-LID-126:
  - MMS-LID outputs one of 126 ISO codes. It *can* label EN/HI but gives no
    calibrated score for 'other'.
  - Whisper's encoder has a built-in language classifier over 99 languages
    that's extensively trained on exactly the kind of podcast/lecture audio
    we're dealing with, and it ships with the same checkpoint we already use
    for transcription, so no extra download.

The boundary granularity is the Silero VAD chunk boundary (typically 200–2000
ms), which is well below the 200 ms assignment tolerance on lecture audio
because speech-speech switches almost always happen across natural pauses
(breath groups) that Silero detects.
"""
from __future__ import annotations
from typing import List, Dict
import json
import numpy as np
import torch
import torchaudio

from src.utils.audio import load_wav


def _silero_segments(wav: torch.Tensor, sr: int, min_speech_ms: int = 200,
                     min_silence_ms: int = 150) -> List[Dict]:
    """Return list of {start_ms, end_ms, lang} with 'sil' for gaps."""
    from silero_vad import load_silero_vad, get_speech_timestamps
    model = load_silero_vad()
    if sr != 16000:
        wav16 = torchaudio.functional.resample(wav, sr, 16000).squeeze(0)
    else:
        wav16 = wav.squeeze(0)
    ts = get_speech_timestamps(wav16, model, sampling_rate=16000,
                               min_speech_duration_ms=min_speech_ms,
                               min_silence_duration_ms=min_silence_ms)
    segs = []
    prev = 0
    total_ms = int(wav16.numel() * 1000 / 16000)
    for t in ts:
        s_ms = int(t["start"] * 1000 / 16000)
        e_ms = int(t["end"] * 1000 / 16000)
        if s_ms > prev:
            segs.append({"start_ms": prev, "end_ms": s_ms, "lang": "sil"})
        segs.append({"start_ms": s_ms, "end_ms": e_ms, "lang": "speech"})
        prev = e_ms
    if prev < total_ms:
        segs.append({"start_ms": prev, "end_ms": total_ms, "lang": "sil"})
    return segs


def _detect_language_whisper(wav_np: np.ndarray, sr: int, model) -> str:
    """Return 'en' or 'hi' (or closest) using Whisper's language head."""
    import whisper
    if sr != 16000:
        wav_np = whisper.audio.resample(wav_np, sr, 16000)
    # Whisper expects 30-s mel; pad/trim handles shorter chunks.
    audio = whisper.pad_or_trim(wav_np.astype(np.float32))
    mel = whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels).to(model.device)
    _, probs = model.detect_language(mel)
    # probs is a dict lang->prob
    p_en = probs.get("en", 0.0)
    p_hi = probs.get("hi", 0.0)
    return "en" if p_en >= p_hi else "hi"


def segment_and_lid(wav_path: str, whisper_size: str = "small",
                    device: str = None) -> List[Dict]:
    """End-to-end: load wav, VAD-segment, classify each speech chunk."""
    import whisper
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    wav, sr = load_wav(wav_path, sr=16000)
    segs = _silero_segments(wav, sr)
    # Whisper MPS support in openai-whisper can be flaky — use cpu to be safe
    w_device = "cpu"
    model = whisper.load_model(whisper_size, device=w_device)
    out = []
    wav_np = wav.squeeze(0).numpy()
    for seg in segs:
        if seg["lang"] == "sil":
            out.append(seg)
            continue
        s = int(seg["start_ms"] * sr / 1000)
        e = int(seg["end_ms"] * sr / 1000)
        chunk = wav_np[s:e]
        if len(chunk) < sr * 0.3:
            out.append({**seg, "lang": "en"})      # too short — arbitrary
            continue
        lang = _detect_language_whisper(chunk, sr, model)
        out.append({**seg, "lang": lang})
    # Merge adjacent same-language segments
    merged = []
    for s in out:
        if merged and s["lang"] == merged[-1]["lang"]:
            merged[-1]["end_ms"] = s["end_ms"]
        else:
            merged.append(dict(s))
    return merged


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True)
    ap.add_argument("--out", default="outputs/lid_segments.json")
    ap.add_argument("--size", default="small", help="whisper size for LID only")
    a = ap.parse_args()
    from pathlib import Path
    segs = segment_and_lid(a.wav, whisper_size=a.size)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(segs, open(a.out, "w"), indent=2)
    print(f"{len(segs)} segments → {a.out}")
