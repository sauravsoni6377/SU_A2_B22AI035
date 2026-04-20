"""Generate a tiny CM dataset from the pipeline outputs.

Slices `student_voice_ref.wav` (60 s) into 4-s bonafide clips (with
optional pitch-shift augmentation), and slices the cloned LRL output
into 4-s spoof clips. Also holds out 20% as a test set.

Usage:
    python scripts/prepare_cm_data.py \\
        --ref data/raw/student_voice_ref.wav \\
        --spoof outputs/output_LRL_cloned.wav
"""
from __future__ import annotations
import argparse
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

from src.utils.audio import load_wav, save_wav


def slice_wav(src: str, out_dir: str, label: str, chunk_s: float = 4.0,
              hop_s: float = 2.0, augment: bool = False, sr: int = 16000):
    wav, _ = load_wav(src, sr=sr)
    wav = wav.squeeze(0)
    n = int(chunk_s * sr); hop = int(hop_s * sr)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    idx = 0
    for s in range(0, wav.numel() - n + 1, hop):
        chunk = wav[s:s + n]
        save_wav(f"{out_dir}/{label}_{idx:04d}.wav", chunk.unsqueeze(0), sr)
        idx += 1
        if augment:
            # Pitch-shift ±1 semitone for augmentation (bonafide only)
            for semis in (-1, 1):
                shifted = torchaudio.functional.pitch_shift(chunk.unsqueeze(0), sr, semis)
                save_wav(f"{out_dir}/{label}_{idx:04d}.wav", shifted, sr)
                idx += 1
    return idx


def split_trainval(in_dir: str, train_dir: str, val_dir: str, val_frac: float = 0.2,
                   seed: int = 42):
    random.seed(seed)
    Path(train_dir).mkdir(parents=True, exist_ok=True)
    Path(val_dir).mkdir(parents=True, exist_ok=True)
    files = sorted(Path(in_dir).glob("*.wav"))
    random.shuffle(files)
    n_val = max(1, int(len(files) * val_frac))
    for f in files[:n_val]:
        f.rename(Path(val_dir) / f.name)
    for f in files[n_val:]:
        f.rename(Path(train_dir) / f.name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="data/raw/student_voice_ref.wav")
    ap.add_argument("--spoof", default="outputs/output_LRL_cloned.wav")
    a = ap.parse_args()

    import shutil
    for d in ("data/cm/bonafide", "data/cm/spoof", "data/cm_test/bonafide",
              "data/cm_test/spoof", "data/cm/_all/bonafide", "data/cm/_all/spoof"):
        shutil.rmtree(d, ignore_errors=True)

    n_bf = slice_wav(a.ref, "data/cm/_all/bonafide", "bf", augment=True)
    print(f"bonafide clips: {n_bf}")
    n_sp = slice_wav(a.spoof, "data/cm/_all/spoof", "sp", augment=False)
    print(f"spoof clips:    {n_sp}")

    split_trainval("data/cm/_all/bonafide", "data/cm/bonafide", "data/cm_test/bonafide")
    split_trainval("data/cm/_all/spoof",    "data/cm/spoof",    "data/cm_test/spoof")
    shutil.rmtree("data/cm/_all", ignore_errors=True)
    print("CM data ready.")


if __name__ == "__main__":
    main()
