"""Score arbitrary wav files with the trained CM."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch

from src.antispoofing.features import lfcc, cqcc
from src.antispoofing.model import CMModel, score
from src.utils.audio import load_wav


def load_cm(ckpt_path: str, device: str = "cpu"):
    state = torch.load(ckpt_path, map_location=device)
    model = CMModel(in_dim=state["in_dim"],
                    hidden=state["cfg"]["antispoof"]["hidden"]).to(device)
    model.load_state_dict(state["model"]); model.eval()
    return model, state


def score_file(wav_path: str, ckpt_path: str, device: str = "cpu") -> float:
    m, st = load_cm(ckpt_path, device)
    w, _ = load_wav(wav_path, sr=16000)
    w = w.squeeze(0).numpy()
    feat = lfcc(w) if st["feat"] == "lfcc" else cqcc(w)
    x = torch.from_numpy(feat).unsqueeze(0).to(device)
    with torch.no_grad():
        s = score(m(x))[0]
    return float(s)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True)
    ap.add_argument("--ckpt", default="checkpoints/cm_best.pt")
    a = ap.parse_args()
    s = score_file(a.wav, a.ckpt)
    verdict = "bonafide" if s >= 0.5 else "spoof"
    print(f"{a.wav}: score={s:.3f} → {verdict}")
