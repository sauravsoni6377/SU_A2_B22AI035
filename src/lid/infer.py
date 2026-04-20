"""Run trained LID on a wav and emit (start_ms, end_ms, lang) segments."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch
import yaml

from src.lid.model import MultiHeadLID
from src.utils.audio import load_wav


def load_model(ckpt_path: str, device: str = "cpu") -> MultiHeadLID:
    state = torch.load(ckpt_path, map_location=device)
    cfg = state["cfg"]
    m = MultiHeadLID(n_classes=cfg["lid"]["n_classes"], d=cfg["lid"]["hidden"],
                     n_heads=cfg["lid"]["n_heads"], n_layers=cfg["lid"]["n_layers"],
                     sr=cfg["audio"]["sr"], hop_ms=cfg["audio"]["hop_ms"]).to(device)
    m.load_state_dict(state["model"])
    m.eval()
    return m


def segment(wav_path: str, ckpt_path: str, smoothing_ms: float = 80, device: str = "cpu"):
    model = load_model(ckpt_path, device)
    wav, sr = load_wav(wav_path, sr=16000)
    wav = wav.to(device)
    segs = model.decode(wav, smoothing_ms=smoothing_ms)
    # Merge tiny language islands (<120 ms) into neighbour (cleans flicker)
    merged = []
    for s in segs:
        dur = s["end_ms"] - s["start_ms"]
        if merged and dur < 120 and s["lang"] != merged[-1]["lang"]:
            merged[-1]["end_ms"] = s["end_ms"]
        else:
            merged.append(dict(s))
    return merged


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True)
    ap.add_argument("--ckpt", default="checkpoints/lid_best.pt")
    ap.add_argument("--out", default="outputs/lid_segments.json")
    a = ap.parse_args()
    segs = segment(a.wav, a.ckpt)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(segs, open(a.out, "w"), indent=2)
    print(f"wrote {len(segs)} segments → {a.out}")
