"""Speaker embedding extractor (d-vector / x-vector).

Uses SpeechBrain's pretrained ECAPA-TDNN ("spkrec-ecapa-voxceleb") as the
default high-dimensional embedder (192-dim). ECAPA is the industry standard
for zero-shot speaker verification and voice cloning conditioning.

Given 60 s of student_voice_ref.wav we produce a single L2-normalised 192-dim
vector (averaged over 3-s windows with 50% overlap — more robust than a single
pass on a long utterance).
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
import torchaudio

from src.utils.audio import load_wav


def _load_ecapa(device: str = "cpu"):
    from speechbrain.inference.speaker import EncoderClassifier
    return EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="checkpoints/ecapa",
        run_opts={"device": device},
    )


def extract(wav_path: str, out_path: str, sr: int = 16000,
            win_s: float = 3.0, hop_s: float = 1.5, device: str = None) -> np.ndarray:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _load_ecapa(device)
    wav, _ = load_wav(wav_path, sr=sr)
    wav = wav.squeeze(0)
    win = int(win_s * sr); hop = int(hop_s * sr)
    embs = []
    if wav.numel() <= win:
        embs.append(model.encode_batch(wav.unsqueeze(0).to(device))
                    .squeeze().detach().cpu().numpy())
    else:
        for s in range(0, wav.numel() - win + 1, hop):
            chunk = wav[s:s + win].unsqueeze(0).to(device)
            e = model.encode_batch(chunk).squeeze().detach().cpu().numpy()
            embs.append(e)
    emb = np.mean(np.stack(embs, axis=0), axis=0)
    emb = emb / (np.linalg.norm(emb) + 1e-9)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, emb.astype(np.float32))
    return emb


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True)
    ap.add_argument("--out", default="data/embeddings/spk.npy")
    a = ap.parse_args()
    e = extract(a.wav, a.out)
    print(f"embedding shape={e.shape}  L2={np.linalg.norm(e):.3f}  → {a.out}")
