"""Train the anti-spoofing CM model.

Expected directory layout:
    data/cm/bonafide/*.wav    # your own voice (incl. student_voice_ref slices)
    data/cm/spoof/*.wav       # Part-III synthesised outputs (+ optional augmentation)

Use --split_seed to produce a reproducible train/test split; prints EER on test.
"""
from __future__ import annotations
import argparse
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from src.antispoofing.features import lfcc, cqcc
from src.antispoofing.model import CMModel, focal_loss, score
from src.utils.audio import load_wav
from src.utils.metrics import compute_eer


class CMDataset(Dataset):
    def __init__(self, items: List[Tuple[Path, int]], feat: str, sr: int = 16000, max_s: float = 4.0):
        self.items = items
        self.feat = feat
        self.sr = sr
        self.max_samples = int(max_s * sr)

    def __len__(self): return len(self.items)

    def _slice(self, w: np.ndarray) -> np.ndarray:
        if len(w) >= self.max_samples:
            start = random.randint(0, len(w) - self.max_samples)
            return w[start:start + self.max_samples]
        pad = self.max_samples - len(w)
        return np.pad(w, (0, pad))

    def __getitem__(self, idx):
        p, y = self.items[idx]
        w, _ = load_wav(p, sr=self.sr)
        w = self._slice(w.squeeze(0).numpy())
        feat = lfcc(w, self.sr) if self.feat == "lfcc" else cqcc(w, self.sr)
        return torch.from_numpy(feat), torch.tensor(y, dtype=torch.long)


def collate(batch):
    lens = [b[0].shape[0] for b in batch]
    L = max(lens)
    F = batch[0][0].shape[1]
    x = torch.zeros(len(batch), L, F)
    y = torch.zeros(len(batch), dtype=torch.long)
    for i, (f, t) in enumerate(batch):
        x[i, :f.shape[0]] = f
        y[i] = t
    return x, y


def _collect(root: str, label: int) -> List[Tuple[Path, int]]:
    return [(p, label) for p in Path(root).glob("**/*.wav")]


def train(cfg_path: str, bonafide_dir: str, spoof_dir: str,
          split_seed: int = 42, test_frac: float = 0.2):
    cfg = yaml.safe_load(open(cfg_path))
    random.seed(split_seed)
    items = _collect(bonafide_dir, 1) + _collect(spoof_dir, 0)
    if not items:
        raise RuntimeError("no CM audio files found; populate data/cm/{bonafide,spoof}")
    random.shuffle(items)
    n_te = max(1, int(len(items) * test_frac))
    te, tr = items[:n_te], items[n_te:]
    feat = cfg["antispoof"]["feature"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds_tr = CMDataset(tr, feat); ds_te = CMDataset(te, feat)
    # Probe actual feature dim from the first sample (includes Δ + ΔΔ)
    x0, _ = ds_tr[0]
    in_dim = x0.shape[-1]
    print(f"[cm] feat={feat}  in_dim={in_dim}  n_train={len(ds_tr)}  n_test={len(ds_te)}")
    dl_tr = DataLoader(ds_tr, batch_size=16, shuffle=True, collate_fn=collate)
    dl_te = DataLoader(ds_te, batch_size=16, shuffle=False, collate_fn=collate)

    model = CMModel(in_dim=in_dim, hidden=cfg["antispoof"]["hidden"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_eer = 1.0
    ckpt = Path(cfg["antispoof"]["ckpt"]); ckpt.parent.mkdir(parents=True, exist_ok=True)

    for ep in range(cfg["antispoof"]["epochs"]):
        model.train()
        losses = []
        for x, y in tqdm(dl_tr, desc=f"cm-ep{ep}"):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = focal_loss(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        # Eval
        model.eval()
        scores, labels = [], []
        with torch.no_grad():
            for x, y in dl_te:
                x = x.to(device)
                s = score(model(x))
                scores.extend(s.tolist()); labels.extend(y.tolist())
        eer, thr = compute_eer(np.array(labels), np.array(scores))
        print(f"[ep{ep}] loss={np.mean(losses):.4f}  EER={eer * 100:.2f}%")
        if eer < best_eer:
            best_eer = eer
            torch.save({"model": model.state_dict(), "cfg": cfg,
                        "feat": feat, "in_dim": in_dim, "eer": eer, "thr": thr}, ckpt)
    print(f"best EER = {best_eer * 100:.2f}% → {ckpt}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/config.yaml")
    ap.add_argument("--bonafide", default="data/cm/bonafide")
    ap.add_argument("--spoof", default="data/cm/spoof")
    a = ap.parse_args()
    train(a.cfg, a.bonafide, a.spoof)
