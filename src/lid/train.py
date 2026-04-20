"""Train the frame-level multi-head LID model."""
from __future__ import annotations
import argparse
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from src.lid.model import MultiHeadLID, multi_head_loss
from src.lid.dataset import LIDDataset, scan_clips, collate_fn


def _f1_per_class(pred: torch.Tensor, targ: torch.Tensor, n_classes: int = 3):
    """Macro-F1 and per-class F1 on valid frames (targ != -100)."""
    mask = targ.ne(-100)
    p, t = pred[mask], targ[mask]
    f1s = []
    for c in range(n_classes):
        tp = ((p == c) & (t == c)).sum().float()
        fp = ((p == c) & (t != c)).sum().float()
        fn = ((p != c) & (t == c)).sum().float()
        prec = tp / (tp + fp + 1e-9)
        rec = tp / (tp + fn + 1e-9)
        f1s.append((2 * prec * rec / (prec + rec + 1e-9)).item())
    return sum(f1s) / len(f1s), f1s


def train(cfg_path: str, en_dir: str, hi_dir: str, sil_dir: str):
    cfg = yaml.safe_load(open(cfg_path))
    device = cfg["device"] if torch.cuda.is_available() else "cpu"
    clips = scan_clips(en_dir, hi_dir, sil_dir)
    if not clips:
        raise RuntimeError("no clips found. populate data/lid_raw/{en,hi,sil}/")
    ds = LIDDataset(clips, sr=cfg["audio"]["sr"], hop_ms=cfg["audio"]["hop_ms"])
    n_val = max(1, int(0.1 * len(ds)))
    ds_tr, ds_va = random_split(ds, [len(ds) - n_val, n_val])
    dl_tr = DataLoader(ds_tr, batch_size=cfg["lid"]["batch_size"], shuffle=True,
                       num_workers=2, collate_fn=collate_fn)
    dl_va = DataLoader(ds_va, batch_size=cfg["lid"]["batch_size"], shuffle=False,
                       num_workers=2, collate_fn=collate_fn)

    model = MultiHeadLID(n_classes=cfg["lid"]["n_classes"], d=cfg["lid"]["hidden"],
                         n_heads=cfg["lid"]["n_heads"], n_layers=cfg["lid"]["n_layers"],
                         sr=cfg["audio"]["sr"], hop_ms=cfg["audio"]["hop_ms"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lid"]["lr"],
                            weight_decay=cfg["lid"]["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["lid"]["epochs"])
    best_f1 = 0.0
    ckpt = Path(cfg["lid"]["ckpt"]); ckpt.parent.mkdir(parents=True, exist_ok=True)

    for ep in range(cfg["lid"]["epochs"]):
        model.train()
        loss_sum = 0.0
        for wav, y in tqdm(dl_tr, desc=f"ep{ep}"):
            wav, y = wav.to(device), y.to(device)
            logits, aux = model(wav)
            # Align time dims: truncate to min
            T = min(logits.shape[1], y.shape[1])
            logits = logits[:, :T]; aux = aux[:, :T]; y = y[:, :T]
            loss = multi_head_loss(logits, aux, y)
            opt.zero_grad(); loss.backward(); opt.step()
            loss_sum += loss.item()
        sched.step()

        # Validation
        model.eval()
        preds, targs = [], []
        with torch.no_grad():
            for wav, y in dl_va:
                wav, y = wav.to(device), y.to(device)
                logits, _ = model(wav)
                T = min(logits.shape[1], y.shape[1])
                preds.append(logits[:, :T].argmax(-1).cpu())
                targs.append(y[:, :T].cpu())
        preds = torch.cat([p.reshape(-1) for p in preds])
        targs = torch.cat([t.reshape(-1) for t in targs])
        macro_f1, per = _f1_per_class(preds, targs)
        print(f"[ep{ep}] loss={loss_sum / max(1, len(dl_tr)):.4f}  macro-F1={macro_f1:.3f}  "
              f"EN={per[0]:.3f} HI={per[1]:.3f} SIL={per[2]:.3f}")
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            torch.save({"model": model.state_dict(), "cfg": cfg, "f1": macro_f1}, ckpt)
    print(f"best macro-F1 = {best_f1:.3f} → {ckpt}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/config.yaml")
    ap.add_argument("--en_dir", default="data/lid_raw/en")
    ap.add_argument("--hi_dir", default="data/lid_raw/hi")
    ap.add_argument("--sil_dir", default="data/lid_raw/sil")
    a = ap.parse_args()
    train(a.cfg, a.en_dir, a.hi_dir, a.sil_dir)
