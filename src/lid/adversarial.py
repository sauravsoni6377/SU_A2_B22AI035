"""FGSM on the LID model — find min epsilon that flips Hindi→English prediction
while keeping SNR > 40 dB (imperceptible).

FGSM perturbation:  x_adv = x + eps * sign( ∇_x L(theta, x, y_target) )
We use an *untargeted* flip for frames whose true label is HI; success = argmax
becomes EN. We sweep epsilon from 0 to cfg.max_epsilon in steps, pick smallest
epsilon whose SNR > 40 dB and causes at least 50% of HI frames to flip to EN.
"""
from __future__ import annotations
import argparse
import torch
import torch.nn.functional as F
import yaml

from src.lid.model import MultiHeadLID
from src.lid.infer import load_model
from src.utils.audio import load_wav, save_wav, snr_db


def fgsm(model: MultiHeadLID, wav: torch.Tensor, eps: float) -> torch.Tensor:
    """Targeted-untargeted FGSM: push away from ground-truth argmax per frame."""
    wav = wav.clone().detach().requires_grad_(True)
    logits, _ = model(wav)               # [1, T, 3]
    y = logits.argmax(-1).detach()       # treat current preds as "true"
    loss = F.cross_entropy(logits.reshape(-1, 3), y.reshape(-1))
    model.zero_grad()
    loss.backward()
    pert = eps * wav.grad.sign()
    return (wav + pert).detach().clamp(-1.0, 1.0)


def sweep_min_epsilon(cfg_path: str, wav_path: str, ckpt: str, out_wav: str = None):
    cfg = yaml.safe_load(open(cfg_path))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(ckpt, device=device)
    for p in model.parameters():
        p.requires_grad_(False)

    wav, sr = load_wav(wav_path, sr=cfg["audio"]["sr"])
    wav = wav.to(device)

    # Pick 5-second HI-heavy window
    logits, _ = model(wav)
    cls = logits.argmax(-1)[0]
    hop = model.effective_hop_ms
    win_frames = int(5000 / hop)
    best_start, best_hi = 0, -1
    for s in range(0, cls.numel() - win_frames, win_frames // 4):
        n_hi = (cls[s:s + win_frames] == 1).sum().item()
        if n_hi > best_hi:
            best_hi, best_start = n_hi, s
    samp_start = best_start * hop * sr // 1000
    samp_end = samp_start + 5 * sr
    clip = wav[:, samp_start:samp_end]

    step = cfg["adversarial"]["step"]
    max_eps = cfg["adversarial"]["max_epsilon"]
    target_snr = cfg["adversarial"]["target_snr_db"]

    result = {"min_epsilon": None, "snr_db": None, "flip_rate": None}
    eps = step
    while eps <= max_eps:
        adv = fgsm(model, clip, eps)
        snr = snr_db(clip.cpu().squeeze(), adv.cpu().squeeze())
        logits_adv, _ = model(adv)
        cls_adv = logits_adv.argmax(-1)[0]
        logits_cln, _ = model(clip)
        cls_cln = logits_cln.argmax(-1)[0]
        T = min(cls_adv.numel(), cls_cln.numel())
        hi_mask = (cls_cln[:T] == 1)
        flips = ((cls_adv[:T] == 0) & hi_mask).sum().item()
        flip_rate = flips / max(hi_mask.sum().item(), 1)
        print(f"eps={eps:.4f} SNR={snr:.1f}dB flip_rate={flip_rate:.2f}")
        if flip_rate >= 0.5 and snr > target_snr:
            result.update(min_epsilon=eps, snr_db=snr, flip_rate=flip_rate)
            if out_wav:
                save_wav(out_wav, adv.cpu(), sr)
            return result
        eps += step
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/config.yaml")
    ap.add_argument("--wav", required=True)
    ap.add_argument("--ckpt", default="checkpoints/lid_best.pt")
    ap.add_argument("--out", default="outputs/fgsm_adv.wav")
    a = ap.parse_args()
    r = sweep_min_epsilon(a.cfg, a.wav, a.ckpt, a.out)
    print(r)
