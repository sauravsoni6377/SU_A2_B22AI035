"""FGSM against Whisper's pretrained language-ID head.

We treat Whisper's encoder + language head as the LID classifier and attack
it with Fast Gradient Sign Method:

    x_adv = x + ε · sign(∇_x L(θ, x, ŷ))

where ŷ is Whisper's *current* argmax over {en, hi}. We sweep ε until
the prediction flips from Hindi to English while the signal-to-noise
ratio stays above 40 dB (perceptually inaudible).

The attack operates on raw 16-kHz waveforms because Whisper's log-mel and
encoder are differentiable end-to-end.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import yaml

from src.utils.audio import load_wav, save_wav, snr_db


def _pad_or_trim(wav: torch.Tensor, n: int = 16000 * 30) -> torch.Tensor:
    if wav.numel() >= n:
        return wav[:n]
    return F.pad(wav, (0, n - wav.numel()))


def _language_logits(model, wav: torch.Tensor):
    """wav: [T] float, 16kHz. Returns logits over Whisper's 99 languages."""
    import whisper
    device = next(model.parameters()).device
    padded = _pad_or_trim(wav).to(device)
    mel = whisper.log_mel_spectrogram(padded, n_mels=model.dims.n_mels)
    # language detection uses SOT token at encoder's first decoder position;
    # we compute it directly via the official API in differentiable form.
    mel = mel.unsqueeze(0)
    audio_features = model.encoder(mel)
    # Use the same tokenizer-based indexing as whisper.decoding.detect_language
    tokenizer = whisper.tokenizer.get_tokenizer(model.is_multilingual,
                                                num_languages=model.num_languages)
    sot = torch.tensor([[tokenizer.sot]], device=device)
    logits = model.logits(sot, audio_features)[:, 0]
    mask = torch.ones(logits.shape[-1], dtype=torch.bool, device=device)
    mask[list(tokenizer.all_language_tokens)] = False
    logits = logits.clone()
    logits[..., mask] = -float("inf")
    lang_tokens = list(tokenizer.all_language_tokens)
    return logits[0, lang_tokens], tokenizer.all_language_codes


def _classify_enhi(model, wav: torch.Tensor):
    logits, codes = _language_logits(model, wav)
    idx_en = codes.index("en") if "en" in codes else None
    idx_hi = codes.index("hi") if "hi" in codes else None
    return logits, idx_en, idx_hi


def fgsm_step(model, wav: torch.Tensor, eps: float) -> torch.Tensor:
    wav = wav.clone().detach().requires_grad_(True)
    logits, i_en, i_hi = _classify_enhi(model, wav)
    y = int(torch.argmax(logits).item())
    loss = F.cross_entropy(logits.unsqueeze(0), torch.tensor([y], device=logits.device))
    model.zero_grad()
    loss.backward()
    perturbed = (wav + eps * wav.grad.sign()).detach().clamp(-1.0, 1.0)
    return perturbed


def sweep(cfg_path: str, wav_path: str, whisper_size: str = "small",
          out_wav: str = "outputs/fgsm_adv.wav") -> dict:
    cfg = yaml.safe_load(open(cfg_path))
    import whisper
    device = "cpu"   # whisper MPS grads are unreliable
    print(f"[fgsm] loading whisper-{whisper_size} on {device}")
    model = whisper.load_model(whisper_size, device=device)
    for p in model.parameters():
        p.requires_grad_(False)

    wav, sr = load_wav(wav_path, sr=16000)
    wav = wav.squeeze(0)

    # Pick a 5-s window with the highest Hindi probability under clean
    win = 5 * sr; stride = 2 * sr
    best_p_hi, best_s = -1.0, 0
    for s in range(0, max(1, wav.numel() - win + 1), stride):
        clip = wav[s:s + win]
        with torch.no_grad():
            logits, i_en, i_hi = _classify_enhi(model, clip)
        if i_hi is None:
            continue
        prob = F.softmax(logits, dim=-1)
        p_hi = float(prob[i_hi].item())
        if p_hi > best_p_hi:
            best_p_hi, best_s = p_hi, s
    clip = wav[best_s:best_s + win]
    print(f"[fgsm] starting window @ {best_s / sr:.1f}s with p(hi)={best_p_hi:.3f}")

    target_snr = cfg["adversarial"]["target_snr_db"]
    step = cfg["adversarial"]["step"]
    max_eps = cfg["adversarial"]["max_epsilon"]

    result = {"min_epsilon": None, "snr_db": None,
              "clean_top_lang": None, "adv_top_lang": None,
              "p_hi_clean": best_p_hi}
    with torch.no_grad():
        c_logits, i_en, i_hi = _classify_enhi(model, clip)
    c_codes = list(whisper.tokenizer.get_tokenizer(model.is_multilingual,
                       num_languages=model.num_languages).all_language_codes)
    result["clean_top_lang"] = c_codes[int(c_logits.argmax().item())]

    eps = step
    while eps <= max_eps:
        adv = fgsm_step(model, clip, eps)
        snr = snr_db(clip, adv)
        with torch.no_grad():
            a_logits, _, _ = _classify_enhi(model, adv)
        a_top = c_codes[int(a_logits.argmax().item())]
        print(f"[fgsm] eps={eps:.4f}  SNR={snr:.1f}dB  top=en? {a_top=='en'}  "
              f"clean_top={result['clean_top_lang']}")
        if a_top == "en" and result["clean_top_lang"] == "hi" and snr > target_snr:
            result.update(min_epsilon=eps, snr_db=snr, adv_top_lang=a_top)
            save_wav(out_wav, adv.unsqueeze(0), sr)
            return result
        if a_top != result["clean_top_lang"] and snr > target_snr:
            result.update(min_epsilon=eps, snr_db=snr, adv_top_lang=a_top)
            save_wav(out_wav, adv.unsqueeze(0), sr)
            return result
        eps += step
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/config.yaml")
    ap.add_argument("--wav", required=True)
    ap.add_argument("--size", default="small")
    ap.add_argument("--out", default="outputs/fgsm_adv.wav")
    a = ap.parse_args()
    r = sweep(a.cfg, a.wav, whisper_size=a.size, out_wav=a.out)
    print(r)
