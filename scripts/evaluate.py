"""Compute all evaluation metrics required by the assignment:

  WER (EN & HI)          — from outputs/transcript.json vs ground-truth JSON
  MCD                    — output_LRL_cloned.wav vs student_voice_ref.wav
  LID switch precision   — predicted segments vs hand-annotated segments
  Anti-spoofing EER      — on held-out bona-fide/spoof test set
  FGSM min ε             — reported, plus SNR and flip rate

Ground-truth references are optional; when missing we print 'n/a' but still
produce the other metrics so the evaluator gets a partial report.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from src.utils.metrics import wer_by_language, mcd, switch_timestamp_precision, compute_eer


def _load(p):
    return json.load(open(p)) if Path(p).exists() else None


def _require_wav(p: str):
    from src.utils.audio import load_wav
    w, sr = load_wav(p, sr=22050)
    return w.squeeze(0).numpy(), sr


def eval_wer(transcript_path: str, gt_path: str = "data/gt/transcript_gt.json"):
    hyp = _load(transcript_path)
    ref = _load(gt_path)
    if hyp is None or ref is None:
        return None
    aligned = []
    for h, r in zip(hyp, ref):
        aligned.append(dict(lang=h["lang"], hyp=h["text"], ref=r["text"]))
    return wer_by_language(aligned)


def eval_mcd(cloned: str, reference: str):
    if not Path(cloned).exists() or not Path(reference).exists():
        return None
    c, sr = _require_wav(cloned)
    r, _ = _require_wav(reference)
    return mcd(r, c, sr=sr)


def eval_lid(pred_json: str, gt_json: str = "data/gt/lid_gt.json",
             tol_ms: float = 200):
    pred = _load(pred_json); gt = _load(gt_json)
    if pred is None or gt is None:
        return None
    pred_sw = [s["start_ms"] for s in pred[1:]]
    gt_sw = [s["start_ms"] for s in gt[1:]]
    return switch_timestamp_precision(pred_sw, gt_sw, tolerance_ms=tol_ms)


def eval_cm(ckpt: str, test_dir_bf: str = "data/cm_test/bonafide",
            test_dir_sp: str = "data/cm_test/spoof"):
    if not Path(ckpt).exists():
        return None
    from src.antispoofing.infer import score_file
    import torch
    scores, labels = [], []
    for p in Path(test_dir_bf).glob("**/*.wav"):
        scores.append(score_file(str(p), ckpt)); labels.append(1)
    for p in Path(test_dir_sp).glob("**/*.wav"):
        scores.append(score_file(str(p), ckpt)); labels.append(0)
    if not scores:
        return None
    eer, thr = compute_eer(np.array(labels), np.array(scores))
    return dict(eer=eer, threshold=thr, n=len(scores))


def eval_fgsm(cfg_path: str, wav: str, ckpt: str):
    if not (Path(wav).exists() and Path(ckpt).exists()):
        return None
    from src.lid.adversarial import sweep_min_epsilon
    return sweep_min_epsilon(cfg_path, wav, ckpt, out_wav="outputs/fgsm_adv.wav")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/config.yaml")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.cfg))

    print("\n=====  Evaluation Report  =====")

    wer = eval_wer(cfg["outputs"]["transcript_json"])
    print(f"\n[WER]   {wer if wer else 'n/a (no ground-truth transcript)'}")

    mcd_val = eval_mcd(cfg["outputs"]["cloned_wav"], cfg["data"]["student_ref_wav"])
    print(f"\n[MCD]   {mcd_val:.3f} dB" if mcd_val is not None else "\n[MCD]   n/a")

    lid = eval_lid("outputs/lid_segments.json")
    print(f"\n[LID switching]   {lid if lid else 'n/a (no ground-truth LID)'}")

    cm = eval_cm(cfg["antispoof"]["ckpt"])
    print(f"\n[Anti-spoof EER]  {cm if cm else 'n/a (train CM first)'}")

    fgsm = eval_fgsm(a.cfg, "data/processed/original_denoised.wav",
                     cfg["lid"]["ckpt"])
    print(f"\n[FGSM sweep]      {fgsm if fgsm else 'n/a'}")


if __name__ == "__main__":
    main()
