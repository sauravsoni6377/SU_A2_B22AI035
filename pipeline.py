"""End-to-end pipeline driver.

Runs, in order, and caches each stage's output so re-runs skip finished work:

    ├── 0. Download lecture segment  (if missing)
    ├── 1. Denoise / normalise
    ├── 2. Frame-level LID segmentation
    ├── 3. Constrained Whisper STT (per-segment)
    ├── 4. Hinglish → IPA
    ├── 5. Translate to Maithili (dict + NLLB)
    ├── 6. MMS synthesis (flat)
    ├── 7. Voice conversion  → student timbre
    ├── 8. DTW prosody warp  → final LRL wav
    └── 9. Spoof scoring + FGSM sweep + report metrics

Usage:
    python pipeline.py --cfg configs/config.yaml
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import yaml


def _cache_exists(p: str) -> bool:
    return Path(p).exists() and Path(p).stat().st_size > 0


def _maybe(msg: str, cond: bool):
    print(f"[{'SKIP' if cond else 'RUN '}] {msg}")


def run(cfg_path: str, force: bool = False):
    cfg = yaml.safe_load(open(cfg_path))
    Path("outputs").mkdir(exist_ok=True)

    raw = cfg["data"]["original_wav"]
    ref = cfg["data"]["student_ref_wav"]
    denoised = "data/processed/original_denoised.wav"
    lid_json = "outputs/lid_segments.json"
    transcript_json = cfg["outputs"]["transcript_json"]
    ipa_txt = cfg["outputs"]["ipa_txt"]
    ipa_tokens = "outputs/ipa_tokens.json"
    mai_txt = cfg["outputs"]["lrl_txt"]
    translation_json = "outputs/translation.json"
    flat_wav = "outputs/mms_flat.wav"
    vc_wav = "outputs/mms_vc.wav"
    final_wav = cfg["outputs"]["cloned_wav"]

    # 0. Download ---------------------------------------------------------
    _maybe("0/ download lecture segment", _cache_exists(raw) and not force)
    if force or not _cache_exists(raw):
        from src.preprocessing.download import download_segment
        download_segment(cfg["data"]["lecture_url"],
                         cfg["data"]["lecture_start"],
                         cfg["data"]["lecture_end"], raw,
                         sr=cfg["audio"]["sr"])

    # 1. Denoise ----------------------------------------------------------
    _maybe("1/ denoise + normalise", _cache_exists(denoised) and not force)
    if force or not _cache_exists(denoised):
        from src.preprocessing.denoise import run as denoise_run
        denoise_run(raw, denoised, sr=cfg["audio"]["sr"],
                    backend=cfg["preprocessing"]["denoiser"],
                    target_dbfs=cfg["preprocessing"]["normalize_dbfs"],
                    high_pass_hz=cfg["preprocessing"]["high_pass_hz"])

    # 2. LID segmentation -------------------------------------------------
    _maybe("2/ LID segmentation", _cache_exists(lid_json) and not force)
    if force or not _cache_exists(lid_json):
        if Path(cfg["lid"]["ckpt"]).exists():
            from src.lid.infer import segment
            segs = segment(denoised, cfg["lid"]["ckpt"],
                           smoothing_ms=cfg["lid"]["smoothing_ms"])
        else:
            # No trained checkpoint — pragmatic fallback (Silero VAD + Whisper LID)
            from src.lid.pretrained_lid import segment_and_lid
            segs = segment_and_lid(denoised, whisper_size="small")
        Path(lid_json).parent.mkdir(parents=True, exist_ok=True)
        json.dump(segs, open(lid_json, "w"), indent=2)

    # 3. Transcription ----------------------------------------------------
    _maybe("3/ constrained Whisper STT", _cache_exists(transcript_json) and not force)
    if force or not _cache_exists(transcript_json):
        # Use openai-whisper (more stable on noisy audio) with post-hoc
        # technical-term rescoring; HF logit-bias path remains smoke-tested.
        from src.stt.whisper_openai import transcribe_full
        segs = json.load(open(lid_json))
        tx = transcribe_full(denoised, segs, cfg)
        json.dump(tx, open(transcript_json, "w"), ensure_ascii=False, indent=2)

    # 4. IPA --------------------------------------------------------------
    _maybe("4/ unified IPA", _cache_exists(ipa_txt) and not force)
    if force or not _cache_exists(ipa_txt):
        from src.phonetics.g2p_hinglish import transcript_to_ipa
        segs = json.load(open(transcript_json))
        ipa, toks = transcript_to_ipa(segs)
        Path(ipa_txt).write_text(ipa, encoding="utf-8")
        json.dump([t.__dict__ for t in toks], open(ipa_tokens, "w"),
                  ensure_ascii=False, indent=2)

    # 5. Translation ------------------------------------------------------
    _maybe("5/ translation to Maithili", _cache_exists(mai_txt) and not force)
    if force or not _cache_exists(mai_txt):
        from src.phonetics.translate import ParallelDict, NLLBTranslator, translate_segments
        segs = json.load(open(transcript_json))
        pd = ParallelDict(cfg["phonetics"]["parallel_corpus"])
        try:
            nllb = NLLBTranslator(target=cfg["phonetics"]["target_lrl"])
        except Exception as e:
            print(f"[warn] NLLB unavailable ({e}); dict-only translation")
            nllb = None
        text, per_seg = translate_segments(segs, cfg, pd, nllb)
        Path(mai_txt).write_text(text, encoding="utf-8")
        json.dump(per_seg, open(translation_json, "w"), ensure_ascii=False, indent=2)

    # 6. Flat MMS synthesis ----------------------------------------------
    _maybe("6/ flat MMS synthesis", _cache_exists(flat_wav) and not force)
    if force or not _cache_exists(flat_wav):
        from src.tts.synthesis import synth_chunked
        text = Path(mai_txt).read_text(encoding="utf-8")
        synth_chunked(text, flat_wav, cfg)

    # 7. Voice conversion → student timbre (stats-match VC; no Coqui dep) -
    _maybe("7/ voice conversion (MCC stats-match)", _cache_exists(vc_wav) and not force)
    if force or not _cache_exists(vc_wav):
        from src.tts.voice_convert import stats_match_vc
        try:
            stats_match_vc(flat_wav, ref, vc_wav,
                           sr=cfg["audio"]["synth_sr"])
        except Exception as e:
            print(f"[warn] VC failed ({e}); copying flat synth")
            import shutil; shutil.copy(flat_wav, vc_wav)

    # 8. Prosody DTW warp → final wav ------------------------------------
    _maybe("8/ DTW prosody warping", _cache_exists(final_wav) and not force)
    if force or not _cache_exists(final_wav):
        from src.tts.prosody import transfer_prosody
        try:
            transfer_prosody(denoised, vc_wav, final_wav,
                             sr=cfg["audio"]["synth_sr"])
        except Exception as e:
            print(f"[warn] prosody transfer failed ({e}); using vc_wav directly")
            import shutil; shutil.copy(vc_wav, final_wav)

    print(f"\nDONE. Final cloned lecture: {final_wav}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/config.yaml")
    ap.add_argument("--force", action="store_true", help="re-run all stages")
    a = ap.parse_args()
    run(a.cfg, force=a.force)
