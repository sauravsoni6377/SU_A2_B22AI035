"""Constrained Whisper decoding.

Strategy:
  - Group consecutive LID segments of the same language into ~25-second
    chunks (Whisper's native receptive field is 30 s).
  - Call whisper-medium.generate() once per chunk with:
        forced_decoder_ids = [lang, transcribe]
        LogitsProcessor    = NGramLogitBias(λ·log p_LM + β·tech_whitelist)
        prompt_ids         = technical-term seed (Whisper-native biasing)
  - Re-align Whisper's own sub-segment timestamps back to the original
    timeline and keep per-segment lang labels.

This is both correct and tractable: ~30 generate() calls for a 10-min clip
rather than one per VAD chunk.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import List, Dict

import torch
import yaml

from src.utils.audio import load_wav


def _load_whitelist(path: str | None):
    if not path or not Path(path).exists():
        return set()
    return {w.strip().lower() for w in Path(path).read_text(encoding="utf-8").splitlines() if w.strip()}


def _group_segments(segs: List[Dict], target_s: float = 25.0) -> List[Dict]:
    """Merge consecutive same-lang segments (skip silence) into ~target_s chunks."""
    groups = []
    cur = None
    for s in segs:
        if s["lang"] == "sil":
            if cur is not None:
                cur["end_ms"] = s["end_ms"]    # extend through silence
            continue
        if cur is None:
            cur = {"start_ms": s["start_ms"], "end_ms": s["end_ms"], "lang": s["lang"]}
            continue
        same = s["lang"] == cur["lang"]
        dur = (s["end_ms"] - cur["start_ms"]) / 1000.0
        if same and dur < target_s:
            cur["end_ms"] = s["end_ms"]
        else:
            groups.append(cur)
            cur = {"start_ms": s["start_ms"], "end_ms": s["end_ms"], "lang": s["lang"]}
    if cur is not None:
        groups.append(cur)
    return groups


def transcribe(wav_path: str, lid_segments: List[Dict], cfg: Dict) -> List[Dict]:
    """Returns: list of {start_ms, end_ms, lang, text}."""
    from transformers import WhisperProcessor, WhisperForConditionalGeneration

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[stt] loading {cfg['stt']['backbone']} on {device}...")
    proc = WhisperProcessor.from_pretrained(cfg["stt"]["backbone"])
    model = WhisperForConditionalGeneration.from_pretrained(
        cfg["stt"]["backbone"], torch_dtype=torch.float32
    ).to(device)
    model.eval()

    # Optional n-gram logit bias ------------------------------------------------
    logits_processor = None
    try:
        from src.stt.ngram_lm import NGramLM
        from src.stt.logit_bias import NGramLogitBias
        lm_path = Path(cfg["stt"]["ngram"]["arpa_path"]).with_suffix(".json")
        if lm_path.exists():
            lm = NGramLM.load_json(lm_path)
            whitelist = _load_whitelist(cfg["stt"]["ngram"]["boost_whitelist"])
            from transformers import LogitsProcessorList
            logits_processor = LogitsProcessorList([
                NGramLogitBias(lm, proc.tokenizer, whitelist,
                               lambda_lm=cfg["stt"]["ngram"]["lambda_lm"])
            ])
            print(f"[stt] n-gram logit bias: λ={cfg['stt']['ngram']['lambda_lm']}, "
                  f"|whitelist|={len(whitelist)}")
    except Exception as e:
        print(f"[stt] logit bias disabled ({e})")

    # Technical-term prompt is Whisper's native bias mechanism
    whitelist_txt = " ".join(sorted(_load_whitelist(cfg["stt"]["ngram"]["boost_whitelist"])))

    wav, sr = load_wav(wav_path, sr=16000)
    wav = wav.squeeze(0).numpy()
    groups = _group_segments(lid_segments, target_s=25.0)
    print(f"[stt] grouped {len(lid_segments)} LID segs into {len(groups)} chunks")

    out: List[Dict] = []
    for i, g in enumerate(groups):
        s = int(g["start_ms"] * sr / 1000)
        e = int(g["end_ms"] * sr / 1000)
        chunk = wav[s:e]
        if len(chunk) < sr * 0.3:
            continue
        inputs = proc(chunk, sampling_rate=sr, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        forced = proc.get_decoder_prompt_ids(language=g["lang"], task="transcribe")
        gen_kwargs = dict(
            forced_decoder_ids=forced,
            num_beams=cfg["stt"]["beam_size"],
            max_new_tokens=220,
            prompt_ids=proc.get_prompt_ids(whitelist_txt[:200], return_tensors="pt").to(device),
        )
        if logits_processor is not None:
            gen_kwargs["logits_processor"] = logits_processor
        with torch.no_grad():
            try:
                ids = model.generate(**inputs, **gen_kwargs)
            except Exception as e:
                gen_kwargs.pop("prompt_ids", None)
                ids = model.generate(**inputs, **gen_kwargs)
        text = proc.batch_decode(ids, skip_special_tokens=True)[0].strip()
        out.append(dict(start_ms=g["start_ms"], end_ms=g["end_ms"],
                        lang=g["lang"], text=text))
        print(f"[stt {i+1}/{len(groups)}]  {g['lang']}  {g['start_ms']/1000:.1f}-"
              f"{g['end_ms']/1000:.1f}s:  {text[:80]}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/config.yaml")
    ap.add_argument("--wav", required=True)
    ap.add_argument("--lid_json", required=True)
    ap.add_argument("--out", default="outputs/transcript.json")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.cfg))
    segs = json.load(open(a.lid_json))
    tx = transcribe(a.wav, segs, cfg)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(tx, open(a.out, "w"), ensure_ascii=False, indent=2)
    print(f"wrote {len(tx)} transcript segments → {a.out}")


if __name__ == "__main__":
    main()
