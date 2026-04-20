"""Translate the code-switched (English+Hindi) transcript into Maithili.

Policy:
  1. Tokenise into words/phrases.
  2. For each term, try the user-maintained parallel dictionary first.
     This is where *technical* terms are forced to have a deterministic
     translation — essential because MT models hallucinate on domain words.
  3. Fall back to Facebook NLLB-200 (distilled) for remaining phrases.
     Source language is switched per LID.

Output: single Maithili string in Devanagari.
"""
from __future__ import annotations
import json
import re
from pathlib import Path
from typing import List, Dict, Tuple


class ParallelDict:
    def __init__(self, path: str):
        self.path = path
        if Path(path).exists():
            raw = json.load(open(path, encoding="utf-8"))
        else:
            raw = {}
        self.table: Dict[str, str] = {k.lower(): v for k, v in raw.items()}

    def lookup(self, term: str) -> str | None:
        return self.table.get(term.lower())

    def __len__(self): return len(self.table)


class NLLBTranslator:
    MODEL = "facebook/nllb-200-distilled-600M"
    _MAP = {"en": "eng_Latn", "hi": "hin_Deva", "mai": "mai_Deva"}

    def __init__(self, target: str = "mai"):
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        import torch
        self.tok = AutoTokenizer.from_pretrained(self.MODEL)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.MODEL)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.target = self._MAP[target]

    def translate(self, text: str, src_lang: str = "en") -> str:
        import torch
        src = self._MAP[src_lang]
        self.tok.src_lang = src
        inputs = self.tok(text, return_tensors="pt").to(self.device)
        bos = self.tok.convert_tokens_to_ids(self.target)
        with torch.no_grad():
            out = self.model.generate(**inputs, forced_bos_token_id=bos, max_new_tokens=256)
        return self.tok.batch_decode(out, skip_special_tokens=True)[0]


def translate_segments(segments: List[Dict], cfg: dict,
                       pd: ParallelDict | None = None,
                       nllb: NLLBTranslator | None = None) -> Tuple[str, List[Dict]]:
    """Segments: [{lang, text, ...}]. Returns (joined_mai_text, per_seg_list)."""
    if pd is None:
        pd = ParallelDict(cfg["phonetics"]["parallel_corpus"])
    out_segs, pieces = [], []
    for seg in segments:
        words = re.findall(r"[^\s]+|[.,?!]", seg["text"])
        translated = []
        for w in words:
            direct = pd.lookup(w.strip(".,?!"))
            if direct is not None:
                translated.append(direct)
            else:
                translated.append(w)
        partial = " ".join(translated)
        # For the non-dict residue, send the whole segment through NLLB
        needs_mt = any(pd.lookup(w.strip(".,?!")) is None and re.match(r"\w+", w)
                       for w in words)
        if needs_mt and nllb is not None:
            try:
                translated_full = nllb.translate(seg["text"], src_lang=seg["lang"])
            except Exception as e:
                print(f"[warn] NLLB failed for segment: {e}")
                translated_full = partial
        else:
            translated_full = partial
        out_segs.append(dict(seg, mai_text=translated_full))
        pieces.append(translated_full)
    return " ".join(pieces), out_segs


def main():
    import argparse, yaml
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/config.yaml")
    ap.add_argument("--transcript", required=True)
    ap.add_argument("--out_txt", default="outputs/maithili_text.txt")
    ap.add_argument("--out_json", default="outputs/translation.json")
    ap.add_argument("--no_nllb", action="store_true",
                    help="skip NLLB (dictionary only, for offline demos)")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.cfg))
    segs = json.load(open(a.transcript))
    pd = ParallelDict(cfg["phonetics"]["parallel_corpus"])
    nllb = None if a.no_nllb else NLLBTranslator(target=cfg["phonetics"]["target_lrl"])
    text, per_seg = translate_segments(segs, cfg, pd, nllb)
    Path(a.out_txt).write_text(text, encoding="utf-8")
    json.dump(per_seg, open(a.out_json, "w"), ensure_ascii=False, indent=2)
    print(f"wrote maithili text ({len(text)} chars) with {len(pd)}-entry dict")


if __name__ == "__main__":
    main()
