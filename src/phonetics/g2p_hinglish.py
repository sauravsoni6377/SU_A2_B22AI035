"""Hinglish → Unified IPA.

Motivation: off-the-shelf G2Ps fail on code-switched input because (a) English
G2P mis-predicts Hindi words written in Roman (e.g. "samjhaate"), and (b) Hindi
G2P cannot handle Latin-script input at all. We compose three stages:

    1. Per-segment routing (LID tells us en/hi for each word).
    2. English words  → epitran 'eng-Latn' (CMU-dict-backed).
    3. Hindi words (Devanagari or Roman) → custom rule-based transliterator →
       Hindi phoneme set → IPA.

The Roman-Hindi transliterator uses a greedy longest-match over a hand-built
phoneme map (ITRANS-inspired) plus schwa-deletion after word-final consonants
which is the most frequent Hindi G2P pitfall in practice.

Output: single IPA string with word boundaries preserved as spaces, plus a
token-level alignment list useful for the prosody stage.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import re

# -------- Roman-Hindi -> IPA map (greedy longest-match) --------------------
# Inspired by the ITRANS convention, tuned for Hinglish spelling habits.
_ROMAN_HI = [
    ("aa", "aː"), ("ii", "iː"), ("ee", "iː"), ("oo", "uː"), ("ou", "au"),
    ("ai", "ɛː"), ("au", "ɔː"), ("ri", "ɾɪ"),
    ("kh", "kʰ"), ("gh", "ɡʱ"), ("ch", "tʃ"), ("chh", "tʃʰ"),
    ("jh", "dʒʱ"), ("th", "t̪ʰ"), ("dh", "d̪ʱ"),
    ("Th", "ʈʰ"), ("Dh", "ɖʱ"), ("ph", "pʰ"), ("bh", "bʱ"),
    ("sh", "ʃ"), ("Sh", "ʂ"), ("ng", "ŋ"), ("nj", "ɲ"),
    ("a", "ə"), ("i", "ɪ"), ("u", "ʊ"), ("e", "eː"), ("o", "oː"),
    ("k", "k"), ("g", "ɡ"), ("c", "tʃ"), ("j", "dʒ"),
    ("T", "ʈ"), ("D", "ɖ"), ("N", "ɳ"),
    ("t", "t̪"), ("d", "d̪"), ("n", "n"),
    ("p", "p"), ("b", "b"), ("m", "m"),
    ("y", "j"), ("r", "ɾ"), ("l", "l"), ("v", "ʋ"), ("w", "ʋ"),
    ("s", "s"), ("h", "ɦ"), ("z", "z"), ("f", "f"), ("q", "q"),
]

# -------- Devanagari -> IPA (consonants & vowel signs) ---------------------
_DEV_CONS = {
    "क": "k", "ख": "kʰ", "ग": "ɡ", "घ": "ɡʱ", "ङ": "ŋ",
    "च": "tʃ", "छ": "tʃʰ", "ज": "dʒ", "झ": "dʒʱ", "ञ": "ɲ",
    "ट": "ʈ", "ठ": "ʈʰ", "ड": "ɖ", "ढ": "ɖʱ", "ण": "ɳ",
    "त": "t̪", "थ": "t̪ʰ", "द": "d̪", "ध": "d̪ʱ", "न": "n",
    "प": "p", "फ": "pʰ", "ब": "b", "भ": "bʱ", "म": "m",
    "य": "j", "र": "ɾ", "ल": "l", "व": "ʋ",
    "श": "ʃ", "ष": "ʂ", "स": "s", "ह": "ɦ",
    "क़": "q", "ख़": "x", "ग़": "ɣ", "ज़": "z", "फ़": "f", "ड़": "ɽ", "ढ़": "ɽʱ",
}
_DEV_VOWEL = {
    "अ": "ə", "आ": "aː", "इ": "ɪ", "ई": "iː", "उ": "ʊ", "ऊ": "uː",
    "ऋ": "ɾɪ", "ए": "eː", "ऐ": "ɛː", "ओ": "oː", "औ": "ɔː",
    "ा": "aː", "ि": "ɪ", "ी": "iː", "ु": "ʊ", "ू": "uː", "े": "eː",
    "ै": "ɛː", "ो": "oː", "ौ": "ɔː", "ृ": "ɾɪ",
}
_HALANT = "्"
_ANUSVARA = "ं"
_VISARGA = "ः"
_CHANDRABINDU = "ँ"


def is_devanagari(s: str) -> bool:
    return any("\u0900" <= ch <= "\u097F" for ch in s)


def roman_hi_to_ipa(word: str) -> str:
    """Greedy longest-match from Hinglish Latin spelling."""
    s = word
    out = []
    i = 0
    while i < len(s):
        matched = False
        # Check length 3 then 2 then 1
        for L in (3, 2, 1):
            if i + L > len(s):
                continue
            sub = s[i:i + L]
            for src, tgt in _ROMAN_HI:
                if sub == src or (L == 2 and sub.lower() == src and src.islower()):
                    out.append(tgt)
                    i += L
                    matched = True
                    break
            if matched:
                break
        if not matched:
            out.append(s[i]); i += 1
    ipa = "".join(out)
    # Schwa-deletion: drop final "ə" if preceded by consonant (Hindi phonology)
    if ipa.endswith("ə") and len(ipa) >= 2 and ipa[-2] not in "aeiouəɪʊiːuːoːeːɛːɔː":
        ipa = ipa[:-1]
    return ipa


def devanagari_to_ipa(word: str) -> str:
    out = []
    i = 0
    while i < len(word):
        ch = word[i]
        nxt = word[i + 1] if i + 1 < len(word) else ""
        if ch in _DEV_CONS:
            out.append(_DEV_CONS[ch])
            # Inherent schwa unless suppressed by halant / vowel sign
            if nxt == _HALANT:
                i += 2
                continue
            if nxt not in _DEV_VOWEL and nxt != _ANUSVARA:
                out.append("ə")
            i += 1
        elif ch in _DEV_VOWEL:
            out.append(_DEV_VOWEL[ch]); i += 1
        elif ch == _ANUSVARA:
            out.append("̃"); i += 1           # nasalisation diacritic
        elif ch == _VISARGA:
            out.append("ɦ"); i += 1
        elif ch == _CHANDRABINDU:
            out.append("̃"); i += 1
        else:
            out.append(ch); i += 1
    ipa = "".join(out)
    # Final schwa deletion
    if ipa.endswith("ə"):
        ipa = ipa[:-1]
    return ipa


def english_to_ipa(word: str) -> str:
    """Wrap epitran; fall back to a tiny rule set if unavailable."""
    try:
        import epitran
        if not hasattr(english_to_ipa, "_ep"):
            english_to_ipa._ep = epitran.Epitran("eng-Latn")  # type: ignore
        return english_to_ipa._ep.transliterate(word)         # type: ignore
    except Exception:
        return _fallback_eng_ipa(word)


_EN_FALLBACK = [
    ("tion", "ʃən"), ("ing", "ɪŋ"), ("ck", "k"), ("sh", "ʃ"), ("ch", "tʃ"),
    ("ph", "f"), ("th", "θ"), ("oo", "uː"), ("ee", "iː"), ("ou", "aʊ"),
    ("oi", "ɔɪ"), ("ai", "eɪ"), ("ay", "eɪ"), ("ey", "eɪ"),
    ("a", "æ"), ("e", "ɛ"), ("i", "ɪ"), ("o", "ɒ"), ("u", "ʌ"),
]


def _fallback_eng_ipa(w: str) -> str:
    out, i, s = [], 0, w.lower()
    while i < len(s):
        matched = False
        for L in (4, 3, 2, 1):
            if i + L > len(s):
                continue
            sub = s[i:i + L]
            for src, tgt in _EN_FALLBACK:
                if sub == src:
                    out.append(tgt); i += L; matched = True; break
            if matched:
                break
        if not matched:
            out.append(s[i]); i += 1
    return "".join(out)


# ---------- public entrypoint -----------------------------------------------
@dataclass
class IPAToken:
    word: str
    lang: str           # 'en' | 'hi'
    ipa: str
    start_ms: int
    end_ms: int


def transcript_to_ipa(segments: List[dict]) -> Tuple[str, List[IPAToken]]:
    """segments: list of {start_ms, end_ms, lang, text} from whisper_decode.
    Returns (flat_ipa_string, tokens)."""
    tokens: List[IPAToken] = []
    for seg in segments:
        words = re.findall(r"[^\s]+", seg["text"])
        if not words:
            continue
        step = (seg["end_ms"] - seg["start_ms"]) / len(words)
        for i, w in enumerate(words):
            st = int(seg["start_ms"] + i * step)
            et = int(seg["start_ms"] + (i + 1) * step)
            if seg["lang"] == "en":
                ipa = english_to_ipa(w.strip(".,?!").lower())
            else:   # hi
                if is_devanagari(w):
                    ipa = devanagari_to_ipa(w)
                else:
                    ipa = roman_hi_to_ipa(w.strip(".,?!").lower())
            tokens.append(IPAToken(word=w, lang=seg["lang"], ipa=ipa,
                                   start_ms=st, end_ms=et))
    flat = " ".join(t.ipa for t in tokens)
    return flat, tokens


if __name__ == "__main__":
    import json, argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--transcript", required=True)
    ap.add_argument("--out_txt", default="outputs/ipa_string.txt")
    ap.add_argument("--out_tokens", default="outputs/ipa_tokens.json")
    a = ap.parse_args()
    segs = json.load(open(a.transcript))
    ipa, toks = transcript_to_ipa(segs)
    from pathlib import Path
    Path(a.out_txt).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out_txt).write_text(ipa, encoding="utf-8")
    json.dump([t.__dict__ for t in toks], open(a.out_tokens, "w"), ensure_ascii=False, indent=2)
    print(f"{len(toks)} tokens → {a.out_txt}")
