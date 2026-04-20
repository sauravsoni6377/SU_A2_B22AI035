"""Lightweight N-gram LM trained on the Speech Course Syllabus corpus.

Implements interpolated Kneser-Ney-style smoothing. The math exposed is:

    log p_biased(w_t | x, h_<t) = log p_whisper(w_t | x, h_<t)
                                  + λ · log p_LM(w_t | h_<t)
                                  + β · 𝟙[w_t ∈ TechTerms]

Design notes:
  - Use .get() everywhere so unseen keys return 0 rather than KeyError.
  - Precompute context-marginal counts c(h) once at training time so each
    log_prob call is O(n) instead of O(n · |V|).
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
import math
import json
import re


_WORD_RE = re.compile(r"[\w\-']+", re.UNICODE)


def tokenize(text: str) -> List[str]:
    return _WORD_RE.findall(text.lower())


@dataclass
class NGramLM:
    order: int
    vocab: set
    counts: Dict[int, Dict[Tuple[str, ...], int]]       # n -> (ngram -> count)
    hist_counts: Dict[int, Dict[Tuple[str, ...], int]]  # n -> (history -> c(h))
    n_continuations: Dict[int, Dict[Tuple[str, ...], int]]  # n -> (history -> |{w:c(h,w)>0}|)
    totals: Dict[int, int]

    # --- training ----------------------------------------------------------
    @classmethod
    def train(cls, text_lines: Iterable[str], order: int = 4) -> "NGramLM":
        counts = {n: Counter() for n in range(1, order + 1)}
        vocab = set()
        for line in text_lines:
            toks = ["<s>"] * (order - 1) + tokenize(line) + ["</s>"]
            vocab.update(toks)
            for n in range(1, order + 1):
                for i in range(len(toks) - n + 1):
                    counts[n][tuple(toks[i:i + n])] += 1
        totals = {n: sum(counts[n].values()) for n in counts}

        # Precompute context marginals
        hist_counts: Dict[int, Dict[Tuple[str, ...], int]] = {n: defaultdict(int) for n in range(1, order + 1)}
        n_cont: Dict[int, Dict[Tuple[str, ...], int]] = {n: defaultdict(int) for n in range(1, order + 1)}
        for n in range(2, order + 1):
            for ng, c in counts[n].items():
                h = ng[:-1]
                hist_counts[n][h] += c
                if c > 0:
                    n_cont[n][h] += 1
        return cls(order=order, vocab=vocab, counts=counts,
                   hist_counts={n: dict(d) for n, d in hist_counts.items()},
                   n_continuations={n: dict(d) for n, d in n_cont.items()},
                   totals=totals)

    # --- scoring -----------------------------------------------------------
    def _discount(self, n: int) -> float:
        return 0.75 if n >= 2 else 0.0

    def log_prob(self, token: str, history: Tuple[str, ...]) -> float:
        """log P(token | history) using interpolated KN-style smoothing."""
        n = min(self.order, len(history) + 1)
        history = tuple(history[-(n - 1):]) if n > 1 else ()
        if n == 1:
            c_uni = self.counts[1].get((token,), 0)
            V = max(len(self.vocab), 1)
            return math.log((c_uni + 1) / (self.totals[1] + V))
        d = self._discount(n)
        c_hist = self.hist_counts[n].get(history, 0)
        if c_hist == 0:
            return self.log_prob(token, history[1:])
        c = self.counts[n].get(history + (token,), 0)
        first = max(c - d, 0.0) / c_hist
        n_dot = self.n_continuations[n].get(history, 0)
        lam = (d * n_dot) / c_hist if c_hist > 0 else 1.0
        lower = math.exp(self.log_prob(token, history[1:]))
        return math.log(max(first + lam * lower, 1e-12))

    # --- persistence -------------------------------------------------------
    def save_arpa(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write("\\data\\\n")
            for n in range(1, self.order + 1):
                f.write(f"ngram {n}={len(self.counts[n])}\n")
            for n in range(1, self.order + 1):
                f.write(f"\n\\{n}-grams:\n")
                for ng, c in self.counts[n].items():
                    p = c / max(self.totals[n], 1)
                    f.write(f"{math.log10(max(p, 1e-9)):.5f}\t{' '.join(ng)}\n")
            f.write("\n\\end\\\n")

    def save_json(self, path: str):
        obj = {
            "order": self.order,
            "vocab": list(self.vocab),
            "counts": {n: {"|".join(k): v for k, v in d.items()} for n, d in self.counts.items()},
            "hist_counts": {n: {"|".join(k): v for k, v in d.items()} for n, d in self.hist_counts.items()},
            "n_continuations": {n: {"|".join(k): v for k, v in d.items()} for n, d in self.n_continuations.items()},
            "totals": self.totals,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        json.dump(obj, open(path, "w"), ensure_ascii=False)

    @classmethod
    def load_json(cls, path: str) -> "NGramLM":
        obj = json.load(open(path))

        def _unpack(d):
            return {int(n): {tuple(k.split("|")) if k else (): v for k, v in kv.items()}
                    for n, kv in d.items()}

        counts = _unpack(obj["counts"])
        hist_counts = _unpack(obj.get("hist_counts", {}))
        n_cont = _unpack(obj.get("n_continuations", {}))
        return cls(order=obj["order"], vocab=set(obj["vocab"]),
                   counts=counts, hist_counts=hist_counts,
                   n_continuations=n_cont,
                   totals={int(n): v for n, v in obj["totals"].items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out_arpa", default="checkpoints/ngram.arpa")
    ap.add_argument("--out_json", default="checkpoints/ngram.json")
    ap.add_argument("--order", type=int, default=4)
    a = ap.parse_args()
    lines = Path(a.corpus).read_text(encoding="utf-8").splitlines()
    lm = NGramLM.train(lines, order=a.order)
    lm.save_arpa(a.out_arpa)
    lm.save_json(a.out_json)
    print(f"vocab={len(lm.vocab)}  unigrams={lm.totals[1]}  → {a.out_json}")


if __name__ == "__main__":
    main()
