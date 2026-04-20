"""Hook to inject N-gram log-probabilities into Whisper's beam search.

Whisper (HuggingFace implementation) accepts a `LogitsProcessor`. We build one
that maintains a rolling token history (decoded so far) and adds
    Δ = λ · log p_LM(w_t | h_<t)  +  β · 𝟙[w_t ∈ TechTerms]
to each candidate's logit.

For efficiency, we score only the top-K candidates per step (K=50) — matching
standard beam implementations where the candidate set is pruned before scoring.
"""
from __future__ import annotations
from typing import List, Set

import torch

try:
    from transformers import LogitsProcessor
except ImportError:  # graceful when transformers is missing
    class LogitsProcessor:  # type: ignore
        pass

from src.stt.ngram_lm import NGramLM


class NGramLogitBias(LogitsProcessor):
    def __init__(self, lm: NGramLM, tokenizer, whitelist: Set[str],
                 lambda_lm: float = 0.6, beta_tech: float = 2.5, top_k: int = 50):
        super().__init__()
        self.lm = lm
        self.tok = tokenizer
        self.whitelist = {w.lower() for w in whitelist}
        self.lambda_lm = lambda_lm
        self.beta_tech = beta_tech
        self.top_k = top_k

    def _history_words(self, input_ids: torch.Tensor) -> List[str]:
        text = self.tok.decode(input_ids.tolist(), skip_special_tokens=True)
        return text.split()

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        # input_ids: [batch, seq]; scores: [batch, vocab]
        for b in range(scores.size(0)):
            hist = tuple(self._history_words(input_ids[b])[-3:])
            # Only boost/penalise top-K candidates (fast)
            top = torch.topk(scores[b], self.top_k)
            for idx_raw in top.indices:
                idx = int(idx_raw)
                piece = self.tok.decode([idx], skip_special_tokens=True).strip().lower()
                if not piece or not piece.isalpha():
                    continue
                lp = self.lm.log_prob(piece, hist)
                delta = self.lambda_lm * lp
                if piece in self.whitelist:
                    delta += self.beta_tech
                scores[b, idx] += delta
        return scores
