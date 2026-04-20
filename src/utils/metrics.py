"""Evaluation metrics: WER, MCD, EER, LID timestamp precision."""
from __future__ import annotations
from typing import List, Tuple, Sequence
import numpy as np


# -------- WER ---------------------------------------------------------------
def wer(refs: Sequence[str], hyps: Sequence[str]) -> float:
    """Standard word error rate. refs/hyps may be a single string each."""
    try:
        import jiwer
        return float(jiwer.wer(list(refs), list(hyps)))
    except ImportError:
        # Fallback Levenshtein
        total_edits, total_words = 0, 0
        for r, h in zip(refs, hyps):
            r_tok, h_tok = r.split(), h.split()
            total_words += len(r_tok)
            total_edits += _edit_distance(r_tok, h_tok)
        return total_edits / max(total_words, 1)


def _edit_distance(a, b) -> int:
    n, m = len(a), len(b)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, m + 1):
            cur = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j - 1], dp[j])
            prev = cur
    return dp[m]


def wer_by_language(segments: List[dict]) -> dict:
    """segments: list of {lang, ref, hyp}. Returns {'en': wer, 'hi': wer}."""
    buckets = {}
    for s in segments:
        buckets.setdefault(s["lang"], {"ref": [], "hyp": []})
        buckets[s["lang"]]["ref"].append(s["ref"])
        buckets[s["lang"]]["hyp"].append(s["hyp"])
    return {lang: wer(b["ref"], b["hyp"]) for lang, b in buckets.items()}


# -------- MCD ---------------------------------------------------------------
def mcd(ref_wav: np.ndarray, syn_wav: np.ndarray, sr: int = 22050) -> float:
    """Mel-Cepstral Distortion (dB) using DTW-aligned mel-cepstral coefficients.

    MCD = (10 / ln10) * sqrt(2 * sum_{d=1..D}(mc_ref[d] - mc_syn[d])^2)
    averaged over aligned frames.

    Uses pyworld + pysptk for MCC extraction (librosa-free — compatible with
    Python builds without _lzma).
    """
    import pyworld as pw
    import pysptk
    from scipy.spatial.distance import cdist

    alpha = 0.455 if sr >= 22050 else 0.41
    order = 24

    def mcc(y: np.ndarray) -> np.ndarray:
        y = y.astype(np.float64)
        f0, t = pw.dio(y, sr, frame_period=5.0)
        f0 = pw.stonemask(y, f0, t, sr)
        sp = pw.cheaptrick(y, f0, t, sr)
        mc = np.stack([pysptk.sp2mc(s, order, alpha) for s in sp], axis=0)
        return mc[:, 1:]   # drop c0, shape [T, order]

    r, s = mcc(ref_wav), mcc(syn_wav)
    D = cdist(r, s, metric="euclidean")
    # Simple DTW path with unit step costs
    T1, T2 = D.shape
    acc = np.full((T1 + 1, T2 + 1), np.inf)
    acc[0, 0] = 0
    for i in range(1, T1 + 1):
        for j in range(1, T2 + 1):
            acc[i, j] = D[i - 1, j - 1] + min(acc[i - 1, j - 1], acc[i - 1, j], acc[i, j - 1])
    # Back-trace length
    i, j, path = T1, T2, []
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        step = np.argmin([acc[i - 1, j - 1], acc[i - 1, j], acc[i, j - 1]])
        if step == 0:
            i, j = i - 1, j - 1
        elif step == 1:
            i -= 1
        else:
            j -= 1
    diffs = np.array([np.sum((r[a] - s[b]) ** 2) for a, b in path])
    mcd_val = (10.0 / np.log(10)) * np.sqrt(2.0) * np.mean(np.sqrt(diffs))
    return float(mcd_val)


# -------- EER ---------------------------------------------------------------
def compute_eer(labels: np.ndarray, scores: np.ndarray) -> Tuple[float, float]:
    """labels: 1 = bona-fide, 0 = spoof. Returns (EER, threshold)."""
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores).astype(float)
    order = np.argsort(-scores)  # descending
    labels = labels[order]
    scores = scores[order]

    P = max(1, (labels == 1).sum())
    N = max(1, (labels == 0).sum())
    tpr = np.cumsum(labels == 1) / P
    fpr = np.cumsum(labels == 0) / N
    fnr = 1 - tpr
    # Threshold where FPR ≈ FNR
    diff = np.abs(fpr - fnr)
    idx = np.argmin(diff)
    eer = (fpr[idx] + fnr[idx]) / 2.0
    return float(eer), float(scores[idx])


# -------- LID switch timestamp precision ------------------------------------
def switch_timestamp_precision(
    pred_switches_ms: Sequence[float],
    ref_switches_ms: Sequence[float],
    tolerance_ms: float = 200,
) -> dict:
    """Match each predicted switch to nearest ref switch; report precision/recall
    within tolerance and median absolute error.
    """
    pred = np.asarray(list(pred_switches_ms))
    ref = np.asarray(list(ref_switches_ms))
    if len(pred) == 0 or len(ref) == 0:
        return {"precision": 0.0, "recall": 0.0, "median_abs_err_ms": float("inf")}
    matched_p, matched_r, errs = 0, 0, []
    used = np.zeros(len(ref), dtype=bool)
    for p in pred:
        d = np.abs(ref - p)
        d[used] = np.inf
        j = int(np.argmin(d))
        if d[j] <= tolerance_ms:
            matched_p += 1
            used[j] = True
            errs.append(d[j])
    matched_r = used.sum()
    return {
        "precision": matched_p / len(pred),
        "recall": float(matched_r) / len(ref),
        "median_abs_err_ms": float(np.median(errs)) if errs else float("inf"),
    }
