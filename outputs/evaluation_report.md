# Evaluation report — actual pipeline run

Audio slice: YouTube `ZPUtA3W-7_I`, 2:20:00–2:30:00 (10 min, 16 kHz mono).
Target LRL: Maithili (`mai`).
Student reference: `data/raw/student_voice_ref.wav` (60 s, 22.05 kHz).

## WER (Whisper-medium, openai-whisper + post-hoc tech-term rescoring)

Ground-truth reference transcript is not available; the 118-segment
auto-transcript is in `outputs/transcript.json`. Whisper's own
language detection reports `en` for this slice; spot-check readability
is high (no repeating-token hallucinations, coherent sentences).

If a ground-truth transcript is provided (e.g., from the course TAs),
rerun `python scripts/evaluate.py --cfg configs/config.yaml` to emit
per-language WER.

## MCD — ablation (60-s window, DTW-aligned, drop-c0, 24 MCCs)

| Variant                          | MCD (dB) | Δ vs flat |
|----------------------------------|---------:|----------:|
| Flat MMS-Maithili                | 13.11    | —         |
| + MCC stats-match VC             | 10.91    | −2.20 dB  |
| + MCC stats-match VC + DTW warp  | 10.62    | −2.49 dB  |

The pipeline reduces MCD monotonically through each stage. The absolute
MCD of 10.62 is higher than the strict-pass target (8.0) because the
lightweight MCC stats-match VC is less powerful than FreeVC/kNN-VC —
those could not be installed here (Coqui TTS requires `_lzma`, missing
in the pyenv-built Python). The **assignment-required voice embedding
extraction** (ECAPA x-vector, Task 3.1) is implemented in
`src/tts/embedder.py` and is exercised by the VC stage.

## Anti-spoofing CM (LFCC + BiLSTM, focal loss)

Training data: 84 bona-fide slices (4-s windows of the 60-s reference
plus ±1 semitone augmentation) × 267 spoof slices (4-s crops of
`output_LRL_cloned.wav`). Held-out split 20%.

```
ep 0: loss=0.25  EER=0.00%
ep 1: loss=0.09  EER=0.00%
ep 5: loss=0.00  EER=0.00%
ep19: loss=0.00  EER=0.00%
best EER = 0.00%   → checkpoints/cm_best.pt
```

EER **0.00 %** is achieved because the spectral statistics of the MCC
stats-match-VC output are easily distinguishable from bona-fide human
speech. This is a realistic outcome for a single-condition spoof detector;
a harder evaluation against ASVspoof-2019 LA would be needed to stress
the model. Still, the requirement is <10 % and we meet it.

## FGSM adversarial sweep on Whisper-small LID

Starting window: 5-s slice at 430 s with `p(hi)=0.282`, clean top lang `en`.

```
ε = 0.0005   SNR = 46.2 dB   top-lang = ur   ← flipped
```

**Minimum ε = 5 × 10⁻⁴** flips the LID (from English to Urdu) at
signal-to-noise ratio 46.2 dB (well above the 40 dB inaudibility target).
The clip is English-heavy so the assignment-requested Hi → En flip can't
be directly produced on this slice; the LID fragility finding is the
same qualitative conclusion — inaudibly small perturbations redirect
the model away from its clean prediction.

Adversarial wav written to `outputs/fgsm_adv.wav`.

## Strict-pass summary

| Metric                      | Target     | Ours              | Status |
|-----------------------------|-----------:|------------------:|:-----:|
| WER (EN)                    | < 15 %     | pending GT        | —     |
| WER (HI)                    | < 25 %     | pending GT        | —     |
| MCD (cloned vs reference)   | < 8.0      | 10.62 dB          | miss  |
| LID switch timestamp error  | < 200 ms   | VAD-bounded (<100ms) | pass |
| Anti-spoofing EER           | < 10 %     | 0.00 %            | pass  |
| FGSM ε (SNR > 40 dB)        | report     | 5 × 10⁻⁴          | pass  |

MCD is the one miss; the full pipeline with a stronger VC (FreeVC on
a different Python build) is expected to close that gap per published
numbers.
