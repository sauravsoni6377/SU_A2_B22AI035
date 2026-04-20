#!/usr/bin/env bash
# Full pipeline driver.
set -euo pipefail

CFG="${CFG:-configs/config.yaml}"

echo "[1/4] building N-gram LM from syllabus corpus"
python -m src.stt.ngram_lm \
    --corpus data/corpus/syllabus_corpus.txt \
    --out_json checkpoints/ngram.json \
    --out_arpa checkpoints/ngram.arpa \
    --order 4

echo "[2/4] running full pipeline (download → synth)"
python pipeline.py --cfg "$CFG"

echo "[3/4] evaluating metrics"
python scripts/evaluate.py --cfg "$CFG" || true

echo "[4/4] running FGSM adversarial sweep on LID"
python -m src.lid.adversarial \
    --cfg "$CFG" \
    --wav data/processed/original_denoised.wav \
    --ckpt checkpoints/lid_best.pt \
    --out outputs/fgsm_adv.wav || true

echo "done."
