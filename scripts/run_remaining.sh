#!/usr/bin/env bash
# After transcript.json exists, this finishes the pipeline.
set -euo pipefail
source .venv/bin/activate

echo "[4/9] IPA mapping"
python -u -m src.phonetics.g2p_hinglish \
    --transcript outputs/transcript.json \
    --out_txt outputs/ipa_string.txt \
    --out_tokens outputs/ipa_tokens.json

echo "[5/9] translation (dict + NLLB)"
python -u -m src.phonetics.translate \
    --cfg configs/config.yaml \
    --transcript outputs/transcript.json \
    --out_txt outputs/maithili_text.txt \
    --out_json outputs/translation.json

echo "[6/9] MMS synthesis (chunked)"
python -u -c "
import yaml
from pathlib import Path
from src.tts.synthesis import synth_chunked
cfg = yaml.safe_load(open('configs/config.yaml'))
text = Path('outputs/maithili_text.txt').read_text(encoding='utf-8')
synth_chunked(text, 'outputs/mms_flat.wav', cfg)
print('flat synth done')
"

echo "[7/9] stats-match voice conversion"
python -u -m src.tts.voice_convert \
    --src outputs/mms_flat.wav \
    --ref data/raw/student_voice_ref.wav \
    --out outputs/mms_vc.wav --sr 22050

echo "[8/9] DTW prosody warping"
python -u -m src.tts.prosody \
    --src data/processed/original_denoised.wav \
    --tgt outputs/mms_vc.wav \
    --out outputs/output_LRL_cloned.wav --sr 22050

echo "[9/9] anti-spoof data + train + FGSM"
python -u scripts/prepare_cm_data.py \
    --ref data/raw/student_voice_ref.wav \
    --spoof outputs/output_LRL_cloned.wav
python -u scripts/train_antispoof.py --cfg configs/config.yaml || echo "CM training skipped"
python -u -m src.lid.adversarial_whisper \
    --cfg configs/config.yaml \
    --wav data/processed/original_denoised.wav \
    --size small \
    --out outputs/fgsm_adv.wav || echo "FGSM sweep skipped"

echo "[ok] pipeline complete."
ls -lh outputs/
