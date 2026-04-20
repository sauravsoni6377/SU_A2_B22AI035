# SU Assignment 2 — Code-Switched Lecture Pipeline

End-to-end pipeline: **Hinglish lecture → Transcript → IPA → Low-Resource Language (Maithili) → Voice-cloned synthesis**, with adversarial robustness and spoof detection.

- **Course**: Speech Understanding (IITJ)
- **Target LRL**: Maithili (ISO 639-3: `mai`)
- **Stack**: PyTorch, torchaudio, Whisper, DeepFilterNet, SpeechBrain, MMS

## Directory layout

```
.
├── pipeline.py                 # End-to-end orchestrator
├── configs/config.yaml         # All hyperparameters & paths
├── requirements.txt
├── src/
│   ├── preprocessing/          # YouTube download, denoise, VAD
│   ├── lid/                    # Multi-head frame-level LID + FGSM
│   ├── stt/                    # Whisper + N-gram logit biasing
│   ├── phonetics/              # Hinglish G2P → IPA → Maithili
│   ├── tts/                    # d-vector, DTW prosody, MMS synthesis
│   ├── antispoofing/           # LFCC/CQCC + EER
│   └── utils/                  # Audio, metrics (WER/MCD/EER), viz
├── scripts/
│   ├── run_all.sh              # Full pipeline driver
│   ├── train_lid.py
│   ├── train_antispoof.py
│   └── evaluate.py
├── data/
│   ├── raw/                    # original_segment.wav, student_voice_ref.wav
│   ├── processed/              # Denoised / VAD-segmented
│   └── corpus/
│       ├── syllabus_corpus.txt      # For N-gram LM
│       └── parallel_corpus.json     # Hinglish ↔ Maithili (500+ entries)
├── outputs/
│   ├── transcript.json         # Code-switched transcript with LID
│   ├── ipa_string.txt
│   ├── maithili_text.txt
│   └── output_LRL_cloned.wav
├── report/report.md            # 10-page IEEE-format report (Markdown→PDF)
└── notes/implementation_notes.md  # 1-page non-obvious design notes
```

## Quickstart

```bash
# 1. Install deps
pip install -r requirements.txt

# 2. Fetch 10-min lecture segment (2:20:00 – 2:30:00 of the YouTube video)
python -m src.preprocessing.download \
    --url "https://youtu.be/ZPUtA3W-7_I" \
    --start "02:20:00" --end "02:30:00" \
    --out data/raw/original_segment.wav

# 3. Record your reference voice (60s @ 22.05kHz), save to data/raw/student_voice_ref.wav

# 4. Train LID (FreeSound + Common Voice frames, ~1h on GPU)
python scripts/train_lid.py --cfg configs/config.yaml

# 5. Build N-gram LM from syllabus corpus
python -m src.stt.ngram_lm --corpus data/corpus/syllabus_corpus.txt --out checkpoints/ngram.arpa

# 6. Train anti-spoofing classifier
python scripts/train_antispoof.py --cfg configs/config.yaml

# 7. Run full pipeline
bash scripts/run_all.sh

# 8. Evaluate all metrics (WER/MCD/EER/LID-switch-precision/FGSM epsilon)
python scripts/evaluate.py --cfg configs/config.yaml
```

## Evaluation targets (strict pass)

| Metric                        | Target           |
|-------------------------------|------------------|
| WER (English)                 | < 15%            |
| WER (Hindi)                   | < 25%            |
| MCD (cloned vs reference)     | < 8.0            |
| LID switch timestamp          | within 200 ms    |
| Anti-spoofing EER             | < 10%            |
| FGSM ε flipping LID (Hi→En)   | reported (SNR>40dB) |

## Citations / Libraries

- Whisper v3 — Radford et al. 2022, OpenAI. https://github.com/openai/whisper
- DeepFilterNet — Schröter et al. 2022/2023. https://github.com/Rikorose/DeepFilterNet
- SpeechBrain (x-vector, ECAPA) — Ravanelli et al. 2021. https://speechbrain.github.io
- Meta MMS — Pratap et al. 2023. https://huggingface.co/facebook/mms-tts
- KenLM — Heafield 2011 (N-gram LM).
- epitran — Mortensen et al. 2018 (G2P/IPA).
- pyworld — F0 extraction.
- ASVspoof 2019 baseline (LFCC-GMM/CM) — Todisco et al.

See `report/report.md` for full references.

## Notes

- The 10-minute segment chosen (2:20–2:30) contains continuous Hinglish lecture. Adjust in `configs/config.yaml` if your instructor asks for a different slice.
- The pipeline is designed so that each stage caches its output — re-running `pipeline.py` skips completed stages.
