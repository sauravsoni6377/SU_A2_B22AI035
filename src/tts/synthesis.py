"""Text-to-speech synthesis in the target LRL (Maithili).

Backends:
  - 'mms'    : Meta MMS (facebook/mms-tts-mai). No speaker conditioning, so we
               emit flat synthesis then apply voice conversion via the prosody
               module (and optionally YourTTS for timbre swap).
  - 'yourtts': Coqui YourTTS in zero-shot mode — supports speaker embeddings
               directly. Only English/PT/FR natively, so we use it only for
               TIMBRE by synthesising phonemes (IPA) and relying on the
               cross-lingual transfer paper (Casanova et al., 2022).

Because MMS-Maithili does not accept speaker embeddings, the actual pipeline is:
    1. MMS(text) -> flat_mai.wav (correct pronunciation, wrong voice)
    2. prosody.transfer_prosody(prof_wav, flat_mai.wav) -> copies intonation
    3. voice conversion (optional) -> applies student x-vector timbre via
       KNN-VC or FreeVC (both supported — FreeVC preferred as it is
       fully differentiable and ships with speechbrain).
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
import torchaudio

from src.utils.audio import load_wav, save_wav, peak_normalize


def synth_mms(text: str, out_wav: str, model_id: str = "facebook/mms-tts-mai",
              sr_out: int = 22050) -> str:
    from transformers import VitsTokenizer, VitsModel
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = VitsTokenizer.from_pretrained(model_id)
    model = VitsModel.from_pretrained(model_id).to(device).eval()
    with torch.no_grad():
        inp = tok(text, return_tensors="pt").to(device)
        out = model(**inp).waveform.squeeze(0).cpu()
    sr_native = model.config.sampling_rate
    if sr_native != sr_out:
        out = torchaudio.functional.resample(out, sr_native, sr_out)
    save_wav(out_wav, peak_normalize(out), sr_out)
    return out_wav


def voice_conversion_freevc(src_wav: str, tgt_ref_wav: str, out_wav: str,
                            sr_out: int = 22050) -> str:
    """Convert the timbre of src_wav so it sounds like tgt_ref_wav speaker."""
    try:
        from TTS.api import TTS
    except ImportError as e:
        raise RuntimeError("Coqui TTS is required for FreeVC") from e
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vc = TTS(model_name="voice_conversion_models/multilingual/vctk/freevc24").to(device)
    vc.voice_conversion_to_file(source_wav=src_wav, target_wav=tgt_ref_wav,
                                file_path=out_wav)
    # FreeVC emits 24k — optionally downsample to 22.05k to match spec
    wav, sr = load_wav(out_wav, sr=sr_out)
    save_wav(out_wav, wav, sr_out)
    return out_wav


def synth_chunked(text: str, out_wav: str, cfg: dict,
                  max_chars: int = 400) -> str:
    """Break a long translation into sentence-level chunks to avoid OOM in MMS.
    Each chunk is saved to a tmp file and concatenated."""
    sr = cfg["audio"]["synth_sr"]
    # Simple chunking on Devanagari full-stops + English periods
    import re
    sents = re.split(r"(?<=[।\.])\s+", text)
    chunks = []; cur = ""
    for s in sents:
        if len(cur) + len(s) > max_chars and cur:
            chunks.append(cur); cur = s
        else:
            cur = (cur + " " + s).strip()
    if cur: chunks.append(cur)

    tmp_dir = Path(out_wav).parent / "tts_chunks"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    chunk_files = []
    for i, c in enumerate(chunks):
        p = str(tmp_dir / f"chunk_{i:04d}.wav")
        synth_mms(c, p, model_id=cfg["tts"]["mms_model"], sr_out=sr)
        chunk_files.append(p)

    # Concatenate with a short (100 ms) silence between chunks for naturalness
    gap = torch.zeros(1, int(0.1 * sr))
    acc = []
    for p in chunk_files:
        w, _ = load_wav(p, sr=sr)
        acc.append(w); acc.append(gap)
    full = torch.cat(acc, dim=1)
    save_wav(out_wav, full, sr)
    return out_wav
