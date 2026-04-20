"""LID dataset.

Primary source: Mozilla Common Voice (EN + HI). Silence class is synthesised by
mixing low-energy room-tone noise at near-zero SNR onto padding regions.

Training clips are produced by concatenating random {EN, HI, SIL} chunks into
1–6 s utterances, producing frame-level labels automatically.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
import random
import torch
import torchaudio
from torch.utils.data import Dataset

from src.utils.audio import load_wav


@dataclass
class Clip:
    path: Path
    lang: str          # 'en' | 'hi' | 'sil'
    duration_s: float


class LIDDataset(Dataset):
    """Each sample: (wav[T], frame_labels[T//hop])."""

    LANG2ID = {"en": 0, "hi": 1, "sil": 2}

    def __init__(self, clips: List[Clip], sr: int = 16000, hop_ms: float = 10.0,
                 min_s: float = 2.0, max_s: float = 6.0, frames_per_label_subsample: int = 4):
        self.clips = [c for c in clips if c.duration_s >= 0.5]
        self.by_lang = {l: [c for c in self.clips if c.lang == l] for l in ("en", "hi", "sil")}
        self.sr = sr
        self.hop = int(sr * hop_ms / 1000)
        self.min_s, self.max_s = min_s, max_s
        self.sub = frames_per_label_subsample  # conv subsample factor

    def __len__(self): return max(1, len(self.clips))

    def _rand_chunk(self, lang: str, dur_s: float) -> torch.Tensor:
        bank = self.by_lang.get(lang, [])
        if not bank:
            # fallback silence
            return torch.zeros(int(self.sr * dur_s))
        c = random.choice(bank)
        wav, _ = load_wav(c.path, sr=self.sr)
        wav = wav.squeeze(0)
        need = int(self.sr * dur_s)
        if wav.numel() < need:
            reps = need // wav.numel() + 1
            wav = wav.repeat(reps)
        start = random.randint(0, wav.numel() - need)
        return wav[start:start + need]

    def __getitem__(self, idx):
        total_s = random.uniform(self.min_s, self.max_s)
        n_segs = random.choice([1, 2, 3])
        # Generate segment plan
        boundaries = sorted(random.sample(range(1, 20), n_segs - 1)) if n_segs > 1 else []
        frac = [b / 20 for b in boundaries] + [1.0]
        prev = 0.0
        chunks: List[Tuple[torch.Tensor, str]] = []
        langs_avail = ["en", "hi"]
        for f in frac:
            dur = (f - prev) * total_s
            prev = f
            lang = random.choice(langs_avail + ["sil"] * 0)  # prefer speech
            chunks.append((self._rand_chunk(lang, dur), lang))
        # Sprinkle short silence pad (~5%)
        if random.random() < 0.3:
            chunks.insert(random.randint(0, len(chunks)), (self._rand_chunk("sil", 0.3), "sil"))
        wav = torch.cat([c for c, _ in chunks])
        # Frame labels (at 10-ms grid) then downsample /4 to match model output hop
        labels = []
        for c, lang in chunks:
            n = c.numel() // self.hop
            labels += [self.LANG2ID[lang]] * n
        labels = torch.tensor(labels, dtype=torch.long)
        # Downsample by conv subsample
        t = (labels.numel() // self.sub) * self.sub
        labels = labels[:t].reshape(-1, self.sub).mode(dim=-1).values
        return wav, labels


def collate_fn(batch):
    lens = [b[0].numel() for b in batch]
    L = max(lens)
    wavs = torch.zeros(len(batch), L)
    label_lens = [b[1].numel() for b in batch]
    Lb = max(label_lens)
    labels = torch.full((len(batch), Lb), fill_value=-100, dtype=torch.long)
    for i, (w, y) in enumerate(batch):
        wavs[i, :w.numel()] = w
        labels[i, :y.numel()] = y
    return wavs, labels


def scan_clips(en_dir: str, hi_dir: str, sil_dir: str, max_per_lang: int = 4000) -> List[Clip]:
    """Scan folders of wav/flac/mp3 files. Durations via torchaudio.info (fast)."""
    def collect(root, lang):
        out = []
        for p in Path(root).glob("**/*"):
            if p.suffix.lower() not in (".wav", ".flac", ".mp3"):
                continue
            try:
                info = torchaudio.info(str(p))
                dur = info.num_frames / info.sample_rate
                out.append(Clip(p, lang, dur))
            except Exception:
                continue
            if len(out) >= max_per_lang:
                break
        return out
    return collect(en_dir, "en") + collect(hi_dir, "hi") + collect(sil_dir, "sil")
