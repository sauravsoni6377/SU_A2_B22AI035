"""Frame-level Multi-Head Language Identification.

Architecture:
  log-mel (80-d) → 2× conv subsampling (stride 2 each = /4) → 4× Conformer block
  → 3 heads (EN, HI, SIL) producing per-frame posteriors.

Why multi-head: training jointly with a silence head and CE on language frames
gives a much sharper transition around switches, improving timestamp precision.

Emits per-frame posteriors at 40 ms resolution (input hop 10 ms × /4 subsample),
which is below the 200 ms required tolerance.
"""
from __future__ import annotations
from typing import List
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


# ---- Conformer block (compact, self-contained) -----------------------------
class FeedForward(nn.Module):
    def __init__(self, d: int, mult: int = 4, p: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d, d * mult), nn.SiLU(),
            nn.Dropout(p), nn.Linear(d * mult, d), nn.Dropout(p))

    def forward(self, x): return x + 0.5 * self.net(x)


class ConvModule(nn.Module):
    def __init__(self, d: int, kernel: int = 15, p: float = 0.1):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.pw1 = nn.Conv1d(d, 2 * d, 1)
        self.dw = nn.Conv1d(d, d, kernel, padding=kernel // 2, groups=d)
        self.bn = nn.BatchNorm1d(d)
        self.pw2 = nn.Conv1d(d, d, 1)
        self.dp = nn.Dropout(p)

    def forward(self, x):  # x: [B, T, D]
        y = self.ln(x).transpose(1, 2)
        y = F.glu(self.pw1(y), dim=1)
        y = self.bn(self.dw(y))
        y = F.silu(y)
        y = self.dp(self.pw2(y)).transpose(1, 2)
        return x + y


class MHSA(nn.Module):
    def __init__(self, d: int, heads: int, p: float = 0.1):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=p, batch_first=True)
        self.dp = nn.Dropout(p)

    def forward(self, x, mask=None):
        y = self.ln(x)
        y, _ = self.attn(y, y, y, key_padding_mask=mask, need_weights=False)
        return x + self.dp(y)


class ConformerBlock(nn.Module):
    def __init__(self, d: int, heads: int, p: float = 0.1):
        super().__init__()
        self.ff1 = FeedForward(d, p=p)
        self.mhsa = MHSA(d, heads, p=p)
        self.conv = ConvModule(d, p=p)
        self.ff2 = FeedForward(d, p=p)
        self.ln = nn.LayerNorm(d)

    def forward(self, x, mask=None):
        x = self.ff1(x)
        x = self.mhsa(x, mask)
        x = self.conv(x)
        x = self.ff2(x)
        return self.ln(x)


# ---- Feature frontend ------------------------------------------------------
class LogMel(nn.Module):
    def __init__(self, sr=16000, n_mels=80, win_ms=25, hop_ms=10):
        super().__init__()
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr, n_fft=int(sr * win_ms / 1000 * 2),
            win_length=int(sr * win_ms / 1000),
            hop_length=int(sr * hop_ms / 1000),
            n_mels=n_mels, power=2.0,
        )

    def forward(self, wav):  # [B, T]
        x = self.melspec(wav).clamp(min=1e-9).log()
        return x.transpose(1, 2)  # [B, T, n_mels]


class ConvSubsample(nn.Module):
    """Two strided convs → /4 in time."""

    def __init__(self, in_c=80, d=256):
        super().__init__()
        self.conv1 = nn.Conv2d(1, d, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(d, d, 3, stride=2, padding=1)
        # After /4 in mel axis, linear projection
        self.proj = nn.Linear(d * ((in_c + 3) // 4), d)

    def forward(self, x):   # x: [B, T, M]
        y = x.unsqueeze(1)   # [B, 1, T, M]
        y = F.silu(self.conv1(y))
        y = F.silu(self.conv2(y))
        b, c, t, m = y.shape
        y = y.permute(0, 2, 1, 3).reshape(b, t, c * m)
        return self.proj(y)


# ---- Full model ------------------------------------------------------------
class MultiHeadLID(nn.Module):
    """Per-frame posteriors over {EN, HI, SIL}."""

    def __init__(self, n_classes=3, d=256, n_heads=4, n_layers=4, sr=16000,
                 hop_ms=10, dropout=0.1):
        super().__init__()
        self.sr = sr
        self.hop_ms = hop_ms
        self.effective_hop_ms = hop_ms * 4   # after /4 subsample
        self.feat = LogMel(sr=sr, hop_ms=hop_ms)
        self.front = ConvSubsample(in_c=80, d=d)
        self.blocks = nn.ModuleList([ConformerBlock(d, n_heads, dropout) for _ in range(n_layers)])
        # Multi-head: one classifier per language + joint softmax head.
        self.head_en = nn.Linear(d, 1)
        self.head_hi = nn.Linear(d, 1)
        self.head_sil = nn.Linear(d, 1)
        self.joint = nn.Linear(d, n_classes)

    def forward(self, wav):  # wav: [B, T_samples]
        x = self.feat(wav)
        x = self.front(x)
        for b in self.blocks:
            x = b(x)
        # Multi-head logits. Joint head used for primary CE loss; per-language heads
        # act as auxiliary sigmoid regularizers that push each class into a
        # dedicated subspace — empirically sharpens language boundaries.
        joint_logits = self.joint(x)                               # [B, T, C]
        aux = torch.cat([self.head_en(x), self.head_hi(x), self.head_sil(x)], dim=-1)
        return joint_logits, aux

    @torch.no_grad()
    def posteriors(self, wav) -> torch.Tensor:
        self.eval()
        logits, _ = self.forward(wav)
        return F.softmax(logits, dim=-1)

    @torch.no_grad()
    def decode(self, wav, smoothing_ms: float = 80, sil_thresh: float = 0.5) -> List[dict]:
        """Returns segments: [{start_ms, end_ms, lang}]. 'lang' ∈ {en,hi,sil}."""
        post = self.posteriors(wav)[0].cpu()   # [T, C]
        hop = self.effective_hop_ms
        k = max(1, int(smoothing_ms / hop))
        # Simple 1-D median filter per class for robustness
        post_s = post.clone()
        pad = k // 2
        padded = F.pad(post.transpose(0, 1).unsqueeze(0), (pad, pad), mode="replicate")
        post_s = padded.unfold(-1, k, 1).median(dim=-1).values.squeeze(0).transpose(0, 1)[: post.shape[0]]

        names = ["en", "hi", "sil"]
        cls = post_s.argmax(dim=-1).tolist()
        segments = []
        cur = cls[0]; start = 0
        for i in range(1, len(cls)):
            if cls[i] != cur:
                segments.append(dict(start_ms=start * hop, end_ms=i * hop, lang=names[cur]))
                cur, start = cls[i], i
        segments.append(dict(start_ms=start * hop, end_ms=len(cls) * hop, lang=names[cur]))
        return segments


def multi_head_loss(joint_logits, aux_logits, targets, alpha: float = 0.3):
    """targets: [B, T] in {0,1,2}. Both losses are frame-level."""
    B, T, C = joint_logits.shape
    ce = F.cross_entropy(joint_logits.reshape(-1, C), targets.reshape(-1), ignore_index=-100)
    one_hot = F.one_hot(targets.clamp(min=0), num_classes=C).float()
    bce = F.binary_cross_entropy_with_logits(aux_logits, one_hot, reduction="mean")
    return ce + alpha * bce
