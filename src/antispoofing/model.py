"""Lightweight CM classifier: LFCC/CQCC → BiLSTM + attention pool → sigmoid.

For the assignment's small dataset (your own voice + Part-III synthesised
clips), a compact recurrent model trained with focal loss generalises much
better than a deep ResNet, which overfits.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttnPool(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.q = nn.Linear(d, 1)

    def forward(self, x):   # [B, T, D]
        w = F.softmax(self.q(x).squeeze(-1), dim=1)     # [B, T]
        return torch.einsum("btd,bt->bd", x, w)


class CMModel(nn.Module):
    def __init__(self, in_dim: int = 60, hidden: int = 128, n_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden)
        self.lstm = nn.LSTM(hidden, hidden, num_layers=n_layers,
                            bidirectional=True, batch_first=True, dropout=dropout)
        self.pool = AttnPool(2 * hidden)
        self.head = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.SiLU(),
            nn.Dropout(dropout), nn.Linear(hidden, 1),
        )

    def forward(self, x):    # x: [B, T, F]
        x = self.proj(x)
        x, _ = self.lstm(x)
        x = self.pool(x)
        return self.head(x).squeeze(-1)   # logits


def focal_loss(logits, targets, gamma: float = 2.0):
    """targets: 0 (spoof) / 1 (bonafide)."""
    p = torch.sigmoid(logits)
    pt = torch.where(targets == 1, p, 1 - p)
    bce = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction="none")
    return ((1 - pt) ** gamma * bce).mean()


def score(logits: torch.Tensor) -> np.ndarray:
    """Higher score = more confident BONA FIDE."""
    return torch.sigmoid(logits).detach().cpu().numpy()
