"""Plotting helpers (LID posteriors, prosody warp)."""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_lid_posteriors(post: np.ndarray, hop_ms: float, out_png: str, classes=("EN", "HI", "SIL")):
    """post: [T, C] posteriors."""
    t = np.arange(post.shape[0]) * hop_ms / 1000.0
    plt.figure(figsize=(10, 3))
    for c in range(post.shape[1]):
        plt.plot(t, post[:, c], label=classes[c])
    plt.ylim(0, 1)
    plt.xlabel("time (s)")
    plt.ylabel("p(class)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=120)
    plt.close()


def plot_f0_warp(src_f0: np.ndarray, tgt_f0: np.ndarray, warped: np.ndarray, out_png: str):
    plt.figure(figsize=(10, 3))
    plt.plot(src_f0, label="source F0 (prof)")
    plt.plot(tgt_f0, label="cloned F0 (pre-warp)")
    plt.plot(warped, label="cloned F0 (post-warp)")
    plt.xlabel("frame")
    plt.ylabel("Hz")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=120)
    plt.close()


def plot_confusion(cm: np.ndarray, labels, out_png: str, title="Confusion"):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im)
    plt.tight_layout()
    plt.savefig(out_png, dpi=120)
    plt.close()
