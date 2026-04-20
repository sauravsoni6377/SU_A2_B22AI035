"""CLI wrapper to train the LID model (see src/lid/train.py)."""
from src.lid.train import train
import argparse

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/config.yaml")
    ap.add_argument("--en_dir", default="data/lid_raw/en")
    ap.add_argument("--hi_dir", default="data/lid_raw/hi")
    ap.add_argument("--sil_dir", default="data/lid_raw/sil")
    a = ap.parse_args()
    train(a.cfg, a.en_dir, a.hi_dir, a.sil_dir)
