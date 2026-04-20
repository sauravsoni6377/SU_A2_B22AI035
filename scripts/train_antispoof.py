"""CLI wrapper to train the anti-spoofing CM model."""
from src.antispoofing.train import train
import argparse

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/config.yaml")
    ap.add_argument("--bonafide", default="data/cm/bonafide")
    ap.add_argument("--spoof", default="data/cm/spoof")
    a = ap.parse_args()
    train(a.cfg, a.bonafide, a.spoof)
