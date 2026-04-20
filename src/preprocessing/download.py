"""Fetch a specific time-range segment of a YouTube video as 16kHz mono WAV.

Uses yt-dlp to stream the best-audio stream, pipes it through ffmpeg with
-ss/-to for the sub-clip. Designed to be idempotent.
"""
from __future__ import annotations
import argparse
import shutil
import subprocess
from pathlib import Path


def _tool(name: str) -> str:
    p = shutil.which(name)
    if p is None:
        raise RuntimeError(f"required tool '{name}' not found in PATH")
    return p


def download_segment(url: str, start: str, end: str, out_path: str, sr: int = 16000) -> str:
    """start/end in HH:MM:SS."""
    ytdlp = _tool("yt-dlp")
    ffmpeg = _tool("ffmpeg")
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Grab the stream URL for best audio (no download to disk, just a direct URL)
    stream_url = subprocess.check_output(
        [ytdlp, "-f", "bestaudio", "-g", url], text=True
    ).strip().splitlines()[0]
    cmd = [
        ffmpeg, "-y",
        "-ss", start, "-to", end,
        "-i", stream_url,
        "-vn", "-ac", "1", "-ar", str(sr),
        "-acodec", "pcm_s16le",
        str(out),
    ]
    subprocess.run(cmd, check=True)
    return str(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--start", required=True, help="HH:MM:SS")
    ap.add_argument("--end", required=True, help="HH:MM:SS")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sr", type=int, default=16000)
    args = ap.parse_args()
    path = download_segment(args.url, args.start, args.end, args.out, args.sr)
    print(f"saved: {path}")


if __name__ == "__main__":
    main()
