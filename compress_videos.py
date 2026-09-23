"""Downscale and compress the source videos to match a low-quality camera.

Originals in videos/ are left untouched; results go to videos_lowres/ as .mp4.

Usage:
    python compress_videos.py                          # 640x360 @ 30 fps, CRF 30
    python compress_videos.py --size 640x480 --fps 15  # match a 4:3 webcam
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VIDEO_EXTS = {".mov", ".mp4", ".avi", ".mkv", ".m4v"}


def parse_size(text):
    w, h = (int(v) for v in text.lower().split("x"))
    if w % 2 or h % 2:
        raise argparse.ArgumentTypeError("width and height must be even")
    return w, h


def compress(src, dst, width, height, fps, crf):
    # Scale to cover the target size, then centre-crop, so any aspect ratio works.
    # ffmpeg applies the rotation metadata automatically, so upside-down clips come out upright.
    vf = (f"scale={width}:{height}:force_original_aspect_ratio=increase:flags=area,"
          f"crop={width}:{height},fps={fps}")
    cmd = [
        "ffmpeg", "-hide_banner", "-v", "error", "-y",
        "-i", str(src),
        "-vf", vf,
        "-c:v", "libx264", "-preset", "medium", "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        "-an",
        str(dst),
    ]
    subprocess.run(cmd, check=True)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", type=Path, default=ROOT / "videos")
    p.add_argument("--dst", type=Path, default=ROOT / "videos_lowres")
    p.add_argument("--size", type=parse_size, default=(640, 360),
                   help="output WIDTHxHEIGHT (default 640x360)")
    p.add_argument("--fps", type=float, default=30, help="output frame rate (default 30)")
    p.add_argument("--crf", type=int, default=30,
                   help="x264 quality, 18-35; higher = smaller and worse (default 30)")
    p.add_argument("--overwrite", action="store_true", help="redo files that already exist")
    args = p.parse_args()

    videos = sorted(f for f in args.src.iterdir() if f.suffix.lower() in VIDEO_EXTS)
    if not videos:
        sys.exit(f"no videos found in {args.src}")
    args.dst.mkdir(parents=True, exist_ok=True)

    width, height = args.size
    for i, src in enumerate(videos, 1):
        dst = args.dst / f"{src.stem}.mp4"
        tag = f"[{i}/{len(videos)}] {src.name}"
        if dst.exists() and not args.overwrite:
            print(f"{tag}: exists, skipping")
            continue
        print(f"{tag}: compressing...", flush=True)
        try:
            compress(src, dst, width, height, args.fps, args.crf)
        except subprocess.CalledProcessError:
            dst.unlink(missing_ok=True)
            print(f"{tag}: FAILED", file=sys.stderr)
            continue
        before, after = src.stat().st_size / 1e6, dst.stat().st_size / 1e6
        print(f"{tag}: {before:.0f} MB -> {after:.1f} MB")


if __name__ == "__main__":
    main()
