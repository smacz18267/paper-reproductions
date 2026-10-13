#!/usr/bin/env python3
"""Validate Safe-Sora comparison artifacts and compute simple fidelity metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import av
import numpy as np
from PIL import Image


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fidelity(first: np.ndarray, second: np.ndarray) -> dict:
    error = first.astype(np.float64) - second.astype(np.float64)
    mse = float(np.mean(error * error))
    psnr = float("inf") if mse == 0 else 10 * math.log10((255.0 * 255.0) / mse)
    return {"mse": mse, "psnr_db": psnr, "mean_absolute_error": float(np.mean(np.abs(error)))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    video_path = run / "work" / "results" / "cover_combined.mp4"
    image_path = run / "work" / "results" / "secret_combined.png"
    log_path = run / "run.log"
    for path in (video_path, image_path, log_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Missing or empty artifact: {path}")

    container = av.open(str(video_path))
    stream = container.streams.video[0]
    rate = stream.average_rate or stream.guessed_rate
    frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
    container.close()
    if len(frames) != 16:
        raise RuntimeError(f"Expected 16 frames, found {len(frames)}")
    height, total_width = frames[0].shape[:2]
    if total_width % 3:
        raise RuntimeError(f"Composite video width is not divisible by three: {total_width}")
    width = total_width // 3
    original = np.stack([frame[:, :width] for frame in frames])
    watermarked = np.stack([frame[:, width : 2 * width] for frame in frames])

    combined = np.asarray(Image.open(image_path).convert("RGB"))
    if combined.shape[1] % 3:
        raise RuntimeError("Composite watermark width is not divisible by three")
    watermark_width = combined.shape[1] // 3
    watermark_original = combined[:, :watermark_width]
    watermark_recovered = combined[:, watermark_width : 2 * watermark_width]
    metrics = {
        "video_fidelity": fidelity(original, watermarked),
        "watermark_fidelity": fidelity(watermark_original, watermark_recovered),
        "video": {
            "frames": len(frames),
            "panel_width": width,
            "height": height,
            "fps": float(rate) if rate else None,
        },
        "watermark": {"width": watermark_width, "height": combined.shape[0]},
    }
    (run / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    artifacts = [video_path, image_path, log_path, run / "metrics.json"]
    manifest = {
        "metrics": metrics,
        "files": {
            str(path.relative_to(run)): {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in artifacts
        },
    }
    (run / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (run / "README.md").write_text(
        "# Recorded Safe-Sora demonstration\n\n"
        f"The pretrained workflow produced 16 comparison frames. Original-to-watermarked "
        f"video PSNR was **{metrics['video_fidelity']['psnr_db']:.2f} dB**; original-to-recovered "
        f"graphical-watermark PSNR was **{metrics['watermark_fidelity']['psnr_db']:.2f} dB**. "
        "The composite MP4 shows original, watermarked and amplified difference panels; the PNG "
        "shows original, recovered and amplified difference watermarks.\n"
    )
    print("Safe-Sora artifacts decoded and verified.")


if __name__ == "__main__":
    main()
