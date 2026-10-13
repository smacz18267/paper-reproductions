#!/usr/bin/env python3
"""Validate a recorded VideoSeal run and create compact evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw


def decode(path: Path) -> tuple[dict, list[np.ndarray]]:
    container = av.open(str(path))
    stream = container.streams.video[0]
    rate = stream.average_rate or stream.guessed_rate
    frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
    info = {
        "frames": len(frames),
        "width": stream.width,
        "height": stream.height,
        "fps": float(rate) if rate else None,
    }
    container.close()
    return info, frames


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def comparison(original: list[np.ndarray], marked: list[np.ndarray], output: Path) -> None:
    indices = sorted({0, len(original) // 2, len(original) - 1})
    h, w = original[0].shape[:2]
    scale = min(1.0, 420 / w)
    tw, th = max(1, round(w * scale)), max(1, round(h * scale))
    canvas = Image.new("RGB", (tw * 3, (th + 28) * len(indices)), "white")
    draw = ImageDraw.Draw(canvas)
    for row, index in enumerate(indices):
        a = original[index]
        b = marked[index]
        diff = np.clip(np.abs(a.astype(np.int16) - b.astype(np.int16)) * 5, 0, 255).astype(np.uint8)
        for column, (label, array) in enumerate((("Original", a), ("Watermarked", b), ("Difference ×5", diff))):
            image = Image.fromarray(array).resize((tw, th), Image.Resampling.LANCZOS)
            x, y = column * tw, row * (th + 28)
            draw.text((x + 5, y + 6), f"{label} — frame {index}", fill="black")
            canvas.paste(image, (x, y + 28))
    canvas.save(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    original_path = run / "input.mp4"
    output_path = run / "watermarked.mp4"
    metrics_path = run / "metrics.json"
    message_path = run / "message.txt"
    for path in (original_path, output_path, metrics_path, message_path, run / "run.log"):
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Missing or empty artifact: {path}")

    metrics = json.loads(metrics_path.read_text())
    original_info, original = decode(original_path)
    output_info, marked = decode(output_path)
    if original_info != output_info:
        raise RuntimeError(f"Video properties changed: {original_info} != {output_info}")
    if original_info["frames"] == 0:
        raise RuntimeError("No decoded frames")
    if metrics["input_frames"] != original_info["frames"] or metrics["output_frames"] != output_info["frames"]:
        raise RuntimeError("Recorded frame counts do not match decoded videos")
    if metrics["message_bits"] != 256 or len(message_path.read_text().strip()) != 256:
        raise RuntimeError("Expected a 256-bit VideoSeal message")
    if not 0.0 <= float(metrics["bit_accuracy"]) <= 1.0:
        raise RuntimeError("Invalid bit accuracy")

    comparison(original, marked, run / "comparison.png")
    artifacts = [original_path, output_path, metrics_path, message_path, run / "comparison.png", run / "run.log"]
    manifest = {
        "video": output_info,
        "metrics": metrics,
        "files": {path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)} for path in artifacts},
    }
    (run / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (run / "README.md").write_text(
        "# Recorded VideoSeal inference\n\n"
        f"The pretrained VideoSeal v1.0 model embedded and recovered a 256-bit message "
        f"with **{metrics['bit_accuracy'] * 100:.2f}% bit accuracy**. The output contains "
        f"{output_info['frames']} frames at {output_info['width']} × {output_info['height']} "
        f"and {output_info['fps']:g} FPS. `artifact_manifest.json` records SHA-256 hashes.\n"
    )
    print("VideoSeal artifacts decoded and verified.")


if __name__ == "__main__":
    main()
