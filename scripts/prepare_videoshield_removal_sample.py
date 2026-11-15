#!/usr/bin/env python3
"""Prepare and finalize VideoShield's provided sample for removal testing."""

from __future__ import annotations

import argparse
import json
import shutil
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

from mose_watermark_common import (
    atomic_json,
    copy_atomic,
    decode_rgb,
    encode_binary_masks,
    encode_rgb,
    read_mask_sequence,
    sha256,
    validate_video,
)


EXPECTED_FRAMES = 16
PADDED_FRAMES = 24
EXPECTED_SIZE = (256, 256)


def write_rgb_frames(directory: Path, frames: list[np.ndarray]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for old in directory.glob("*.png"):
        old.unlink()
    for index, frame in enumerate(frames):
        path = directory / f"{index:02d}.png"
        if not cv2.imwrite(str(path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)):
            raise RuntimeError(f"Could not write {path}")


def write_mask_frames(directory: Path, masks: list[np.ndarray]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for old in directory.glob("*.png"):
        old.unlink()
    for index, mask in enumerate(masks):
        path = directory / f"{index:02d}.png"
        binary = np.where(mask > 127, 255, 0).astype(np.uint8)
        if not cv2.imwrite(str(path), binary):
            raise RuntimeError(f"Could not write {path}")


def prepare(args: argparse.Namespace) -> None:
    frames, rate = decode_rgb(args.source_video)
    masks = read_mask_sequence(args.source_mask_frames)
    if len(frames) != EXPECTED_FRAMES or len(masks) != EXPECTED_FRAMES:
        raise RuntimeError(
            f"VideoShield ModelScope sample must have {EXPECTED_FRAMES} frames; "
            f"found video={len(frames)}, masks={len(masks)}"
        )
    if any(frame.shape[:2][::-1] != EXPECTED_SIZE for frame in frames):
        raise RuntimeError(f"VideoShield sample frames must be {EXPECTED_SIZE[0]}x{EXPECTED_SIZE[1]}")
    if any(mask.shape[:2][::-1] != EXPECTED_SIZE for mask in masks):
        raise RuntimeError(f"VideoShield masks must be {EXPECTED_SIZE[0]}x{EXPECTED_SIZE[1]}")
    masks = [np.where(mask > 127, 255, 0).astype(np.uint8) for mask in masks]
    if any(not np.any(mask) for mask in masks):
        raise RuntimeError("Every official VideoShield sample mask must contain a tampered region")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    copy_atomic(args.source_video, output / "input-watermarked.mp4")
    copy_atomic(args.source_watermark_info, output / "wm_info.bin")
    write_rgb_frames(output / "input-frames", frames)
    write_mask_frames(output / "gt-mask-frames", masks)
    encode_binary_masks(output / "gt-mask.mp4", masks, rate)

    padded_frames = frames + [frames[-1].copy() for _ in range(PADDED_FRAMES - EXPECTED_FRAMES)]
    padded_masks = masks + [masks[-1].copy() for _ in range(PADDED_FRAMES - EXPECTED_FRAMES)]
    encode_rgb(output / "padded-input.mp4", padded_frames, rate, crf=18)
    encode_binary_masks(output / "padded-mask.mp4", padded_masks, rate)

    metadata = {
        "status": "prepared",
        "source": "VideoShield provided ModelScope sample",
        "protocol": "watermarked sample -> same official mask -> ProPainter and DiffuEraser",
        "frames": EXPECTED_FRAMES,
        "size": list(EXPECTED_SIZE),
        "fps": float(rate),
        "padding": {
            "reason": "DiffuEraser requires at least 22 frames",
            "method": "repeat final input frame and final mask only for inference",
            "padded_frames": PADDED_FRAMES,
            "reported_frames": EXPECTED_FRAMES,
        },
        "input_sha256": sha256(args.source_video),
        "watermark_info_sha256": sha256(args.source_watermark_info),
        "mask_foreground_pixels": [int((mask > 127).sum()) for mask in masks],
    }
    atomic_json(output / "sample.json", metadata)
    validate_video(output / "input-watermarked.mp4", frames=EXPECTED_FRAMES, size=EXPECTED_SIZE, fps=None)
    validate_video(output / "gt-mask.mp4", frames=EXPECTED_FRAMES, size=EXPECTED_SIZE, fps=float(rate))
    validate_video(output / "padded-input.mp4", frames=PADDED_FRAMES, size=EXPECTED_SIZE, fps=float(rate))
    validate_video(output / "padded-mask.mp4", frames=PADDED_FRAMES, size=EXPECTED_SIZE, fps=float(rate))
    print(f"Prepared VideoShield sample in {output}")


def finalize(args: argparse.Namespace) -> None:
    sample = json.loads(args.sample_json.read_text())
    frames, _raw_rate = decode_rgb(args.source_video)
    expected = int(sample["frames"])
    if len(frames) < expected:
        raise RuntimeError(f"Removal output has {len(frames)} frames; expected at least {expected}")
    reported = frames[:expected]
    size = tuple(int(value) for value in sample["size"])
    if any(frame.shape[:2][::-1] != size for frame in reported):
        raise RuntimeError(f"Removal output does not preserve the {size[0]}x{size[1]} sample size")
    rate = Fraction(str(sample["fps"]))
    encode_rgb(args.output_video, reported, rate, crf=18)
    write_rgb_frames(args.output_frames, reported)
    info = validate_video(args.output_video, frames=expected, size=size, fps=float(rate))
    atomic_json(
        args.output_video.with_suffix(".json"),
        {
            "status": "complete",
            "source_video": str(args.source_video),
            "trimmed_compatibility_padding": len(frames) - expected,
            "output": info,
            "sha256": sha256(args.output_video),
        },
    )
    print(f"Finalized {args.output_video}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--source-video", required=True, type=Path)
    prepare_parser.add_argument("--source-mask-frames", required=True, type=Path)
    prepare_parser.add_argument("--source-watermark-info", required=True, type=Path)
    prepare_parser.add_argument("--output", required=True, type=Path)
    prepare_parser.set_defaults(handler=prepare)

    finalize_parser = subparsers.add_parser("finalize")
    finalize_parser.add_argument("--source-video", required=True, type=Path)
    finalize_parser.add_argument("--sample-json", required=True, type=Path)
    finalize_parser.add_argument("--output-video", required=True, type=Path)
    finalize_parser.add_argument("--output-frames", required=True, type=Path)
    finalize_parser.set_defaults(handler=finalize)

    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
