#!/usr/bin/env python3
"""Prepare any authentic watermarked video for the two removal methods.

The preparation is deliberately watermark-agnostic.  It preserves the source
video, converts the supplied attack mask to strict binary frames, and applies
the one compatibility adapter required by DiffuEraser: sequences shorter than
22 frames are padded by repeating the final video frame and final mask to 24
frames.  Reported outputs are always trimmed back to the authentic source
length by the ``finalize`` command.
"""

from __future__ import annotations

import argparse
import json
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
    utc_now,
    validate_video,
)


DIFFUERASER_MINIMUM_FRAMES = 22
PADDED_FRAMES = 24


def write_rgb_frames(directory: Path, frames: list[np.ndarray]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for index, frame in enumerate(frames):
        path = directory / f"{index:05d}.png"
        if not cv2.imwrite(str(path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)):
            raise RuntimeError(f"Could not write {path}")


def write_mask_frames(directory: Path, masks: list[np.ndarray]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for index, mask in enumerate(masks):
        path = directory / f"{index:05d}.png"
        binary = np.where(mask > 127, 255, 0).astype(np.uint8)
        if not cv2.imwrite(str(path), binary):
            raise RuntimeError(f"Could not write {path}")


def load_masks(path: Path) -> tuple[list[np.ndarray], Fraction | None]:
    if path.is_dir():
        return read_mask_sequence(path), None
    if path.is_file():
        frames, rate = decode_rgb(path)
        masks = [cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in frames]
        return masks, rate
    raise FileNotFoundError(f"Mask input does not exist: {path}")


def prepare(args: argparse.Namespace) -> None:
    frames, rate = decode_rgb(args.source_video)
    masks, mask_rate = load_masks(args.source_mask)
    if len(frames) != len(masks):
        raise RuntimeError(
            f"Video/mask frame mismatch: video={len(frames)}, mask={len(masks)}"
        )
    size = frames[0].shape[:2][::-1]
    if any(frame.shape[:2][::-1] != size for frame in frames):
        raise RuntimeError("Source video dimensions change between frames")
    if any(mask.shape[:2][::-1] != size for mask in masks):
        raise RuntimeError(f"Every attack mask must match the source size {size}")
    if mask_rate is not None and abs(float(mask_rate) - float(rate)) > 0.01:
        raise RuntimeError(f"Mask FPS {float(mask_rate)} differs from video FPS {float(rate)}")

    masks = [np.where(mask > 127, 255, 0).astype(np.uint8) for mask in masks]
    if not any(np.any(mask) for mask in masks):
        raise RuntimeError("The attack mask is empty in every frame")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    sample_json = output / "sample.json"
    if sample_json.exists() and not args.force:
        raise FileExistsError(f"{sample_json} already exists; pass --force to replace this run")

    copy_atomic(args.source_video.resolve(), output / "input-watermarked.mp4")
    write_rgb_frames(output / "input-frames", frames)
    write_mask_frames(output / "gt-mask-frames", masks)
    encode_binary_masks(output / "gt-mask.mp4", masks, rate)

    padded = len(frames) < DIFFUERASER_MINIMUM_FRAMES
    if padded:
        attack_frames = frames + [frames[-1].copy() for _ in range(PADDED_FRAMES - len(frames))]
        attack_masks = masks + [masks[-1].copy() for _ in range(PADDED_FRAMES - len(masks))]
        encode_rgb(output / "attack-input.mp4", attack_frames, rate, crf=18)
    else:
        attack_frames = frames
        attack_masks = masks
        copy_atomic(args.source_video.resolve(), output / "attack-input.mp4")
    encode_binary_masks(output / "attack-mask.mp4", attack_masks, rate)

    extra = {}
    if args.provenance_json is not None:
        value = json.loads(args.provenance_json.read_text())
        if not isinstance(value, dict):
            raise TypeError("--provenance-json must contain a JSON object")
        extra = value

    metadata = {
        "status": "prepared",
        "prepared_at": utc_now(),
        "watermark_method": args.watermark_method,
        "sample_id": args.sample_id,
        "sample_source": args.sample_source,
        "source_kind": args.source_kind,
        "protocol": "authentic watermarked sample -> one aligned mask -> ProPainter and DiffuEraser",
        "frames": len(frames),
        "attack_frames": len(attack_frames),
        "size": list(size),
        "fps": float(rate),
        "padding": None if not padded else {
            "reason": "DiffuEraser requires at least 22 frames",
            "method": "repeat final input frame and final mask only for inference",
            "padded_frames": PADDED_FRAMES,
            "reported_frames": len(frames),
        },
        "input_sha256": sha256(args.source_video.resolve()),
        "mask_input_sha256": sha256(args.source_mask.resolve()) if args.source_mask.is_file() else None,
        "mask_foreground_pixels": [int((mask > 127).sum()) for mask in masks],
        "provenance": extra,
    }
    atomic_json(sample_json, metadata)
    validate_video(output / "input-watermarked.mp4", frames=len(frames), size=size, fps=float(rate))
    validate_video(output / "gt-mask.mp4", frames=len(frames), size=size, fps=float(rate))
    validate_video(output / "attack-input.mp4", frames=len(attack_frames), size=size, fps=float(rate))
    validate_video(output / "attack-mask.mp4", frames=len(attack_masks), size=size, fps=float(rate))
    print(f"Prepared {args.watermark_method}/{args.sample_id} in {output}")


def finalize(args: argparse.Namespace) -> None:
    sample = json.loads(args.sample_json.read_text())
    frames, _raw_rate = decode_rgb(args.source_video)
    expected = int(sample["frames"])
    if len(frames) < expected:
        raise RuntimeError(f"Removal output has {len(frames)} frames; expected at least {expected}")
    reported = frames[:expected]
    size = tuple(int(value) for value in sample["size"])
    source_sizes = {frame.shape[:2][::-1] for frame in reported}
    if len(source_sizes) != 1:
        raise RuntimeError("Removal output dimensions change between frames")
    source_size = next(iter(source_sizes))
    resized = source_size != size
    if resized:
        source_ratio = source_size[0] / source_size[1]
        expected_ratio = size[0] / size[1]
        if abs(source_ratio - expected_ratio) > 1e-6:
            raise RuntimeError(
                f"Removal output aspect ratio {source_size[0]}x{source_size[1]} "
                f"does not match {size[0]}x{size[1]}"
            )
        # DiffuEraser's official pipeline runs its ProPainter prior with the
        # upstream default resize_ratio=0.6. Restore that prior to the source
        # dimensions for aligned evaluation, while recording the normalization.
        reported = [
            cv2.resize(frame, size, interpolation=cv2.INTER_LINEAR)
            for frame in reported
        ]
    rate = Fraction(str(sample["fps"]))
    encode_rgb(args.output_video, reported, rate, crf=18)
    if args.output_frames is not None:
        write_rgb_frames(args.output_frames, reported)
    info = validate_video(args.output_video, frames=expected, size=size, fps=float(rate))
    atomic_json(
        args.output_video.with_suffix(".json"),
        {
            "status": "complete",
            "case": args.case,
            "source_video": str(args.source_video.resolve()),
            "trimmed_compatibility_padding": len(frames) - expected,
            "dimension_normalization": {
                "source_size": list(source_size),
                "expected_size": list(size),
                "resized": resized,
                "interpolation": "cv2.INTER_LINEAR" if resized else None,
                "reason": (
                    "restore the official ProPainter prior from its default "
                    "resize_ratio=0.6 to the source dimensions"
                    if resized
                    else None
                ),
            },
            "output": info,
            "sha256": sha256(args.output_video),
            "completed_at": utc_now(),
        },
    )
    print(f"Finalized {args.case}: {args.output_video}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--source-video", required=True, type=Path)
    prepare_parser.add_argument("--source-mask", required=True, type=Path)
    prepare_parser.add_argument("--output", required=True, type=Path)
    prepare_parser.add_argument("--watermark-method", required=True)
    prepare_parser.add_argument("--sample-id", required=True)
    prepare_parser.add_argument("--sample-source", required=True)
    prepare_parser.add_argument(
        "--source-kind",
        required=True,
        choices=("authors_provided", "authors_generated", "mose_post_hoc"),
    )
    prepare_parser.add_argument("--provenance-json", type=Path)
    prepare_parser.add_argument("--force", action="store_true")
    prepare_parser.set_defaults(handler=prepare)

    finalize_parser = subparsers.add_parser("finalize")
    finalize_parser.add_argument("--source-video", required=True, type=Path)
    finalize_parser.add_argument("--sample-json", required=True, type=Path)
    finalize_parser.add_argument("--output-video", required=True, type=Path)
    finalize_parser.add_argument("--output-frames", type=Path)
    finalize_parser.add_argument("--case", required=True, choices=("propainter", "diffueraser"))
    finalize_parser.set_defaults(handler=finalize)

    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
