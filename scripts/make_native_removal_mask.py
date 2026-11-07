#!/usr/bin/env python3
"""Create a recorded attack mask for an authors' native watermarked sample.

Use ``resize-sequence`` when a genuinely aligned, published mask exists (for
example the VideoShield red-panda mask reused on the same ModelScope prompt).
Use ``center-box`` only when the watermark authors released no object mask; the
result is then explicitly a deterministic region-inpainting protocol, not an
author-provided object annotation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from mose_watermark_common import atomic_json, decode_rgb, encode_binary_masks, read_mask_sequence, sha256, utc_now


def write_frames(directory: Path, masks: list[np.ndarray]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for index, mask in enumerate(masks):
        if not cv2.imwrite(str(directory / f"{index:05d}.png"), np.where(mask, 255, 0).astype(np.uint8)):
            raise RuntimeError("Could not write mask frame")


def save(args: argparse.Namespace, masks: list[np.ndarray], rate, metadata: dict) -> None:
    args.output_video.parent.mkdir(parents=True, exist_ok=True)
    encode_binary_masks(args.output_video, masks, rate)
    write_frames(args.output_frames, masks)
    atomic_json(
        args.output_video.with_suffix(".json"),
        {
            "status": "complete",
            "video": str(args.video.resolve()),
            "video_sha256": sha256(args.video.resolve()),
            "mask_sha256": sha256(args.output_video),
            "frames": len(masks),
            "foreground_pixels": [int(mask.sum()) for mask in masks],
            "created_at": utc_now(),
            **metadata,
        },
    )


def center_box(args: argparse.Namespace) -> None:
    frames, rate = decode_rgb(args.video)
    height, width = frames[0].shape[:2]
    box_width = max(1, round(width * args.width_fraction))
    box_height = max(1, round(height * args.height_fraction))
    left = (width - box_width) // 2
    top = (height - box_height) // 2
    mask = np.zeros((height, width), dtype=bool)
    mask[top : top + box_height, left : left + box_width] = True
    masks = [mask.copy() for _ in frames]
    save(
        args,
        masks,
        rate,
        {
            "provenance": "experiment-defined deterministic region mask; not supplied by the watermark authors",
            "protocol": "static centered rectangle",
            "box": {"left": left, "top": top, "width": box_width, "height": box_height},
        },
    )


def resize_sequence(args: argparse.Namespace) -> None:
    frames, rate = decode_rgb(args.video)
    height, width = frames[0].shape[:2]
    if args.source_mask.is_dir():
        sources = read_mask_sequence(args.source_mask)
        source_hash = None
    else:
        source_frames, _source_rate = decode_rgb(args.source_mask)
        sources = [cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in source_frames]
        source_hash = sha256(args.source_mask.resolve())
    if len(sources) != len(frames):
        raise RuntimeError(
            f"Published mask/video frame mismatch: mask={len(sources)}, video={len(frames)}"
        )
    masks = [
        cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST) > 127 for mask in sources
    ]
    save(
        args,
        masks,
        rate,
        {
            "provenance": args.provenance,
            "protocol": "nearest-neighbor spatial resize; frame indices unchanged",
            "source_mask": str(args.source_mask.resolve()),
            "source_mask_sha256": source_hash,
            "source_size": list(sources[0].shape[:2][::-1]),
            "output_size": [width, height],
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--video", required=True, type=Path)
    common.add_argument("--output-video", required=True, type=Path)
    common.add_argument("--output-frames", required=True, type=Path)

    box = subparsers.add_parser("center-box", parents=[common])
    box.add_argument("--width-fraction", type=float, default=0.35)
    box.add_argument("--height-fraction", type=float, default=0.35)
    box.set_defaults(handler=center_box)

    resize = subparsers.add_parser("resize-sequence", parents=[common])
    resize.add_argument("--source-mask", required=True, type=Path)
    resize.add_argument("--provenance", required=True)
    resize.set_defaults(handler=resize_sequence)

    args = parser.parse_args()
    if hasattr(args, "width_fraction"):
        if not 0 < args.width_fraction <= 1 or not 0 < args.height_fraction <= 1:
            raise ValueError("Box fractions must be in (0, 1]")
    args.handler(args)


if __name__ == "__main__":
    main()
