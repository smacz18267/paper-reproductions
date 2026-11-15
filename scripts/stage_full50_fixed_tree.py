#!/usr/bin/env python3
"""Stage the validated review archive for fixed-MOSE E2FGVI jobs.

The review archive is the source of truth.  This creates only symlinks plus a
lossless PNG expansion of each reviewed ground-truth mask, matching the input
layout expected by the original MOSE batch runners.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import av
import cv2
import numpy as np


METHODS = ("editguard", "omniguard", "videoseal", "safe-sora")


def load_object(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return value


def relative_symlink(source: Path, destination: Path) -> None:
    source = source.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    target = os.path.relpath(source, destination.parent.resolve())
    if destination.is_symlink():
        if destination.resolve() == source:
            return
        destination.unlink()
    elif destination.exists():
        raise RuntimeError(f"Refusing to replace non-symlink path: {destination}")
    destination.symlink_to(target, target_is_directory=source.is_dir())


def decode_masks(source: Path, destination: Path, expected_frames: int) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    existing = sorted(destination.glob("*.png"))
    if len(existing) == expected_frames:
        return
    if existing:
        raise RuntimeError(
            f"{destination}: found {len(existing)} existing PNG masks, expected {expected_frames}; "
            "use a fresh staging directory"
        )
    count = 0
    with av.open(str(source)) as container:
        for count, frame in enumerate(container.decode(video=0), 1):
            gray = frame.to_ndarray(format="gray")
            binary = np.where(gray > 127, 255, 0).astype(np.uint8)
            output = destination / f"{count - 1:05d}.png"
            if not cv2.imwrite(str(output), binary, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
                raise RuntimeError(f"Could not write {output}")
    if count != expected_frames:
        raise RuntimeError(f"{source}: decoded {count} mask frames, expected {expected_frames}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--review-root", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()

    selection = load_object(args.selection.resolve())
    review_root = args.review_root.resolve()
    review = load_object(review_root / "manifest.json")
    selected = selection.get("clips", [])
    samples = review.get("samples", [])
    if len(selected) != 50 or len(samples) != 50:
        raise RuntimeError(f"Expected 50 selected/reviewed samples, found {len(selected)}/{len(samples)}")
    selected_by_id = {item["video_id"]: item for item in selected}
    sample_by_id = {item["clip_id"]: item for item in samples}
    if len(selected_by_id) != 50 or set(selected_by_id) != set(sample_by_id):
        raise RuntimeError("Selection and review manifest do not contain the same 50 unique clips")

    args.dataset_root.mkdir(parents=True, exist_ok=True)
    args.output_root.mkdir(parents=True, exist_ok=True)
    for ordinal, clip_id in enumerate(selected_by_id, 1):
        selected_item = selected_by_id[clip_id]
        reviewed = sample_by_id[clip_id]
        expected_frames = int(selected_item["frame_count"])
        if int(reviewed["frame_count"]) != expected_frames:
            raise RuntimeError(f"{clip_id}: selection/review frame-count mismatch")
        media = reviewed["media"]
        original = review_root / media["original"]
        mask_video = review_root / media["gt_mask"]
        relative_symlink(original, args.dataset_root / "batch_inputs" / clip_id / "video.mp4")
        decode_masks(mask_video, args.dataset_root / "binary_masks" / clip_id, expected_frames)
        for method in METHODS:
            source = review_root / "media" / clip_id / "methods" / method
            if not source.is_dir():
                raise FileNotFoundError(f"Missing reviewed method directory: {source}")
            relative_symlink(source, args.output_root / "clips" / clip_id / "methods" / method)
        print(f"[{ordinal}/50] staged {clip_id}", flush=True)

    print(
        json.dumps(
            {
                "status": "complete",
                "samples": 50,
                "methods": list(METHODS),
                "dataset_root": str(args.dataset_root.resolve()),
                "output_root": str(args.output_root.resolve()),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
