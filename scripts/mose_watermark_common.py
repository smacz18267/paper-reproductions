#!/usr/bin/env python3
"""Shared paths, validation, and atomic I/O for the MOSE watermark batch."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable

import av
import cv2
import numpy as np


FPS = 25
EXPECTED_CLIPS = 50


@dataclass(frozen=True)
class Clip:
    clip_id: str
    frame_count: int


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_clips(selection: Path) -> list[Clip]:
    payload = json.loads(selection.read_text())
    clips = [Clip(item["video_id"], int(item["frame_count"])) for item in payload.get("clips", [])]
    if len(clips) != EXPECTED_CLIPS or len({clip.clip_id for clip in clips}) != EXPECTED_CLIPS:
        raise RuntimeError(f"Expected {EXPECTED_CLIPS} unique clips in {selection}, found {len(clips)}")
    return clips


def method_dir(output_root: Path, clip_id: str, method: str) -> Path:
    return output_root / "clips" / clip_id / "methods" / method


def clip_dir(output_root: Path, clip_id: str) -> Path:
    return output_root / "clips" / clip_id


def original_video(dataset_root: Path, clip_id: str) -> Path:
    return dataset_root / "batch_inputs" / clip_id / "video.mp4"


def raw_mask_frames(dataset_root: Path, clip_id: str) -> Path:
    return dataset_root / "binary_masks" / clip_id


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_atomic(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".part")
    shutil.copy2(source, temporary)
    temporary.replace(destination)


def video_info(path: Path) -> dict[str, int | float]:
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"Missing or empty video: {path}")
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        rate = stream.average_rate or stream.guessed_rate
        frames = sum(1 for _ in container.decode(video=0))
        return {
            "frames": frames,
            "width": int(stream.width),
            "height": int(stream.height),
            "fps": float(rate) if rate is not None else 0.0,
            "bytes": path.stat().st_size,
        }


def validate_video(
    path: Path,
    *,
    frames: int | None = None,
    size: tuple[int, int] | None = None,
    fps: float | None = FPS,
) -> dict[str, int | float]:
    info = video_info(path)
    if frames is not None and info["frames"] != frames:
        raise RuntimeError(f"{path}: expected {frames} frames, got {info['frames']}")
    if size is not None and (info["width"], info["height"]) != size:
        raise RuntimeError(f"{path}: expected {size}, got {(info['width'], info['height'])}")
    if fps is not None and abs(float(info["fps"]) - fps) > 0.01:
        raise RuntimeError(f"{path}: expected {fps} FPS, got {info['fps']}")
    return info


def decode_rgb(path: Path) -> tuple[list[np.ndarray], Fraction]:
    frames: list[np.ndarray] = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        rate = stream.average_rate or stream.guessed_rate or Fraction(FPS, 1)
        for frame in container.decode(video=0):
            frames.append(frame.to_ndarray(format="rgb24"))
    if not frames:
        raise RuntimeError(f"No frames decoded from {path}")
    return frames, Fraction(rate.numerator, rate.denominator)


def encode_rgb(path: Path, frames: Iterable[np.ndarray], rate: Fraction, *, crf: int = 18) -> int:
    arrays = list(frames)
    if not arrays:
        raise RuntimeError("Cannot encode an empty frame sequence")
    height, width = arrays[0].shape[:2]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".part.mp4")
    if temporary.exists():
        temporary.unlink()
    output = av.open(str(temporary), mode="w")
    stream = output.add_stream("libx264", rate=rate)
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": str(crf), "preset": "medium"}
    try:
        for index, array in enumerate(arrays):
            if array.shape[:2] != (height, width):
                raise RuntimeError("Frame sizes changed within one video")
            frame = av.VideoFrame.from_ndarray(array, format="rgb24")
            frame.pts = index
            frame.time_base = Fraction(rate.denominator, rate.numerator)
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
    finally:
        output.close()
    temporary.replace(path)
    return len(arrays)


def encode_binary_masks(path: Path, masks: Iterable[np.ndarray], rate: Fraction = Fraction(FPS, 1)) -> int:
    rgb = []
    for mask in masks:
        # Effective attack masks are boolean arrays while detector masks are
        # uint8 arrays. Treat every non-zero value as foreground so both are
        # encoded faithfully as strict black/white videos.
        binary = np.where(mask != 0, 255, 0).astype(np.uint8)
        rgb.append(cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB))
    return encode_rgb(path, rgb, rate, crf=0)


def read_mask_sequence(directory: Path, indices: list[int] | None = None) -> list[np.ndarray]:
    paths = sorted(directory.glob("*.png"))
    if indices is not None:
        paths = [paths[index] for index in indices]
    masks = []
    for path in paths:
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Could not read mask {path}")
        masks.append(mask)
    if not masks:
        raise RuntimeError(f"No masks found in {directory}")
    return masks
