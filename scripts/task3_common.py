#!/usr/bin/env python3
"""Shared utilities for the Task 3 watermark-baseline runs."""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class InputVideo:
    method: str
    clip_id: str
    path: Path

    @property
    def key(self) -> str:
        return f"{self.method}/{self.clip_id}"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def discover_inputs(dataset_root: Path) -> list[InputVideo]:
    patterns = {
        "propainter": ("inpaint_out.mp4", dataset_root / "results50" / "propainter"),
        "diffueraser": ("diffueraser_result.mp4", dataset_root / "results50" / "diffueraser"),
    }
    videos: list[InputVideo] = []
    for method, (filename, root) in patterns.items():
        for path in sorted(root.glob(f"*/{filename}")):
            videos.append(InputVideo(method=method, clip_id=path.parent.name, path=path.resolve()))
    counts = {method: sum(video.method == method for video in videos) for method in patterns}
    if counts != {"propainter": 50, "diffueraser": 50}:
        raise RuntimeError(f"Expected 50 inputs per method, found {counts}")
    return sorted(videos, key=lambda video: (video.method, video.clip_id))


def select_inputs(
    videos: Iterable[InputVideo],
    only: list[str] | None,
    limit: int | None,
) -> list[InputVideo]:
    selected = list(videos)
    if only:
        requested = set(only)
        selected = [video for video in selected if video.key in requested]
        missing = requested - {video.key for video in selected}
        if missing:
            raise RuntimeError(f"Unknown --only inputs: {sorted(missing)}")
    if limit is not None:
        selected = selected[:limit]
    if not selected:
        raise RuntimeError("No Task 3 inputs selected")
    return selected


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def append_jsonl(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(value, sort_keys=True) + "\n")
        output.flush()
        os.fsync(output.fileno())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fidelity(first: np.ndarray, second: np.ndarray) -> dict[str, float]:
    if first.shape != second.shape:
        raise RuntimeError(f"Fidelity inputs differ: {first.shape} != {second.shape}")
    error = first.astype(np.float64) - second.astype(np.float64)
    mse = float(np.mean(error * error))
    return {
        "mse": mse,
        "psnr_db": math.inf if mse == 0 else 10 * math.log10((255.0 * 255.0) / mse),
        "mean_absolute_error": float(np.mean(np.abs(error))),
    }


def output_directory(root: Path, baseline: str, video: InputVideo) -> Path:
    return root / baseline / video.method / video.clip_id

