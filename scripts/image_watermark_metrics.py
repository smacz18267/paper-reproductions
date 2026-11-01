#!/usr/bin/env python3
"""Shared metrics and mask adapters for framewise image-watermark methods."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import scipy.ndimage

from mose_watermark_common import decode_rgb


def fidelity(reference: list[np.ndarray], candidate: list[np.ndarray]) -> dict[str, float | None]:
    if len(reference) != len(candidate) or not reference:
        raise RuntimeError("Fidelity inputs must contain the same non-zero number of frames")
    squared_error = 0.0
    absolute_error = 0.0
    values = 0
    for expected, actual in zip(reference, candidate):
        if expected.shape != actual.shape:
            raise RuntimeError(f"Fidelity frame shapes differ: {expected.shape} and {actual.shape}")
        difference = expected.astype(np.float64) - actual.astype(np.float64)
        squared_error += float(np.square(difference).sum())
        absolute_error += float(np.abs(difference).sum())
        values += difference.size
    mse = squared_error / values
    return {
        "psnr_db": None if mse == 0 else float(10.0 * np.log10((255.0 ** 2) / mse)),
        "mse": mse,
        "mean_absolute_error": absolute_error / values,
    }


def mask_video(path: Path) -> list[np.ndarray]:
    frames, _rate = decode_rgb(path)
    return [cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) > 127 for frame in frames]


def processing_size(metadata_path: Path, native_size: tuple[int, int]) -> tuple[int, int]:
    payload = json.loads(metadata_path.read_text())
    adapter = payload.get("processing_adapter")
    if not adapter:
        return native_size
    return tuple(int(value) for value in adapter["processing_size"])


def effective_attack_mask(
    mask: np.ndarray,
    remover: str,
    output_size: tuple[int, int],
    inference_size: tuple[int, int],
) -> np.ndarray:
    resized = cv2.resize(mask.astype(np.uint8), inference_size, interpolation=cv2.INTER_NEAREST) > 0
    if remover == "propainter":
        effective = scipy.ndimage.binary_dilation(resized, iterations=8)
    elif remover == "diffueraser":
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        eroded = cv2.erode(resized.astype(np.uint8), kernel, iterations=1)
        effective = cv2.dilate(eroded, kernel, iterations=8).astype(bool)
    else:
        raise ValueError(f"Unknown remover: {remover}")
    if inference_size != output_size:
        effective = cv2.resize(
            effective.astype(np.uint8), output_size, interpolation=cv2.INTER_NEAREST
        ) > 0
    return effective


def localization_metrics(
    predicted: list[np.ndarray], expected: list[np.ndarray]
) -> dict[str, float | int | list[float]]:
    if len(predicted) != len(expected) or not predicted:
        raise RuntimeError("Localization inputs must contain the same non-zero number of frames")
    intersection = 0
    union = 0
    per_frame: list[float] = []
    for pred, truth in zip(predicted, expected):
        if pred.shape != truth.shape:
            raise RuntimeError(f"Localization shapes differ: {pred.shape} and {truth.shape}")
        frame_intersection = int(np.logical_and(pred, truth).sum())
        frame_union = int(np.logical_or(pred, truth).sum())
        intersection += frame_intersection
        union += frame_union
        per_frame.append(1.0 if frame_union == 0 else frame_intersection / frame_union)
    return {
        "global_iou": 1.0 if union == 0 else intersection / union,
        "mean_frame_iou": float(np.mean(per_frame)),
        "minimum_frame_iou": float(np.min(per_frame)),
        "maximum_frame_iou": float(np.max(per_frame)),
        "intersection_pixels": intersection,
        "union_pixels": union,
        "per_frame_iou": per_frame,
    }
