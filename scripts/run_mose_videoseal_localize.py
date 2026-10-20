#!/usr/bin/env python3
"""Detect VideoSeal absence after removal and score it against the true masks."""

from __future__ import annotations

import argparse
import gc
import os
import time
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np
import scipy.ndimage
import torch
import torch.nn.functional as F
import videoseal

from mose_watermark_common import (
    FPS,
    append_jsonl,
    atomic_json,
    decode_rgb,
    encode_binary_masks,
    load_clips,
    method_dir,
    raw_mask_frames,
    read_mask_sequence,
    sha256,
    utc_now,
    validate_video,
)


def attack_mask(
    mask: np.ndarray,
    remover: str,
    size: tuple[int, int],
    processing_size: tuple[int, int] | None = None,
) -> np.ndarray:
    inference_size = processing_size or size
    resized = cv2.resize(mask, inference_size, interpolation=cv2.INTER_NEAREST) > 0
    if remover == "propainter":
        effective = scipy.ndimage.binary_dilation(resized, iterations=8)
    else:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        value = cv2.erode(resized.astype(np.uint8), kernel, iterations=1)
        effective = cv2.dilate(value, kernel, iterations=8).astype(bool)
    if inference_size != size:
        effective = cv2.resize(effective.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0
    return effective


def detect_tamper(model, frames: list[np.ndarray], chunk_size: int, grid_size: int) -> tuple[list[np.ndarray], str]:
    masks: list[np.ndarray] = []
    height, width = frames[0].shape[:2]
    probe = torch.from_numpy(frames[0]).float().permute(2, 0, 1)[None].cuda() / 255.0
    with torch.inference_mode():
        probe_logits = model.detect(probe, is_video=True)["preds"][:, 0:1]
    native_spatial = probe_logits.ndim == 4
    del probe, probe_logits

    if not native_spatial:
        y_bounds = np.linspace(0, height, grid_size + 1).round().astype(int)
        x_bounds = np.linspace(0, width, grid_size + 1).round().astype(int)
        for start in range(0, len(frames), chunk_size):
            batch_frames = frames[start : start + chunk_size]
            crops: list[np.ndarray] = []
            for frame in batch_frames:
                for y_index in range(grid_size):
                    for x_index in range(grid_size):
                        crop = frame[y_bounds[y_index] : y_bounds[y_index + 1], x_bounds[x_index] : x_bounds[x_index + 1]]
                        crops.append(cv2.resize(crop, (256, 256), interpolation=cv2.INTER_AREA))
            tensor = torch.from_numpy(np.stack(crops)).float().permute(0, 3, 1, 2).cuda() / 255.0
            with torch.inference_mode():
                logits = model.detect(tensor, is_video=False)["preds"][:, 0]
                absent = (logits <= 0.0).byte().cpu().numpy()
            offset = 0
            for _frame in batch_frames:
                mask = np.zeros((height, width), dtype=np.uint8)
                for y_index in range(grid_size):
                    for x_index in range(grid_size):
                        if absent[offset]:
                            mask[y_bounds[y_index] : y_bounds[y_index + 1], x_bounds[x_index] : x_bounds[x_index + 1]] = 255
                        offset += 1
                masks.append(mask)
            del tensor, logits
        return masks, f"derived_{grid_size}x{grid_size}_patch_grid_from_global_presence_detector"

    for start in range(0, len(frames), chunk_size):
        arrays = np.stack(frames[start : start + chunk_size])
        tensor = torch.from_numpy(arrays).float().permute(0, 3, 1, 2).cuda() / 255.0
        with torch.inference_mode():
            logits = model.detect(tensor, is_video=True)["preds"][:, 0:1]
            # The first VideoSeal channel is a watermark-presence logit. The
            # object-removal prediction is therefore its thresholded absence.
            absence = logits <= 0.0
            absence = F.interpolate(absence.float(), size=(height, width), mode="nearest") > 0.5
        masks.extend((absence[:, 0].byte().cpu().numpy() * 255).astype(np.uint8))
        del tensor, logits, absence
    return masks, "native_pixelwise_presence_detector"


def iou_metrics(predicted: list[np.ndarray], expected: list[np.ndarray]) -> dict:
    if len(predicted) != len(expected):
        raise RuntimeError("Prediction and GT mask frame counts differ")
    intersections: list[int] = []
    unions: list[int] = []
    per_frame: list[float] = []
    for pred, gt in zip(predicted, expected):
        pred_bool = pred > 0
        gt_bool = gt > 0
        intersection = int(np.logical_and(pred_bool, gt_bool).sum())
        union = int(np.logical_or(pred_bool, gt_bool).sum())
        intersections.append(intersection)
        unions.append(union)
        per_frame.append(1.0 if union == 0 else intersection / union)
    total_union = sum(unions)
    return {
        "global_iou": 1.0 if total_union == 0 else sum(intersections) / total_union,
        "mean_frame_iou": float(np.mean(per_frame)),
        "minimum_frame_iou": float(np.min(per_frame)),
        "maximum_frame_iou": float(np.max(per_frame)),
        "per_frame_iou": per_frame,
        "intersection_pixels": sum(intersections),
        "union_pixels": total_union,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--grid-size", type=int, default=4)
    parser.add_argument("--only", action="append")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("VideoSeal localization requires CUDA")
    clips = load_clips(args.selection)
    if args.only:
        requested = set(args.only)
        clips = [clip for clip in clips if clip.clip_id in requested]
        if {clip.clip_id for clip in clips} != requested:
            raise RuntimeError(f"Unknown --only clip(s): {sorted(requested - {clip.clip_id for clip in clips})}")
    previous = Path.cwd()
    os.chdir(args.source.resolve())
    try:
        model = videoseal.load("videoseal")
    finally:
        os.chdir(previous)
    model.eval().cuda()
    events = args.output_root / "videoseal-localization-events.jsonl"
    failures = 0

    for ordinal, clip in enumerate(clips, 1):
        destination = method_dir(args.output_root, clip.clip_id, "videoseal")
        summary_path = destination / "localization.json"
        try:
            if summary_path.is_file() and not args.force:
                existing = __import__("json").loads(summary_path.read_text())
                for remover in ("propainter", "diffueraser"):
                    validate_video(destination / f"pred-mask-{remover}.mp4", frames=clip.frame_count)
                    if existing[remover]["global_iou"] is None:
                        raise RuntimeError("Stored localization has no IoU")
                print(f"[{ordinal}/50] skip VideoSeal localization {clip.clip_id}", flush=True)
                continue

            masks_raw = read_mask_sequence(raw_mask_frames(args.dataset_root, clip.clip_id))
            if len(masks_raw) != clip.frame_count:
                raise RuntimeError(f"Unexpected raw mask count for {clip.clip_id}")
            result: dict[str, object] = {
                "status": "complete",
                "method": "videoseal",
                "clip_id": clip.clip_id,
                "detector_semantics": "preds[:,0] > 0 is watermark present; reported mask is preds[:,0] <= 0",
                "threshold": 0.0,
                "prediction_resize": "nearest from detector resolution to attack-video resolution",
                "started_at": utc_now(),
            }
            started = time.monotonic()
            for remover in ("propainter", "diffueraser"):
                attack_path = destination / f"{remover}.mp4"
                info = validate_video(attack_path, frames=clip.frame_count, fps=FPS)
                size = (int(info["width"]), int(info["height"]))
                remover_metadata = __import__("json").loads((destination / f"{remover}.json").read_text())
                adapter = remover_metadata.get("processing_adapter")
                processing_size = tuple(adapter["processing_size"]) if adapter else None
                frames, _rate = decode_rgb(attack_path)
                predicted, localization_mode = detect_tamper(model, frames, args.chunk_size, args.grid_size)
                expected = [attack_mask(mask, remover, size, processing_size) for mask in masks_raw]
                pred_path = destination / f"pred-mask-{remover}.mp4"
                gt_path = destination / f"effective-gt-mask-{remover}.mp4"
                encode_binary_masks(pred_path, predicted, Fraction(FPS, 1))
                encode_binary_masks(gt_path, expected, Fraction(FPS, 1))
                metrics = iou_metrics(predicted, expected)
                metrics.update({
                    "pred_mask_sha256": sha256(pred_path),
                    "effective_gt_mask_sha256": sha256(gt_path),
                    "frames": clip.frame_count,
                    "width": size[0],
                    "height": size[1],
                    "effective_gt": (
                        "scipy binary_dilation, 8 iterations"
                        if remover == "propainter"
                        else "OpenCV 3x3 rectangle erosion once, then dilation 8 iterations"
                    ),
                    "localization_mode": localization_mode,
                    "processing_adapter": adapter,
                })
                result[remover] = metrics
            result["elapsed_seconds"] = round(time.monotonic() - started, 3)
            result["completed_at"] = utc_now()
            atomic_json(summary_path, result)
            append_jsonl(events, result)
            print(
                f"[{ordinal}/50] complete VideoSeal localization {clip.clip_id}: "
                f"P={result['propainter']['global_iou']:.4f} D={result['diffueraser']['global_iou']:.4f}",
                flush=True,
            )
        except Exception as error:
            failures += 1
            append_jsonl(events, {"status": "failed", "clip_id": clip.clip_id, "error": repr(error), "timestamp": utc_now()})
            print(f"[{ordinal}/50] FAILED VideoSeal localization {clip.clip_id}: {error!r}", flush=True)
        finally:
            gc.collect()
            torch.cuda.empty_cache()
    if failures:
        raise SystemExit(f"VideoSeal localization failed for {failures} clips")


if __name__ == "__main__":
    main()
