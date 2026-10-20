#!/usr/bin/env python3
"""Embed Safe-Sora into the official 16-frame sample of each MOSE clip."""

from __future__ import annotations

import argparse
import gc
import os
import random
import sys
import time
from fractions import Fraction
from pathlib import Path

import av
import cv2
import numpy as np
import torch

from mose_watermark_common import (
    FPS,
    append_jsonl,
    atomic_json,
    encode_binary_masks,
    encode_rgb,
    load_clips,
    method_dir,
    original_video,
    raw_mask_frames,
    read_mask_sequence,
    sha256,
    utc_now,
    validate_video,
)


TARGET_HEIGHT = 320
TARGET_WIDTH = 512
TARGET_FRAMES = 16
LATENT_SCALE = 0.18215


def sample_video(path: Path) -> tuple[list[np.ndarray], list[int], float]:
    frames: list[np.ndarray] = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        rate = stream.average_rate or stream.guessed_rate
        for frame in container.decode(video=0):
            rgb = frame.to_ndarray(format="rgb24")
            frames.append(cv2.resize(rgb, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_AREA))
    if not frames:
        raise RuntimeError(f"No frames decoded from {path}")
    indices = np.linspace(0, len(frames) - 1, TARGET_FRAMES).round().astype(int).tolist()
    return [frames[index] for index in indices], indices, float(rate) if rate is not None else FPS


def read_safe_sora_result(composite: Path) -> tuple[list[np.ndarray], list[np.ndarray]]:
    covers: list[np.ndarray] = []
    marked: list[np.ndarray] = []
    with av.open(str(composite)) as container:
        for frame in container.decode(video=0):
            array = frame.to_ndarray(format="rgb24")
            if array.shape[:2] != (TARGET_HEIGHT, TARGET_WIDTH * 3):
                raise RuntimeError(f"Unexpected Safe-Sora composite shape: {array.shape}")
            covers.append(array[:, :TARGET_WIDTH].copy())
            marked.append(array[:, TARGET_WIDTH : 2 * TARGET_WIDTH].copy())
    if len(marked) != TARGET_FRAMES:
        raise RuntimeError(f"Expected {TARGET_FRAMES} Safe-Sora result frames, got {len(marked)}")
    return covers, marked


def fidelity(first: np.ndarray, second: np.ndarray) -> dict[str, float]:
    if first.shape != second.shape:
        raise RuntimeError(f"Fidelity shapes differ: {first.shape} vs {second.shape}")
    error = first.astype(np.float64) - second.astype(np.float64)
    mse = float(np.mean(error * error))
    return {
        "mse": mse,
        "psnr_db": float("inf") if mse == 0 else float(10 * np.log10((255.0 * 255.0) / mse)),
        "mean_absolute_error": float(np.mean(np.abs(error))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--watermark", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--only", action="append")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("Safe-Sora embedding requires CUDA")
    clips = load_clips(args.selection)
    if args.only:
        requested = set(args.only)
        clips = [clip for clip in clips if clip.clip_id in requested]
        if {clip.clip_id for clip in clips} != requested:
            raise RuntimeError(f"Unknown --only clip(s): {sorted(requested - {clip.clip_id for clip in clips})}")
    source = args.source.resolve()
    watermark = args.watermark.resolve()
    if not watermark.is_file():
        raise RuntimeError(f"Missing watermark image: {watermark}")

    sys.path.insert(0, str(source))
    previous = Path.cwd()
    os.chdir(source)
    try:
        import test as safe_test
    finally:
        os.chdir(previous)

    events = args.output_root / "safe-sora-embed-events.jsonl"
    failures = 0
    for ordinal, clip in enumerate(clips, 1):
        destination = method_dir(args.output_root, clip.clip_id, "safe-sora")
        marked_path = destination / "watermarked.mp4"
        original_path = destination / "sampled-original.mp4"
        gt_path = destination / "gt-mask.mp4"
        metrics_path = destination / "embed.json"
        try:
            if metrics_path.is_file() and marked_path.is_file() and gt_path.is_file() and not args.force:
                validate_video(marked_path, frames=TARGET_FRAMES, size=(TARGET_WIDTH, TARGET_HEIGHT))
                validate_video(gt_path, frames=TARGET_FRAMES, size=(TARGET_WIDTH, TARGET_HEIGHT))
                print(f"[{ordinal}/50] skip Safe-Sora embed {clip.clip_id}", flush=True)
                continue

            started = time.monotonic()
            seed = args.seed + ordinal
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            selected, indices, source_fps = sample_video(original_video(args.dataset_root, clip.clip_id))
            pixels = torch.from_numpy(np.stack(selected)).float().permute(0, 3, 1, 2).cuda()
            pixels = pixels / 127.5 - 1.0
            with torch.inference_mode():
                latents = safe_test.vae.encode(pixels).mode() * LATENT_SCALE
                video_latent = latents.unsqueeze(0).permute(0, 2, 1, 3, 4).contiguous()
            del pixels, latents

            work = destination / "work"
            (work / "results").mkdir(parents=True, exist_ok=True)
            os.chdir(work)
            try:
                safe_test.Eval(video_latent, str(watermark))
            finally:
                os.chdir(previous)
                del video_latent
            composite = work / "results" / "cover_combined.mp4"
            recovered = work / "results" / "secret_combined.png"
            if not composite.is_file() or not recovered.is_file():
                raise RuntimeError("Safe-Sora did not create its expected result files")
            covers, marked = read_safe_sora_result(composite)
            encode_rgb(original_path, selected, Fraction(FPS, 1), crf=18)
            encode_rgb(marked_path, marked, Fraction(FPS, 1), crf=18)

            masks = read_mask_sequence(raw_mask_frames(args.dataset_root, clip.clip_id), indices)
            masks = [cv2.resize(mask, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_NEAREST) for mask in masks]
            encode_binary_masks(gt_path, masks)
            marked_info = validate_video(marked_path, frames=TARGET_FRAMES, size=(TARGET_WIDTH, TARGET_HEIGHT))
            metrics = {
                "status": "complete",
                "method": "safe-sora",
                "protocol": "native_16_frame_postprocess_original_then_inpaint",
                "clip_id": clip.clip_id,
                "seed": seed,
                "sampling": {
                    "frames": TARGET_FRAMES,
                    "indices": indices,
                    "source_fps": source_fps,
                    "output_fps": FPS,
                    "resize": [TARGET_WIDTH, TARGET_HEIGHT],
                },
                "input_to_vae_reconstruction": fidelity(np.stack(selected), np.stack(covers)),
                "watermark_embedding_fidelity": fidelity(np.stack(covers), np.stack(marked)),
                "watermarked": marked_info,
                "files": {
                    "watermarked.mp4": sha256(marked_path),
                    "sampled-original.mp4": sha256(original_path),
                    "gt-mask.mp4": sha256(gt_path),
                    "recovery-composite.png": sha256(recovered),
                },
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "completed_at": utc_now(),
            }
            atomic_json(metrics_path, metrics)
            append_jsonl(events, metrics)
            print(f"[{ordinal}/50] complete Safe-Sora embed {clip.clip_id}", flush=True)
        except Exception as error:
            failures += 1
            append_jsonl(events, {"status": "failed", "clip_id": clip.clip_id, "error": repr(error), "timestamp": utc_now()})
            print(f"[{ordinal}/50] FAILED Safe-Sora embed {clip.clip_id}: {error!r}", flush=True)
        finally:
            gc.collect()
            torch.cuda.empty_cache()
    if failures:
        raise SystemExit(f"Safe-Sora embedding failed for {failures} clips")


if __name__ == "__main__":
    main()
