#!/usr/bin/env python3
"""Run VideoShield detection on Task 1 outputs as an explicit negative control.

VideoShield embeds a watermark in a diffusion model's initial Gaussian noise and
cannot post-process an already-rendered video. Task 1 outputs were not generated
from VideoShield-marked noise. This runner therefore applies the official
encoder, DDIM inversion, decryption, and recovery metric to those outputs using
one fixed reference key. The expected accuracy is chance-level; it must not be
described as watermark embedding or robustness.
"""

from __future__ import annotations

import argparse
import gc
import random
import sys
import time
from pathlib import Path

import av
import cv2
import numpy as np
import torch
from diffusers import TextToVideoSDPipeline
from diffusers.schedulers import DDIMInverseScheduler

from task3_common import (
    append_jsonl,
    discover_inputs,
    output_directory,
    select_inputs,
    sha256,
    utc_now,
    write_json,
)


TARGET_FRAMES = 16
TARGET_HEIGHT = 256
TARGET_WIDTH = 256


def load_uniform_frames(path: Path) -> tuple[list[np.ndarray], list[int], float]:
    container = av.open(str(path))
    stream = container.streams.video[0]
    rate = stream.average_rate or stream.guessed_rate
    frames: list[np.ndarray] = []
    try:
        for frame in container.decode(video=0):
            rgb = frame.to_ndarray(format="rgb24")
            frames.append(cv2.resize(rgb, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_AREA))
    finally:
        container.close()
    if not frames:
        raise RuntimeError(f"No frames decoded from {path}")
    indices = np.linspace(0, len(frames) - 1, TARGET_FRAMES).round().astype(int).tolist()
    selected = [frames[index] for index in indices]
    normalized = [frame.astype(np.float32) / 255.0 for frame in selected]
    return normalized, indices, float(rate) if rate is not None else 25.0


def process_one(pipe, inverse_scheduler, VideoShield, get_video_latents, transform_video, reference: dict, video, output_root: Path, steps: int, seed: int) -> dict:
    destination = output_directory(output_root, "videoshield", video)
    destination.mkdir(parents=True, exist_ok=True)
    frames, indices, source_fps = load_uniform_frames(video.path)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    watermark = VideoShield(
        ch_factor=1,
        hw_factor=4,
        frame_factor=8,
        height=TARGET_HEIGHT // 8,
        width=TARGET_WIDTH // 8,
        num_frames=TARGET_FRAMES,
        device="cuda:0",
    )
    watermark.m = reference["m"].to("cuda:0")
    watermark.watermark = reference["watermark"].to("cuda:0")
    watermark.key = reference["key"]
    watermark.nonce = reference["nonce"]

    tensor = transform_video(frames).to(pipe.vae.dtype).to("cuda:0")
    with torch.inference_mode():
        video_latents = get_video_latents(pipe.vae, tensor, sample=False, permute=True)
        pipe.scheduler = inverse_scheduler
        reversed_latents = pipe(
            prompt="",
            latents=video_latents,
            num_inference_steps=steps,
            guidance_scale=1.0,
            output_type="latent",
        ).frames
        accuracy = watermark.eval_watermark(reversed_latents)
    del tensor, video_latents, reversed_latents, watermark

    metrics = {
        "status": "complete",
        "baseline": "videoshield",
        "protocol": "unwatermarked_negative_control_detection",
        "interpretation": "Task 1 videos were not generated from VideoShield-marked diffusion noise; chance-level recovery is expected and is not a robustness score.",
        "input": {"method": video.method, "clip_id": video.clip_id, "path": str(video.path)},
        "sampling": {
            "frames": TARGET_FRAMES,
            "indices": indices,
            "source_fps": source_fps,
            "resize": [TARGET_WIDTH, TARGET_HEIGHT],
        },
        "seed": seed,
        "ddim_inversion_steps": steps,
        "reference_watermark_accuracy": float(accuracy),
        "completed_at": utc_now(),
    }
    write_json(destination / "metrics.json", metrics)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--reference-watermark", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--only", action="append")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("VideoShield Task 3 batch requires CUDA")
    source = args.source.resolve()
    model_path = args.model_path.resolve()
    reference_path = args.reference_watermark.resolve()
    videos = select_inputs(discover_inputs(args.dataset_root.resolve()), args.only, args.limit)
    output_root = args.output_root.resolve()
    events = output_root / "videoshield-events.jsonl"

    sys.path.insert(0, str(source))
    from utils import get_video_latents, transform_video
    from watermark import VideoShield

    try:
        reference = torch.load(reference_path, map_location="cpu", weights_only=False)
    except TypeError:
        reference = torch.load(reference_path, map_location="cpu")
    expected = {"m", "watermark", "key", "nonce"}
    if set(reference) != expected:
        raise RuntimeError(f"Unexpected VideoShield reference fields: {sorted(reference)}")

    pipe = TextToVideoSDPipeline.from_pretrained(model_path, torch_dtype=torch.float16).to("cuda:0")
    pipe.safety_checker = None
    inverse_scheduler = DDIMInverseScheduler.from_pretrained(model_path, subfolder="scheduler")

    failures = 0
    for index, video in enumerate(videos, 1):
        metrics_path = output_directory(output_root, "videoshield", video) / "metrics.json"
        if metrics_path.is_file() and not args.force:
            print(f"[{index}/{len(videos)}] skip complete {video.key}", flush=True)
            continue
        started = time.monotonic()
        append_jsonl(events, {"status": "running", "input": video.key, "timestamp": utc_now()})
        try:
            metrics = process_one(pipe, inverse_scheduler, VideoShield, get_video_latents, transform_video, reference, video, output_root, args.steps, args.seed + index)
            elapsed = time.monotonic() - started
            append_jsonl(events, {"status": "complete", "input": video.key, "elapsed_seconds": elapsed, "accuracy": metrics["reference_watermark_accuracy"], "timestamp": utc_now()})
            print(f"[{index}/{len(videos)}] complete {video.key}: negative-control accuracy {metrics['reference_watermark_accuracy']:.4f} in {elapsed:.1f}s", flush=True)
        except Exception as error:
            failures += 1
            append_jsonl(events, {"status": "failed", "input": video.key, "error": repr(error), "timestamp": utc_now()})
            print(f"[{index}/{len(videos)}] FAILED {video.key}: {error!r}", flush=True)
        finally:
            gc.collect()
            torch.cuda.empty_cache()
    print(f"VideoShield Task 3 negative-control batch finished: {len(videos) - failures}/{len(videos)} succeeded", flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
