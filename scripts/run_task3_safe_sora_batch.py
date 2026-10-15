#!/usr/bin/env python3
"""Apply Safe-Sora to 16-frame samples from all Task 1 output videos."""

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

from task3_common import (
    append_jsonl,
    discover_inputs,
    fidelity,
    output_directory,
    select_inputs,
    sha256,
    utc_now,
    write_json,
)


TARGET_HEIGHT = 320
TARGET_WIDTH = 512
TARGET_FRAMES = 16
LATENT_SCALE = 0.18215


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
    return selected, indices, float(rate) if rate is not None else 25.0


def split_composite(composite_path: Path, output_path: Path) -> tuple[np.ndarray, np.ndarray]:
    source = av.open(str(composite_path))
    stream = source.streams.video[0]
    rate = stream.average_rate or stream.guessed_rate or Fraction(10, 1)
    temporary = output_path.with_name(output_path.stem + ".part" + output_path.suffix)
    if temporary.exists():
        temporary.unlink()
    target = av.open(str(temporary), mode="w")
    try:
        try:
            target_stream = target.add_stream("libx264", rate=rate)
        except av.codec.codec.UnknownCodecError:
            target_stream = target.add_stream("mpeg4", rate=rate)
        target_stream.width = TARGET_WIDTH
        target_stream.height = TARGET_HEIGHT
        target_stream.pix_fmt = "yuv420p"
        target_stream.options = {"crf": "10", "preset": "medium"}
        covers: list[np.ndarray] = []
        marked: list[np.ndarray] = []
        for frame in source.decode(video=0):
            array = frame.to_ndarray(format="rgb24")
            if array.shape[:2] != (TARGET_HEIGHT, TARGET_WIDTH * 3):
                raise RuntimeError(f"Unexpected Safe-Sora composite shape: {array.shape}")
            covers.append(array[:, :TARGET_WIDTH].copy())
            marked_frame = array[:, TARGET_WIDTH : 2 * TARGET_WIDTH].copy()
            marked.append(marked_frame)
            encoded = av.VideoFrame.from_ndarray(marked_frame, format="rgb24")
            for packet in target_stream.encode(encoded):
                target.mux(packet)
        for packet in target_stream.encode():
            target.mux(packet)
    finally:
        target.close()
        source.close()
    if len(covers) != TARGET_FRAMES:
        raise RuntimeError(f"Expected {TARGET_FRAMES} Safe-Sora frames, got {len(covers)}")
    temporary.replace(output_path)
    return np.stack(covers), np.stack(marked)


def inspect_video(path: Path) -> dict[str, float | int]:
    container = av.open(str(path))
    stream = container.streams.video[0]
    rate = stream.average_rate or stream.guessed_rate
    frames = sum(1 for _ in container.decode(video=0))
    result = {
        "frames": frames,
        "width": stream.width,
        "height": stream.height,
        "fps": float(rate) if rate is not None else None,
        "bytes": path.stat().st_size,
    }
    container.close()
    return result


def process_one(safe_test, video, output_root: Path, watermark_path: Path, seed: int) -> dict:
    destination = output_directory(output_root, "safe-sora", video)
    work = destination / "work"
    results = work / "results"
    results.mkdir(parents=True, exist_ok=True)
    output_path = destination / "watermarked.mp4"

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    selected, indices, input_fps = load_uniform_frames(video.path)
    pixels = torch.from_numpy(np.stack(selected)).float().permute(0, 3, 1, 2).cuda()
    pixels = pixels / 127.5 - 1.0
    with torch.inference_mode():
        latents = safe_test.vae.encode(pixels).mode() * LATENT_SCALE
        video_latent = latents.unsqueeze(0).permute(0, 2, 1, 3, 4).contiguous()
    del pixels, latents

    previous = Path.cwd()
    os.chdir(work)
    try:
        safe_test.Eval(video_latent, str(watermark_path))
    finally:
        os.chdir(previous)
        del video_latent

    composite_path = results / "cover_combined.mp4"
    secret_path = results / "secret_combined.png"
    for artifact in (composite_path, secret_path):
        if not artifact.is_file() or artifact.stat().st_size == 0:
            raise RuntimeError(f"Safe-Sora did not create {artifact}")

    covers, marked = split_composite(composite_path, output_path)
    original = np.stack(selected)
    combined = cv2.cvtColor(cv2.imread(str(secret_path)), cv2.COLOR_BGR2RGB)
    if combined.shape != (256, 768, 3):
        raise RuntimeError(f"Unexpected recovered watermark composite: {combined.shape}")
    watermark_original = combined[:, :256]
    watermark_recovered = combined[:, 256:512]

    metrics = {
        "status": "complete",
        "baseline": "safe-sora",
        "protocol": "postprocess_embed_then_recover",
        "sampling": {
            "frames": TARGET_FRAMES,
            "indices": indices,
            "source_fps": input_fps,
            "resize": [TARGET_WIDTH, TARGET_HEIGHT],
        },
        "input": {"method": video.method, "clip_id": video.clip_id, "path": str(video.path)},
        "seed": seed,
        "input_to_vae_reconstruction": fidelity(original, covers),
        "watermark_embedding_fidelity": fidelity(covers, marked),
        "watermark_recovery_fidelity": fidelity(watermark_original, watermark_recovered),
        "watermarked_video": inspect_video(output_path),
        "files": {
            "watermarked.mp4": {"bytes": output_path.stat().st_size, "sha256": sha256(output_path)},
            "work/results/cover_combined.mp4": {"bytes": composite_path.stat().st_size, "sha256": sha256(composite_path)},
            "work/results/secret_combined.png": {"bytes": secret_path.stat().st_size, "sha256": sha256(secret_path)},
        },
        "completed_at": utc_now(),
    }
    write_json(destination / "metrics.json", metrics)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--watermark", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--only", action="append")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("Safe-Sora Task 3 batch requires CUDA")
    source = args.source.resolve()
    watermark = args.watermark.resolve()
    if not watermark.is_file():
        raise RuntimeError(f"Missing watermark image: {watermark}")
    videos = select_inputs(discover_inputs(args.dataset_root.resolve()), args.only, args.limit)
    output_root = args.output_root.resolve()
    events = output_root / "safe-sora-events.jsonl"

    sys.path.insert(0, str(source))
    previous = Path.cwd()
    os.chdir(source)
    try:
        import test as safe_test
    finally:
        os.chdir(previous)

    failures = 0
    for index, video in enumerate(videos, 1):
        metrics_path = output_directory(output_root, "safe-sora", video) / "metrics.json"
        if metrics_path.is_file() and not args.force:
            print(f"[{index}/{len(videos)}] skip complete {video.key}", flush=True)
            continue
        started = time.monotonic()
        append_jsonl(events, {"status": "running", "input": video.key, "timestamp": utc_now()})
        try:
            metrics = process_one(safe_test, video, output_root, watermark, args.seed + index)
            elapsed = time.monotonic() - started
            append_jsonl(events, {"status": "complete", "input": video.key, "elapsed_seconds": elapsed, "video_psnr_db": metrics["watermark_embedding_fidelity"]["psnr_db"], "watermark_psnr_db": metrics["watermark_recovery_fidelity"]["psnr_db"], "timestamp": utc_now()})
            print(f"[{index}/{len(videos)}] complete {video.key}: video {metrics['watermark_embedding_fidelity']['psnr_db']:.2f} dB, watermark {metrics['watermark_recovery_fidelity']['psnr_db']:.2f} dB in {elapsed:.1f}s", flush=True)
        except Exception as error:
            failures += 1
            append_jsonl(events, {"status": "failed", "input": video.key, "error": repr(error), "timestamp": utc_now()})
            print(f"[{index}/{len(videos)}] FAILED {video.key}: {error!r}", flush=True)
        finally:
            gc.collect()
            torch.cuda.empty_cache()
    print(f"Safe-Sora Task 3 batch finished: {len(videos) - failures}/{len(videos)} succeeded", flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
