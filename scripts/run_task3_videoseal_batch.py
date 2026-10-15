#!/usr/bin/env python3
"""Apply VideoSeal to all ProPainter and DiffuEraser Task 1 outputs."""

from __future__ import annotations

import argparse
import os
import random
import time
from fractions import Fraction
from itertools import zip_longest
from pathlib import Path

import av
import numpy as np
import torch

import videoseal
from videoseal.evals.metrics import bit_accuracy

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


def frame_rate(stream: av.video.stream.VideoStream) -> Fraction:
    rate = stream.average_rate or stream.guessed_rate
    if rate is None:
        raise RuntimeError("Input video does not declare a frame rate")
    return Fraction(rate.numerator, rate.denominator)


def embed_chunk(model, frames: list[np.ndarray], message: torch.Tensor) -> list[np.ndarray]:
    tensor = torch.from_numpy(np.stack(frames)).float().permute(0, 3, 1, 2) / 255.0
    with torch.inference_mode():
        marked = model.embed(
            tensor,
            msgs=message,
            is_video=True,
            lowres_attenuation=True,
        )["imgs_w"].clamp(0, 1)
    return list((marked * 255).byte().permute(0, 2, 3, 1).cpu().numpy())


def detect_chunk(model, frames: list[np.ndarray]) -> torch.Tensor:
    tensor = torch.from_numpy(np.stack(frames)).float().permute(0, 3, 1, 2) / 255.0
    with torch.inference_mode():
        return model.detect(tensor, is_video=True)["preds"][:, 1:].detach().cpu()


def encode_frames(stream, container, frames: list[np.ndarray]) -> None:
    for array in frames:
        frame = av.VideoFrame.from_ndarray(array, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)


def inspect_video(path: Path) -> dict[str, float | int]:
    container = av.open(str(path))
    stream = container.streams.video[0]
    rate = frame_rate(stream)
    frames = sum(1 for _ in container.decode(video=0))
    result = {
        "frames": frames,
        "width": stream.width,
        "height": stream.height,
        "fps": float(rate),
        "bytes": path.stat().st_size,
    }
    container.close()
    return result


def decoded_fidelity(first_path: Path, second_path: Path) -> dict[str, float]:
    first = av.open(str(first_path))
    second = av.open(str(second_path))
    sum_squared = 0.0
    sum_absolute = 0.0
    values = 0
    try:
        pairs = zip_longest(first.decode(video=0), second.decode(video=0))
        for left, right in pairs:
            if left is None or right is None:
                raise RuntimeError("Input and output frame counts differ")
            left_array = left.to_ndarray(format="rgb24")
            right_array = right.to_ndarray(format="rgb24")
            if left_array.shape != right_array.shape:
                raise RuntimeError("Input and output frame dimensions differ")
            error = left_array.astype(np.float64) - right_array.astype(np.float64)
            sum_squared += float(np.sum(error * error))
            sum_absolute += float(np.sum(np.abs(error)))
            values += error.size
    finally:
        first.close()
        second.close()
    if values == 0:
        raise RuntimeError("No frames decoded for fidelity measurement")
    mse = sum_squared / values
    return {
        "mse": mse,
        "psnr_db": float("inf") if mse == 0 else 10 * np.log10((255.0 * 255.0) / mse),
        "mean_absolute_error": sum_absolute / values,
    }


def process_one(model, video, output_root: Path, chunk_size: int, seed: int) -> dict:
    destination = output_directory(output_root, "videoseal", video)
    destination.mkdir(parents=True, exist_ok=True)
    output_path = destination / "watermarked.mp4"
    temporary_path = destination / "watermarked.part.mp4"
    metrics_path = destination / "metrics.json"
    message_path = destination / "message.txt"

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    message = model.get_random_msg().to("cuda")

    input_container = av.open(str(video.path))
    input_stream = input_container.streams.video[0]
    rate = frame_rate(input_stream)
    width, height = input_stream.width, input_stream.height
    if width % 2 or height % 2:
        raise RuntimeError(f"VideoSeal requires even dimensions, got {width}x{height}")

    if temporary_path.exists():
        temporary_path.unlink()
    output_container = av.open(str(temporary_path), mode="w")
    try:
        try:
            output_stream = output_container.add_stream("libx264", rate=rate)
        except av.codec.codec.UnknownCodecError:
            output_stream = output_container.add_stream("mpeg4", rate=rate)
        output_stream.width = width
        output_stream.height = height
        output_stream.pix_fmt = "yuv420p"
        output_stream.options = {"crf": "18", "preset": "medium"}

        chunk: list[np.ndarray] = []
        input_frames = 0
        for decoded in input_container.decode(video=0):
            chunk.append(decoded.to_ndarray(format="rgb24"))
            input_frames += 1
            if len(chunk) == chunk_size:
                encode_frames(output_stream, output_container, embed_chunk(model, chunk, message))
                chunk.clear()
        if chunk:
            encode_frames(output_stream, output_container, embed_chunk(model, chunk, message))
        for packet in output_stream.encode():
            output_container.mux(packet)
    finally:
        output_container.close()
        input_container.close()
    temporary_path.replace(output_path)

    predictions: list[torch.Tensor] = []
    marked_container = av.open(str(output_path))
    chunk = []
    output_frames = 0
    try:
        for decoded in marked_container.decode(video=0):
            chunk.append(decoded.to_ndarray(format="rgb24"))
            output_frames += 1
            if len(chunk) == chunk_size:
                predictions.append(detect_chunk(model, chunk))
                chunk.clear()
        if chunk:
            predictions.append(detect_chunk(model, chunk))
    finally:
        marked_container.close()

    if input_frames != output_frames:
        raise RuntimeError(f"Frame count changed: {input_frames} -> {output_frames}")
    soft_message = torch.cat(predictions, dim=0).mean(dim=0, keepdim=True)
    accuracy = float(bit_accuracy(soft_message, message.detach().cpu()).item())
    bits = "".join(str(int(value)) for value in message[0].detach().cpu())
    message_path.write_text(bits + "\n")

    metrics = {
        "status": "complete",
        "baseline": "videoseal_1.0",
        "protocol": "postprocess_embed_then_detect",
        "input": {"method": video.method, "clip_id": video.clip_id, "path": str(video.path)},
        "seed": seed,
        "message_bits": len(bits),
        "bit_accuracy": accuracy,
        "input_video": inspect_video(video.path),
        "watermarked_video": inspect_video(output_path),
        "decoded_fidelity": decoded_fidelity(video.path, output_path),
        "files": {
            "watermarked.mp4": {"bytes": output_path.stat().st_size, "sha256": sha256(output_path)},
            "message.txt": {"bytes": message_path.stat().st_size, "sha256": sha256(message_path)},
        },
        "completed_at": utc_now(),
    }
    write_json(metrics_path, metrics)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--only", action="append")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("VideoSeal Task 3 batch requires CUDA")
    videos = select_inputs(discover_inputs(args.dataset_root.resolve()), args.only, args.limit)
    output_root = args.output_root.resolve()
    events = output_root / "videoseal-events.jsonl"

    previous = Path.cwd()
    os.chdir(args.source.resolve())
    try:
        model = videoseal.load("videoseal")
    finally:
        os.chdir(previous)
    model.eval().to("cuda")
    failures = 0
    for index, video in enumerate(videos, 1):
        metrics_path = output_directory(output_root, "videoseal", video) / "metrics.json"
        if metrics_path.is_file() and not args.force:
            print(f"[{index}/{len(videos)}] skip complete {video.key}", flush=True)
            continue
        started = time.monotonic()
        append_jsonl(events, {"status": "running", "input": video.key, "timestamp": utc_now()})
        try:
            metrics = process_one(model, video, output_root, args.chunk_size, args.seed + index)
            elapsed = time.monotonic() - started
            append_jsonl(events, {"status": "complete", "input": video.key, "elapsed_seconds": elapsed, "bit_accuracy": metrics["bit_accuracy"], "timestamp": utc_now()})
            print(f"[{index}/{len(videos)}] complete {video.key}: {metrics['bit_accuracy']:.4f} bits in {elapsed:.1f}s", flush=True)
        except Exception as error:
            failures += 1
            append_jsonl(events, {"status": "failed", "input": video.key, "error": repr(error), "timestamp": utc_now()})
            print(f"[{index}/{len(videos)}] FAILED {video.key}: {error!r}", flush=True)
            torch.cuda.empty_cache()
    print(f"VideoSeal Task 3 batch finished: {len(videos) - failures}/{len(videos)} succeeded", flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
