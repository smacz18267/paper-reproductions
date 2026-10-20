#!/usr/bin/env python3
"""Embed one fixed VideoSeal binary message into all 50 MOSE originals."""

from __future__ import annotations

import argparse
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

import videoseal

from mose_watermark_common import (
    append_jsonl,
    atomic_json,
    decode_rgb,
    encode_rgb,
    load_clips,
    method_dir,
    original_video,
    sha256,
    utc_now,
    validate_video,
)


def fixed_message(model, path: Path, seed: int) -> torch.Tensor:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    generated = model.get_random_msg().detach().cpu()
    bits = "".join(str(int(value)) for value in generated[0])
    if path.exists():
        saved = path.read_text().strip()
        if saved != bits:
            raise RuntimeError(f"Saved VideoSeal message does not match seed {seed}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(bits + "\n")
    return generated.to("cuda")


def embed(model, source: Path, destination: Path, message: torch.Tensor, chunk_size: int) -> int:
    frames, rate = decode_rgb(source)
    marked: list[np.ndarray] = []
    for start in range(0, len(frames), chunk_size):
        chunk = np.stack(frames[start : start + chunk_size])
        tensor = torch.from_numpy(chunk).float().permute(0, 3, 1, 2).to("cuda") / 255.0
        with torch.inference_mode():
            output = model.embed(
                tensor,
                msgs=message,
                is_video=True,
                lowres_attenuation=True,
            )["imgs_w"].clamp(0, 1)
        marked.extend((output * 255).byte().permute(0, 2, 3, 1).cpu().numpy())
        del tensor, output
    encode_rgb(destination, marked, rate, crf=18)
    return len(frames)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--only", action="append")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("VideoSeal embedding requires CUDA")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu))
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
    model.eval().to("cuda")
    message_path = args.output_root / "videoseal-message.txt"
    message = fixed_message(model, message_path, args.seed)
    events = args.output_root / "videoseal-embed-events.jsonl"

    failures = 0
    for index, clip in enumerate(clips, 1):
        source = original_video(args.dataset_root, clip.clip_id)
        destination_dir = method_dir(args.output_root, clip.clip_id, "videoseal")
        destination = destination_dir / "watermarked.mp4"
        metrics_path = destination_dir / "embed.json"
        try:
            source_info = validate_video(source, frames=clip.frame_count)
            expected_size = (int(source_info["width"]), int(source_info["height"]))
            if destination.is_file() and metrics_path.is_file() and not args.force:
                validate_video(destination, frames=clip.frame_count, size=expected_size)
                print(f"[{index}/50] skip VideoSeal embed {clip.clip_id}", flush=True)
                continue
            started = time.monotonic()
            count = embed(model, source, destination, message, args.chunk_size)
            marked_info = validate_video(destination, frames=count, size=expected_size)
            metrics = {
                "status": "complete",
                "method": "videoseal",
                "protocol": "postprocess_original_then_inpaint",
                "clip_id": clip.clip_id,
                "seed": args.seed,
                "message_bits": int(message.shape[-1]),
                "source": source_info,
                "watermarked": marked_info,
                "sha256": sha256(destination),
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "completed_at": utc_now(),
            }
            atomic_json(metrics_path, metrics)
            append_jsonl(events, metrics)
            print(f"[{index}/50] complete VideoSeal embed {clip.clip_id}", flush=True)
        except Exception as error:
            failures += 1
            append_jsonl(events, {"status": "failed", "clip_id": clip.clip_id, "error": repr(error), "timestamp": utc_now()})
            print(f"[{index}/50] FAILED VideoSeal embed {clip.clip_id}: {error!r}", flush=True)
    if failures:
        raise SystemExit(f"VideoSeal embedding failed for {failures} clips")


if __name__ == "__main__":
    main()
