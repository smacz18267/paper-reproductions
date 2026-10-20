#!/usr/bin/env python3
"""Run VideoShield's detector on MOSE originals as an honest negative control.

VideoShield embeds into a diffusion model's initial Gaussian noise. It cannot
post-process arbitrary MOSE videos, so watermarked/removal/mask artifacts are
explicitly unavailable. This run only measures the official inverse-diffusion
detector against one fixed reference key.
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

from mose_watermark_common import append_jsonl, atomic_json, load_clips, method_dir, original_video, utc_now


TARGET_FRAMES = 16
TARGET_HEIGHT = 256
TARGET_WIDTH = 256


def sample(path: Path) -> tuple[list[np.ndarray], list[int], float]:
    decoded: list[np.ndarray] = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        rate = stream.average_rate or stream.guessed_rate
        for frame in container.decode(video=0):
            rgb = frame.to_ndarray(format="rgb24")
            decoded.append(cv2.resize(rgb, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_AREA))
    indices = np.linspace(0, len(decoded) - 1, TARGET_FRAMES).round().astype(int).tolist()
    return [decoded[index].astype(np.float32) / 255.0 for index in indices], indices, float(rate or 25.0)


def fixed_reference(VideoShield, path: Path, seed: int) -> dict:
    if path.is_file():
        with np.load(path, allow_pickle=False) as saved:
            return {
                "m": torch.from_numpy(saved["m"]),
                "watermark": torch.from_numpy(saved["watermark"]),
                "key": saved["key"].tobytes(),
                "nonce": saved["nonce"].tobytes(),
            }
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    reference = VideoShield(
        ch_factor=1, hw_factor=4, frame_factor=8,
        height=TARGET_HEIGHT // 8, width=TARGET_WIDTH // 8,
        num_frames=TARGET_FRAMES, device="cuda:0",
    )
    reference.create_watermark_and_return_w()
    payload = {
        "m": reference.m.cpu().numpy(),
        "watermark": reference.watermark.cpu().numpy(),
        "key": np.frombuffer(reference.key, dtype=np.uint8),
        "nonce": np.frombuffer(reference.nonce, dtype=np.uint8),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)
    return {
        "m": reference.m.cpu(), "watermark": reference.watermark.cpu(),
        "key": reference.key, "nonce": reference.nonce,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--only", action="append")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("VideoShield control requires CUDA")
    source = args.source.resolve()
    sys.path.insert(0, str(source))
    from utils import get_video_latents, transform_video
    from watermark import VideoShield

    reference = fixed_reference(VideoShield, args.output_root / "videoshield-reference.npz", args.seed)
    pipe = TextToVideoSDPipeline.from_pretrained(args.model_path.resolve(), torch_dtype=torch.float16).to("cuda:0")
    pipe.safety_checker = None
    inverse_scheduler = DDIMInverseScheduler.from_pretrained(args.model_path.resolve(), subfolder="scheduler")
    clips = load_clips(args.selection)
    if args.only:
        requested = set(args.only)
        clips = [clip for clip in clips if clip.clip_id in requested]
        if {clip.clip_id for clip in clips} != requested:
            raise RuntimeError(f"Unknown --only clip(s): {sorted(requested - {clip.clip_id for clip in clips})}")
    events = args.output_root / "videoshield-control-events.jsonl"
    failures = 0

    for ordinal, clip in enumerate(clips, 1):
        destination = method_dir(args.output_root, clip.clip_id, "videoshield")
        metrics_path = destination / "availability.json"
        try:
            if metrics_path.is_file() and not args.force:
                print(f"[{ordinal}/50] skip VideoShield control {clip.clip_id}", flush=True)
                continue
            started = time.monotonic()
            frames, indices, source_fps = sample(original_video(args.dataset_root, clip.clip_id))
            random.seed(args.seed)
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
            detector = VideoShield(
                ch_factor=1, hw_factor=4, frame_factor=8,
                height=TARGET_HEIGHT // 8, width=TARGET_WIDTH // 8,
                num_frames=TARGET_FRAMES, device="cuda:0",
            )
            detector.m = reference["m"].to("cuda:0")
            detector.watermark = reference["watermark"].to("cuda:0")
            detector.key = reference["key"]
            detector.nonce = reference["nonce"]
            tensor = transform_video(frames).to(pipe.vae.dtype).to("cuda:0")
            with torch.inference_mode():
                latents = get_video_latents(pipe.vae, tensor, sample=False, permute=True)
                pipe.scheduler = inverse_scheduler
                reversed_latents = pipe(
                    prompt="", latents=latents, num_inference_steps=args.steps,
                    guidance_scale=1.0, output_type="latent",
                ).frames
                accuracy = float(detector.eval_watermark(reversed_latents))
            metrics = {
                "status": "complete",
                "method": "videoshield",
                "protocol": "unwatermarked_mose_negative_control_detection",
                "clip_id": clip.clip_id,
                "availability": {
                    "watermarked_video": "not_applicable",
                    "propainter_video": "not_applicable",
                    "diffueraser_video": "not_applicable",
                    "pred_mask": "not_applicable",
                    "iou": "not_applicable",
                },
                "reason": "VideoShield embeds in diffusion-generation noise and cannot post-process an existing MOSE video.",
                "reference_watermark_accuracy": accuracy,
                "interpretation": "Expected chance-level negative control; not an embedding or robustness score.",
                "sampling": {"frames": TARGET_FRAMES, "indices": indices, "resize": [TARGET_WIDTH, TARGET_HEIGHT], "source_fps": source_fps},
                "ddim_inversion_steps": args.steps,
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "completed_at": utc_now(),
            }
            atomic_json(metrics_path, metrics)
            append_jsonl(events, metrics)
            print(f"[{ordinal}/50] complete VideoShield control {clip.clip_id}: {accuracy:.4f}", flush=True)
            del tensor, latents, reversed_latents, detector
        except Exception as error:
            failures += 1
            append_jsonl(events, {"status": "failed", "clip_id": clip.clip_id, "error": repr(error), "timestamp": utc_now()})
            print(f"[{ordinal}/50] FAILED VideoShield control {clip.clip_id}: {error!r}", flush=True)
        finally:
            gc.collect()
            torch.cuda.empty_cache()
    if failures:
        raise SystemExit(f"VideoShield control failed for {failures} clips")


if __name__ == "__main__":
    main()
