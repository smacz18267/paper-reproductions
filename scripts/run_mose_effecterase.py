#!/usr/bin/env python3
"""Run the official EffectErase remover on baseline or watermarked MOSE videos."""

from __future__ import annotations

import argparse
import gc
import math
import os
import sys
import time
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from mose_watermark_common import (
    FPS,
    append_jsonl,
    atomic_json,
    clip_dir,
    decode_rgb,
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


METHODS = ("baseline", "videoseal", "safe-sora", "editguard", "omniguard")
DEFAULT_REPOSITORY_COMMIT = "bcee0a5da5ef387c2ba39390dc4d579503669fb8"


def choose_clips(args: argparse.Namespace):
    clips = load_clips(args.selection)
    if not args.only:
        return clips
    requested = set(args.only)
    selected = [clip for clip in clips if clip.clip_id in requested]
    missing = requested - {clip.clip_id for clip in selected}
    if missing:
        raise RuntimeError(f"Unknown --only clip(s): {sorted(missing)}")
    return selected


def processing_size(native: tuple[int, int], max_side: int) -> tuple[int, int]:
    width, height = native
    if max(width, height) <= max_side and width % 16 == 0 and height % 16 == 0:
        return native
    scale = min(1.0, max_side / max(width, height))
    return (
        max(16, int(width * scale) // 16 * 16),
        max(16, int(height * scale) // 16 * 16),
    )


def padded_frame_count(value: int) -> int:
    """Wan video latents represent temporal lengths of the form 4n+1."""
    if value < 1:
        raise ValueError("EffectErase chunks must contain at least one frame")
    return math.ceil((value - 1) / 4) * 4 + 1


def chunk_ranges(frame_count: int, max_model_frames: int) -> list[tuple[int, int, int]]:
    if max_model_frames < 5 or max_model_frames % 4 != 1:
        raise ValueError("--max-model-frames must be at least 5 and satisfy 4n+1")
    ranges: list[tuple[int, int, int]] = []
    cursor = 0
    while cursor < frame_count:
        start = 0 if cursor == 0 else cursor - 1
        end = min(frame_count, start + max_model_frames)
        drop = 0 if cursor == 0 else 1
        ranges.append((start, end, drop))
        cursor += end - start - drop
    return ranges


def input_paths(args: argparse.Namespace, clip_id: str) -> tuple[Path, Path | None, Path]:
    if args.watermark_method == "baseline":
        destination = clip_dir(args.output_root, clip_id) / "baseline"
        return original_video(args.dataset_root, clip_id), None, destination
    destination = method_dir(args.output_root, clip_id, args.watermark_method)
    mask_video = destination / "gt-mask.mp4" if args.watermark_method in ("safe-sora", "editguard") else None
    return destination / "watermarked.mp4", mask_video, destination


def masks_for_clip(args: argparse.Namespace, clip_id: str, mask_video: Path | None) -> list[np.ndarray]:
    if mask_video is not None:
        frames, _rate = decode_rgb(mask_video)
        return [cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in frames]
    return read_mask_sequence(raw_mask_frames(args.dataset_root, clip_id))


def resized_tensor(frames: list[np.ndarray], size: tuple[int, int], *, mask: bool = False) -> torch.Tensor:
    width, height = size
    tensors: list[torch.Tensor] = []
    for frame in frames:
        if mask:
            array = np.where(frame > 127, 255, 0).astype(np.uint8)
            image = Image.fromarray(cv2.cvtColor(array, cv2.COLOR_GRAY2RGB))
        else:
            image = Image.fromarray(frame)
        image = image.resize((width, height), resample=Image.Resampling.BILINEAR)
        value = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float() / 127.5 - 1.0
        tensors.append(value)
    return torch.stack(tensors).permute(1, 0, 2, 3).contiguous()


def reference_crop(
    frames: list[np.ndarray], masks: list[np.ndarray], size: tuple[int, int]
) -> torch.Tensor | None:
    width, height = size
    reference_index = next((index for index, mask in enumerate(masks) if np.any(mask > 127)), None)
    if reference_index is None:
        return None
    mask = cv2.resize(masks[reference_index], (width, height), interpolation=cv2.INTER_NEAREST)
    frame = cv2.resize(frames[reference_index], (width, height), interpolation=cv2.INTER_LANCZOS4)
    ys, xs = np.where(mask > 127)
    x0, x1 = xs.min(), xs.max() + 1
    y0, y1 = ys.min(), ys.max() + 1
    side = max(x1 - x0, y1 - y0)
    center_x = (x0 + x1) / 2.0
    center_y = (y0 + y1) / 2.0
    sx0 = int(np.floor(center_x - side / 2))
    sy0 = int(np.floor(center_y - side / 2))
    sx1, sy1 = sx0 + side, sy0 + side
    pad_left, pad_top = max(0, -sx0), max(0, -sy0)
    pad_right, pad_bottom = max(0, sx1 - width), max(0, sy1 - height)
    if pad_left or pad_top or pad_right or pad_bottom:
        frame = np.pad(frame, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)))
        mask = np.pad(mask, ((pad_top, pad_bottom), (pad_left, pad_right)))
        sx0, sx1 = sx0 + pad_left, sx1 + pad_left
        sy0, sy1 = sy0 + pad_top, sy1 + pad_top
    crop = frame[sy0:sy1, sx0:sx1] * (mask[sy0:sy1, sx0:sx1, None] > 127)
    crop = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_LINEAR)
    return torch.from_numpy(crop.copy()).permute(2, 0, 1).float() / 127.5 - 1.0


def load_pipeline(args: argparse.Namespace):
    source = args.source.resolve()
    os.chdir(source)
    sys.path.insert(0, str(source))
    from diffsynth import ModelManager, WanRemovePipeline

    base = source / "Wan-AI" / "Wan2.1-Fun-1.3B-InP"
    paths = {
        "dit": base / "diffusion_pytorch_model.safetensors",
        "text_encoder": base / "models_t5_umt5-xxl-enc-bf16.pth",
        "vae": base / "Wan2.1_VAE.pth",
        "image_encoder": base / "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
        "effecterase": source / "EffectErase.ckpt",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise RuntimeError(f"Missing EffectErase model files: {missing}")
    manager = ModelManager(device="cuda")
    manager.load_models(
        [
            str(paths["dit"]),
            str(paths["text_encoder"]),
            str(paths["vae"]),
            str(paths["image_encoder"]),
        ],
        torch_dtype=torch.bfloat16,
    )
    manager.load_lora_v2(str(paths["effecterase"]), lora_alpha=args.lora_alpha)
    pipeline = WanRemovePipeline.from_model_manager(manager, torch_dtype=torch.bfloat16, device="cuda")
    pipeline.enable_vram_management(num_persistent_param_in_dit=6 * 10**9)
    record = {
        "repository": "https://github.com/FudanCVL/EffectErase",
        "repository_commit": args.repository_commit,
        "effecterase_checkpoint": {
            "path": str(paths["effecterase"]),
            "sha256": sha256(paths["effecterase"]),
        },
        "base_model_revision": "9bcbbcf673c0e6a996df6f3f25bee596ffa36d47",
        "effecterase_revision": "95c9d455300c265c57159767ba3e3e1eebb347b0",
    }
    return pipeline, record


def run_chunk(
    pipeline,
    frames: list[np.ndarray],
    masks: list[np.ndarray],
    size: tuple[int, int],
    args: argparse.Namespace,
    seed: int,
) -> tuple[list[np.ndarray], int]:
    actual_frames = len(frames)
    target_frames = padded_frame_count(actual_frames)
    padded_frames = frames + [frames[-1].copy() for _ in range(target_frames - actual_frames)]
    padded_masks = masks + [masks[-1].copy() for _ in range(target_frames - actual_frames)]
    reference = reference_crop(padded_frames, padded_masks, size)
    if reference is None:
        return frames, target_frames
    video = resized_tensor(padded_frames, size)
    mask = resized_tensor(padded_masks, size, mask=True)
    remove_prompt = "Remove the specified object and all related effects, then restore a clean background."
    negative_prompt = (
        "细节模糊不清，字幕，作品，画作，画面，静止，最差质量，低质量，JPEG压缩残留，"
        "丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，"
        "形态畸形的肢体，手指融合，杂乱的背景，三条腿，背景人很多，倒着走"
    )
    width, height = size
    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        generated, _attention = pipeline(
            video_mask=mask.cuda(),
            video_fg_bg=video.cuda(),
            video_bg=None,
            task="remove",
            fg_first_img=reference.cuda(),
            prompt_remove=remove_prompt,
            negative_prompt=negative_prompt,
            num_frames=target_frames,
            num_inference_steps=args.num_inference_steps,
            cfg_scale=args.cfg,
            seed=seed,
            tiled=args.tiled,
            height=height,
            width=width,
            tea_cache_l1_thresh=0.3 if args.use_teacache else None,
            tea_cache_model_id="Wan2.1-T2V-1.3B" if args.use_teacache else "",
        )
    arrays = [np.asarray(frame.convert("RGB")) for frame in generated[:actual_frames]]
    del video, mask, reference, generated, _attention
    gc.collect()
    torch.cuda.empty_cache()
    return arrays, target_frames


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--watermark-method", required=True, choices=METHODS)
    parser.add_argument("--repository-commit", default=DEFAULT_REPOSITORY_COMMIT)
    parser.add_argument("--max-side", type=int, default=832)
    parser.add_argument("--max-model-frames", type=int, default=81)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--cfg", type=float, default=1.0)
    parser.add_argument("--lora-alpha", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--tiled", action="store_true")
    parser.add_argument("--use-teacache", action="store_true")
    parser.add_argument("--only", action="append")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("EffectErase inference requires CUDA")
    clips = choose_clips(args)
    pipeline, model_record = load_pipeline(args)
    events = args.output_root / f"{args.watermark_method}-effecterase-events.jsonl"
    failures = 0
    for ordinal, clip in enumerate(clips, 1):
        source_path, mask_video, destination = input_paths(args, clip.clip_id)
        output_path = destination / "effecterase.mp4"
        metadata_path = destination / "effecterase.json"
        try:
            source_info = validate_video(source_path, fps=FPS)
            expected_frames = 16 if args.watermark_method == "safe-sora" else clip.frame_count
            native_size = (int(source_info["width"]), int(source_info["height"]))
            if int(source_info["frames"]) != expected_frames:
                raise RuntimeError(f"{source_path}: expected {expected_frames} frames")
            if output_path.is_file() and metadata_path.is_file() and not args.force:
                validate_video(output_path, frames=expected_frames, size=native_size)
                print(f"[{ordinal}/{len(clips)}] skip {args.watermark_method}/effecterase {clip.clip_id}", flush=True)
                continue
            started = time.monotonic()
            frames, rate = decode_rgb(source_path)
            masks = masks_for_clip(args, clip.clip_id, mask_video)
            if len(masks) != expected_frames:
                raise RuntimeError(f"{clip.clip_id}: expected {expected_frames} masks, got {len(masks)}")
            size = processing_size(native_size, args.max_side)
            output_frames: list[np.ndarray] = []
            temporal_chunks: list[dict[str, int]] = []
            blank_chunks = 0
            for chunk_index, (start, end, drop) in enumerate(
                chunk_ranges(expected_frames, args.max_model_frames)
            ):
                chunk_frames = frames[start:end]
                chunk_masks = masks[start:end]
                is_blank = not any(np.any(mask > 127) for mask in chunk_masks)
                generated, padded = run_chunk(
                    pipeline,
                    chunk_frames,
                    chunk_masks,
                    size,
                    args,
                    args.seed + chunk_index,
                )
                blank_chunks += int(is_blank)
                restored = [
                    frame
                    if frame.shape[1::-1] == native_size
                    else cv2.resize(frame, native_size, interpolation=cv2.INTER_LANCZOS4)
                    for frame in generated
                ]
                output_frames.extend(restored[drop:])
                temporal_chunks.append(
                    {"start": start, "end_exclusive": end, "drop_prefix": drop, "padded_frames": padded}
                )
                print(
                    f"  {clip.clip_id}: EffectErase chunk {chunk_index + 1} "
                    f"({start}:{end}, padded={padded})",
                    flush=True,
                )
            if len(output_frames) != expected_frames:
                raise RuntimeError(
                    f"{clip.clip_id}: assembled {len(output_frames)} frames, expected {expected_frames}"
                )
            encode_rgb(output_path, output_frames, rate or Fraction(FPS, 1), crf=18)
            metrics = {
                "status": "complete",
                "watermark_method": args.watermark_method,
                "remover": "effecterase",
                "protocol": "watermark_original_then_remove_object",
                "clip_id": clip.clip_id,
                "input": source_info,
                "output": validate_video(output_path, frames=expected_frames, size=native_size),
                "sha256": sha256(output_path),
                "model": model_record,
                "processing_adapter": {
                    "reason": "Preserve aspect ratio within EffectErase's released Wan inference scale.",
                    "processing_size": list(size),
                    "reported_size": list(native_size),
                    "restoration": "Lanczos resize to the aligned input size",
                },
                "temporal_adapter": {
                    "reason": "EffectErase uses Wan 4n+1 latent windows; long clips use one-frame overlap.",
                    "maximum_model_frames": args.max_model_frames,
                    "chunks": temporal_chunks,
                    "trimmed_back_to_source_frames": True,
                    "blank_mask_chunks_copied_without_generation": blank_chunks,
                },
                "settings": {
                    "num_inference_steps": args.num_inference_steps,
                    "cfg": args.cfg,
                    "seed": args.seed,
                    "tiled": args.tiled,
                    "teacache": args.use_teacache,
                },
                "mask_preprocessing": "binary source mask, bilinear resize by the released input transform, no dilation",
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "completed_at": utc_now(),
            }
            atomic_json(metadata_path, metrics)
            append_jsonl(events, metrics)
            print(
                f"[{ordinal}/{len(clips)}] complete {args.watermark_method}/effecterase {clip.clip_id}",
                flush=True,
            )
        except Exception as error:
            failures += 1
            append_jsonl(
                events,
                {"status": "failed", "clip_id": clip.clip_id, "error": repr(error), "timestamp": utc_now()},
            )
            print(
                f"[{ordinal}/{len(clips)}] FAILED {args.watermark_method}/effecterase "
                f"{clip.clip_id}: {error!r}",
                flush=True,
            )
    if failures:
        raise SystemExit(f"{args.watermark_method}/effecterase failed for {failures} clips")


if __name__ == "__main__":
    main()
