#!/usr/bin/env python3
"""Run ProPainter or DiffuEraser on genuinely watermarked MOSE inputs."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import time
from fractions import Fraction
from pathlib import Path

import cv2

from mose_watermark_common import (
    FPS,
    append_jsonl,
    atomic_json,
    decode_rgb,
    encode_binary_masks,
    encode_rgb,
    load_clips,
    method_dir,
    raw_mask_frames,
    sha256,
    utc_now,
    validate_video,
)


def command_to_log(command: list[str], cwd: Path, log: Path) -> tuple[int, float]:
    log.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with log.open("a", encoding="utf-8") as handle:
        handle.write(f"\n[{utc_now()}] COMMAND: {json.dumps(command)}\n")
        handle.flush()
        result = subprocess.run(
            command,
            cwd=str(cwd),
            env=os.environ.copy(),
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        elapsed = time.monotonic() - started
        handle.write(f"[{utc_now()}] EXIT: {result.returncode}; ELAPSED_SECONDS: {elapsed:.3f}\n")
    return result.returncode, elapsed


def safe_sora_masks(dataset_root: Path, embed: dict, output: Path, clip_id: str) -> Path:
    directory = output / ".mask-frames"
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
    paths = sorted(raw_mask_frames(dataset_root, clip_id).glob("*.png"))
    indices = [int(index) for index in embed["sampling"]["indices"]]
    for ordinal, source_index in enumerate(indices):
        mask = cv2.imread(str(paths[source_index]), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Could not read {paths[source_index]}")
        mask = cv2.resize(mask, (512, 320), interpolation=cv2.INTER_NEAREST)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        if not cv2.imwrite(str(directory / f"{ordinal:05d}.png"), mask):
            raise RuntimeError("Could not write temporary Safe-Sora mask")
    return directory


def clean_temporary(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--watermark-method", required=True, choices=("videoseal", "safe-sora"))
    parser.add_argument("--remover", required=True, choices=("propainter", "diffueraser"))
    parser.add_argument("--propainter", type=Path, default=Path("/home/smayan/research/ProPainter"))
    parser.add_argument("--diffueraser", type=Path, default=Path("/home/smayan/research/DiffuEraser"))
    parser.add_argument("--propainter-subvideo-length", type=int, default=40)
    parser.add_argument(
        "--processing-max-side",
        type=int,
        default=960,
        help="Optional lower-resolution inference adapter; output is restored to the watermarked input size.",
    )
    parser.add_argument("--only", action="append")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    clips = load_clips(args.selection)
    if args.only:
        requested = set(args.only)
        clips = [clip for clip in clips if clip.clip_id in requested]
        if {clip.clip_id for clip in clips} != requested:
            raise RuntimeError(f"Unknown --only clip(s): {sorted(requested - {clip.clip_id for clip in clips})}")
    propainter = args.propainter.resolve()
    diffueraser = args.diffueraser.resolve()
    propainter_python = propainter / ".venv" / "bin" / "python"
    diffueraser_python = diffueraser / ".venv" / "bin" / "python"
    for required in (propainter_python, diffueraser_python):
        if not required.is_file():
            raise RuntimeError(f"Missing required executable: {required}")

    events = args.output_root / f"{args.watermark_method}-{args.remover}-events.jsonl"
    failures = 0
    for ordinal, clip in enumerate(clips, 1):
        destination = method_dir(args.output_root, clip.clip_id, args.watermark_method)
        watermarked = destination / "watermarked.mp4"
        final = destination / f"{args.remover}.mp4"
        metrics_path = destination / f"{args.remover}.json"
        temporary_masks: Path | None = None
        try:
            source_info = validate_video(watermarked, fps=FPS)
            expected_frames = int(source_info["frames"])
            size = (int(source_info["width"]), int(source_info["height"]))
            if args.watermark_method == "videoseal" and expected_frames != clip.frame_count:
                raise RuntimeError(f"VideoSeal frame count differs for {clip.clip_id}")
            if args.watermark_method == "safe-sora" and expected_frames != 16:
                raise RuntimeError(f"Safe-Sora must contain 16 frames for {clip.clip_id}")
            if final.is_file() and metrics_path.is_file() and not args.force:
                validate_video(final, frames=expected_frames, size=size)
                print(f"[{ordinal}/50] skip {args.watermark_method}/{args.remover} {clip.clip_id}", flush=True)
                continue

            if final.exists():
                final.unlink()
            started = time.monotonic()
            log = destination / "logs" / f"{args.remover}.log"
            processing_size = size
            if max(size) > args.processing_max_side:
                scale = args.processing_max_side / max(size)
                processing_size = (
                    max(16, int(size[0] * scale) // 16 * 16),
                    max(16, int(size[1] * scale) // 16 * 16),
                )
            processing_adapter: dict[str, object] | None = None
            if args.watermark_method == "safe-sora":
                embed = json.loads((destination / "embed.json").read_text())
                temporary_masks = safe_sora_masks(args.dataset_root, embed, destination, clip.clip_id)
                mask_directory = temporary_masks
                mask_video = destination / "gt-mask.mp4"
            else:
                mask_directory = raw_mask_frames(args.dataset_root, clip.clip_id)
                mask_video = args.dataset_root / "batch_inputs" / clip.clip_id / "mask.mp4"

            if args.remover == "propainter":
                temporary_output = destination / ".propainter-output"
                clean_temporary(temporary_output)
                command = [
                    str(propainter_python), "-u", str(propainter / "inference_propainter.py"),
                    "--video", str(watermarked), "--mask", str(mask_directory),
                    "--output", str(temporary_output), "--height", str(processing_size[1]),
                    "--width", str(processing_size[0]), "--mask_dilation", "8",
                    "--subvideo_length", str(args.propainter_subvideo_length),
                    "--save_fps", str(FPS), "--fp16",
                ]
                code, elapsed = command_to_log(command, propainter, log)
                generated = temporary_output / "watermarked" / "inpaint_out.mp4"
                if code != 0 or not generated.is_file():
                    raise RuntimeError(f"ProPainter exited {code}; see {log}")
                if processing_size != size:
                    generated_frames, generated_rate = decode_rgb(generated)
                    restored = [
                        cv2.resize(frame, size, interpolation=cv2.INTER_LANCZOS4)
                        for frame in generated_frames
                    ]
                    encode_rgb(final, restored, generated_rate, crf=18)
                    processing_adapter = {
                        "reason": "Shared-GPU memory limit at native spatial resolution.",
                        "processing_size": list(processing_size),
                        "reported_size": list(size),
                        "restoration": "Lanczos resize of inference result",
                    }
                else:
                    generated.replace(final)
                clean_temporary(temporary_output)
            else:
                temporary_output = destination / ".diffueraser-output"
                clean_temporary(temporary_output)
                temporary_output.mkdir(parents=True)
                diffueraser_video = watermarked
                diffueraser_mask = mask_video
                compatibility: dict[str, object] | None = None
                if expected_frames < 22:
                    # DiffuEraser rejects sequences shorter than 22 frames,
                    # while Safe-Sora's official protocol is exactly 16. Pad
                    # only for inference, then trim the result back to the 16
                    # genuine watermarked frames.
                    padded_frames = 24
                    video_frames, _video_rate = decode_rgb(watermarked)
                    mask_frames, _mask_rate = decode_rgb(mask_video)
                    video_frames.extend([video_frames[-1].copy() for _ in range(padded_frames - expected_frames)])
                    gray_masks = [cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in mask_frames]
                    gray_masks.extend([gray_masks[-1].copy() for _ in range(padded_frames - expected_frames)])
                    diffueraser_video = temporary_output / "padded-watermarked.mp4"
                    diffueraser_mask = temporary_output / "padded-mask.mp4"
                    encode_rgb(diffueraser_video, video_frames, Fraction(FPS, 1), crf=18)
                    encode_binary_masks(diffueraser_mask, gray_masks, Fraction(FPS, 1))
                    compatibility = {
                        "reason": "DiffuEraser requires at least 22 frames; Safe-Sora natively produces 16.",
                        "padding": "repeat final frame and final mask",
                        "padded_frames": padded_frames,
                        "reported_frames": expected_frames,
                    }
                command = [
                    str(diffueraser_python), "-u", str(diffueraser / "run_diffueraser.py"),
                    "--input_video", str(diffueraser_video), "--input_mask", str(diffueraser_mask),
                    "--video_length", str(max(1, math.ceil(max(expected_frames, 24 if compatibility else 0) / FPS))),
                    "--mask_dilation_iter", "8", "--max_img_size", str(args.processing_max_side),
                    "--subvideo_length", str(args.propainter_subvideo_length),
                    "--save_path", str(temporary_output),
                ]
                code, elapsed = command_to_log(command, diffueraser, log)
                generated = temporary_output / "diffueraser_result.mp4"
                if code != 0 or not generated.is_file():
                    raise RuntimeError(f"DiffuEraser exited {code}; see {log}")
                if compatibility or processing_size != size:
                    generated_frames, generated_rate = decode_rgb(generated)
                    if len(generated_frames) < expected_frames:
                        raise RuntimeError("DiffuEraser result is shorter than the reported clip")
                    reported_frames = generated_frames[:expected_frames]
                    if processing_size != size:
                        reported_frames = [
                            cv2.resize(frame, size, interpolation=cv2.INTER_LANCZOS4)
                            for frame in reported_frames
                        ]
                        processing_adapter = {
                            "reason": "Shared-GPU memory limit at native spatial resolution.",
                            "processing_size": list(processing_size),
                            "reported_size": list(size),
                            "restoration": "Lanczos resize of inference result",
                        }
                    encode_rgb(final, reported_frames, generated_rate, crf=18)
                else:
                    generated.replace(final)
                clean_temporary(temporary_output)

            output_info = validate_video(final, frames=expected_frames, size=size)
            metrics = {
                "status": "complete",
                "watermark_method": args.watermark_method,
                "remover": args.remover,
                "protocol": "watermark_original_then_remove_object",
                "clip_id": clip.clip_id,
                "input": source_info,
                "output": output_info,
                "sha256": sha256(final),
                "mask_dilation_iterations": 8,
                "compatibility_adapter": compatibility if args.remover == "diffueraser" else None,
                "processing_adapter": processing_adapter,
                "elapsed_seconds": round(elapsed, 3),
                "completed_at": utc_now(),
            }
            atomic_json(metrics_path, metrics)
            append_jsonl(events, metrics)
            print(f"[{ordinal}/50] complete {args.watermark_method}/{args.remover} {clip.clip_id}", flush=True)
        except Exception as error:
            failures += 1
            append_jsonl(events, {"status": "failed", "clip_id": clip.clip_id, "error": repr(error), "timestamp": utc_now()})
            print(f"[{ordinal}/50] FAILED {args.watermark_method}/{args.remover} {clip.clip_id}: {error!r}", flush=True)
        finally:
            if temporary_masks is not None:
                clean_temporary(temporary_masks)
    if failures:
        raise SystemExit(f"{args.watermark_method}/{args.remover} failed for {failures} clips")


if __name__ == "__main__":
    main()
