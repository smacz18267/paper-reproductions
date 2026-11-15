#!/usr/bin/env python3
"""Run the official E2FGVI-HQ test entry point on one prepared sample."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import cv2

from mose_watermark_common import (
    atomic_json,
    decode_rgb,
    encode_rgb,
    read_mask_sequence,
    sha256,
    utc_now,
    validate_video,
)


def git_revision(path: Path) -> str:
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def create_browser_preview(
    source: Path,
    destination: Path,
    *,
    frames: int,
    size: tuple[int, int],
    fps: float,
) -> dict[str, int | float]:
    """Create an H.264 playback copy without replacing the evaluated output."""
    decoded, rate = decode_rgb(source)
    encode_rgb(destination, decoded, rate, crf=18)
    return validate_video(destination, frames=frames, size=size, fps=fps)


def prepare_inference_inputs(
    *,
    input_video: Path,
    mask_frames: Path,
    work: Path,
    size: tuple[int, int],
    fps: float,
    max_side: int,
) -> tuple[Path, Path, dict[str, object]]:
    """Bound E2FGVI spatial memory while preserving aligned final outputs."""
    if max(size) <= max_side:
        return input_video, mask_frames, {
            "applied": False,
            "source_size": list(size),
            "inference_size": list(size),
            "max_inference_side": max_side,
        }

    scale = max_side / max(size)
    # The official model downsamples spatially by four.  Preserve the source
    # ratio as closely as possible while keeping both axes model-compatible.
    inference_size = tuple(
        max(4, int(round(dimension * scale / 4.0)) * 4)
        for dimension in size
    )
    inference_video = work / "inference-input.mp4"
    inference_masks = work / "inference-mask-frames"
    frames, rate = decode_rgb(input_video)
    resized_frames = [
        cv2.resize(frame, inference_size, interpolation=cv2.INTER_AREA)
        for frame in frames
    ]
    encode_rgb(inference_video, resized_frames, rate, crf=18)

    masks = read_mask_sequence(mask_frames)
    if len(masks) != len(frames):
        raise RuntimeError(
            f"E2FGVI video/mask frame mismatch: video={len(frames)}, mask={len(masks)}"
        )
    if inference_masks.exists():
        shutil.rmtree(inference_masks)
    inference_masks.mkdir(parents=True)
    for index, mask in enumerate(masks):
        resized = cv2.resize(mask, inference_size, interpolation=cv2.INTER_NEAREST)
        path = inference_masks / f"{index:05d}.png"
        if not cv2.imwrite(str(path), resized):
            raise RuntimeError(f"Could not write {path}")
    validate_video(
        inference_video,
        frames=len(frames),
        size=inference_size,
        fps=fps,
    )
    return inference_video, inference_masks, {
        "applied": True,
        "source_size": list(size),
        "inference_size": list(inference_size),
        "max_inference_side": max_side,
        "video_interpolation": "cv2.INTER_AREA",
        "mask_interpolation": "cv2.INTER_NEAREST",
        "reason": "bound official E2FGVI-HQ transformer memory on high-resolution clips",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--python", type=Path, default=Path(os.environ.get("PYTHON", "python3")))
    parser.add_argument("--device", default="0")
    parser.add_argument("--max-inference-side", type=int, default=720)
    parser.add_argument("--oom-retry-max-inference-side", type=int, default=576)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    source = args.source.resolve()
    checkpoint = args.checkpoint.resolve()
    run = args.run.resolve()
    sample = json.loads((run / "sample.json").read_text())
    input_video = run / "input-watermarked.mp4"
    mask_frames = run / "gt-mask-frames"
    output_video = run / "e2fgvi.mp4"
    browser_video = run / "e2fgvi-browser.mp4"
    metadata_path = run / "e2fgvi.json"
    expected_frames = int(sample["frames"])
    size = tuple(int(value) for value in sample["size"])
    fps = float(sample["fps"])

    for path, label in ((source / "test.py", "E2FGVI test.py"), (checkpoint, "E2FGVI-HQ checkpoint"), (mask_frames, "mask frames")):
        if not path.exists():
            raise FileNotFoundError(f"Missing {label}: {path}")
    validate_video(input_video, frames=expected_frames, size=size, fps=fps)
    if output_video.is_file() and metadata_path.is_file() and not args.force:
        validate_video(output_video, frames=expected_frames, size=size, fps=fps)
        if not browser_video.is_file():
            create_browser_preview(
                output_video,
                browser_video,
                frames=expected_frames,
                size=size,
                fps=fps,
            )
        print(f"E2FGVI-HQ already complete: {output_video}")
        return

    work = run / "work/e2fgvi-hq"
    results = work / "results"
    work.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    # Preserve the physical-GPU mapping exported by the queue worker. The
    # official child sees that physical device as logical device 0.
    environment.setdefault("CUDA_VISIBLE_DEVICES", args.device)
    environment["MPLBACKEND"] = "Agg"
    log = run / "logs/e2fgvi.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    sides = list(
        dict.fromkeys(
            [
                args.max_inference_side,
                min(args.max_inference_side, args.oom_retry_max_inference_side),
            ]
        )
    )
    attempted_commands: list[list[str]] = []
    for attempt, max_side in enumerate(sides, 1):
        inference_video, inference_masks, dimension_normalization = prepare_inference_inputs(
            input_video=input_video,
            mask_frames=mask_frames,
            work=work,
            size=size,
            fps=fps,
            max_side=max_side,
        )
        generated = results / f"{inference_video.stem}_results.mp4"
        if generated.exists():
            generated.unlink()
        command = [
            str(args.python),
            str(source / "test.py"),
            "--model", "e2fgvi_hq",
            "--video", str(inference_video),
            "--mask", str(inference_masks),
            "--ckpt", str(checkpoint),
            "--savefps", str(max(1, round(fps))),
        ]
        attempted_commands.append(command)
        log_start = log.stat().st_size if attempt > 1 and log.exists() else 0
        with log.open("w" if attempt == 1 else "a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {"attempt": attempt, "cwd": str(work), "command": command},
                    indent=2,
                )
                + "\n"
            )
            handle.flush()
            result = subprocess.run(
                command,
                cwd=work,
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if result.returncode == 0:
            break
        attempt_log = log.read_text(encoding="utf-8", errors="replace")[log_start:]
        out_of_memory = "OutOfMemoryError" in attempt_log
        if not out_of_memory or attempt == len(sides):
            raise subprocess.CalledProcessError(result.returncode, command)
    if not generated.is_file():
        raise RuntimeError(f"Official E2FGVI test did not create {generated}")
    if dimension_normalization["applied"]:
        generated_frames, generated_rate = decode_rgb(generated)
        restored = [
            cv2.resize(frame, size, interpolation=cv2.INTER_LINEAR)
            for frame in generated_frames
        ]
        encode_rgb(output_video, restored, generated_rate, crf=18)
    else:
        temporary = output_video.with_suffix(".tmp.mp4")
        shutil.copy2(generated, temporary)
        temporary.replace(output_video)
    output_info = validate_video(output_video, frames=expected_frames, size=size, fps=fps)
    browser_info = create_browser_preview(
        output_video,
        browser_video,
        frames=expected_frames,
        size=size,
        fps=fps,
    )
    atomic_json(
        metadata_path,
        {
            "status": "complete",
            "completed_at": utc_now(),
            "watermark_method": sample["watermark_method"],
            "sample_id": sample["sample_id"],
            "remover": "E2FGVI-HQ",
            "protocol": "official test.py with the same authentic watermarked video and binary mask frames",
            "repository": "https://github.com/MCG-NKU/E2FGVI",
            "repository_commit": git_revision(source),
            "checkpoint": {"path": str(checkpoint), "sha256": sha256(checkpoint)},
            "input": {"path": input_video.name, "sha256": sha256(input_video)},
            "mask": {
                "path": str(mask_frames.relative_to(run)),
                "preprocessing": "official read_mask: nearest resize, binary threshold, 3x3 cross dilation x4",
            },
            "output": {"path": output_video.name, "sha256": sha256(output_video), "video": output_info},
            "dimension_normalization": dimension_normalization,
            "browser_preview": {
                "path": browser_video.name,
                "sha256": sha256(browser_video),
                "video": browser_info,
                "encoding": "H.264/yuv420p playback derivative; evaluation uses output.path",
            },
            "settings": {"visible_device": args.device, "model": "e2fgvi_hq", "save_fps": max(1, round(fps))},
            "command": command,
            "attempted_commands": attempted_commands,
            "elapsed_seconds": round(time.monotonic() - started, 3),
        },
    )
    print(f"Completed E2FGVI-HQ for {sample['watermark_method']}/{sample['sample_id']}")


if __name__ == "__main__":
    main()
