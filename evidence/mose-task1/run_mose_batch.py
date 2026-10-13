#!/usr/bin/env python3
"""Prepare and run the 50-clip MOSEv1 ProPainter/DiffuEraser batch.

Run this with DiffuEraser's virtual-environment Python so PyAV and OpenCV
are available. The runner is resumable: it validates existing outputs,
skips completed clips, writes one log per method/clip, and records state.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any

import av
import cv2


FPS = 25
MAX_SIDE = 960
MASK_DILATION = 8
EXPECTED_CLIPS = 50
MIN_FRAMES = 22


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def say(message: str) -> None:
    print(f"[{timestamp()}] {message}", flush=True)


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def append_event(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")


def load_selection(root: Path) -> list[dict[str, Any]]:
    path = root / "selection50.json"
    selection = json.loads(path.read_text())
    clips = selection.get("clips", [])
    if len(clips) != EXPECTED_CLIPS:
        raise RuntimeError(f"Expected {EXPECTED_CLIPS} clips in {path}, found {len(clips)}")

    seen: set[str] = set()
    for clip in clips:
        video = clip["video_id"]
        count = int(clip["frame_count"])
        if video in seen:
            raise RuntimeError(f"Duplicate clip in selection: {video}")
        if count < MIN_FRAMES:
            raise RuntimeError(f"Clip has fewer than {MIN_FRAMES} frames: {video}")
        seen.add(video)
    return sorted(clips, key=lambda item: item["video_id"])


def target_size(width: int, height: int) -> tuple[int, int]:
    longest = max(width, height)
    if longest > MAX_SIDE:
        ratio = longest / MAX_SIDE
        width = int(width / ratio)
        height = int(height / ratio)
    # ProPainter's imageio writer encodes in 16-pixel macroblocks and rounds
    # nonconforming dimensions up. Prepare both methods at that final size so
    # their inputs and validated outputs remain pixel-aligned.
    width = min(MAX_SIDE, math.ceil(width / 16) * 16)
    height = min(MAX_SIDE, math.ceil(height / 16) * 16)
    if min(width, height) < 16:
        raise RuntimeError(f"Invalid target size: {width}x{height}")
    return width, height


def source_paths(root: Path, video: str, expected: int) -> tuple[list[Path], list[Path], tuple[int, int]]:
    frames = sorted((root / "selected50" / "JPEGImages" / video).glob("*.jpg"))
    masks = sorted((root / "binary_masks" / video).glob("*.png"))
    if len(frames) != expected or len(masks) != expected:
        raise RuntimeError(
            f"Unexpected source count for {video}: frames={len(frames)}, masks={len(masks)}, expected={expected}"
        )
    if [path.stem for path in frames] != [path.stem for path in masks]:
        raise RuntimeError(f"Frame/mask filenames differ for {video}")

    first = cv2.imread(str(frames[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise RuntimeError(f"Could not read {frames[0]}")
    height, width = first.shape[:2]
    return frames, masks, target_size(width, height)


def video_info(path: Path) -> dict[str, Any]:
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        width = int(stream.width)
        height = int(stream.height)
        rate = float(stream.average_rate) if stream.average_rate else 0.0
        frames = sum(1 for _frame in container.decode(video=0))
    return {"frames": frames, "fps": rate, "width": width, "height": height, "bytes": path.stat().st_size}


def valid_video(path: Path, expected: int, size: tuple[int, int]) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        info = video_info(path)
    except Exception:
        return False
    return (
        info["frames"] == expected
        and abs(info["fps"] - FPS) < 0.01
        and (info["width"], info["height"]) == size
    )


def write_mp4(
    destination: Path,
    sources: list[Path],
    size: tuple[int, int],
    *,
    mask: bool,
) -> None:
    if "libx264" not in av.codecs_available:
        raise RuntimeError("PyAV does not provide libx264; lossless mask encoding is unavailable")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.stem + ".part.mp4")
    if temporary.exists():
        temporary.unlink()

    width, height = size
    container = av.open(str(temporary), mode="w")
    stream = container.add_stream("libx264", rate=FPS)
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "0" if mask else "18", "preset": "medium"}

    try:
        for index, source in enumerate(sources):
            if mask:
                image = cv2.imread(str(source), cv2.IMREAD_GRAYSCALE)
                if image is None:
                    raise RuntimeError(f"Could not read {source}")
                image = cv2.resize(image, size, interpolation=cv2.INTER_NEAREST)
                _threshold, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            else:
                image = cv2.imread(str(source), cv2.IMREAD_COLOR)
                if image is None:
                    raise RuntimeError(f"Could not read {source}")
                image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)

            frame = av.VideoFrame.from_ndarray(image, format="bgr24")
            frame.pts = index
            frame.time_base = Fraction(1, FPS)
            for packet in stream.encode(frame):
                container.mux(packet)

        for packet in stream.encode():
            container.mux(packet)
        container.close()
        temporary.replace(destination)
    except Exception:
        container.close()
        if temporary.exists():
            temporary.unlink()
        raise


def prepare_clip(root: Path, video: str, expected: int) -> tuple[Path, Path, tuple[int, int]]:
    frames, masks, size = source_paths(root, video, expected)
    directory = root / "batch_inputs" / video
    input_video = directory / "video.mp4"
    input_mask = directory / "mask.mp4"

    if not valid_video(input_video, expected, size):
        say(f"Encoding input video for {video} at {size[0]}x{size[1]}")
        write_mp4(input_video, frames, size, mask=False)
    if not valid_video(input_mask, expected, size):
        say(f"Encoding mask video for {video} at {size[0]}x{size[1]}")
        write_mp4(input_mask, masks, size, mask=True)

    if not valid_video(input_video, expected, size) or not valid_video(input_mask, expected, size):
        raise RuntimeError(f"Prepared inputs failed validation for {video}")
    return input_video, input_mask, size


def gpu_state(index: int) -> tuple[int, int]:
    command = [
        "nvidia-smi",
        "--query-gpu=memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
        "-i",
        str(index),
    ]
    output = subprocess.check_output(command, text=True).strip()
    free_text, utilization_text = [part.strip() for part in output.split(",")]
    return int(free_text), int(utilization_text)


def wait_for_gpu(index: int, minimum_free: int, maximum_utilization: int, poll_seconds: int) -> None:
    while True:
        free, utilization = gpu_state(index)
        if free >= minimum_free and utilization <= maximum_utilization:
            say(f"GPU {index} ready: {free} MiB free, {utilization}% utilization")
            return
        say(
            f"Waiting for GPU {index}: {free} MiB free, {utilization}% utilization "
            f"(need >= {minimum_free} MiB and <= {maximum_utilization}%)"
        )
        time.sleep(poll_seconds)


def quarantine(path: Path) -> None:
    if not path.exists():
        return
    suffix = datetime.now().strftime("%Y%m%dT%H%M%S")
    target = path.with_name(path.name + f".invalid-{suffix}")
    path.replace(target)


def seed_test_results(root: Path, clips: list[dict[str, Any]], result_root: Path) -> None:
    first = next((item for item in clips if item["video_id"] == "02b72b8d"), None)
    if first is None:
        return
    expected = int(first["frame_count"])
    _frames, _masks, size = source_paths(root, first["video_id"], expected)
    pairs = [
        (
            root / "test_outputs" / "propainter" / "02b72b8d" / "inpaint_out.mp4",
            result_root / "propainter" / "02b72b8d" / "inpaint_out.mp4",
        ),
        (
            root / "test_outputs" / "diffueraser" / "02b72b8d" / "diffueraser_result.mp4",
            result_root / "diffueraser" / "02b72b8d" / "diffueraser_result.mp4",
        ),
    ]
    for source, destination in pairs:
        if valid_video(destination, expected, size):
            continue
        if valid_video(source, expected, size):
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            say(f"Seeded completed test result: {destination}")


def load_state(path: Path, gpu: int) -> dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text())
    return {
        "schema_version": 1,
        "created_at": timestamp(),
        "gpu": gpu,
        "fps": FPS,
        "max_side": MAX_SIDE,
        "mask_dilation": MASK_DILATION,
        "clips": {},
    }


def run_command(command: list[str], cwd: Path, log_path: Path, environment: dict[str, str]) -> tuple[int, float]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n[{timestamp()}] COMMAND: {json.dumps(command)}\n")
        handle.flush()
        result = subprocess.run(
            command,
            cwd=str(cwd),
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        elapsed = time.monotonic() - started
        handle.write(f"[{timestamp()}] EXIT: {result.returncode}; ELAPSED_SECONDS: {elapsed:.3f}\n")
    return result.returncode, elapsed


def mark(
    state: dict[str, Any],
    state_path: Path,
    events_path: Path,
    video: str,
    method: str,
    status: str,
    **details: Any,
) -> None:
    record = {"status": status, "updated_at": timestamp(), **details}
    state.setdefault("clips", {}).setdefault(video, {})[method] = record
    atomic_json(state_path, state)
    append_event(events_path, {"timestamp": timestamp(), "video_id": video, "method": method, **record})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/home/smayan/research/datasets/MOSEv1"))
    parser.add_argument("--propainter", type=Path, default=Path("/home/smayan/research/ProPainter"))
    parser.add_argument("--diffueraser", type=Path, default=Path("/home/smayan/research/DiffuEraser"))
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--min-free-mb", type=int, default=24576)
    parser.add_argument("--max-utilization", type=int, default=20)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    propainter = args.propainter.resolve()
    diffueraser = args.diffueraser.resolve()
    propainter_python = propainter / ".venv" / "bin" / "python"
    diffueraser_python = diffueraser / ".venv" / "bin" / "python"
    for required in (propainter_python, diffueraser_python, root / "selection50.json"):
        if not required.exists():
            raise RuntimeError(f"Required path does not exist: {required}")

    clips = load_selection(root)
    result_root = root / "results50"
    logs_root = root / "batch_logs"
    state_path = result_root / "batch_state.json"
    events_path = result_root / "batch_events.jsonl"
    state = load_state(state_path, args.gpu)

    say(f"Preparing aligned MP4 inputs for {len(clips)} clips")
    prepared: dict[str, tuple[Path, Path, tuple[int, int]]] = {}
    for index, clip in enumerate(clips, 1):
        video = clip["video_id"]
        expected = int(clip["frame_count"])
        prepared[video] = prepare_clip(root, video, expected)
        say(f"Prepared {index}/{len(clips)}: {video}")

    seed_test_results(root, clips, result_root)
    if args.prepare_only:
        say("Preparation complete; no GPU inference was started")
        return 0

    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    failures = 0

    for index, clip in enumerate(clips, 1):
        video = clip["video_id"]
        expected = int(clip["frame_count"])
        input_video, input_mask, size = prepared[video]
        width, height = size
        say(f"Clip {index}/{len(clips)}: {video} ({expected} frames, {width}x{height})")

        propainter_output = result_root / "propainter" / video / "inpaint_out.mp4"
        if valid_video(propainter_output, expected, size):
            mark(
                state,
                state_path,
                events_path,
                video,
                "propainter",
                "complete",
                action="skipped_valid",
                output=str(propainter_output),
                video_info=video_info(propainter_output),
            )
            say(f"ProPainter already complete: {video}")
        else:
            quarantine(propainter_output)
            wait_for_gpu(args.gpu, args.min_free_mb, args.max_utilization, args.poll_seconds)
            output_root = result_root / "propainter"
            command = [
                str(propainter_python),
                "-u",
                str(propainter / "inference_propainter.py"),
                "--video",
                str(root / "selected50" / "JPEGImages" / video),
                "--mask",
                str(root / "binary_masks" / video),
                "--output",
                str(output_root),
                "--height",
                str(height),
                "--width",
                str(width),
                "--mask_dilation",
                str(MASK_DILATION),
                "--save_fps",
                str(FPS),
                "--fp16",
            ]
            mark(state, state_path, events_path, video, "propainter", "running")
            code, elapsed = run_command(
                command,
                propainter,
                logs_root / "propainter" / f"{video}.log",
                environment,
            )
            if code == 0 and valid_video(propainter_output, expected, size):
                mark(
                    state,
                    state_path,
                    events_path,
                    video,
                    "propainter",
                    "complete",
                    action="generated",
                    elapsed_seconds=round(elapsed, 3),
                    output=str(propainter_output),
                    video_info=video_info(propainter_output),
                )
                say(f"ProPainter complete: {video} in {elapsed:.1f}s")
            else:
                failures += 1
                mark(
                    state,
                    state_path,
                    events_path,
                    video,
                    "propainter",
                    "failed",
                    returncode=code,
                    elapsed_seconds=round(elapsed, 3),
                    log=str(logs_root / "propainter" / f"{video}.log"),
                )
                say(f"ProPainter FAILED: {video}; see its log")

        diffueraser_output = result_root / "diffueraser" / video / "diffueraser_result.mp4"
        if valid_video(diffueraser_output, expected, size):
            mark(
                state,
                state_path,
                events_path,
                video,
                "diffueraser",
                "complete",
                action="skipped_valid",
                output=str(diffueraser_output),
                video_info=video_info(diffueraser_output),
            )
            say(f"DiffuEraser already complete: {video}")
        else:
            quarantine(diffueraser_output)
            wait_for_gpu(args.gpu, args.min_free_mb, args.max_utilization, args.poll_seconds)
            save_path = result_root / "diffueraser" / video
            save_path.mkdir(parents=True, exist_ok=True)
            duration_seconds = max(1, math.ceil(expected / FPS))
            command = [
                str(diffueraser_python),
                "-u",
                str(diffueraser / "run_diffueraser.py"),
                "--input_video",
                str(input_video),
                "--input_mask",
                str(input_mask),
                "--video_length",
                str(duration_seconds),
                "--mask_dilation_iter",
                str(MASK_DILATION),
                "--max_img_size",
                str(MAX_SIDE),
                "--save_path",
                str(save_path),
            ]
            mark(state, state_path, events_path, video, "diffueraser", "running")
            code, elapsed = run_command(
                command,
                diffueraser,
                logs_root / "diffueraser" / f"{video}.log",
                environment,
            )
            if code == 0 and valid_video(diffueraser_output, expected, size):
                mark(
                    state,
                    state_path,
                    events_path,
                    video,
                    "diffueraser",
                    "complete",
                    action="generated",
                    elapsed_seconds=round(elapsed, 3),
                    output=str(diffueraser_output),
                    video_info=video_info(diffueraser_output),
                )
                say(f"DiffuEraser complete: {video} in {elapsed:.1f}s")
            else:
                failures += 1
                mark(
                    state,
                    state_path,
                    events_path,
                    video,
                    "diffueraser",
                    "failed",
                    returncode=code,
                    elapsed_seconds=round(elapsed, 3),
                    log=str(logs_root / "diffueraser" / f"{video}.log"),
                )
                say(f"DiffuEraser FAILED: {video}; see its log")

    complete_propainter = 0
    complete_diffueraser = 0
    for clip in clips:
        video = clip["video_id"]
        expected = int(clip["frame_count"])
        _input_video, _input_mask, size = prepared[video]
        complete_propainter += int(
            valid_video(result_root / "propainter" / video / "inpaint_out.mp4", expected, size)
        )
        complete_diffueraser += int(
            valid_video(result_root / "diffueraser" / video / "diffueraser_result.mp4", expected, size)
        )

    state["summary"] = {
        "updated_at": timestamp(),
        "propainter_complete": complete_propainter,
        "diffueraser_complete": complete_diffueraser,
        "failures_this_run": failures,
    }
    atomic_json(state_path, state)
    say(
        f"Batch summary: ProPainter {complete_propainter}/{len(clips)}, "
        f"DiffuEraser {complete_diffueraser}/{len(clips)}, failures this run={failures}"
    )
    return 0 if complete_propainter == len(clips) and complete_diffueraser == len(clips) else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        say("Interrupted; completed outputs remain resumable")
        raise SystemExit(130)
