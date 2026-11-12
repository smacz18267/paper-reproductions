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

from mose_watermark_common import (
    atomic_json,
    decode_rgb,
    encode_rgb,
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--python", type=Path, default=Path(os.environ.get("PYTHON", "python3")))
    parser.add_argument("--device", default="0")
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
    generated = results / "input-watermarked_results.mp4"
    if generated.exists():
        generated.unlink()
    command = [
        str(args.python),
        str(source / "test.py"),
        "--model", "e2fgvi_hq",
        "--video", str(input_video),
        "--mask", str(mask_frames),
        "--ckpt", str(checkpoint),
        "--savefps", str(max(1, round(fps))),
    ]
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = args.device
    environment["MPLBACKEND"] = "Agg"
    log = run / "logs/e2fgvi.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with log.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"cwd": str(work), "command": command}, indent=2) + "\n")
        handle.flush()
        subprocess.run(command, cwd=work, env=environment, stdout=handle, stderr=subprocess.STDOUT, check=True)
    if not generated.is_file():
        raise RuntimeError(f"Official E2FGVI test did not create {generated}")
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
            "browser_preview": {
                "path": browser_video.name,
                "sha256": sha256(browser_video),
                "video": browser_info,
                "encoding": "H.264/yuv420p playback derivative; evaluation uses output.path",
            },
            "settings": {"visible_device": args.device, "model": "e2fgvi_hq", "save_fps": max(1, round(fps))},
            "command": command,
            "elapsed_seconds": round(time.monotonic() - started, 3),
        },
    )
    print(f"Completed E2FGVI-HQ for {sample['watermark_method']}/{sample['sample_id']}")


if __name__ == "__main__":
    main()
