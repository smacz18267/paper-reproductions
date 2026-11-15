#!/usr/bin/env python3
"""Run E2FGVI-HQ for all 50 fixed MOSE inputs of the four existing methods.

This augments the already-complete ProPainter, DiffuEraser, and EffectErase
batch without touching those artifacts. Detector/recovery commands are kept
separate because each watermark method uses its own official environment.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

METHODS = ("editguard", "omniguard", "videoseal", "safe-sora")


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--e2fgvi", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--e2fgvi-python", required=True, type=Path)
    parser.add_argument("--device", default="0")
    parser.add_argument("--method", action="append", choices=METHODS)
    parser.add_argument("--only", action="append")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    # Keep command discovery and --help usable on orchestration hosts that do
    # not have the media environment. Actual execution still fails fast here.
    from mose_watermark_common import (
        append_jsonl,
        copy_atomic,
        load_clips,
        method_dir,
        raw_mask_frames,
        utc_now,
        validate_video,
    )

    root = Path(__file__).resolve().parents[1]
    prepare = root / "scripts/prepare_watermark_removal_sample.py"
    execute = root / "scripts/run_prepared_e2fgvi.py"
    clips = load_clips(args.selection)
    if args.only:
        requested = set(args.only)
        clips = [clip for clip in clips if clip.clip_id in requested]
        missing = requested - {clip.clip_id for clip in clips}
        if missing:
            raise RuntimeError(f"Unknown --only clip(s): {sorted(missing)}")
    selected_methods = tuple(args.method or METHODS)
    events = args.output_root / "e2fgvi-full50-events.jsonl"
    failures = 0

    for method in selected_methods:
        for ordinal, clip in enumerate(clips, 1):
            destination = method_dir(args.output_root, clip.clip_id, method)
            final_video = destination / "e2fgvi.mp4"
            final_browser = destination / "e2fgvi-browser.mp4"
            final_json = destination / "e2fgvi.json"
            try:
                if final_video.is_file() and final_json.is_file() and not args.force:
                    source_info = validate_video(destination / "watermarked.mp4", fps=25)
                    validate_video(
                        final_video,
                        frames=int(source_info["frames"]),
                        size=(int(source_info["width"]), int(source_info["height"])),
                        fps=25,
                    )
                    print(f"[{ordinal}/{len(clips)}] skip {method}/e2fgvi {clip.clip_id}", flush=True)
                    continue

                source_video = destination / "watermarked.mp4"
                source_mask = (
                    destination / "gt-mask.mp4"
                    if method in {"editguard", "safe-sora"}
                    else raw_mask_frames(args.dataset_root, clip.clip_id)
                )
                run_root = destination / ".e2fgvi-full50"
                prepare_command = [
                    sys.executable,
                    str(prepare),
                    "prepare",
                    "--source-video",
                    str(source_video),
                    "--source-mask",
                    str(source_mask),
                    "--output",
                    str(run_root),
                    "--watermark-method",
                    method,
                    "--sample-id",
                    clip.clip_id,
                    "--sample-source",
                    "fixed MOSE full-50 selection",
                    "--source-kind",
                    "mose_framewise" if method != "videoseal" else "mose_post_hoc",
                ]
                if args.force or not (run_root / "sample.json").is_file():
                    if args.force:
                        prepare_command.append("--force")
                    run(prepare_command)
                execute_command = [
                    sys.executable,
                    str(execute),
                    "--source",
                    str(args.e2fgvi),
                    "--checkpoint",
                    str(args.checkpoint),
                    "--python",
                    str(args.e2fgvi_python),
                    "--run",
                    str(run_root),
                    "--device",
                    args.device,
                ]
                if args.force:
                    execute_command.append("--force")
                run(execute_command)
                copy_atomic(run_root / "e2fgvi.mp4", final_video)
                copy_atomic(run_root / "e2fgvi-browser.mp4", final_browser)
                metadata = json.loads((run_root / "e2fgvi.json").read_text())
                metadata["published_output"] = {
                    "video": final_video.name,
                    "browser_preview": final_browser.name,
                    "working_directory": str(run_root),
                }
                temporary = final_json.with_name(final_json.name + ".tmp")
                temporary.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
                temporary.replace(final_json)
                append_jsonl(
                    events,
                    {
                        "status": "complete",
                        "method": method,
                        "clip_id": clip.clip_id,
                        "remover": "e2fgvi",
                        "completed_at": utc_now(),
                    },
                )
                print(f"[{ordinal}/{len(clips)}] complete {method}/e2fgvi {clip.clip_id}", flush=True)
            except Exception as error:
                failures += 1
                append_jsonl(
                    events,
                    {
                        "status": "failed",
                        "method": method,
                        "clip_id": clip.clip_id,
                        "remover": "e2fgvi",
                        "error": repr(error),
                        "timestamp": utc_now(),
                    },
                )
                print(f"[{ordinal}/{len(clips)}] FAILED {method}/e2fgvi {clip.clip_id}: {error!r}", flush=True)
    if failures:
        raise SystemExit(f"E2FGVI-HQ failed for {failures} method/sample jobs")


if __name__ == "__main__":
    main()
