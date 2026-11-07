#!/usr/bin/env python3
"""Stage one attacked SIGMark video for the authors' extraction entry point.

The native extractor expects the same output-tree layout created by
``main.py --mode=gen``.  This helper copies the maintained watermark state and
ground-truth message, substitutes exactly one video, and writes an identity
disturbance map so the released extraction path handles the sequence without
changing its frame order.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np

from mose_watermark_common import atomic_json, copy_atomic, decode_rgb, sha256, utc_now


def exactly_one(paths: list[Path], label: str) -> Path:
    if len(paths) != 1:
        raise RuntimeError(f"Expected one {label}, found {len(paths)}: {paths}")
    return paths[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated-output", required=True, type=Path)
    parser.add_argument("--sample-key", required=True, help="Relative video key without .mp4, e.g. Dimension/name")
    parser.add_argument("--replacement-video", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    source = args.generated_output.resolve()
    destination = args.destination.resolve()
    if destination.exists():
        if not args.force:
            raise FileExistsError(f"{destination} exists; pass --force to replace it")
        shutil.rmtree(destination)
    destination.mkdir(parents=True)

    maintained = exactly_one(list(source.glob("*-maintained_info.pkl")), "maintained-info file")
    gt_path = exactly_one(list(source.glob("*-gt_watermark_messages.npz")), "GT-message file")
    gt_all = dict(np.load(gt_path))
    if args.sample_key not in gt_all:
        raise KeyError(f"{args.sample_key!r} is not present in {gt_path}")
    copy_atomic(maintained, destination / maintained.name)
    np.savez(destination / gt_path.name, **{args.sample_key: gt_all[args.sample_key]})

    frames, _rate = decode_rgb(args.replacement_video.resolve())
    relative_video = Path(args.sample_key + ".mp4")
    copy_atomic(args.replacement_video.resolve(), destination / relative_video)
    np.savez(
        destination / "disturbance_info.npz",
        **{str(relative_video): np.arange(len(frames), dtype=np.int64)},
    )
    atomic_json(
        destination / "staged-case.json",
        {
            "status": "ready",
            "sample_key": args.sample_key,
            "frames": len(frames),
            "replacement_video": str(args.replacement_video.resolve()),
            "replacement_sha256": sha256(args.replacement_video.resolve()),
            "maintained_info_sha256": sha256(maintained),
            "gt_message_archive_sha256": sha256(destination / gt_path.name),
            "staged_at": utc_now(),
        },
    )
    print(f"Staged SIGMark detector case at {destination}")


if __name__ == "__main__":
    main()
