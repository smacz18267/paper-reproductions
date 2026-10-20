#!/usr/bin/env python3
"""Extract only a fixed MOSE selection from the official train.tar.gz."""

from __future__ import annotations

import argparse
import json
import shutil
import tarfile
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    selection = json.loads(args.selection.read_text())
    clips = selection.get("clips", [])
    expected = {item["video_id"]: int(item["frame_count"]) for item in clips}
    if len(expected) != 50:
        raise RuntimeError(f"Expected 50 unique clips, found {len(expected)}")
    args.output.mkdir(parents=True, exist_ok=True)

    found = {clip_id: 0 for clip_id in expected}
    visited = 0
    last_report = time.monotonic()
    with tarfile.open(args.archive, mode="r|gz") as archive:
        for member in archive:
            visited += 1
            if time.monotonic() - last_report >= 30:
                total = sum(found.values())
                print(f"Scanned {visited:,} archive members; extracted {total:,}/{sum(expected.values()):,} selected frames", flush=True)
                last_report = time.monotonic()
            if not member.isfile():
                continue
            parts = Path(member.name).parts
            try:
                image_index = parts.index("JPEGImages")
            except ValueError:
                continue
            if len(parts) != image_index + 3:
                continue
            clip_id, filename = parts[image_index + 1 :]
            if clip_id not in expected or Path(filename).suffix.lower() not in {".jpg", ".jpeg"}:
                continue
            if Path(filename).name != filename:
                raise RuntimeError(f"Unsafe archive member: {member.name}")
            try:
                frame_index = int(Path(filename).stem)
            except ValueError:
                continue
            # The official archive contains more JPEGs than the annotated
            # sequences used by the existing inpainting run.  Keep the exact
            # zero-based prefix that was paired with its masks and outputs.
            if frame_index >= expected[clip_id]:
                continue
            destination = args.output / clip_id / filename
            destination.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                raise RuntimeError(f"Could not read archive member: {member.name}")
            if destination.is_file() and destination.stat().st_size == member.size:
                source.close()
            else:
                temporary = destination.with_suffix(destination.suffix + ".part")
                with source, temporary.open("wb") as output:
                    shutil.copyfileobj(source, output)
                temporary.replace(destination)
            found[clip_id] += 1

    errors: list[str] = []
    for clip_id, frame_count in expected.items():
        frames = sorted((args.output / clip_id).glob("*.jpg"))
        expected_names = [f"{index:05d}.jpg" for index in range(frame_count)]
        if [frame.name for frame in frames] != expected_names:
            errors.append(f"{clip_id}: expected {frame_count} complete frames, found {len(frames)}")
    if errors:
        raise RuntimeError("Selected-frame validation failed:\n" + "\n".join(errors))
    print(f"Extracted and validated {len(expected)} clips with {sum(expected.values()):,} frames in {args.output}")


if __name__ == "__main__":
    main()
