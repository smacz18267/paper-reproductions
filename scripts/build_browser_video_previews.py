#!/usr/bin/env python3
"""Create H.264 viewing copies for canonical MP4s that browsers cannot decode.

The evaluated source is never replaced. Each derivative records both hashes,
frame count, dimensions, FPS, and codec so notebooks can show the derivative
while continuing to link to the exact evaluated artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2

from watermark_artifact_validation import require_materialized


BROWSER_CODEC = "avc1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fourcc(value: float) -> str:
    integer = int(value)
    return "".join(chr((integer >> (8 * index)) & 0xFF) for index in range(4)).strip("\x00")


def inspect(path: Path) -> dict[str, int | float | str]:
    require_materialized(path, "video")
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    result = {
        "codec": fourcc(capture.get(cv2.CAP_PROP_FOURCC)),
        "frames": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": float(capture.get(cv2.CAP_PROP_FPS)),
    }
    capture.release()
    if result["frames"] <= 0 or result["width"] <= 0 or result["height"] <= 0 or result["fps"] <= 0:
        raise RuntimeError(f"Invalid video metadata for {path}: {result}")
    return result


def validate_preview(source: Path, destination: Path) -> None:
    metadata = destination.with_suffix(".json")
    require_materialized(destination, "browser preview")
    if not metadata.is_file():
        raise RuntimeError(f"Missing browser-preview metadata: {metadata}")
    record = json.loads(metadata.read_text())
    if record.get("source", {}).get("sha256") != sha256(source):
        raise RuntimeError(f"Browser preview is stale for source: {source}")
    if record.get("preview", {}).get("sha256") != sha256(destination):
        raise RuntimeError(f"Browser-preview checksum mismatch: {destination}")
    source_info = inspect(source)
    preview_info = inspect(destination)
    for key in ("frames", "width", "height"):
        if preview_info[key] != source_info[key]:
            raise RuntimeError(f"Preview changed {key}: source={source_info[key]}, preview={preview_info[key]}")
    if abs(float(preview_info["fps"]) - float(source_info["fps"])) > 0.01:
        raise RuntimeError(f"Preview changed FPS: source={source_info['fps']}, preview={preview_info['fps']}")
    if str(preview_info["codec"]).lower() not in {"avc1", "h264"}:
        raise RuntimeError(f"Preview is not H.264: {preview_info}")


def transcode(source: Path, destination: Path, force: bool) -> None:
    source_info = inspect(source)
    metadata = destination.with_suffix(".json")
    source_hash = sha256(source)
    if destination.is_file() and metadata.is_file() and not force:
        existing = json.loads(metadata.read_text())
        if existing.get("source", {}).get("sha256") == source_hash:
            preview_info = inspect(destination)
            if str(preview_info["codec"]).lower() in {"avc1", "h264"}:
                print(f"skip current preview: {destination}")
                return

    capture = cv2.VideoCapture(str(source))
    temporary = destination.with_name(destination.stem + ".part.mp4")
    if temporary.exists():
        temporary.unlink()
    writer = cv2.VideoWriter(
        str(temporary),
        cv2.VideoWriter_fourcc(*BROWSER_CODEC),
        float(source_info["fps"]),
        (int(source_info["width"]), int(source_info["height"])),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(
            "OpenCV could not open an H.264 encoder. Run this command in the maintained notebook "
            "environment or install an OpenCV build with avc1 encoding support."
        )
    written = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        writer.write(frame)
        written += 1
    capture.release()
    writer.release()
    if written != source_info["frames"]:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Decoded {written} frames from {source}; expected {source_info['frames']}")
    temporary.replace(destination)
    preview_info = inspect(destination)
    for key in ("frames", "width", "height"):
        if preview_info[key] != source_info[key]:
            raise RuntimeError(f"Preview changed {key}: source={source_info[key]}, preview={preview_info[key]}")
    if abs(float(preview_info["fps"]) - float(source_info["fps"])) > 0.01:
        raise RuntimeError(f"Preview changed FPS: source={source_info['fps']}, preview={preview_info['fps']}")
    if str(preview_info["codec"]).lower() not in {"avc1", "h264"}:
        raise RuntimeError(f"Preview is not H.264: {preview_info}")
    payload = {
        "status": "complete",
        "semantics": "browser-playback derivative only; evaluation uses the source video",
        "source": {"path": source.name, "sha256": source_hash, "video": source_info},
        "preview": {"path": destination.name, "sha256": sha256(destination), "video": preview_info},
        "completed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    temporary_metadata = metadata.with_name(metadata.name + ".tmp")
    temporary_metadata.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary_metadata.replace(metadata)
    print(f"created {destination} from {source_info['codec']} source")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=Path("evidence/four-inpainter-matrix.json"))
    parser.add_argument("--stem", action="append", default=["diffueraser"])
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--check", action="store_true", help="Validate required derivatives without writing files")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    matrix_path = args.matrix if args.matrix.is_absolute() else root / args.matrix
    matrix = json.loads(matrix_path.read_text())
    completed = 0
    for method, record in matrix["methods"].items():
        if record.get("status") != "complete":
            continue
        attack = root / record["attack_directory"]
        for stem in dict.fromkeys(args.stem):
            source = attack / f"{stem}.mp4"
            if not source.is_file():
                raise FileNotFoundError(f"Missing canonical source for {method}/{stem}: {source}")
            info = inspect(source)
            if str(info["codec"]).lower() in {"avc1", "h264"}:
                print(f"already browser-safe: {source}")
                continue
            destination = attack / f"{stem}-browser.mp4"
            if args.check:
                validate_preview(source, destination)
                print(f"validated browser preview: {destination}")
            else:
                transcode(source, destination, args.force)
            completed += 1
    print(f"Validated or created {completed} browser-preview derivative(s)")


if __name__ == "__main__":
    main()
