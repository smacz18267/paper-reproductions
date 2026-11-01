#!/usr/bin/env python3
"""Validate EditGuard and OmniGuard media used by the three-sample notebooks."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2


SAMPLES = ("02b72b8d", "0ab75563", "0b02886a")
METHODS = ("editguard", "omniguard")


def inspect(path: Path, *, mask: bool = False) -> dict[str, float | int]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {path}")
    result = {
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": float(capture.get(cv2.CAP_PROP_FPS)),
        "frames": 0,
        "foreground_pixels": 0,
        "bytes": path.stat().st_size,
    }
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        result["frames"] += 1
        if mask:
            result["foreground_pixels"] += int((cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) > 127).sum())
    capture.release()
    return result


def validate_video(
    path: Path,
    *,
    frames: int,
    size: tuple[int, int],
    mask: bool = False,
    foreground: bool = False,
) -> dict[str, float | int]:
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"Missing or empty video: {path}")
    info = inspect(path, mask=mask)
    if info["frames"] != frames:
        raise RuntimeError(f"{path}: expected {frames} frames, got {info['frames']}")
    if (info["width"], info["height"]) != size:
        raise RuntimeError(f"{path}: expected {size}, got {(info['width'], info['height'])}")
    if abs(float(info["fps"]) - 25.0) > 0.01:
        raise RuntimeError(f"{path}: expected 25 FPS, got {info['fps']}")
    if foreground and int(info["foreground_pixels"]) == 0:
        raise RuntimeError(f"{path}: expected a non-empty mask")
    return info


def displayed_iou(predicted: Path, expected: Path) -> float:
    pred_capture = cv2.VideoCapture(str(predicted))
    gt_capture = cv2.VideoCapture(str(expected))
    intersection = 0
    union = 0
    while True:
        pred_ok, pred_frame = pred_capture.read()
        gt_ok, gt_frame = gt_capture.read()
        if pred_ok != gt_ok:
            raise RuntimeError(f"Mask streams differ in length: {predicted}, {expected}")
        if not pred_ok:
            break
        pred = cv2.cvtColor(pred_frame, cv2.COLOR_BGR2GRAY) > 127
        truth = cv2.cvtColor(gt_frame, cv2.COLOR_BGR2GRAY) > 127
        intersection += int((pred & truth).sum())
        union += int((pred | truth).sum())
    pred_capture.release()
    gt_capture.release()
    return 1.0 if union == 0 else intersection / union


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("review_root", type=Path)
    args = parser.parse_args()
    root = args.review_root.resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    records = {item["clip_id"]: item for item in manifest["samples"]}
    report: dict[str, object] = {
        "status": "complete",
        "validated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "samples": list(SAMPLES),
        "methods": {},
        "videos": 0,
        "decoded_frames": 0,
    }

    for method in METHODS:
        report["methods"][method] = {}
        for sample_id in SAMPLES:
            record = records[sample_id]
            frames = int(record["frame_count"])
            if method == "editguard":
                size = (512, 512)
                names = ["aligned-original.mp4", "gt-mask.mp4", "watermarked.mp4"]
            else:
                size = (int(record["width"]), int(record["height"]))
                names = ["watermarked.mp4"]
            names.extend(
                [
                    "propainter.mp4",
                    "diffueraser.mp4",
                    "effective-gt-mask-propainter.mp4",
                    "effective-gt-mask-diffueraser.mp4",
                    "pred-mask-propainter.mp4",
                    "pred-mask-diffueraser.mp4",
                ]
            )
            directory = root / "media" / sample_id / "methods" / method
            localization = json.loads((directory / "localization.json").read_text())
            embed = json.loads((directory / "embed.json").read_text())
            if embed.get("status") != "complete" or localization.get("status") != "complete":
                raise RuntimeError(f"{method}/{sample_id}: incomplete metadata")
            media: dict[str, object] = {}
            for name in names:
                is_mask = "mask" in name
                info = validate_video(
                    directory / name,
                    frames=frames,
                    size=size,
                    mask=is_mask,
                    foreground=name.startswith("effective-gt") or name == "gt-mask.mp4",
                )
                media[name] = info
                report["videos"] += 1
                report["decoded_frames"] += int(info["frames"])
            for remover in ("propainter", "diffueraser"):
                measured = displayed_iou(
                    directory / f"pred-mask-{remover}.mp4",
                    directory / f"effective-gt-mask-{remover}.mp4",
                )
                claimed = float(localization[remover]["global_iou"])
                if abs(measured - claimed) > 1e-9:
                    raise RuntimeError(
                        f"{method}/{sample_id}/{remover}: displayed IoU {measured} != {claimed}"
                    )
                if method == "editguard":
                    accuracy = float(localization[remover]["mean_message_bit_accuracy"])
                    if not 0.0 <= accuracy <= 1.0:
                        raise RuntimeError(f"Invalid EditGuard bit accuracy: {accuracy}")
            report["methods"][method][sample_id] = {
                "status": "valid",
                "size": list(size),
                "frames": frames,
                "media": media,
            }
            print(f"valid {method}/{sample_id}")

    destination = root / "image-method-notebook-validation.json"
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"Validated {report['videos']} videos and {report['decoded_frames']} decoded frames")


if __name__ == "__main__":
    main()
