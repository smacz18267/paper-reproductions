#!/usr/bin/env python3
"""Validate and summarize the VideoShield removal-verification artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from mose_watermark_common import atomic_json, decode_rgb, sha256, validate_video


EXPECTED_FRAMES = 16
EXPECTED_SIZE = (256, 256)


def decoded_masks(path: Path) -> list[np.ndarray]:
    frames, _rate = decode_rgb(path)
    return [cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) > 127 for frame in frames]


def global_iou(predicted: Path, expected: Path) -> float:
    pred = decoded_masks(predicted)
    gt = decoded_masks(expected)
    if len(pred) != len(gt):
        raise RuntimeError(f"Mask length mismatch: {predicted}, {expected}")
    intersection = sum(int(np.logical_and(p, g).sum()) for p, g in zip(pred, gt))
    union = sum(int(np.logical_or(p, g).sum()) for p, g in zip(pred, gt))
    return 1.0 if union == 0 else intersection / union


def labelled(frame: np.ndarray, label: str) -> np.ndarray:
    image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    canvas = np.zeros((image.shape[0] + 42, image.shape[1], 3), dtype=np.uint8)
    canvas[42:] = image
    cv2.putText(canvas, label, (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def middle_frame(path: Path) -> np.ndarray:
    frames, _rate = decode_rgb(path)
    if len(frames) != EXPECTED_FRAMES:
        raise RuntimeError(f"Expected {EXPECTED_FRAMES} frames in {path}, found {len(frames)}")
    return frames[len(frames) // 2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    run = args.run.resolve()

    videos = {
        "input": run / "input-watermarked.mp4",
        "gt_mask": run / "gt-mask.mp4",
        "propainter": run / "propainter.mp4",
        "diffueraser": run / "diffueraser.mp4",
        "original_pred_mask": run / "detection/original/pred-mask.mp4",
        "propainter_pred_mask": run / "detection/propainter/pred-mask.mp4",
        "diffueraser_pred_mask": run / "detection/diffueraser/pred-mask.mp4",
    }
    inventory = {
        name: validate_video(path, frames=EXPECTED_FRAMES, size=EXPECTED_SIZE, fps=None)
        for name, path in videos.items()
    }
    input_fps = float(inventory["input"]["fps"])
    for name, info in inventory.items():
        if abs(float(info["fps"]) - input_fps) > 0.01:
            raise RuntimeError(f"{name} FPS differs from the VideoShield sample")

    detection = json.loads((run / "detection/summary.json").read_text())
    if detection.get("status") != "complete" or set(detection.get("cases", {})) != {
        "original", "propainter", "diffueraser"
    }:
        raise RuntimeError("Detection summary is incomplete")

    recomputed = {}
    for method in ("propainter", "diffueraser"):
        value = global_iou(videos[f"{method}_pred_mask"], videos["gt_mask"])
        reported = float(detection["cases"][method]["spatial_localization"]["iou"])
        if abs(value - reported) > 1e-9:
            raise RuntimeError(f"{method} IoU mismatch: reported={reported}, recomputed={value}")
        recomputed[method] = value

    blank = np.zeros((EXPECTED_SIZE[1], EXPECTED_SIZE[0], 3), dtype=np.uint8)
    sheet = np.vstack(
        [
            np.hstack(
                [
                    labelled(middle_frame(videos["input"]), "Watermarked input"),
                    labelled(middle_frame(videos["gt_mask"]), "Official mask"),
                    labelled(middle_frame(videos["propainter"]), "ProPainter"),
                    labelled(middle_frame(videos["diffueraser"]), "DiffuEraser"),
                ]
            ),
            np.hstack(
                [
                    labelled(middle_frame(videos["original_pred_mask"]), "Detector: original"),
                    labelled(cv2.cvtColor(blank, cv2.COLOR_BGR2RGB), "White = detected tamper"),
                    labelled(middle_frame(videos["propainter_pred_mask"]), "Detector: ProPainter"),
                    labelled(middle_frame(videos["diffueraser_pred_mask"]), "Detector: DiffuEraser"),
                ]
            ),
        ]
    )
    if not cv2.imwrite(str(run / "comparison.png"), sheet):
        raise RuntimeError("Could not write comparison.png")

    cases = detection["cases"]
    aggregate = {
        "status": "complete",
        "sample": "VideoShield ModelScope/a_red_panda_eating_leaves",
        "frames": EXPECTED_FRAMES,
        "size": list(EXPECTED_SIZE),
        "fps": input_fps,
        "methods": {
            method: {
                "watermark_accuracy": cases[method]["watermark_accuracy"],
                "spatial_localization": cases[method]["spatial_localization"],
                "independently_recomputed_iou": recomputed[method],
            }
            for method in ("propainter", "diffueraser")
        },
        "baseline": {
            "watermark_accuracy": cases["original"]["watermark_accuracy"],
            "predicted_foreground_pixels": cases["original"]["baseline_predicted_foreground_pixels"],
        },
        "videos": {name: str(path.relative_to(run)) for name, path in videos.items()},
        "inventory": inventory,
    }
    atomic_json(run / "metrics.json", aggregate)

    table_rows = []
    for method, label in (("propainter", "ProPainter"), ("diffueraser", "DiffuEraser")):
        item = aggregate["methods"][method]
        spatial = item["spatial_localization"]
        table_rows.append(
            f"| {label} | {item['watermark_accuracy']:.4f} | {spatial['f1']:.4f} | "
            f"{spatial['precision']:.4f} | {spatial['recall']:.4f} | {spatial['iou']:.4f} |"
        )
    readme = (
        "# VideoShield removal verification\n\n"
        "This run uses VideoShield's provided `a red panda eating leaves` ModelScope sample "
        "and the exact spatial mask produced by the authors' sample workflow. The same "
        "watermarked frames and mask were passed to both methods. ProPainter is the upstream "
        "prior generated inside the official DiffuEraser pipeline; DiffuEraser is its final output.\n\n"
        "[Comparison](comparison.png) · [Watermarked input](input-watermarked.mp4) · "
        "[GT mask](gt-mask.mp4) · [ProPainter](propainter.mp4) · "
        "[DiffuEraser](diffueraser.mp4)\n\n"
        "![Input, removal outputs, and VideoShield predicted masks](comparison.png)\n\n"
        "## Results\n\n"
        f"The untouched sample baseline recovered the watermark with accuracy "
        f"**{aggregate['baseline']['watermark_accuracy']:.4f}**.\n\n"
        "| Removal method | Watermark accuracy | Spatial F1 | Precision | Recall | IoU |\n"
        "| --- | ---: | ---: | ---: | ---: | ---: |\n"
        + "\n".join(table_rows)
        + "\n\n"
        "IoU is independently recomputed from the committed predicted-mask and official "
        "GT-mask videos. White pixels denote the removed/detected region.\n\n"
        "## Compatibility note\n\n"
        "The provided VideoShield sample contains 16 frames, while DiffuEraser requires at "
        "least 22. Inference therefore repeats only the final frame and mask to 24 frames. "
        "Both reported outputs are trimmed back to the original 16 genuine sample frames "
        "before VideoShield detection and scoring.\n"
    )
    (run / "README.md").write_text(readme)

    manifest = {}
    excluded = {"artifact-manifest.json"}
    for path in sorted(run.rglob("*")):
        if path.is_file() and path.name not in excluded and "work" not in path.relative_to(run).parts:
            manifest[str(path.relative_to(run))] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    atomic_json(run / "artifact-manifest.json", manifest)
    print(
        f"Validated VideoShield removal run: ProPainter IoU={recomputed['propainter']:.4f}, "
        f"DiffuEraser IoU={recomputed['diffueraser']:.4f}"
    )


if __name__ == "__main__":
    main()
