#!/usr/bin/env python3
"""Validate VideoShield outputs and extract reported metrics from logs."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import av

NUMBER = r"([0-9]+(?:\.[0-9]+)?)"


def inspect_video(path: Path) -> dict:
    container = av.open(str(path))
    stream = container.streams.video[0]
    rate = stream.average_rate or stream.guessed_rate
    frames = sum(1 for _ in container.decode(video=0))
    container.close()
    return {"frames": frames, "width": stream.width, "height": stream.height, "fps": float(rate) if rate else None}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def one(pattern: str, text: str, label: str) -> float:
    match = re.search(pattern, text)
    if not match:
        raise RuntimeError(f"Could not find {label} in recorded log")
    return float(match.group(1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    prompt_dir = run / "results" / "modelscope" / "a_red_panda_eating_leaves"
    spatial_dir = prompt_dir / "wm_spatial_tampered"
    videos = {
        "watermarked": prompt_dir / "wm.mp4",
        "tampered": spatial_dir / "wm_tampered.mp4",
        "mask_ground_truth": spatial_dir / "mask_gt.mp4",
        "mask_predicted": spatial_dir / "mask_pred.mp4",
    }
    logs = {name: run / f"{name}.log" for name in ("embed", "temporal", "spatial")}
    required = [*videos.values(), *logs.values(), prompt_dir / "wm_info.bin"]
    for path in required:
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Missing or empty artifact: {path}")

    embed_text = logs["embed"].read_text(errors="replace")
    temporal_text = logs["temporal"].read_text(errors="replace")
    spatial_text = logs["spatial"].read_text(errors="replace")
    extraction = one(r"Watermark extraction accuracy:\s*" + NUMBER, embed_text, "extraction accuracy")
    temporal = [float(value) for value in re.findall(r"Localization accuracy:\s*" + NUMBER, temporal_text)]
    if len(temporal) != 3:
        raise RuntimeError(f"Expected three temporal localization metrics, found {len(temporal)}")
    spatial_match = re.search(
        r"Localization performance:\s*f1=" + NUMBER
        + r",\s*precision=" + NUMBER
        + r",\s*recall=" + NUMBER
        + r",\s*iou=" + NUMBER
        + r",\s*auc=" + NUMBER,
        spatial_text,
    )
    if not spatial_match:
        raise RuntimeError("Could not parse spatial localization metrics")
    spatial = dict(zip(("f1", "precision", "recall", "iou", "auc"), map(float, spatial_match.groups())))
    metrics = {
        "watermark_extraction_accuracy": extraction,
        "temporal_localization_accuracy": {
            "frame_swap": temporal[0],
            "frame_insert": temporal[1],
            "frame_drop": temporal[2],
        },
        "spatial_localization": spatial,
    }
    for value in [extraction, *temporal, *spatial.values()]:
        if not 0.0 <= value <= 1.0:
            raise RuntimeError(f"Metric outside [0, 1]: {value}")

    video_info = {name: inspect_video(path) for name, path in videos.items()}
    for name, info in video_info.items():
        if info["frames"] != 16 or info["width"] != 256 or info["height"] != 256:
            raise RuntimeError(f"Unexpected {name} video: {info}")
    (run / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    artifact_paths = [*videos.values(), *logs.values(), prompt_dir / "wm_info.bin", run / "metrics.json"]
    manifest = {
        "videos": video_info,
        "metrics": metrics,
        "files": {
            str(path.relative_to(run)): {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in artifact_paths
        },
    }
    (run / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (run / "README.md").write_text(
        "# Recorded VideoShield demonstration\n\n"
        f"The ModelScope path produced 16 watermarked frames at 256 × 256. Watermark "
        f"extraction accuracy was **{extraction:.3f}**. Temporal localization accuracies "
        f"(swap/insert/drop) were **{temporal[0]:.3f} / {temporal[1]:.3f} / {temporal[2]:.3f}**. "
        f"Spatial localization F1 was **{spatial['f1']:.3f}** with IoU **{spatial['iou']:.3f}**.\n"
    )
    print("VideoShield videos and metrics verified.")


if __name__ == "__main__":
    main()
