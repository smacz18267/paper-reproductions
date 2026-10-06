#!/usr/bin/env python3
"""Validate the supplied demo and describe artifacts from real EffectErase runs."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

FRAMES = 81
WIDTH, HEIGHT = 832, 480
SAMPLES = (0, 40, 80)


def file_info(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return {"filename": path.name, "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def video_info(path, expected_frames=None, prefix_only=False):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot decode video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not math.isfinite(fps) or fps <= 0:
        cap.release()
        raise RuntimeError(f"Invalid video FPS: {path}")
    count, samples, size = 0, {}, None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            current_size = (frame.shape[1], frame.shape[0])
            if size is not None and size != current_size:
                raise RuntimeError(f"Frame dimensions changed in {path}")
            size = current_size
            if count in SAMPLES:
                samples[count] = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            count += 1
            if prefix_only and count == FRAMES:
                break
    finally:
        cap.release()
    if count < FRAMES:
        raise RuntimeError(f"Only {count} decodable frames in {path}; at least {FRAMES} required")
    if expected_frames is not None and count != expected_frames:
        raise RuntimeError(f"Expected {expected_frames} output frames, decoded {count}")
    metadata = {**file_info(path), "decoded_frames": count, "fps": fps,
                "width": size[0], "height": size[1], "decoded_prefix_only": prefix_only}
    return metadata, samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    args.run.mkdir(parents=True, exist_ok=True)
    source = args.source.resolve()
    video = source / "demo/FG_BG/WILD_ENV006_00042.mp4"
    mask = source / "demo/MASK/WILD_ENV006_00042.mp4"
    original, before = video_info(video, prefix_only=True)
    mask_info, masks = video_info(mask, prefix_only=True)
    if (original["width"], original["height"]) != (mask_info["width"], mask_info["height"]):
        raise RuntimeError("Demo video and mask dimensions differ")
    if not math.isclose(original["fps"], mask_info["fps"], abs_tol=0.05):
        raise RuntimeError("Demo video and mask FPS differ")
    if not np.any(np.asarray(masks[0]) > 0):
        raise RuntimeError("The first mask frame is empty")
    inputs = {"video": original, "mask": mask_info}
    (args.run / "input_metadata.json").write_text(json.dumps(inputs, indent=2) + "\n")
    if args.preflight:
        print("Input and mask passed decoding, dimensions, FPS and first-mask checks.")
        return

    output_path = args.run / "removed.mp4"
    output, after = video_info(output_path, expected_frames=FRAMES)
    if (output["width"], output["height"]) != (WIDTH, HEIGHT):
        raise RuntimeError(f"Unexpected output size: {output['width']}x{output['height']}")
    if not math.isclose(original["fps"], output["fps"], abs_tol=0.05):
        raise RuntimeError("Output FPS differs from the input video")

    tile_w, tile_h, label_h = 416, 240, 30
    comparison = Image.new("RGB", (tile_w * 3, (tile_h + label_h) * len(SAMPLES)), "#101010")
    draw = ImageDraw.Draw(comparison)
    for row, index in enumerate(SAMPLES):
        y = row * (tile_h + label_h)
        for col, (label, frame) in enumerate((("Input", before[index]), ("Object mask", masks[index]), ("EffectErase output", after[index]))):
            draw.text((col * tile_w + 8, y + 8), f"{label} | frame {index}", fill="white")
            comparison.paste(frame.resize((tile_w, tile_h), Image.Resampling.BILINEAR), (col * tile_w, y + label_h))
    comparison.save(args.run / "comparison.png")

    model_dir = source / "Wan-AI/Wan2.1-Fun-1.3B-InP"
    model_paths = [model_dir / name for name in (
        "diffusion_pytorch_model.safetensors", "Wan2.1_VAE.pth",
        "models_t5_umt5-xxl-enc-bf16.pth", "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth")]
    model_paths.append(source / "EffectErase.ckpt")
    print("Hashing the actual model files for the execution record...")
    manifest = {"task": "object and effect removal", "inputs": inputs,
                "settings": {"num_frames": FRAMES, "height": HEIGHT, "width": WIDTH,
                             "num_inference_steps": 50, "seed": 2025, "cfg": 1.0,
                             "lora_alpha": 1.0, "dtype": "bfloat16", "frame_interval": 1,
                             "teacache": False, "tiled": False},
                "output": output, "comparison": file_info(args.run / "comparison.png"),
                "models": [file_info(path) for path in model_paths]}
    (args.run / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.run / "README.md").write_text(
        "# EffectErase recorded inference\n\n"
        "The authors' supplied video and object mask were processed with their pretrained model. "
        "This is a demonstration of inference, with no training or quantitative benchmark evaluation.\n\n"
        "[Output video](removed.mp4) · [Comparison](comparison.png) · [Command](command.sh) · "
        "[Log](run.log) · [Environment](environment.freeze.txt) · [Hashes and video metadata](artifact_manifest.json)\n\n"
        f"Decoded output: {FRAMES} frames, {WIDTH} × {HEIGHT}, {output['fps']:g} FPS. "
        "Comparison rows show frames 0, 40 and 80.\n\n"
        "![Input, object mask, and EffectErase output](comparison.png)\n"
    )
    paper_readme = Path(__file__).resolve().parents[1] / "papers/effecterase/README.md"
    if paper_readme.is_file():
        text = paper_readme.read_text()
        start, end = "<!-- effecterase-results:start -->", "<!-- effecterase-results:end -->"
        if start in text and end in text:
            before_marker, rest = text.split(start, 1)
            _, after_marker = rest.split(end, 1)
            link = f"../../evidence/effecterase/recorded/{args.run.name}"
            paper_readme.write_text(before_marker + start + "\n\n"
                f"The latest recorded inference produced {FRAMES} frames at {WIDTH} × {HEIGHT} "
                f"and {output['fps']:g} FPS. [Video, comparison and execution evidence]({link}/README.md).\n\n"
                + end + after_marker)
            root_readme = paper_readme.parents[2] / "README.md"
            if root_readme.is_file():
                lines = root_readme.read_text().splitlines(keepends=True)
                for index, line in enumerate(lines):
                    if line.startswith("| EffectErase (CVPR 2026) |"):
                        lines[index] = "| EffectErase (CVPR 2026) | [Implementation and setup](papers/effecterase/README.md) | " + f"[Recorded inference](evidence/effecterase/recorded/{args.run.name}/README.md) |\n"
                root_readme.write_text("".join(lines))
    print(f"Verified all {FRAMES} output frames. Saved artifacts to {args.run}")


if __name__ == "__main__":
    main()
