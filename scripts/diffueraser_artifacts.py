#!/usr/bin/env python3
"""Validate DiffuEraser inputs and outputs and record evidence from the actual run."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

SECONDS = 10
MAX_SIZE = 960


def file_info(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return {"filename": path.name, "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def video_info(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not math.isfinite(fps) or fps <= 0:
        cap.release()
        raise RuntimeError(f"Invalid frame rate: {path}")
    count, size = 0, None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            current_size = (frame.shape[1], frame.shape[0])
            if size is not None and size != current_size:
                raise RuntimeError(f"Frame dimensions changed: {path}")
            size = current_size
            count += 1
    finally:
        cap.release()
    if count == 0:
        raise RuntimeError(f"Video has no decodable frames: {path}")
    return {**file_info(path), "decoded_frames": count, "fps": fps,
            "width": size[0], "height": size[1]}


def sampled_frames(path, indices):
    cap = cv2.VideoCapture(str(path))
    result = {}
    try:
        for index in range(max(indices) + 1):
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"Cannot decode frame {index}: {path}")
            if index in indices:
                result[index] = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
    return result


def expected_size(width, height):
    largest = max(width, height)
    if largest < 256 or largest > 4096:
        raise RuntimeError("Input's largest side must be between 256 and 4096 pixels")
    if largest > MAX_SIZE:
        ratio = largest / MAX_SIZE
        width, height = int(width / ratio), int(height / ratio)
    width, height = width - width % 8, height - height % 8
    if width <= 0 or height <= 0:
        raise RuntimeError("Invalid resized frame dimensions")
    return width, height


def validate_output(metadata, frames, fps, size, label):
    if metadata["decoded_frames"] != frames:
        raise RuntimeError(f"{label}: expected {frames} frames, decoded {metadata['decoded_frames']}")
    if not math.isclose(metadata["fps"], fps, rel_tol=0, abs_tol=0.001):
        raise RuntimeError(f"{label}: output frame rate differs from input")
    if (metadata["width"], metadata["height"]) != size:
        raise RuntimeError(f"{label}: unexpected output size")


def expected_prior_size(width, height):
    # Upstream ProPainter uses resize_ratio=0.6 and writes back to the
    # requested pre-rounding dimensions, while DiffuEraser upsamples the prior.
    ratio = 0.6 * min(1.0, MAX_SIZE / max(width, height))
    return int(ratio * width), int(ratio * height)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    run, source = args.run.resolve(), args.source.resolve()
    video, mask = run / "input_video.mp4", run / "input_mask.mp4"
    original, mask_meta = video_info(video), video_info(mask)
    if original["fps"] != mask_meta["fps"]:
        raise RuntimeError("Input video and mask frame rates differ; upstream requires an exact match")
    # Both upstream stages use the common leading sequence. The bundled input
    # and mask may have different tails; compare only frames present in both,
    # bounded by the requested maximum duration. Do not pad or resample masks.
    frames = min(original["decoded_frames"], mask_meta["decoded_frames"],
                 int(SECONDS * original["fps"]))
    if frames < 22:
        raise RuntimeError("DiffuEraser needs at least 22 aligned frames")
    size = expected_size(original["width"], original["height"])
    inputs = {"video": original, "mask": mask_meta, "expected_output_frames": frames,
              "expected_output_size": list(size), "selection": "common leading frames, up to 10 seconds",
              "mask_resampling": "nearest neighbour to processed video dimensions, as in upstream",
              "excluded_video_tail_frames": original["decoded_frames"] - frames,
              "excluded_mask_tail_frames": mask_meta["decoded_frames"] - frames}
    (run / "input_metadata.json").write_text(json.dumps(inputs, indent=2) + "\n")
    samples = sorted({0, frames // 2, frames - 1})
    masks = sampled_frames(mask, samples)
    if not any(np.any(np.asarray(frame) > 0) for frame in masks.values()):
        raise RuntimeError("All sampled masks are empty")
    downloads = json.loads((source / "model_downloads.json").read_text())
    model_paths = [source / entry["destination"] / name for key, entry in downloads.items()
                   if key != "propainter" for name in entry["files"]]
    model_paths += [source / "weights/propainter" / name for name in (
        "ProPainter.pth", "raft-things.pth", "recurrent_flow_completion.pth")]
    for path in model_paths:
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Missing model component: {path}")
    if args.preflight:
        print(f"Decoded input lengths: video={original['decoded_frames']}, mask={mask_meta['decoded_frames']}.")
        if original["decoded_frames"] != mask_meta["decoded_frames"]:
            print(f"Using the first {frames} frames present in both inputs, matching upstream inference.")
        if (original["width"], original["height"]) != (mask_meta["width"], mask_meta["height"]):
            print(f"Upstream will resize the mask to {size[0]}x{size[1]} with nearest-neighbour sampling.")
        print(f"Inputs passed: {frames} aligned frames, output size {size[0]}x{size[1]}.")
        return

    output_path, prior_path = run / "diffueraser_result.mp4", run / "propainter_prior.mp4"
    output, prior = video_info(output_path), video_info(prior_path)
    validate_output(output, frames, original["fps"], size, "DiffuEraser")
    # ProPainter's inclusive end_pts can include one boundary frame; DiffuEraser
    # then truncates to int(video_length * FPS). Preserve this original behaviour.
    if prior["decoded_frames"] not in {frames, frames + 1}:
        raise RuntimeError("ProPainter prior has an unexpected decoded frame count")
    if not math.isclose(prior["fps"], original["fps"], rel_tol=0, abs_tol=0.001):
        raise RuntimeError("ProPainter prior frame rate differs from input")
    prior_size = expected_prior_size(original["width"], original["height"])
    if (prior["width"], prior["height"]) != prior_size:
        raise RuntimeError("ProPainter prior has an unexpected output size")
    before = sampled_frames(video, samples)
    after, priors = sampled_frames(output_path, samples), sampled_frames(prior_path, samples)
    tile_w, tile_h, label_h = 320, max(1, round(320 * size[1] / size[0])), 30
    comparison = Image.new("RGB", (tile_w * 4, (tile_h + label_h) * len(samples)), "#101010")
    draw = ImageDraw.Draw(comparison)
    for row, index in enumerate(samples):
        y = row * (tile_h + label_h)
        for col, (label, image) in enumerate((("Input", before[index]), ("Mask", masks[index]),
                                              ("ProPainter prior", priors[index]), ("DiffuEraser", after[index]))):
            draw.text((col * tile_w + 8, y + 8), f"{label} | frame {index}", fill="white")
            comparison.paste(image.resize((tile_w, tile_h), Image.Resampling.BILINEAR),
                             (col * tile_w, y + label_h))
    comparison.save(run / "comparison.png")
    print("Hashing the actual downloaded model files...", flush=True)
    models = []
    for path in model_paths:
        models.append({**file_info(path), "path": str(path.relative_to(source))})
    helper_root = Path(__file__).resolve().parent
    helpers = [{**file_info(helper_root / name), "path": "scripts/" + name} for name in (
        "setup_diffueraser.sh", "download_diffueraser_models.py", "capture_diffueraser.sh",
        "run_diffueraser_recorded.py", "diffueraser_artifacts.py")]
    manifest = {"task": "pretrained video inpainting demonstration", "inputs": inputs,
                "settings": {"example": "example3", "video_length": SECONDS, "max_img_size": MAX_SIZE,
                             "seed": 2025, "mask_dilation_iter": 8, "pcm": "2-Step", "dtype": "float16",
                             "guidance_scale": 0, "ref_stride": 10, "neighbor_length": 10, "subvideo_length": 50},
                "output": output, "prior": prior, "comparison": file_info(run / "comparison.png"),
                "models": models, "reproduction_scripts": helpers}
    (run / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (run / "README.md").write_text(
        "# DiffuEraser recorded inference\n\n"
        "Pretrained inference on the authors' supplied example3. No training or quantitative benchmark evaluation.\n\n"
        "[Input](input_video.mp4) · [Mask](input_mask.mp4) · [ProPainter prior](propainter_prior.mp4) · "
        "[DiffuEraser output](diffueraser_result.mp4) · [Command](command.sh) · [Log](run.log) · "
        "[Environment](environment.freeze.txt) · [Hashes and metadata](artifact_manifest.json)\n\n"
        f"Final output: {frames} frames, {size[0]} × {size[1]}, {output['fps']:g} FPS. "
        f"Comparison frames: {', '.join(map(str, samples))}.\n\n"
        "![Input, mask, ProPainter prior and DiffuEraser output](comparison.png)\n")
    paper = helper_root.parent / "papers/diffueraser/README.md"
    text = paper.read_text()
    start, end = "<!-- diffueraser-results:start -->", "<!-- diffueraser-results:end -->"
    before_marker, rest = text.split(start, 1)
    _, after_marker = rest.split(end, 1)
    paper.write_text(before_marker + start + "\n\n" +
                     f"Recorded inference: {frames} frames at {size[0]} × {size[1]}, {output['fps']:g} FPS. " +
                     f"[Outputs and execution evidence](../../evidence/diffueraser/recorded/{run.name}/README.md).\n\n" +
                     end + after_marker)
    root_readme = helper_root.parent / "README.md"
    lines = root_readme.read_text().splitlines(keepends=True)
    for index, line in enumerate(lines):
        if line.startswith("| DiffuEraser (2025) |"):
            lines[index] = "| DiffuEraser (2025) | [Implementation and setup](papers/diffueraser/README.md) | " + \
                           f"[Recorded inference](evidence/diffueraser/recorded/{run.name}/README.md) |\n"
    root_readme.write_text("".join(lines))
    print(f"Verified output and prior. Evidence: {run}")


if __name__ == "__main__":
    main()
