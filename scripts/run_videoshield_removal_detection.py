#!/usr/bin/env python3
"""Run VideoShield extraction/localization on object-removal outputs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from mose_watermark_common import atomic_json, decode_rgb, encode_binary_masks, sha256, validate_video


NUM_FRAMES = 16
SIZE = (256, 256)
LOCALE_THRESHOLDS = {
    "loc4": [0.5703, 0.6640],
    "loc2": [0.5625, 0.6875],
    "loc1": [0.2500, 0.7500],
}


def parse_case(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError("Cases must use NAME=/path/to/video.mp4")
    return name, Path(path)


def binary_metrics(predicted: list[np.ndarray], expected: list[np.ndarray]) -> dict[str, float | int]:
    pred = np.stack([frame > 127 for frame in predicted]).flatten()
    gt = np.stack([frame > 127 for frame in expected]).flatten()
    true_positive = int(np.logical_and(pred, gt).sum())
    false_positive = int(np.logical_and(pred, np.logical_not(gt)).sum())
    false_negative = int(np.logical_and(np.logical_not(pred), gt).sum())
    true_negative = int(np.logical_and(np.logical_not(pred), np.logical_not(gt)).sum())
    f1 = 2 * true_positive / (2 * true_positive + false_positive + false_negative + 1e-6)
    precision = true_positive / (true_positive + false_positive + 1e-6)
    recall = true_positive / (true_positive + false_negative + 1e-6)
    iou = true_positive / (true_positive + false_positive + false_negative + 1e-6)
    specificity = true_negative / (true_negative + false_positive + 1e-6)
    # Upstream passes a binary predicted mask (rather than continuous scores)
    # to roc_auc_score. For a binary predictor, ROC AUC is exactly balanced
    # accuracy: the mean of sensitivity and specificity.
    auc = (recall + specificity) / 2
    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "iou": iou,
        "auc": auc,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "predicted_foreground_pixels": int(pred.sum()),
        "expected_foreground_pixels": int(gt.sum()),
    }


def read_masks(directory: Path) -> list[np.ndarray]:
    paths = sorted(directory.glob("*.png"))
    if len(paths) != NUM_FRAMES:
        raise RuntimeError(f"Expected {NUM_FRAMES} GT masks in {directory}, found {len(paths)}")
    masks = []
    for path in paths:
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Could not read {path}")
        masks.append(np.where(mask > 127, 255, 0).astype(np.uint8))
    return masks


def postprocess_mask(video_frames_mask: np.ndarray) -> list[np.ndarray]:
    masks = []
    for frame in video_frames_mask:
        gray = cv2.cvtColor(np.clip(frame * 255, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        # This exactly matches upstream video_mask_postprocess: values at or
        # below 128 are the predicted tamper region.
        masks.append(np.where(gray <= 128, 255, 0).astype(np.uint8))
    return masks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--watermark-info", required=True, type=Path)
    parser.add_argument("--gt-mask-frames", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--case", action="append", required=True, type=parse_case)
    parser.add_argument("--baseline-case", default="original")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-inversion-steps", type=int, default=25)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("VideoShield removal verification requires a CUDA GPU")
    source = args.source.resolve()
    sys.path.insert(0, str(source))
    os.chdir(source)
    from diffusers import DDIMInverseScheduler, TextToVideoSDPipeline
    from utils import get_video_latents, latents_to_video, transform_video
    from watermark import VideoShield

    pipeline = TextToVideoSDPipeline.from_pretrained(
        args.model_path.resolve(),
        torch_dtype=torch.float16,
    ).to(args.device)
    pipeline.safety_checker = None
    pipeline.scheduler = DDIMInverseScheduler.from_pretrained(
        args.model_path.resolve(),
        subfolder="scheduler",
    )
    watermark = VideoShield(
        ch_factor=1,
        hw_factor=4,
        frame_factor=8,
        height=SIZE[1] // 8,
        width=SIZE[0] // 8,
        num_frames=NUM_FRAMES,
        device=args.device,
    )
    watermark_info = torch.load(args.watermark_info.resolve(), map_location=args.device)
    for key in ("m", "watermark", "key", "nonce"):
        setattr(watermark, key, watermark_info[key])

    gt_masks = read_masks(args.gt_mask_frames)
    args.output.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {
        "status": "complete",
        "protocol": "VideoShield provided sample -> object removal -> VideoShield inversion/localization",
        "source_commit": subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
        ).strip(),
        "model_path": str(args.model_path.resolve()),
        "num_inversion_steps": args.num_inversion_steps,
        "localization_thresholds": LOCALE_THRESHOLDS,
        "cases": {},
    }

    for name, path in args.case:
        path = path.resolve()
        info = validate_video(path, frames=NUM_FRAMES, size=SIZE, fps=None)
        frames, rate = decode_rgb(path)
        normalized = [frame.astype(np.float32) / 255.0 for frame in frames]
        tensor = transform_video(normalized).to(pipeline.vae.dtype).to(args.device)
        with torch.inference_mode():
            latents = get_video_latents(pipeline.vae, tensor, sample=False, permute=True)
            reversed_latents = pipeline(
                prompt="",
                latents=latents,
                num_inference_steps=args.num_inversion_steps,
                guidance_scale=1.0,
                output_type="latent",
            ).frames
            watermark_accuracy = watermark.eval_watermark(reversed_latents)
            video_mask, localization_info = watermark.tamper_localization(
                reversed_latents,
                LOCALE_THRESHOLDS,
                hierarchical_level=3,
            )
            decoded_mask = latents_to_video(pipeline, video_mask.half())[0]
        predicted_masks = postprocess_mask(decoded_mask)
        case_dir = args.output / name
        case_dir.mkdir(parents=True, exist_ok=True)
        encode_binary_masks(case_dir / "pred-mask.mp4", predicted_masks, rate)
        frame_dir = case_dir / "pred-mask-frames"
        frame_dir.mkdir(exist_ok=True)
        for old in frame_dir.glob("*.png"):
            old.unlink()
        for index, mask in enumerate(predicted_masks):
            if not cv2.imwrite(str(frame_dir / f"{index:02d}.png"), mask):
                raise RuntimeError("Could not save predicted mask frame")

        metrics = {
            "status": "complete",
            "case": name,
            "input": info,
            "input_sha256": sha256(path),
            "watermark_accuracy": watermark_accuracy,
            "spatial_localization": None if name == args.baseline_case else binary_metrics(predicted_masks, gt_masks),
            "baseline_predicted_foreground_pixels": (
                int(sum((mask > 127).sum() for mask in predicted_masks))
                if name == args.baseline_case
                else None
            ),
            "localization_evidence": {
                key: {
                    "minimum": float(value.min().item()),
                    "maximum": float(value.max().item()),
                    "mean": float(value.float().mean().item()),
                }
                for key, value in localization_info.items()
            },
            "predicted_mask_sha256": sha256(case_dir / "pred-mask.mp4"),
        }
        atomic_json(case_dir / "metrics.json", metrics)
        summary["cases"][name] = metrics
        print(
            f"{name}: watermark_accuracy={watermark_accuracy:.4f}; "
            f"iou={metrics['spatial_localization']['iou']:.4f}"
            if metrics["spatial_localization"] is not None
            else f"{name}: watermark_accuracy={watermark_accuracy:.4f}",
            flush=True,
        )
        del tensor, latents, reversed_latents, video_mask, decoded_mask
        torch.cuda.empty_cache()

    atomic_json(args.output / "summary.json", summary)
    print(f"VideoShield removal verification written to {args.output}")


if __name__ == "__main__":
    main()
