#!/usr/bin/env python3
"""Framewise Watermark Anything embedding and native detection for video.

WAM is an image method, so the video adapter fixes one 32-bit message and
applies the unchanged official image preprocessing independently to each
frame.  Detection reports native watermark-presence masks, recovered bits,
and a tamper mask defined as absence of the full-frame watermark.
"""

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
import torch.nn.functional as functional
from PIL import Image

from mose_watermark_common import atomic_json, decode_rgb, encode_binary_masks, encode_rgb, sha256, utc_now


def message_tensor(value: str, device: torch.device) -> torch.Tensor:
    if len(value) != 32 or set(value) - {"0", "1"}:
        raise ValueError("--message must contain exactly 32 binary digits")
    return torch.tensor([[int(bit) for bit in value]], dtype=torch.float32, device=device)


def load_wam(source: Path, checkpoint: Path, params: Path, device: torch.device):
    checkpoint = checkpoint.resolve()
    params = params.resolve()
    sys.path.insert(0, str(source))
    old_cwd = Path.cwd()
    os.chdir(source)
    try:
        from notebooks.inference_utils import load_model_from_checkpoint

        model = load_model_from_checkpoint(str(params), str(checkpoint))
    finally:
        os.chdir(old_cwd)
    return model.to(device).eval()


def frame_tensor(frame: np.ndarray, device: torch.device) -> torch.Tensor:
    from watermark_anything.data.transforms import default_transform

    return default_transform(Image.fromarray(frame)).unsqueeze(0).to(device)


def tensor_frame(value: torch.Tensor) -> np.ndarray:
    from watermark_anything.data.transforms import unnormalize_img

    array = unnormalize_img(value).clamp(0, 1)[0].permute(1, 2, 0)
    return torch.round(array * 255).byte().cpu().numpy()


def pixel_metrics(predicted: list[np.ndarray], expected: list[np.ndarray]) -> dict[str, float | int]:
    pred = np.stack(predicted).astype(bool).reshape(-1)
    gt = np.stack(expected).astype(bool).reshape(-1)
    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())
    tn = int(np.logical_and(~pred, ~gt).sum())
    return {
        "iou": tp / (tp + fp + fn + 1e-6),
        "f1": 2 * tp / (2 * tp + fp + fn + 1e-6),
        "precision": tp / (tp + fp + 1e-6),
        "recall": tp / (tp + fn + 1e-6),
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
    }


def embed(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    source = args.source.resolve()
    sys.path.insert(0, str(source))
    model = load_wam(source, args.checkpoint, args.params, device)
    message = message_tensor(args.message, device)
    frames, rate = decode_rgb(args.input_video)
    encoded = []
    with torch.inference_mode():
        for frame in frames:
            image = frame_tensor(frame, device)
            encoded.append(tensor_frame(model.embed(image, message)["imgs_w"]))
    encode_rgb(args.output_video, encoded, rate, crf=18)
    atomic_json(
        args.output_video.with_suffix(".json"),
        {
            "status": "complete",
            "method": "Watermark Anything",
            "adapter": "official image embedder independently per frame; one fixed 32-bit message",
            "source_commit": subprocess.check_output(
                ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
            ).strip(),
            "message": args.message,
            "frames": len(encoded),
            "fps": float(rate),
            "input_sha256": sha256(args.input_video),
            "output_sha256": sha256(args.output_video),
            "completed_at": utc_now(),
        },
    )
    print(f"Embedded WAM into {len(encoded)} frames: {args.output_video}")


def detect(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    source = args.source.resolve()
    sys.path.insert(0, str(source))
    from watermark_anything.data.metrics import msg_predict_inference

    model = load_wam(source, args.checkpoint, args.params, device)
    target = message_tensor(args.message, device)
    gt_frames, _ = decode_rgb(args.gt_mask)
    gt_masks = [cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) > 127 for frame in gt_frames]
    summary = {
        "status": "complete",
        "method": "Watermark Anything",
        "message": args.message,
        "tamper_definition": "logical absence of the native WAM watermark-presence mask at threshold 0.5",
        "cases": {},
    }
    args.output.mkdir(parents=True, exist_ok=True)
    for case in args.case:
        name, separator, raw_path = case.partition("=")
        if not separator:
            raise ValueError("--case must use NAME=/path/to/video.mp4")
        path = Path(raw_path).resolve()
        frames, rate = decode_rgb(path)
        if len(frames) != len(gt_masks):
            raise RuntimeError(f"{name}: expected {len(gt_masks)} frames, found {len(frames)}")
        predicted_bits = []
        predicted_tamper = []
        per_frame_accuracy = []
        with torch.inference_mode():
            for frame in frames:
                image = frame_tensor(frame, device)
                preds = model.detect(image)["preds"]
                presence = torch.sigmoid(preds[:, :1])
                bits = msg_predict_inference(preds[:, 1:], presence).float()
                predicted_bits.append(bits[0].cpu().numpy().astype(np.uint8))
                per_frame_accuracy.append(float((bits == target).float().mean().item()))
                absence = functional.interpolate(
                    (presence <= 0.5).float(),
                    size=frame.shape[:2],
                    mode="nearest",
                )[0, 0]
                predicted_tamper.append(absence.byte().cpu().numpy())
        majority = (np.stack(predicted_bits).sum(axis=0) > len(predicted_bits) / 2).astype(np.uint8)
        target_np = target[0].byte().cpu().numpy()
        case_dir = args.output / name
        case_dir.mkdir(parents=True, exist_ok=True)
        encode_binary_masks(case_dir / "pred-tamper.mp4", predicted_tamper, rate)
        metrics = {
            "input": str(path),
            "input_sha256": sha256(path),
            "frame_bit_accuracy_mean": float(np.mean(per_frame_accuracy)),
            "video_majority_bit_accuracy": float((majority == target_np).mean()),
            "predicted_message": "".join(str(int(value)) for value in majority),
            "spatial_localization": pixel_metrics(predicted_tamper, gt_masks),
            "pred_tamper_sha256": sha256(case_dir / "pred-tamper.mp4"),
        }
        atomic_json(case_dir / "metrics.json", metrics)
        summary["cases"][name] = metrics
        print(
            f"{name}: majority bit accuracy={metrics['video_majority_bit_accuracy']:.4f}; "
            f"IoU={metrics['spatial_localization']['iou']:.4f}"
        )
    summary["completed_at"] = utc_now()
    atomic_json(args.output / "summary.json", summary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--source", required=True, type=Path)
    common.add_argument("--checkpoint", required=True, type=Path)
    common.add_argument("--params", required=True, type=Path)
    common.add_argument("--message", default="10100101010000111101010101111110")
    common.add_argument("--device", default="cuda:0")

    embed_parser = subparsers.add_parser("embed", parents=[common])
    embed_parser.add_argument("--input-video", required=True, type=Path)
    embed_parser.add_argument("--output-video", required=True, type=Path)
    embed_parser.set_defaults(handler=embed)

    detect_parser = subparsers.add_parser("detect", parents=[common])
    detect_parser.add_argument("--gt-mask", required=True, type=Path)
    detect_parser.add_argument("--output", required=True, type=Path)
    detect_parser.add_argument("--case", action="append", required=True)
    detect_parser.set_defaults(handler=detect)

    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
