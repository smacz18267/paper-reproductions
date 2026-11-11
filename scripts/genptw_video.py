#!/usr/bin/env python3
"""Framewise GenPTW embedding, extraction, and localization for MOSE video.

GenPTW's released training/evaluation path embeds into existing COCO images by
VAE-encoding them and decoding the same latents through ``decode_wm``.  This
adapter applies that released post-generation path independently to video
frames.  It uses the paper's 512 x 512 image protocol, so both video and mask
are center-cropped to a square and resized to 512 before the two video
inpainting methods are run.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from diffusers import AutoencoderKL
from safetensors.torch import load_file

from mose_watermark_common import atomic_json, decode_rgb, encode_binary_masks, encode_rgb, sha256, utc_now


RESOLUTION = 512
LATENT_SCALE = 0.18215


def message_tensor(value: str, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if len(value) != 64 or set(value) - {"0", "1"}:
        raise ValueError("--message must contain exactly 64 binary digits")
    return torch.tensor([[int(bit) for bit in value]], device=device, dtype=dtype)


def align_frame(frame: np.ndarray, *, mask: bool = False) -> tuple[np.ndarray, dict[str, int]]:
    height, width = frame.shape[:2]
    side = min(height, width)
    left = (width - side) // 2
    top = (height - side) // 2
    cropped = frame[top : top + side, left : left + side]
    interpolation = cv2.INTER_NEAREST if mask else cv2.INTER_LANCZOS4
    resized = cv2.resize(cropped, (RESOLUTION, RESOLUTION), interpolation=interpolation)
    return resized, {
        "left": left,
        "top": top,
        "side": side,
        "source_width": width,
        "source_height": height,
        "output_width": RESOLUTION,
        "output_height": RESOLUTION,
    }


def dtype_from_name(name: str) -> torch.dtype:
    return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[name]


def load_models(args: argparse.Namespace, *, with_detection: bool):
    source = args.source.resolve()
    sys.path.insert(0, str(source))
    from aaai_final_adapter import Decoder, inject_wmadapter

    device = torch.device(args.device)
    dtype = dtype_from_name(args.dtype)
    vae = AutoencoderKL.from_pretrained(args.pretrained_vae.resolve())
    vae = inject_wmadapter(vae, bit_dim=64, image_size=RESOLUTION)
    vae.load_state_dict(load_file(str(args.checkpoint / "diffusion_pytorch_model.safetensors")), strict=False)
    vae.eval().requires_grad_(False).to(device=device, dtype=dtype)
    if not with_detection:
        return vae, None, None, device, dtype

    from Localizer.model import Localizer

    decoder = Decoder(input_channels=3, output_length=64, image_size=RESOLUTION)
    decoder.load_state_dict(torch.load(args.checkpoint / "msg_decoder.pth", map_location="cpu"))
    localizer = Localizer(
        conv_pretrain=True,
        conv_ckpt=args.convnext_checkpoint.resolve(),
        image_size=RESOLUTION,
    )
    localizer.load_state_dict(torch.load(args.checkpoint / "localizer.pth", map_location="cpu"))
    decoder.eval().requires_grad_(False).to(device=device, dtype=dtype)
    localizer.eval().requires_grad_(False).to(device=device, dtype=dtype)
    return vae, decoder, localizer, device, dtype


def image_tensor(frame: np.ndarray, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    value = torch.from_numpy(frame.copy()).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=dtype)
    return value / 127.5 - 1.0


def tensor_image(value: torch.Tensor) -> np.ndarray:
    image = (value[0].float().clamp(-1, 1) / 2 + 0.5).permute(1, 2, 0)
    return torch.round(image * 255).byte().cpu().numpy()


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
    vae, _decoder, _localizer, device, dtype = load_models(args, with_detection=False)
    target = message_tensor(args.message, device, dtype)
    frames, rate = decode_rgb(args.input_video)
    mask_frames, mask_rate = decode_rgb(args.input_mask)
    if len(frames) != len(mask_frames) or abs(float(rate) - float(mask_rate)) > 0.01:
        raise RuntimeError("Input video and mask must have identical frame count and FPS")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    output_frames = []
    output_masks = []
    geometry = None
    with torch.inference_mode():
        for frame, mask_frame in zip(frames, mask_frames):
            aligned, current_geometry = align_frame(frame)
            gray = cv2.cvtColor(mask_frame, cv2.COLOR_RGB2GRAY)
            aligned_mask, mask_geometry = align_frame(gray, mask=True)
            if current_geometry != mask_geometry:
                raise RuntimeError("Video and mask geometry differs")
            if geometry is None:
                geometry = current_geometry
            image = image_tensor(aligned, device, dtype)
            latent = vae.encode(image).latent_dist.sample() * LATENT_SCALE
            watermarked = vae.decode_wm(latent / LATENT_SCALE, target, return_dict=False)[0]
            output_frames.append(tensor_image(watermarked))
            output_masks.append(aligned_mask > 127)
    encode_rgb(args.output_video, output_frames, rate, crf=18)
    encode_binary_masks(args.output_mask, output_masks, rate)
    atomic_json(
        args.output_video.with_suffix(".json"),
        {
            "status": "complete",
            "method": "GenPTW",
            "adapter": "released post-generation VAE encode/decode_wm path, independently per frame",
            "geometry": geometry,
            "message": args.message,
            "seed": args.seed,
            "source_commit": subprocess.check_output(
                ["git", "-C", str(args.source.resolve()), "rev-parse", "HEAD"], text=True
            ).strip(),
            "checkpoint_files": {
                name: sha256(args.checkpoint / name)
                for name in ("diffusion_pytorch_model.safetensors", "msg_decoder.pth", "localizer.pth")
            },
            "output_sha256": sha256(args.output_video),
            "mask_sha256": sha256(args.output_mask),
            "completed_at": utc_now(),
        },
    )
    print(f"Embedded GenPTW into {len(output_frames)} aligned frames")


def detect(args: argparse.Namespace) -> None:
    _vae, decoder, localizer, device, dtype = load_models(args, with_detection=True)
    target = message_tensor(args.message, device, dtype)
    gt_frames, _ = decode_rgb(args.gt_mask)
    gt_masks = [cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) > 127 for frame in gt_frames]
    args.output.mkdir(parents=True, exist_ok=True)
    summary = {"status": "complete", "method": "GenPTW", "message": args.message, "cases": {}}
    for case in args.case:
        name, separator, raw_path = case.partition("=")
        if not separator:
            raise ValueError("--case must use NAME=/path/to/video.mp4")
        path = Path(raw_path).resolve()
        frames, rate = decode_rgb(path)
        if len(frames) != len(gt_masks):
            raise RuntimeError(f"{name}: frame count differs from aligned GT mask")
        bit_predictions = []
        predicted_masks = []
        with torch.inference_mode():
            for frame, gt_mask in zip(frames, gt_masks):
                image = image_tensor(frame, device, dtype)
                mask = torch.from_numpy(gt_mask.copy()).unsqueeze(0).unsqueeze(0).to(device=device, dtype=dtype)
                detected, watermark_features = decoder(image, mask, 0)
                localized = localizer(image, mask, watermark_features)["pred_mask"]
                bit_predictions.append((detected[0] > 0.5).byte().cpu().numpy())
                predicted_masks.append((localized[0, 0] > 0.5).byte().cpu().numpy())
        predictions = np.stack(bit_predictions)
        target_np = target[0].byte().cpu().numpy()
        majority = (predictions.sum(axis=0) > len(predictions) / 2).astype(np.uint8)
        case_dir = args.output / name
        case_dir.mkdir(parents=True, exist_ok=True)
        encode_binary_masks(case_dir / "pred-mask.mp4", predicted_masks, rate)
        metrics = {
            "input": str(path),
            "input_sha256": sha256(path),
            "frame_bit_accuracy_mean": float((predictions == target_np).mean(axis=1).mean()),
            "video_majority_bit_accuracy": float((majority == target_np).mean()),
            "predicted_bits": "".join(str(int(value)) for value in majority),
            "spatial_localization": pixel_metrics(predicted_masks, gt_masks),
            "pred_mask_sha256": sha256(case_dir / "pred-mask.mp4"),
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
    common.add_argument("--pretrained-vae", required=True, type=Path)
    common.add_argument("--convnext-checkpoint", required=True, type=Path)
    common.add_argument("--device", default="cuda:0")
    common.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    common.add_argument(
        "--message",
        default="1010010101000011110101010111111010100101010000111101010101111110",
    )

    embed_parser = subparsers.add_parser("embed", parents=[common])
    embed_parser.add_argument("--input-video", required=True, type=Path)
    embed_parser.add_argument("--input-mask", required=True, type=Path)
    embed_parser.add_argument("--output-video", required=True, type=Path)
    embed_parser.add_argument("--output-mask", required=True, type=Path)
    embed_parser.add_argument("--seed", type=int, default=2026)
    embed_parser.set_defaults(handler=embed)

    detect_parser = subparsers.add_parser("detect", parents=[common])
    detect_parser.add_argument("--gt-mask", required=True, type=Path)
    detect_parser.add_argument("--output", required=True, type=Path)
    detect_parser.add_argument("--case", action="append", required=True)
    detect_parser.set_defaults(handler=detect)

    args = parser.parse_args()
    args.checkpoint = args.checkpoint.resolve()
    args.handler(args)


if __name__ == "__main__":
    main()
