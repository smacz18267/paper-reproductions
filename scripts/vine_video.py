#!/usr/bin/env python3
"""Use the official VINE-R image models on a video frame sequence.

The upstream encoder center-crops non-square images before embedding.  This
adapter performs that same crop on every frame and on the attack mask.  If the
fixed center square would exclude the selected object in every frame, the
square window moves only enough to preserve that object.  The geometry change
is recorded, and one fixed 100-bit text payload is used for the sequence. VINE
exposes a global message decoder but no native pixel tamper localizer.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from mose_watermark_common import atomic_json, decode_rgb, encode_binary_masks, encode_rgb, sha256, utc_now


def message_bits(text: str, device: torch.device) -> torch.Tensor:
    if len(text) > 12:
        raise ValueError("VINE text payloads are limited to 12 UTF-8 characters")
    data = bytearray(text + " " * (12 - len(text)), "utf-8")
    bits = [int(value) for value in "".join(format(value, "08b") for value in data)]
    bits.extend([0, 0, 0, 0])
    if len(bits) != 100:
        raise ValueError("This adapter requires an ASCII payload that produces exactly 100 bits")
    return torch.tensor(bits, dtype=torch.float32, device=device).unsqueeze(0)


def crop_geometry(
    height: int,
    width: int,
    *,
    left: int,
    top: int,
    strategy: str,
) -> dict[str, int | str]:
    side = min(height, width)
    return {
        "left": left,
        "top": top,
        "side": side,
        "source_width": width,
        "source_height": height,
        "strategy": strategy,
    }


def crop(array: np.ndarray, geometry: dict[str, int | str]) -> np.ndarray:
    left = int(geometry["left"])
    top = int(geometry["top"])
    side = int(geometry["side"])
    return array[top : top + side, left : left + side]


def choose_crop(masks: list[np.ndarray], height: int, width: int) -> dict[str, int | str]:
    side = min(height, width)
    center_left = (width - side) // 2
    center_top = (height - side) // 2
    center = crop_geometry(
        height,
        width,
        left=center_left,
        top=center_top,
        strategy="upstream_center_square",
    )
    if any(np.any(crop(mask, center) > 127) for mask in masks):
        return center

    foreground = np.zeros((height, width), dtype=bool)
    for mask in masks:
        foreground |= mask > 127
    ys, xs = np.nonzero(foreground)
    if not len(xs):
        raise RuntimeError("The source attack mask is empty in every frame")
    # Rare MOSE objects can sit entirely outside VINE's normal center square.
    # Move only the square crop window enough to retain that object's union,
    # keeping the released square-input encoder and the benchmark attack valid.
    center_x = float(xs.mean())
    center_y = float(ys.mean())
    left = int(np.clip(round(center_x - side / 2), 0, width - side))
    top = int(np.clip(round(center_y - side / 2), 0, height - side))
    fallback = crop_geometry(
        height,
        width,
        left=left,
        top=top,
        strategy="object_preserving_square_fallback",
    )
    if not any(np.any(crop(mask, fallback) > 127) for mask in masks):
        raise RuntimeError("Object-preserving VINE crop still produced an empty attack mask")
    return fallback


def embed(args: argparse.Namespace) -> None:
    source = args.source.resolve()
    sys.path.insert(0, str(source))
    from vine.src.vine_turbo import VINE_Turbo

    device = torch.device(args.device)
    encoder = VINE_Turbo.from_pretrained(args.encoder_model).to(device).eval()
    payload = message_bits(args.message, device)
    frames, rate = decode_rgb(args.input_video)
    mask_frames, mask_rate = decode_rgb(args.input_mask)
    if len(frames) != len(mask_frames) or abs(float(rate) - float(mask_rate)) > 0.01:
        raise RuntimeError("Input video and mask must have identical frame count and FPS")

    if not frames:
        raise RuntimeError("Input video is empty")
    height, width = frames[0].shape[:2]
    if any(frame.shape[:2] != (height, width) for frame in frames + mask_frames):
        raise RuntimeError("Video or mask dimensions change between frames")
    grayscale_masks = [cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in mask_frames]
    crop_record = choose_crop(grayscale_masks, height, width)

    output_frames: list[np.ndarray] = []
    output_masks: list[np.ndarray] = []
    resize_256 = transforms.Compose(
        [transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC), transforms.ToTensor()]
    )
    with torch.inference_mode():
        for frame, mask in zip(frames, grayscale_masks):
            cropped = crop(frame, crop_record)
            mask = crop(mask, crop_record)
            pil = Image.fromarray(cropped)
            resized = 2.0 * resize_256(pil).unsqueeze(0).to(device) - 1.0
            original = 2.0 * transforms.ToTensor()(pil).unsqueeze(0).to(device) - 1.0
            encoded_256 = encoder(resized, payload)
            residual = transforms.Resize(
                (int(crop_record["side"]), int(crop_record["side"])),
                interpolation=transforms.InterpolationMode.BICUBIC,
            )(encoded_256 - resized)
            encoded = torch.clamp((residual + original) * 0.5 + 0.5, 0, 1)
            array = torch.round(encoded[0].permute(1, 2, 0) * 255).byte().cpu().numpy()
            output_frames.append(array)
            output_masks.append(np.where(mask > 127, 255, 0).astype(np.uint8))

    encode_rgb(args.output_video, output_frames, rate, crf=18)
    encode_binary_masks(args.output_mask, output_masks, rate)
    atomic_json(
        args.output_video.with_suffix(".json"),
        {
            "status": "complete",
            "method": "VINE-R",
            "adapter": "official framewise encoder with upstream center-square crop",
            "crop": crop_record,
            "message": args.message,
            "frames": len(output_frames),
            "fps": float(rate),
            "source_commit": subprocess.check_output(
                ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
            ).strip(),
            "input_sha256": sha256(args.input_video),
            "output_sha256": sha256(args.output_video),
            "mask_sha256": sha256(args.output_mask),
            "completed_at": utc_now(),
        },
    )
    print(f"Embedded VINE-R into {len(output_frames)} frames: {args.output_video}")


def detect(args: argparse.Namespace) -> None:
    source = args.source.resolve()
    sys.path.insert(0, str(source))
    from vine.src.stega_encoder_decoder import CustomConvNeXt

    device = torch.device(args.device)
    decoder = CustomConvNeXt.from_pretrained(args.decoder_model).to(device).eval()
    target = message_bits(args.message, device)[0].byte().cpu().numpy()
    transform = transforms.Compose(
        [transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC), transforms.ToTensor()]
    )
    summary = {
        "status": "complete",
        "method": "VINE-R",
        "message": args.message,
        "localization": None,
        "localization_note": "The released VINE decoder recovers a global 100-bit payload and does not output a pixel mask.",
        "cases": {},
    }
    args.output.mkdir(parents=True, exist_ok=True)
    for case in args.case:
        name, separator, raw_path = case.partition("=")
        if not separator:
            raise ValueError("--case must use NAME=/path/to/video.mp4")
        path = Path(raw_path).resolve()
        frames, _rate = decode_rgb(path)
        predicted = []
        with torch.inference_mode():
            for frame in frames:
                image = transform(Image.fromarray(frame)).unsqueeze(0).to(device)
                bits = decoder(image)[0].detach().cpu().numpy()
                predicted.append(np.round(bits).astype(np.uint8))
        predictions = np.stack(predicted)
        majority = (predictions.sum(axis=0) > len(predictions) / 2).astype(np.uint8)
        metrics = {
            "input": str(path),
            "input_sha256": sha256(path),
            "frame_bit_accuracy_mean": float((predictions == target).mean(axis=1).mean()),
            "video_majority_bit_accuracy": float((majority == target).mean()),
            "predicted_bits": "".join(str(int(value)) for value in majority),
        }
        atomic_json(args.output / f"{name}.json", metrics)
        summary["cases"][name] = metrics
        print(f"{name}: majority bit accuracy={metrics['video_majority_bit_accuracy']:.4f}")
    summary["completed_at"] = utc_now()
    atomic_json(args.output / "summary.json", summary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--source", required=True, type=Path)
    common.add_argument("--message", default="Hello World!")
    common.add_argument("--device", default="cuda:0")

    embed_parser = subparsers.add_parser("embed", parents=[common])
    embed_parser.add_argument("--input-video", required=True, type=Path)
    embed_parser.add_argument("--input-mask", required=True, type=Path)
    embed_parser.add_argument("--output-video", required=True, type=Path)
    embed_parser.add_argument("--output-mask", required=True, type=Path)
    embed_parser.add_argument("--encoder-model", default="Shilin-LU/VINE-R-Enc")
    embed_parser.set_defaults(handler=embed)

    detect_parser = subparsers.add_parser("detect", parents=[common])
    detect_parser.add_argument("--output", required=True, type=Path)
    detect_parser.add_argument("--case", action="append", required=True)
    detect_parser.add_argument("--decoder-model", default="Shilin-LU/VINE-R-Dec")
    detect_parser.set_defaults(handler=detect)

    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
