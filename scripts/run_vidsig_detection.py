#!/usr/bin/env python3
"""Run Video Signature's released TorchScript message decoder on videos."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import numpy as np
import torch
from torchvision import transforms

from mose_watermark_common import atomic_json, decode_rgb, sha256, utc_now


def parse_case(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator:
        raise argparse.ArgumentTypeError("Cases must use NAME=/path/to/video.mp4")
    return name, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--decoder", required=True, type=Path)
    parser.add_argument("--key", default="100011100001001101101100100011111101111110000000")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--case", action="append", required=True, type=parse_case)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if len(args.key) != 48 or set(args.key) - {"0", "1"}:
        raise ValueError("--key must contain exactly 48 binary digits")

    device = torch.device(args.device)
    decoder = torch.jit.load(str(args.decoder.resolve()), map_location=device).eval()
    target = torch.tensor([int(bit) for bit in args.key], device=device, dtype=torch.bool)
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    args.output.mkdir(parents=True, exist_ok=True)
    summary = {
        "status": "complete",
        "method": "VidSig / Video Signature",
        "protocol": "authors' generated sample -> removal -> released frame decoder and majority vote",
        "key": args.key,
        "decoder_sha256": sha256(args.decoder.resolve()),
        "source_commit": subprocess.check_output(
            ["git", "-C", str(args.source.resolve()), "rev-parse", "HEAD"], text=True
        ).strip(),
        "localization": None,
        "localization_note": "The released Video Signature detector recovers a global 48-bit key per frame.",
        "cases": {},
    }
    for name, raw_path in args.case:
        path = raw_path.resolve()
        frames, _rate = decode_rgb(path)
        tensor = torch.stack([transform(frame) for frame in frames]).to(device)
        with torch.inference_mode():
            logits = decoder(tensor)
        predicted = logits > 0
        frame_accuracy = (predicted == target.unsqueeze(0)).float().mean(dim=1)
        majority = predicted.sum(dim=0) > (predicted.shape[0] // 2)
        metrics = {
            "input": str(path),
            "input_sha256": sha256(path),
            "frames": len(frames),
            "frame_bit_accuracy_mean": float(frame_accuracy.mean().item()),
            "video_majority_bit_accuracy": float((majority == target).float().mean().item()),
            "matched_bits": int((majority == target).sum().item()),
            "predicted_bits": "".join(str(int(value)) for value in majority.int().cpu().tolist()),
        }
        atomic_json(args.output / f"{name}.json", metrics)
        summary["cases"][name] = metrics
        print(f"{name}: majority bit accuracy={metrics['video_majority_bit_accuracy']:.4f}")
    summary["completed_at"] = utc_now()
    atomic_json(args.output / "summary.json", summary)


if __name__ == "__main__":
    main()
