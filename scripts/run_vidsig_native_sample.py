#!/usr/bin/env python3
"""Generate one native Video Signature sample from the authors' prompt data."""

from __future__ import annotations

import argparse
import copy
import subprocess
from pathlib import Path

import numpy as np
import torch
from diffusers import TextToVideoSDPipeline
from diffusers.utils import export_to_video
from torchvision import transforms

from mose_watermark_common import atomic_json, sha256, utc_now


def decoder_input(frames, device: torch.device) -> torch.Tensor:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return torch.stack([transform(frame) for frame in frames]).to(device)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--vae-checkpoint", required=True, type=Path)
    parser.add_argument("--message-decoder", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model-id", default="damo-vilab/text-to-video-ms-1.7b")
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument("--prompt-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--num-inference-steps", type=int)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--key", default="100011100001001101101100100011111101111110000000")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if len(args.key) != 48 or set(args.key) - {"0", "1"}:
        raise ValueError("--key must contain exactly 48 binary digits")
    prompt_file = (args.prompt_file or (args.source / "prompt.txt")).resolve()
    prompts = [line.strip() for line in prompt_file.read_text().splitlines() if line.strip()]
    prompt = prompts[args.prompt_index]
    device = torch.device(args.device)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    decoder = torch.jit.load(str(args.message_decoder.resolve()), map_location=device).eval()
    pipeline = TextToVideoSDPipeline.from_pretrained(
        args.model_id, torch_dtype=torch.float16
    ).to(device)
    watermarked_vae = copy.deepcopy(pipeline.vae)
    watermarked_vae.load_state_dict(torch.load(args.vae_checkpoint.resolve(), map_location="cpu"))
    watermarked_vae.to(device=device, dtype=torch.float16).eval()

    call = {
        "prompt": prompt,
        "height": args.height,
        "width": args.width,
        "num_frames": args.num_frames,
        "generator": torch.Generator(device=device).manual_seed(args.seed),
    }
    if args.num_inference_steps is not None:
        call["num_inference_steps"] = args.num_inference_steps
    clean_frames = pipeline(**call).frames[0]
    export_to_video(clean_frames, str(output / "clean.mp4"), fps=args.fps)
    np.save(output / "clean.npy", clean_frames)

    pipeline.vae = watermarked_vae
    call["generator"] = torch.Generator(device=device).manual_seed(args.seed)
    watermarked_frames = pipeline(**call).frames[0]
    export_to_video(watermarked_frames, str(output / "watermarked.mp4"), fps=args.fps)
    np.save(output / "watermarked.npy", watermarked_frames)

    key = torch.tensor([int(bit) for bit in args.key], device=device, dtype=torch.bool)
    with torch.inference_mode():
        logits = decoder(decoder_input(watermarked_frames, device))
    predicted = logits > 0
    majority = predicted.sum(dim=0) > (predicted.shape[0] // 2)
    atomic_json(
        output / "generation.json",
        {
            "status": "complete",
            "method": "VidSig / Video Signature",
            "protocol": "authors' ModelScope pipeline, fine-tuned VAE, prompt file, key, and seed",
            "prompt": prompt,
            "prompt_index": args.prompt_index,
            "seed": args.seed,
            "key": args.key,
            "frames": args.num_frames,
            "size": [args.width, args.height],
            "fps": args.fps,
            "native_majority_bit_accuracy": float((majority == key).float().mean().item()),
            "source_commit": subprocess.check_output(
                ["git", "-C", str(args.source.resolve()), "rev-parse", "HEAD"], text=True
            ).strip(),
            "vae_checkpoint_sha256": sha256(args.vae_checkpoint.resolve()),
            "message_decoder_sha256": sha256(args.message_decoder.resolve()),
            "watermarked_sha256": sha256(output / "watermarked.mp4"),
            "completed_at": utc_now(),
        },
    )
    print(f"Generated native VidSig sample: {output / 'watermarked.mp4'}")


if __name__ == "__main__":
    main()
