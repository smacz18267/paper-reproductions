#!/usr/bin/env python3
"""Test whether VideoShield can be adapted to an existing MOSE video.

VideoShield normally samples an encrypted, truncated-Gaussian starting noise and
uses it to generate a new video.  The released code has no pixel-domain embedder.
This experiment therefore tests two explicit alternatives:

1. ``canonical``: pass VideoShield's sampled starting noise to the pipeline
   unchanged (the released embedding path).
2. ``alpha-*``: DDIM-invert an existing MOSE clip, impose the VideoShield sign
   pattern while preserving the inverted noise magnitudes, and interpolate
   between the original and sign-imposed noises.

The output records watermark recovery and frame-aligned PSNR/SSIM, making the
trade-off between detection and preservation directly inspectable.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from diffusers import DDIMInverseScheduler, TextToVideoSDPipeline
from diffusers.utils import export_to_video


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_sampled_video(path: Path, count: int, size: int) -> tuple[np.ndarray, dict]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {path}")
    source_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames: list[np.ndarray] = []
    while True:
        ok, bgr = capture.read()
        if not ok:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        frames.append(rgb)
    capture.release()
    if len(frames) < count:
        raise RuntimeError(f"{path} has {len(frames)} frames; {count} are required")
    indices = np.linspace(0, len(frames) - 1, count).round().astype(int)
    sampled = np.stack(
        [cv2.resize(frames[index], (size, size), interpolation=cv2.INTER_AREA) for index in indices]
    )
    metadata = {
        "source_frame_count_reported": source_count,
        "source_frame_count_decoded": len(frames),
        "source_fps": source_fps,
        "source_width": source_width,
        "source_height": source_height,
        "sample_indices": indices.tolist(),
        "adapter_width": size,
        "adapter_height": size,
    }
    return sampled.astype(np.float32) / 255.0, metadata


def frame_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict:
    reference = np.asarray(reference, dtype=np.float32)
    candidate = np.asarray(candidate, dtype=np.float32)
    if reference.shape != candidate.shape:
        raise RuntimeError(f"Metric shape mismatch: {reference.shape} != {candidate.shape}")
    candidate = np.clip(candidate, 0.0, 1.0)
    mse = float(np.mean((reference - candidate) ** 2))
    psnr = float("inf") if mse == 0 else -10.0 * math.log10(mse)
    c1 = 0.01**2
    c2 = 0.03**2
    per_frame_ssim = []
    for reference_frame, candidate_frame in zip(reference, candidate):
        mu_reference = cv2.GaussianBlur(reference_frame, (11, 11), 1.5)
        mu_candidate = cv2.GaussianBlur(candidate_frame, (11, 11), 1.5)
        sigma_reference = cv2.GaussianBlur(reference_frame**2, (11, 11), 1.5) - mu_reference**2
        sigma_candidate = cv2.GaussianBlur(candidate_frame**2, (11, 11), 1.5) - mu_candidate**2
        covariance = (
            cv2.GaussianBlur(reference_frame * candidate_frame, (11, 11), 1.5)
            - mu_reference * mu_candidate
        )
        ssim_map = ((2 * mu_reference * mu_candidate + c1) * (2 * covariance + c2)) / (
            (mu_reference**2 + mu_candidate**2 + c1)
            * (sigma_reference + sigma_candidate + c2)
        )
        per_frame_ssim.append(float(np.mean(ssim_map[5:-5, 5:-5])))
    return {
        "mse": mse,
        "psnr_db": psnr,
        "ssim": float(np.mean(per_frame_ssim)),
    }


def make_contact_sheet(videos: list[tuple[str, np.ndarray]], path: Path) -> None:
    selected = (0, len(videos[0][1]) // 2, len(videos[0][1]) - 1)
    rows: list[np.ndarray] = []
    for label, video in videos:
        panels: list[np.ndarray] = []
        for index in selected:
            frame = np.clip(video[index] * 255.0, 0, 255).astype(np.uint8)
            panel = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.rectangle(panel, (0, 0), (panel.shape[1], 27), (0, 0, 0), -1)
            cv2.putText(
                panel,
                f"{label} frame {index}",
                (7, 19),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            panels.append(panel)
        rows.append(np.concatenate(panels, axis=1))
    cv2.imwrite(str(path), np.concatenate(rows, axis=0))


def latent_sign_match(latents: torch.Tensor, target: torch.Tensor) -> float:
    return float(((latents > 0) == (target > 0)).float().mean().item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--input-video", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=(0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.50, 0.75, 1.0),
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("This experiment requires CUDA")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.source.resolve()))
    from utils import get_video_latents, transform_video
    from watermark import VideoShield

    seed_everything(args.seed)
    input_frames, input_metadata = read_sampled_video(args.input_video, args.frames, args.size)
    export_to_video(input_frames, output_video_path=str(args.output_dir / "input-sampled.mp4"), fps=args.fps)

    pipe = TextToVideoSDPipeline.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
    ).to(args.device)
    pipe.safety_checker = None
    forward_scheduler = pipe.scheduler
    inverse_scheduler = DDIMInverseScheduler.from_pretrained(args.model_path, subfolder="scheduler")

    watermark = VideoShield(
        ch_factor=1,
        hw_factor=4,
        frame_factor=8,
        height=args.size // 8,
        width=args.size // 8,
        num_frames=args.frames,
        device=args.device,
    )
    canonical_w = watermark.create_watermark_and_return_w()
    torch.save(
        {
            "m": watermark.m,
            "watermark": watermark.watermark,
            "key": watermark.key,
            "nonce": watermark.nonce,
        },
        args.output_dir / "wm_info.bin",
    )

    input_tensor = transform_video(input_frames).to(pipe.vae.dtype).to(args.device)
    video_latents = get_video_latents(pipe.vae, input_tensor, sample=False, permute=True)
    pipe.scheduler = inverse_scheduler
    inverted = pipe(
        prompt="",
        latents=video_latents,
        num_inference_steps=args.steps,
        guidance_scale=1.0,
        output_type="latent",
    ).frames
    sign_forced = torch.where(canonical_w > 0, inverted.abs(), -inverted.abs())

    variants: list[tuple[str, torch.Tensor, str, float | None]] = []
    for alpha in args.alphas:
        blended = (1.0 - alpha) * inverted + alpha * sign_forced
        # The interpolation shrinks variance. Restore the inverted latent's
        # global standard deviation without changing embedded signs.
        blended = blended * (inverted.std() / blended.std().clamp_min(1e-6))
        variants.append((f"alpha-{alpha:.2f}", blended, "inversion-sign-adapter", alpha))
    variants.append(("canonical", canonical_w, "released-generation-path", None))

    results: list[dict] = []
    rendered: list[tuple[str, np.ndarray]] = [("MOSE input", input_frames)]
    for index, (name, starting_noise, route, alpha) in enumerate(variants, start=1):
        print(f"[{index}/{len(variants)}] generating {name}", flush=True)
        pipe.scheduler = forward_scheduler
        frames = pipe(
            prompt="",
            latents=starting_noise,
            num_frames=args.frames,
            height=args.size,
            width=args.size,
            num_inference_steps=args.steps,
            guidance_scale=1.0,
        ).frames[0]
        frames = np.asarray(frames, dtype=np.float32)
        export_to_video(frames, output_video_path=str(args.output_dir / f"{name}.mp4"), fps=args.fps)

        rendered_tensor = transform_video(frames).to(pipe.vae.dtype).to(args.device)
        rendered_latents = get_video_latents(pipe.vae, rendered_tensor, sample=False, permute=True)
        pipe.scheduler = inverse_scheduler
        recovered_noise = pipe(
            prompt="",
            latents=rendered_latents,
            num_inference_steps=args.steps,
            guidance_scale=1.0,
            output_type="latent",
        ).frames
        result = {
            "name": name,
            "route": route,
            "alpha": alpha,
            "starting_noise_sign_match": latent_sign_match(starting_noise, canonical_w),
            "recovered_noise_sign_match": latent_sign_match(recovered_noise, canonical_w),
            "watermark_accuracy": float(watermark.eval_watermark(recovered_noise)),
            "versus_mose_input": frame_metrics(input_frames, frames),
        }
        if len(rendered) > 1:
            baseline = rendered[1][1]
            result["versus_alpha_0_reconstruction"] = frame_metrics(baseline, frames)
        results.append(result)
        rendered.append((name, frames))
        print(json.dumps(result, sort_keys=True), flush=True)

    commit = subprocess.run(
        ["git", "-C", str(args.source), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    report = {
        "purpose": "Empirical compatibility check for applying VideoShield to an existing MOSE video",
        "upstream_commit": commit,
        "seed": args.seed,
        "device": args.device,
        "gpu": torch.cuda.get_device_name(torch.device(args.device)),
        "input_video": str(args.input_video),
        "model_path": str(args.model_path),
        "frames": args.frames,
        "size": args.size,
        "inference_and_inversion_steps": args.steps,
        "prompt": "",
        "input": input_metadata,
        "adapter": {
            "description": "DDIM inversion followed by VideoShield encrypted-sign imposition with inverted magnitudes preserved",
            "formula": "z_alpha = normalize((1-alpha)*z_inverted + alpha*sign(w)*abs(z_inverted))",
            "status": "experimental adapter; not provided or validated by upstream VideoShield",
        },
        "variants": results,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    make_contact_sheet(rendered, args.output_dir / "contact-sheet.png")
    print(f"Wrote {args.output_dir / 'metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
