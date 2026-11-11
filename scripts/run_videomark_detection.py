#!/usr/bin/env python3
"""Run VideoMark's native inversion/PRC decoder on removal outputs.

This is extraction-only: it accepts the authors' provided ModelScope sample,
its committed shift value and PRC key, then evaluates the untouched sample and
both object-removal outputs without regenerating or re-watermarking them.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from mose_watermark_common import atomic_json, decode_rgb, sha256, utc_now


def parse_case(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator:
        raise argparse.ArgumentTypeError("Cases must use NAME=/path/to/video.mp4")
    return name, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--keys", required=True, type=Path)
    parser.add_argument("--shift", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--case", action="append", required=True, type=parse_case)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--num-bits", type=int, default=512)
    parser.add_argument("--num-inversion-steps", type=int, default=50)
    args = parser.parse_args()

    source = args.source.resolve()
    sys.path.insert(0, str(source))
    os.chdir(source)
    from diffusers import TextToVideoSDPipeline
    from diffusers.schedulers import DDIMInverseScheduler
    from src.prc import Decode, Detect
    import src.pseudogaussians as prc_gaussians
    from utils import get_video_latents, recover, transform_video

    pipeline = TextToVideoSDPipeline.from_pretrained(
        args.model_path.resolve(), torch_dtype=torch.float16
    ).to(args.device)
    pipeline.safety_checker = None
    pipeline.scheduler = DDIMInverseScheduler.from_pretrained(
        args.model_path.resolve(), subfolder="scheduler"
    )
    with args.keys.resolve().open("rb") as handle:
        _encoding_key, decoding_key = pickle.load(handle)
    np.random.seed(11111)
    message_sequence = np.random.randint(0, 2, size=(500, args.num_bits))
    shift = int(np.load(args.shift.resolve()).reshape(-1)[0])
    target_window = message_sequence[shift : shift + args.num_frames]
    target_strings = ["".join(map(str, row)) for row in message_sequence]

    args.output.mkdir(parents=True, exist_ok=True)
    summary = {
        "status": "complete",
        "method": "VideoMark",
        "protocol": "authors' provided watermarked sample -> removal -> native DDIM inversion and PRC decoding",
        "source_commit": subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
        ).strip(),
        "model_path": str(args.model_path.resolve()),
        "key_sha256": sha256(args.keys.resolve()),
        "shift_sha256": sha256(args.shift.resolve()),
        "shift": shift,
        "cases": {},
    }
    for name, raw_path in args.case:
        path = raw_path.resolve()
        frames, _rate = decode_rgb(path)
        if len(frames) != args.num_frames:
            raise RuntimeError(f"{name}: expected {args.num_frames} frames, found {len(frames)}")
        tensor = transform_video(frames).to(pipeline.vae.dtype).to(args.device)
        with torch.inference_mode():
            latents = get_video_latents(pipeline.vae, tensor, sample=False, permute=True)
            reversed_latents = pipeline(
                prompt="",
                latents=latents,
                num_inference_steps=args.num_inversion_steps,
                guidance_scale=1.0,
                output_type="latent",
            ).frames
        # Match the authors' process_frames_in_parallel implementation: their
        # SciPy-based posterior recovery receives CPU tensors, not CUDA tensors.
        reversed_latents_cpu = reversed_latents.cpu()

        index_list: list[int] = []
        message_list: list[np.ndarray] = []
        distance_list: list[int] = []
        detected_frames = 0
        for frame_index in range(args.num_frames):
            posterior = prc_gaussians.recover_posteriors(
                reversed_latents_cpu[:, :, frame_index].to(torch.float64).flatten(), variances=1.5
            ).flatten()
            detected = bool(Detect(decoding_key, posterior, false_positive_rate=0.01))
            if not detected:
                index_list.append(-1)
                message_list.append(np.full(args.num_bits, -1, dtype=np.int16))
                distance_list.append(args.num_bits)
                continue
            decoded = Decode(decoding_key, posterior)
            if decoded is None:
                index_list.append(-1)
                message_list.append(np.full(args.num_bits, -1, dtype=np.int16))
                distance_list.append(args.num_bits)
                continue
            detected_frames += 1
            decoded = np.asarray(decoded[: args.num_bits], dtype=np.uint8)
            decoded_string = "".join(map(str, decoded))
            distances = np.asarray(
                [sum(left != right for left, right in zip(decoded_string, target)) for target in target_strings]
            )
            minimum = int(distances.min())
            nearest = -1 if len(np.unique(distances)) == 1 else int(distances.argmin())
            index_list.append(nearest)
            message_list.append(decoded)
            distance_list.append(minimum)

        recovered_indices, recovered_messages, recovered_distances = recover(
            index_list, message_list, distance_list, args.num_bits
        )
        full_accuracy = float((recovered_messages == target_window).mean())
        upstream_accuracy = float((recovered_messages[1:] == target_window[1:]).mean())
        metrics = {
            "input": str(path),
            "input_sha256": sha256(path),
            "detected_frames": detected_frames,
            "total_frames": args.num_frames,
            "frame_detection_rate": detected_frames / args.num_frames,
            "bit_accuracy_all_frames": full_accuracy,
            "bit_accuracy_authors_protocol": upstream_accuracy,
            "authors_protocol_note": "Matches embedding_and_extraction.py by excluding index 0.",
            "recovered_sequence_indices": recovered_indices.tolist(),
            "recovered_hamming_distances": recovered_distances.tolist(),
        }
        atomic_json(args.output / f"{name}.json", metrics)
        summary["cases"][name] = metrics
        print(
            f"{name}: detection={detected_frames}/{args.num_frames}; "
            f"authors-protocol bit accuracy={upstream_accuracy:.4f}"
        )
        del tensor, latents, reversed_latents, reversed_latents_cpu
        torch.cuda.empty_cache()
    summary["completed_at"] = utc_now()
    atomic_json(args.output / "summary.json", summary)


if __name__ == "__main__":
    main()
