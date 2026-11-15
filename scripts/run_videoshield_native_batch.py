#!/usr/bin/env python3
"""Run one resumable chunk of authentic VideoShield native generations."""

from __future__ import annotations

import argparse
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from mose_watermark_common import atomic_json, sha256, utc_now


def prompt_id(prompt: str) -> str:
    return prompt.replace(" ", "_")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--prompt-file", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--start", required=True, type=int)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    source = args.source.resolve()
    model_path = args.model_path.resolve()
    prompts = [line.strip() for line in args.prompt_file.read_text().splitlines() if line.strip()]
    selected = prompts[args.start : args.start + args.count]
    if not selected:
        raise RuntimeError(f"No prompts selected at index {args.start}")
    if len(set(prompts)) != len(prompts):
        raise RuntimeError("VideoShield prompt file contains duplicate prompts")

    output = args.output.resolve()
    batch = output / "batches" / f"batch-{args.start:03d}-{args.start + len(selected) - 1:03d}"
    results = batch / "results"
    expected = [results / "modelscope" / prompt_id(prompt) / "wm.mp4" for prompt in selected]
    expected_info = [path.with_name("wm_info.bin") for path in expected]
    metadata_path = batch / "generation.json"
    if metadata_path.is_file() and all(path.is_file() for path in expected + expected_info):
        print(f"VideoShield batch already complete: {batch}")
        return

    input_dir = batch / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    (input_dir / "test_prompts.txt").write_text("\n".join(selected) + "\n")
    random.seed(args.seed + args.start)
    np.random.seed(args.seed + args.start)
    torch.manual_seed(args.seed + args.start)
    torch.cuda.manual_seed_all(args.seed + args.start)

    sys.path.insert(0, str(source))
    from watermark_embedding_and_extraction import main as upstream_main

    previous = Path.cwd()
    os.chdir(batch)
    try:
        upstream_main(
            argparse.Namespace(
                device=args.device,
                num_frames=16,
                height=256,
                width=256,
                guidance_scale=9.0,
                num_inference_steps=25,
                num_inversion_steps=25,
                channel_copy=1,
                frames_copy=8,
                hw_copy=4,
                output_dir=str(results),
                data_dir="input",
                model_name="modelscope",
                model_path=str(model_path),
            )
        )
    finally:
        os.chdir(previous)

    missing = [str(path) for path in expected + expected_info if not path.is_file()]
    if missing:
        raise RuntimeError(f"VideoShield batch did not produce expected artifacts: {missing}")
    source_commit = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    atomic_json(
        metadata_path,
        {
            "status": "complete",
            "method": "videoshield",
            "source_commit": source_commit,
            "prompt_file": str(args.prompt_file.resolve()),
            "prompt_start": args.start,
            "prompts": selected,
            "rng_seed": args.seed + args.start,
            "model_path": str(model_path),
            "samples": [
                {
                    "prompt": prompt,
                    "watermarked_video": str(video),
                    "watermark_info": str(info),
                    "watermarked_sha256": sha256(video),
                    "watermark_info_sha256": sha256(info),
                }
                for prompt, video, info in zip(selected, expected, expected_info)
            ],
            "completed_at": utc_now(),
        },
    )
    print(f"Generated {len(selected)} authentic VideoShield samples in {batch}")


if __name__ == "__main__":
    main()
