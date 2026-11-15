#!/usr/bin/env python3
"""Generate the authentic VideoMark pool used by the full-50 manifest."""

from __future__ import annotations

import argparse
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from mose_watermark_common import atomic_json, utc_now


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--keys", required=True, type=Path)
    parser.add_argument("--prompt-file", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--prompt-count", type=int, default=13)
    parser.add_argument("--seed", type=int, default=11111)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    source = args.source.resolve()
    prompts = [line.strip() for line in args.prompt_file.read_text().splitlines() if line.strip()]
    selected = prompts[: args.prompt_count]
    if len(selected) != args.prompt_count or len(set(selected)) != len(selected):
        raise RuntimeError(f"Expected {args.prompt_count} unique VideoMark prompts")
    output = args.output.resolve()
    result_root = output / "videomark" / "modelscope" / "512bit"
    summary = output / "generation.json"
    existing = sorted(result_root.glob("*/wm.mp4"))
    shifts = sorted(result_root.glob("*/shift_value.npy"))
    if summary.is_file() and len(existing) >= 50 and len(shifts) >= 50:
        print(f"VideoMark native pool already complete: {output}")
        return

    input_dir = output / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    (input_dir / "test_prompts.txt").write_text("\n".join(selected) + "\n")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    sys.path.insert(0, str(source))
    from embedding_and_extraction import main as upstream_main

    previous = Path.cwd()
    os.chdir(output)
    try:
        upstream_main(
            argparse.Namespace(
                device=args.device,
                num_frames=16,
                height=64,
                width=64,
                guidance_scale=9.0,
                num_inference_steps=50,
                num_inversion_steps=50,
                output_dir=str(output),
                data_dir="input",
                model_name="modelscope",
                use_watermark=True,
                keys_path=str(args.keys.resolve().parent),
                num_bit=512,
                model_path=str(args.model_path.resolve()),
            )
        )
    finally:
        os.chdir(previous)

    videos = sorted(result_root.glob("*/wm.mp4"))
    shifts = [video.with_name("shift_value.npy") for video in videos]
    if len(videos) < 50 or any(not shift.is_file() for shift in shifts):
        raise RuntimeError(f"VideoMark produced only {len(videos)} complete native samples")
    source_commit = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    atomic_json(
        summary,
        {
            "status": "complete",
            "method": "videomark",
            "source_commit": source_commit,
            "prompt_file": str(args.prompt_file.resolve()),
            "prompts": selected,
            "rng_seed": args.seed,
            "generated_samples": len(videos),
            "manifest_samples": 50,
            "selection": "first 50 upstream outputs in video_results.json insertion order",
            "completed_at": utc_now(),
        },
    )
    print(f"Generated {len(videos)} authentic VideoMark samples; 50 will be selected")


if __name__ == "__main__":
    main()
