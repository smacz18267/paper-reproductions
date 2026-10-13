#!/usr/bin/env python3
"""Seed and invoke one unchanged VideoShield upstream stage."""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("embed", "temporal", "spatial"))
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--frames-dir", type=Path)
    parser.add_argument("--prompt", default="a red panda eating leaves")
    parser.add_argument("--seed", type=int, default=2025)
    args = parser.parse_args()

    source = args.source.resolve()
    run_dir = args.run_dir.resolve()
    model_path = args.model_path.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(source))
    seed_everything(args.seed)

    if args.stage == "embed":
        input_dir = run_dir / "input"
        input_dir.mkdir(exist_ok=True)
        (input_dir / "test_prompts.txt").write_text(args.prompt + "\n")
        os.chdir(run_dir)
        from watermark_embedding_and_extraction import main as upstream_main

        upstream_main(
            argparse.Namespace(
                device="cuda:0",
                num_frames=16,
                height=256,
                width=256,
                guidance_scale=9.0,
                num_inference_steps=25,
                num_inversion_steps=25,
                channel_copy=1,
                frames_copy=8,
                hw_copy=4,
                output_dir=str(run_dir / "results"),
                data_dir="input",
                model_name="modelscope",
                model_path=str(model_path),
            )
        )
        return

    if args.frames_dir is None:
        raise RuntimeError("--frames-dir is required for localization stages")
    common = argparse.Namespace(
        device="cuda:0",
        model_name="modelscope",
        model_path=str(model_path),
        num_inversion_steps=25,
        video_frames_dir=str(args.frames_dir.resolve()),
        video_frames_path=str(args.frames_dir.resolve()),
    )
    if args.stage == "temporal":
        from temporal_tamper_localization import main as upstream_main
    else:
        from spatial_tamper_localization import main as upstream_main
    upstream_main(common)


if __name__ == "__main__":
    main()
