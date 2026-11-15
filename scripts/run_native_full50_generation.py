#!/usr/bin/env python3
"""Generate, mask, and register one 50-sample native watermark method."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
from pathlib import Path


METHODS = ("videoshield", "videomark", "sigmark", "vidsig")
EXPECTED_COMMITS = {
    "videoshield": "a61efa73abb15d30f50ee535c3301cea7dec3075",
    "videomark": "9f8d78b73ab9f9f055651b1b4f37d68bdb05e7be",
    "sigmark": "3713243f2e002cb21bbc3da896094d39b2b528c9",
    "vidsig": "9d219c47670c30fdaaccd17b24e7783c976a8c66",
}
SOURCE_NAMES = {
    "videoshield": "VideoShield",
    "videomark": "VideoMark",
    "sigmark": "SIGMark-release",
    "vidsig": "Video-Signature",
}


def run(command: list[Path | str], *, cwd: Path | None = None) -> None:
    values = [str(value) for value in command]
    print("RUN", json.dumps(values), flush=True)
    subprocess.run(values, cwd=cwd, check=True)


def require_commit(source: Path, expected: str) -> None:
    actual = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != expected:
        raise RuntimeError(f"Expected {expected} in {source}, found {actual}")


def generate_videoshield(args, source: Path, native: Path, prompts: Path) -> None:
    python = args.source_root / "VideoMark" / ".venv" / "bin" / "python"
    for start in range(0, 50, 10):
        run(
            [
                python,
                args.scripts / "run_videoshield_native_batch.py",
                "--source", source,
                "--model-path", args.models / "modelscope-text-to-video-ms-1.7b",
                "--prompt-file", prompts,
                "--output", native / "videoshield",
                "--start", str(start),
                "--count", "10",
                "--seed", "2025",
                "--device", "cuda:0",
            ]
        )


def generate_videomark(args, source: Path, native: Path, prompts: Path) -> None:
    python = source / ".venv" / "bin" / "python"
    run(
        [
            python,
            args.scripts / "run_videomark_native_batch.py",
            "--source", source,
            "--model-path", args.models / "modelscope-text-to-video-ms-1.7b",
            "--keys", source / "keys" / "64_64_512bit.pkl",
            "--prompt-file", prompts,
            "--output", native / "videomark",
            "--prompt-count", "13",
            "--seed", "11111",
            "--device", "cuda:0",
        ]
    )


def generate_vidsig(args, source: Path, native: Path, prompts: Path) -> None:
    python = source / ".venv" / "bin" / "python"
    for index in range(50):
        output = native / "vidsig" / f"vidsig-{index:03d}"
        if (output / "generation.json").is_file() and (output / "watermarked.mp4").is_file():
            print(f"SKIP complete VidSig sample {index}/49", flush=True)
            continue
        run(
            [
                python,
                args.scripts / "run_vidsig_native_sample.py",
                "--source", source,
                "--vae-checkpoint", source / "ckpts" / "vae" / "modelscope" / "checkpoint.pth",
                "--message-decoder", source / "ckpts" / "msg_decoder" / "dec_48b_whit.torchscript.pt",
                "--output", output,
                "--model-id", args.models / "modelscope-text-to-video-ms-1.7b",
                "--prompt-file", prompts,
                "--prompt-index", str(index),
                "--seed", str(2025 + index),
                "--device", "cuda:0",
            ]
        )


def generate_sigmark(args, source: Path, native: Path, _prompts: Path) -> None:
    output = native / "sigmark"
    output.mkdir(parents=True, exist_ok=True)
    videos = list(output.glob("*/*.mp4"))
    archives = list(output.glob("*-gt_watermark_messages.npz"))
    if len(videos) >= 50 and len(archives) == 1:
        print(f"SIGMark generation pool already has {len(videos)} videos", flush=True)
        return
    torchrun = source / ".venv" / "bin" / "torchrun"
    run(
        [
            torchrun,
            "--nproc_per_node=1",
            source / "main.py",
            "--mode", "gen",
            "--model_base_path", args.models,
            "--model_name", "HunyuanVideo-community",
            "--prompt_set", "VBench2_aug",
            "--watermark_method", "sigmark",
            "--quant_text_encoder", "1",
            "--seed", "2025",
            "--debug", "1",
            "--ch_factor", "2",
            "--hw_factor", "8",
            "--fr_factor", "1",
            "--batch_size", "1",
            "--num_prompts_per_dimension", "3",
            "--num_videos_per_prompt", "1",
            "--num_prompts_diversity", "3",
            "--num_videos_per_prompt_diversity", "1",
            "--small_scale_test", "49",
            "--output_path", output,
        ],
        cwd=source,
    )
    videos = list(output.glob("*/*.mp4"))
    if len(videos) != 50:
        raise RuntimeError(f"SIGMark should produce exactly 50 videos, found {len(videos)}")


def finalize_plan_if_ready(args: argparse.Namespace) -> None:
    manifests = {method: args.manifest_root / f"{method}.json" for method in METHODS}
    if not all(path.is_file() for path in manifests.values()):
        missing = [method for method, path in manifests.items() if not path.is_file()]
        print(f"Native plan still waiting for manifests: {missing}", flush=True)
        return
    state = args.stage / "state"
    state.mkdir(parents=True, exist_ok=True)
    with (state / "native-plan.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        command: list[Path | str] = [
            sys.executable,
            args.scripts / "build_full50_watermark_plan.py",
        ]
        for method in METHODS:
            command.extend(["--native-manifest", f"{method}={manifests[method]}"])
        command.append("--require-ready")
        run(command, cwd=args.stage)
        (state / "native-plan.ready").write_text("ready\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--stage", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    args = parser.parse_args()
    args.stage = args.stage.resolve()
    args.source_root = args.source_root.resolve()
    args.scripts = args.stage / "scripts"
    args.models = args.source_root / "remaining-watermark-experiments" / "models"
    args.manifest_root = args.stage / "native-manifests"
    args.manifest_root.mkdir(parents=True, exist_ok=True)
    native = args.stage / "native-full50"
    prompts = args.source_root / "Video-Signature" / "prompt.txt"
    source = args.source_root / SOURCE_NAMES[args.method]
    require_commit(source, EXPECTED_COMMITS[args.method])
    generators = {
        "videoshield": generate_videoshield,
        "videomark": generate_videomark,
        "sigmark": generate_sigmark,
        "vidsig": generate_vidsig,
    }
    generators[args.method](args, source, native, prompts)
    run(
        [
            sys.executable,
            args.scripts / "build_native_full50_manifest.py",
            "--method", args.method,
            "--native-root", native,
            "--source-root", args.source_root,
            "--output", args.manifest_root / f"{args.method}.json",
        ]
    )
    finalize_plan_if_ready(args)


if __name__ == "__main__":
    main()
