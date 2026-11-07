#!/usr/bin/env python3
"""Run ProPainter and DiffuEraser from one prepared, authentic sample.

DiffuEraser's official pipeline first creates a ProPainter prior and then
refines it.  ``run_diffueraser_recorded.py`` preserves that exact prior.  This
runner therefore produces both requested methods with identical inputs,
masks, dilation, dimensions, and seed in a single invocation.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

from mose_watermark_common import atomic_json, sha256, utc_now, validate_video
from prepare_watermark_removal_sample import finalize


def git_revision(path: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def finalize_case(run: Path, generated: Path, case: str) -> None:
    namespace = argparse.Namespace(
        source_video=generated,
        sample_json=run / "sample.json",
        output_video=run / f"{case}.mp4",
        output_frames=run / f"{case}-frames",
        case=case,
    )
    finalize(namespace)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--diffueraser", required=True, type=Path)
    parser.add_argument("--diffueraser-python", type=Path)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--mask-dilation-iter", type=int, default=8)
    parser.add_argument("--max-img-size", type=int, default=960)
    parser.add_argument("--ref-stride", type=int, default=10)
    parser.add_argument("--neighbor-length", type=int, default=10)
    parser.add_argument("--subvideo-length", type=int, default=40)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--finalize-existing",
        action="store_true",
        help="Finalize a previously completed official inference without rerunning the GPU pipeline",
    )
    args = parser.parse_args()
    if args.force and args.finalize_existing:
        parser.error("--force and --finalize-existing are mutually exclusive")

    run = args.run.resolve()
    source = args.diffueraser.resolve()
    # Keep the virtual-environment launcher path intact. Path.resolve() follows
    # the venv's ``python`` symlink to the system interpreter, which drops the
    # venv site-packages and makes imports such as torch fail.
    python = (args.diffueraser_python or Path(sys.executable)).expanduser().absolute()
    sample = json.loads((run / "sample.json").read_text())
    attack_input = run / "attack-input.mp4"
    attack_mask = run / "attack-mask.mp4"
    expected_size = tuple(int(value) for value in sample["size"])
    validate_video(
        attack_input,
        frames=int(sample["attack_frames"]),
        size=expected_size,
        fps=float(sample["fps"]),
    )
    validate_video(
        attack_mask,
        frames=int(sample["attack_frames"]),
        size=expected_size,
        fps=float(sample["fps"]),
    )

    required = [
        python,
        source / "run_diffueraser.py",
        source / "weights/stable-diffusion-v1-5",
        source / "weights/sd-vae-ft-mse",
        source / "weights/diffuEraser",
        source / "weights/propainter",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing DiffuEraser prerequisite(s):\n" + "\n".join(missing))

    work = run / "work/removal-pair"
    log = run / "logs/diffueraser.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    wrapper = Path(__file__).resolve().with_name("run_diffueraser_recorded.py")
    duration = max(1, math.ceil(int(sample["attack_frames"]) / float(sample["fps"])))
    command = [
        str(python),
        "-u",
        str(wrapper),
        "--source",
        str(source),
        "--seed",
        str(args.seed),
        "--input_video",
        str(attack_input),
        "--input_mask",
        str(attack_mask),
        "--video_length",
        str(duration),
        "--mask_dilation_iter",
        str(args.mask_dilation_iter),
        "--max_img_size",
        str(args.max_img_size),
        "--ref_stride",
        str(args.ref_stride),
        "--neighbor_length",
        str(args.neighbor_length),
        "--subvideo_length",
        str(args.subvideo_length),
        "--save_path",
        str(work),
        "--base_model_path",
        str(source / "weights/stable-diffusion-v1-5"),
        "--vae_path",
        str(source / "weights/sd-vae-ft-mse"),
        "--diffueraser_path",
        str(source / "weights/diffuEraser"),
        "--propainter_model_dir",
        str(source / "weights/propainter"),
    ]
    propainter = work / "propainter_prior.mp4"
    diffueraser = work / "diffueraser_result.mp4"
    if args.finalize_existing:
        if not propainter.is_file() or not diffueraser.is_file():
            raise FileNotFoundError(
                "--finalize-existing requires both propainter_prior.mp4 and "
                "diffueraser_result.mp4 from a completed official execution"
            )
    else:
        if work.exists():
            if not args.force:
                raise FileExistsError(f"{work} already exists; pass --force to replace it")
            shutil.rmtree(work)
        work.mkdir(parents=True)
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = args.device
        with log.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps({"cwd": str(source), "command": command}, indent=2) + "\n")
            handle.flush()
            result = subprocess.run(
                command,
                cwd=source,
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        if result.returncode != 0:
            raise RuntimeError(f"DiffuEraser exited {result.returncode}; see {log}")

    if not propainter.is_file() or not diffueraser.is_file():
        raise RuntimeError("DiffuEraser did not produce both the ProPainter prior and final result")
    finalize_case(run, propainter, "propainter")
    finalize_case(run, diffueraser, "diffueraser")

    outputs = {}
    for case in ("propainter", "diffueraser"):
        path = run / f"{case}.mp4"
        outputs[case] = {
            "path": path.name,
            "sha256": sha256(path),
            "video": validate_video(
                path,
                frames=int(sample["frames"]),
                size=expected_size,
                fps=float(sample["fps"]),
            ),
        }
    atomic_json(
        run / "removal-pair.json",
        {
            "status": "complete",
            "completed_at": utc_now(),
            "protocol": "one official DiffuEraser execution; preserved ProPainter prior and final DiffuEraser output",
            "finalized_existing_official_execution": args.finalize_existing,
            "seed": args.seed,
            "mask_dilation_iterations": args.mask_dilation_iter,
            "diffueraser_source": str(source),
            "diffueraser_commit": git_revision(source),
            "command": command,
            "outputs": outputs,
        },
    )
    print(f"Completed ProPainter and DiffuEraser for {sample['watermark_method']}/{sample['sample_id']}")


if __name__ == "__main__":
    main()
