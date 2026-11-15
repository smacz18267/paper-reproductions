#!/usr/bin/env python3
"""Resumable embedding, four-remover, and detection jobs for the full-50 plan.

The four fixed-MOSE methods already present in the review archive are handled
by ``run_mose_e2fgvi_full50.py`` plus their existing batch detectors. This
runner covers the three post-hoc methods and four authentic native-generation
methods, preserving one result directory per method/sample.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from full50_watermark_contract import REMOVERS
from mose_watermark_common import decode_rgb


RUNNABLE = (
    "watermark-anything",
    "vine",
    "genptw",
    "videoshield",
    "videomark",
    "sigmark",
    "vidsig",
)


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return value


def configured_path(config: dict[str, Any], key: str, *, base: Path, executable: bool = False) -> Path:
    raw = config.get(key)
    if raw is None or not str(raw).strip():
        raise RuntimeError(f"Missing runtime configuration {key!r}")
    path = Path(str(raw)).expanduser()
    # Do not call Path.resolve() here: virtual-environment Python executables
    # are symlinks, and resolving them replaces the venv entry point with the
    # base/system interpreter (and therefore loses the venv site-packages).
    path = Path(os.path.abspath(path if path.is_absolute() else base / path))
    if not path.exists():
        raise FileNotFoundError(f"Configured path {key!r} does not exist: {path}")
    if executable and (not path.is_file() or not os.access(path, os.X_OK)):
        raise RuntimeError(f"Configured executable {key!r} is not executable: {path}")
    return path


def run(command: list[Path | str], *, cwd: Path | None = None, dry_run: bool = False, log: Path | None = None) -> None:
    values = [str(value) for value in command]
    print("DRY-RUN" if dry_run else "RUN", json.dumps(values), flush=True)
    if dry_run:
        return
    if log is None:
        subprocess.run(values, cwd=cwd, check=True)
        return
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"command": values, "cwd": str(cwd) if cwd else None}) + "\n")
        handle.flush()
        subprocess.run(values, cwd=cwd, stdout=handle, stderr=subprocess.STDOUT, check=True)


def cases(attack: Path) -> list[str]:
    return [
        f"original={attack / 'input-watermarked.mp4'}",
        *[f"{remover}={attack / (remover + '.mp4')}" for remover in REMOVERS],
    ]


class JobRunner:
    def __init__(self, root: Path, config_path: Path, config: dict[str, Any], dry_run: bool, force: bool):
        self.root = root
        self.scripts = root / "scripts"
        self.config_path = config_path
        self.config = config
        self.base = config_path.parent
        self.dry_run = dry_run
        self.force = force
        self.gpu = str(config.get("gpu", "0"))
        raw_runner = config.get("runner_python")
        self.runner_python = (
            configured_path(config, "runner_python", base=self.base, executable=True)
            if raw_runner
            else Path(sys.executable)
        )

    def method_config(self, slug: str) -> dict[str, Any]:
        value = self.config.get("methods", {}).get(slug)
        if not isinstance(value, dict):
            raise RuntimeError(f"Missing runtime configuration for method {slug!r}")
        return value

    def job_path(self, value: str) -> Path:
        path = Path(value).expanduser()
        return (path if path.is_absolute() else self.root / path).resolve()

    def embed(self, slug: str, job: dict[str, Any]) -> tuple[Path, Path]:
        result = self.job_path(job["result_root"])
        result.mkdir(parents=True, exist_ok=True)
        source_video = self.job_path(job["source_video"])
        source_mask = self.job_path(job["source_mask"])
        if slug not in {"watermark-anything", "vine", "genptw"}:
            return source_video, source_mask
        config = self.method_config(slug)
        python = configured_path(config, "python", base=self.base, executable=True)
        repository = configured_path(config, "root", base=self.base)
        output_video = result / "watermarked.mp4"
        output_mask = source_mask if slug == "watermark-anything" else result / "aligned-mask.mp4"
        if output_video.is_file() and output_mask.exists() and not self.force:
            if slug not in {"vine", "genptw"}:
                return output_video, output_mask
            mask_frames, _mask_rate = decode_rgb(output_mask)
            if any(int(frame.max()) > 127 for frame in mask_frames):
                return output_video, output_mask
            print(
                f"REBUILD {slug}/{job['sample_id']}: center crop removed the full attack mask",
                flush=True,
            )
        device = f"cuda:{self.gpu}"
        if slug == "watermark-anything":
            command = [
                python, self.scripts / "watermark_anything_video.py", "embed",
                "--source", repository,
                "--checkpoint", configured_path(config, "checkpoint", base=self.base),
                "--params", configured_path(config, "params", base=self.base),
                "--input-video", source_video, "--output-video", output_video,
                "--device", device,
            ]
        elif slug == "vine":
            command = [
                python, self.scripts / "vine_video.py", "embed", "--source", repository,
                "--input-video", source_video, "--input-mask", source_mask,
                "--output-video", output_video, "--output-mask", output_mask,
                "--encoder-model", str(config.get("encoder_model", "Shilin-LU/VINE-R-Enc")),
                "--message", str(config.get("message", "Hello World!")), "--device", device,
            ]
        else:
            command = [
                python, self.scripts / "genptw_video.py", "embed", "--source", repository,
                "--checkpoint", configured_path(config, "checkpoint", base=self.base),
                "--pretrained-vae", configured_path(config, "pretrained_vae", base=self.base),
                "--convnext-checkpoint", configured_path(config, "convnext_checkpoint", base=self.base),
                "--input-video", source_video, "--input-mask", source_mask,
                "--output-video", output_video, "--output-mask", output_mask,
                "--device", device, "--seed", str(config.get("seed", 2026)),
            ]
        run(command, dry_run=self.dry_run)
        return output_video, output_mask

    def attack(self, slug: str, job: dict[str, Any], video: Path, mask: Path) -> Path:
        result = self.job_path(job["result_root"])
        attack = result / "attack"
        sample_json = attack / "sample.json"
        if self.force or not sample_json.is_file():
            command: list[Path | str] = [
                self.runner_python, self.scripts / "prepare_watermark_removal_sample.py", "prepare",
                "--source-video", video, "--source-mask", mask, "--output", attack,
                "--watermark-method", slug, "--sample-id", job["sample_id"],
                "--sample-source", job.get("provenance", {}).get("native_manifest", "fixed MOSE full-50 selection"),
                "--source-kind", job["source_kind"],
            ]
            if self.force:
                command.append("--force")
            run(command, dry_run=self.dry_run)

        diffueraser = configured_path(self.config, "diffueraser_root", base=self.base)
        diffueraser_python = configured_path(self.config, "diffueraser_python", base=self.base, executable=True)
        if self.force or not (attack / "removal-pair.json").is_file():
            command = [
                self.runner_python, self.scripts / "run_watermark_removal_pair.py",
                "--run", attack, "--diffueraser", diffueraser,
                "--diffueraser-python", diffueraser_python, "--device", self.gpu,
            ]
            if self.force:
                command.append("--force")
            run(command, dry_run=self.dry_run)

        if self.force or not (attack / "effecterase.json").is_file():
            command = [
                configured_path(self.config, "effecterase_python", base=self.base, executable=True),
                self.scripts / "run_prepared_effecterase.py",
                "--source", configured_path(self.config, "effecterase_root", base=self.base),
                "--run", attack, "--device", self.gpu,
            ]
            if self.force:
                command.append("--force")
            run(command, dry_run=self.dry_run)

        if self.force or not (attack / "e2fgvi.json").is_file():
            command = [
                self.runner_python, self.scripts / "run_prepared_e2fgvi.py",
                "--source", configured_path(self.config, "e2fgvi_root", base=self.base),
                "--checkpoint", configured_path(self.config, "e2fgvi_checkpoint", base=self.base),
                "--python", configured_path(self.config, "e2fgvi_python", base=self.base, executable=True),
                "--run", attack, "--device", self.gpu,
            ]
            if self.force:
                command.append("--force")
            run(command, dry_run=self.dry_run)
        return attack

    def detect(self, slug: str, job: dict[str, Any], attack: Path) -> None:
        output = self.job_path(job["result_root"]) / "detection"
        summary = output / "summary.json"
        if summary.is_file() and not self.force:
            return
        config = self.method_config(slug)
        python = configured_path(config, "python", base=self.base, executable=True)
        repository = configured_path(config, "root", base=self.base)
        device = f"cuda:{self.gpu}"
        case_args = []
        for value in cases(attack):
            case_args.extend(["--case", value])
        if slug == "watermark-anything":
            command = [
                python, self.scripts / "watermark_anything_video.py", "detect", "--source", repository,
                "--checkpoint", configured_path(config, "checkpoint", base=self.base),
                "--params", configured_path(config, "params", base=self.base),
                "--gt-mask", attack / "gt-mask.mp4", "--output", output, "--device", device,
                *case_args,
            ]
        elif slug == "vine":
            command = [
                python, self.scripts / "vine_video.py", "detect", "--source", repository,
                "--decoder-model", str(config.get("decoder_model", "Shilin-LU/VINE-R-Dec")),
                "--message", str(config.get("message", "Hello World!")),
                "--output", output, "--device", device, *case_args,
            ]
        elif slug == "genptw":
            command = [
                python, self.scripts / "genptw_video.py", "detect", "--source", repository,
                "--checkpoint", configured_path(config, "checkpoint", base=self.base),
                "--pretrained-vae", configured_path(config, "pretrained_vae", base=self.base),
                "--convnext-checkpoint", configured_path(config, "convnext_checkpoint", base=self.base),
                "--gt-mask", attack / "gt-mask.mp4", "--output", output,
                "--device", device, *case_args,
            ]
        elif slug == "videoshield":
            watermark_info = job.get("method_inputs", {}).get("watermark_info")
            if not watermark_info:
                raise RuntimeError(f"{slug}/{job['sample_id']}: native manifest lacks method_inputs.watermark_info")
            command = [
                python, self.scripts / "run_videoshield_removal_detection.py",
                "--source", repository, "--model-path", configured_path(config, "model_path", base=self.base),
                "--watermark-info", self.job_path(watermark_info), "--gt-mask-frames", attack / "gt-mask-frames",
                "--output", output, "--device", device, *case_args,
            ]
        elif slug == "videomark":
            shift = job.get("method_inputs", {}).get("shift")
            if not shift:
                raise RuntimeError(f"{slug}/{job['sample_id']}: native manifest lacks method_inputs.shift")
            command = [
                python, self.scripts / "run_videomark_detection.py",
                "--source", repository, "--model-path", configured_path(config, "model_path", base=self.base),
                "--keys", configured_path(config, "keys", base=self.base), "--shift", self.job_path(shift),
                "--output", output, "--device", device, *case_args,
            ]
        elif slug == "vidsig":
            command = [
                python, self.scripts / "run_vidsig_detection.py", "--source", repository,
                "--decoder", configured_path(config, "decoder", base=self.base),
                "--output", output, "--device", device, *case_args,
            ]
        elif slug == "sigmark":
            self.detect_sigmark(job, attack, output, config, repository)
            return
        else:
            raise ValueError(slug)
        run(command, dry_run=self.dry_run)

    def detect_sigmark(self, job, attack: Path, output: Path, config, repository: Path) -> None:
        generated = job.get("method_inputs", {}).get("generated_output")
        sample_key = job.get("method_inputs", {}).get("sample_key")
        if not generated or not sample_key:
            raise RuntimeError("SIGMark native manifest needs method_inputs.generated_output and sample_key")
        torchrun = configured_path(config, "torchrun", base=self.base, executable=True)
        model_base = configured_path(config, "model_base", base=self.base)
        prompt_counts = {
            "num_prompts_per_dimension": int(config.get("num_prompts_per_dimension", 3)),
            "num_videos_per_prompt": int(config.get("num_videos_per_prompt", 1)),
            "num_prompts_diversity": int(config.get("num_prompts_diversity", 3)),
            "num_videos_per_prompt_diversity": int(
                config.get("num_videos_per_prompt_diversity", 1)
            ),
        }
        normalized_cases = []
        for case_value in cases(attack):
            name, raw_video = case_value.split("=", 1)
            staged = output / name
            if self.force or not (staged / "staged-case.json").is_file():
                run([
                    self.runner_python, self.scripts / "stage_sigmark_detection_case.py",
                    "--generated-output", self.job_path(generated), "--sample-key", sample_key,
                    "--replacement-video", raw_video, "--destination", staged,
                    *( ["--force"] if self.force else [] ),
                ], dry_run=self.dry_run)
            accuracy_archives = list(staged.glob("*bit_accuracy.npz"))
            if self.force or len(accuracy_archives) != 1:
                command = [
                    torchrun, "--nproc_per_node=1", repository / "main.py", "--mode", "extract",
                    "--model_base_path", model_base, "--model_name", "HunyuanVideo-community",
                    "--prompt_set", "VBench2_aug", "--watermark_method", "sigmark",
                    "--quant_text_encoder", "1", "--ch_factor", "2", "--hw_factor", "8",
                    "--fr_factor", "1", "--batch_size", "1", "--num_prompts_per_dimension",
                    str(prompt_counts["num_prompts_per_dimension"]),
                    "--num_videos_per_prompt", str(prompt_counts["num_videos_per_prompt"]),
                    "--num_prompts_diversity", str(prompt_counts["num_prompts_diversity"]),
                    "--num_videos_per_prompt_diversity",
                    str(prompt_counts["num_videos_per_prompt_diversity"]),
                    # The staged tree contains only this sample. Iterate the full
                    # official prompt index so samples beyond the first batches are
                    # discovered, while all non-staged entries are skipped.
                    "--small_scale_test", "-1",
                    "--output_path", staged,
                ]
                run(command, cwd=repository, dry_run=self.dry_run, log=staged / "extract.log")
            else:
                print(f"SKIP SIGMark extraction already complete: {accuracy_archives[0]}", flush=True)
            normalized_cases.extend(["--case", f"{name}={staged}"])
        run([
            self.runner_python, self.scripts / "normalize_sigmark_detection.py",
            *normalized_cases, "--output", output / "summary.json",
        ], dry_run=self.dry_run)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=Path("runs/full50-watermark-evaluation/plan.json"))
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--method", action="append", choices=RUNNABLE)
    parser.add_argument("--only", action="append", help="Sample id; repeat to select multiple")
    parser.add_argument("--stage", action="append", choices=("embed", "attack", "detect"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    plan_path = args.plan if args.plan.is_absolute() else root / args.plan
    config_path = args.config.resolve()
    plan = load_object(plan_path)
    config = load_object(config_path)
    runner = JobRunner(root, config_path, config, args.dry_run, args.force)
    selected_methods = tuple(args.method or RUNNABLE)
    selected_stages = set(args.stage or ("embed", "attack", "detect"))
    selected_samples = set(args.only or [])
    seen_samples: set[str] = set()
    failures = []
    for slug in selected_methods:
        for job in plan["methods"][slug]["samples"]:
            if selected_samples and job["sample_id"] not in selected_samples:
                continue
            seen_samples.add(job["sample_id"])
            if job["status"] != "ready":
                failures.append(f"{slug}/{job['sample_id']}: {job['status']}")
                continue
            try:
                if "embed" in selected_stages:
                    video, mask = runner.embed(slug, job)
                else:
                    video = runner.job_path(job["result_root"]) / "watermarked.mp4"
                    mask = (
                        runner.job_path(job["source_mask"])
                        if slug == "watermark-anything"
                        else runner.job_path(job["result_root"]) / "aligned-mask.mp4"
                    )
                if slug not in {"watermark-anything", "vine", "genptw"}:
                    video, mask = runner.job_path(job["source_video"]), runner.job_path(job["source_mask"])
                attack = (
                    runner.attack(slug, job, video, mask)
                    if "attack" in selected_stages
                    else runner.job_path(job["result_root"]) / "attack"
                )
                if "detect" in selected_stages:
                    runner.detect(slug, job, attack)
            except Exception as error:
                failures.append(f"{slug}/{job['sample_id']}: {error!r}")
                print(f"FAILED {failures[-1]}", flush=True)
    missing_samples = selected_samples - seen_samples
    if missing_samples:
        failures.append(f"unknown --only sample ids: {sorted(missing_samples)}")
    status_name = (
        "runner-status.json"
        if not args.method
        else "runner-status-" + "-".join(selected_methods) + ".json"
    )
    status_path = plan_path.parent / status_name
    status = {
        "status": "complete" if not failures else "failed",
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "methods": list(selected_methods),
        "stages": sorted(selected_stages),
        "failures": failures,
    }
    if not args.dry_run:
        status_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = status_path.with_name(status_path.name + ".tmp")
        temporary.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
        temporary.replace(status_path)
    if failures:
        raise SystemExit(f"Full-50 runner encountered {len(failures)} blocked/failed jobs; see {status_path}")


if __name__ == "__main__":
    main()
