#!/usr/bin/env python3
"""Run official E2FGVI detector/recovery stages for the four fixed-MOSE methods."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any


METHODS = ("editguard", "omniguard", "videoseal", "safe-sora")


def path(value: Any, label: str, base: Path) -> Path:
    if value is None or not str(value).strip():
        raise RuntimeError(f"Missing runtime configuration {label!r}")
    result = Path(str(value)).expanduser()
    # Preserve virtual-environment entry-point symlinks. Resolving one to the
    # base interpreter silently drops the environment's installed packages.
    result = Path(os.path.abspath(result if result.is_absolute() else base / result))
    if not result.exists():
        raise FileNotFoundError(f"Configured path {label!r} does not exist: {result}")
    return result


def run(command: list[Path | str], dry_run: bool) -> None:
    values = [str(value) for value in command]
    print(("DRY-RUN " if dry_run else "RUN ") + json.dumps(values), flush=True)
    if not dry_run:
        subprocess.run(values, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--method", action="append", choices=METHODS)
    parser.add_argument("--only", action="append")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config_path = args.config.resolve()
    base = config_path.parent
    config = json.loads(config_path.read_text())
    selection = path(config.get("mose_selection"), "mose_selection", base)
    dataset = path(config.get("mose_dataset_root"), "mose_dataset_root", base)
    output = path(config.get("mose_output_root"), "mose_output_root", base)
    scripts = root / "scripts"
    suffix: list[Path | str] = ["--remover", "e2fgvi"]
    for sample_id in args.only or []:
        suffix.extend(["--only", sample_id])
    if args.force:
        suffix.append("--force")

    for method in args.method or METHODS:
        method_config = config.get("methods", {}).get(method, {})
        python = path(method_config.get("python"), f"methods.{method}.python", base)
        source = path(method_config.get("root"), f"methods.{method}.root", base)
        if method == "editguard":
            command = [
                python, scripts / "run_mose_editguard.py", "localize", "--source", source,
                "--checkpoint", path(method_config.get("checkpoint"), "methods.editguard.checkpoint", base),
                "--selection", selection, "--dataset-root", dataset, "--output-root", output, *suffix,
            ]
        elif method == "omniguard":
            command = [
                python, scripts / "run_mose_omniguard.py", "localize", "--source", source,
                "--checkpoint", path(method_config.get("checkpoint"), "methods.omniguard.checkpoint", base),
                "--detector-checkpoint", path(method_config.get("detector_checkpoint"), "methods.omniguard.detector_checkpoint", base),
                "--selection", selection, "--dataset-root", dataset, "--output-root", output, *suffix,
            ]
        elif method == "videoseal":
            command = [
                python, scripts / "run_mose_videoseal_localize.py", "--source", source,
                "--selection", selection, "--dataset-root", dataset, "--output-root", output,
                "--chunk-size", str(method_config.get("chunk_size", 1)), "--grid-size", "4", *suffix,
            ]
        else:
            command = [
                python, scripts / "run_mose_safe_sora_recover.py", "--source", source,
                "--selection", selection, "--output-root", output,
                "--watermark", path(method_config.get("watermark"), "methods.safe-sora.watermark", base),
                *suffix,
            ]
        run(command, args.dry_run)


if __name__ == "__main__":
    main()
