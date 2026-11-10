#!/usr/bin/env python3
"""Validate the exact 13-notebook, four-inpainter one-clip matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


METHODS = {
    "editguard": "evidence/mose-task1/review50/media/02b72b8d/methods/editguard",
    "omniguard": "evidence/mose-task1/review50/media/02b72b8d/methods/omniguard",
    "safe_sora": "evidence/mose-task1/review50/media/02b72b8d/methods/safe-sora",
    "videoseal": "evidence/mose-task1/review50/media/02b72b8d/methods/videoseal",
    "watermark_anything": "runs/remaining-watermarks/watermark-anything/02b72b8d/attack",
    "vine": "runs/remaining-watermarks/vine/02b72b8d/attack",
    "genptw": "runs/remaining-watermarks/genptw/02b72b8d/attack",
    "videomark": "runs/remaining-watermarks/videomark/a_red_panda_eating_leaves/attack",
    "sigmark": "runs/remaining-watermarks/sigmark/native-vbench2/attack",
    "vidsig": "runs/remaining-watermarks/vidsig/red-panda-seed42/attack",
    "videoshield": "runs/remaining-watermarks/videoshield/modelscope-red-panda/attack",
}

REMOVERS = {
    "propainter": ("propainter.mp4", "propainter.json"),
    "diffueraser": ("diffueraser.mp4", "diffueraser.json"),
    "effecterase": ("effecterase.mp4", "effecterase.json"),
    "e2fgvi_hq": ("e2fgvi.mp4", "e2fgvi.json"),
}

DEFERRED = {
    "rivagan": "deferred_by_user_and_no_official_pretrained_model",
    "lvmark": "blocked_official_weight_server_and_missing_release_prompt_inputs",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("evidence/four-inpainter-matrix.json"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    notebook_dir = root / "notebooks/mose-watermark-samples"
    failures: list[str] = []
    matrix: dict[str, dict[str, object]] = {}

    expected_notebooks = {*(f"{name}.ipynb" for name in METHODS), *(f"{name}.ipynb" for name in DEFERRED)}
    actual_notebooks = {path.name for path in notebook_dir.glob("*.ipynb")}
    if actual_notebooks != expected_notebooks:
        failures.append(
            f"notebook set mismatch: missing={sorted(expected_notebooks - actual_notebooks)}, "
            f"unexpected={sorted(actual_notebooks - expected_notebooks)}"
        )

    for method, relative_attack in METHODS.items():
        attack = root / relative_attack
        notebook = notebook_dir / f"{method}.ipynb"
        combined = notebook.read_text() if notebook.is_file() else ""
        method_record: dict[str, object] = {"status": "complete", "attack_directory": relative_attack, "pairs": {}}
        for remover, (video_name, json_name) in REMOVERS.items():
            video = attack / video_name
            metadata = attack / json_name
            complete = video.is_file() and metadata.is_file()
            method_record["pairs"][remover] = {
                "status": "complete" if complete else "missing",
                "video": str(video.relative_to(root)),
                "metadata": str(metadata.relative_to(root)),
            }
            if not complete:
                failures.append(f"{method}/{remover}: missing {video_name} or {json_name}")
        for text in ("ProPainter", "DiffuEraser", "EffectErase", "E2FGVI-HQ"):
            if text not in combined:
                failures.append(f"{notebook}: missing {text}")
        matrix[method] = method_record

    for method, reason in DEFERRED.items():
        notebook = notebook_dir / f"{method}.ipynb"
        if not notebook.is_file():
            failures.append(f"missing deferred notebook: {notebook}")
        matrix[method] = {
            "status": "deferred" if method == "rivagan" else "blocked",
            "reason": reason,
            "pairs": {
                remover: {"status": "deferred" if method == "rivagan" else "blocked"}
                for remover in REMOVERS
            },
        }

    report = {
        "status": "complete" if not failures else "failed",
        "requirements": {"watermark_methods": 13, "inpainting_methods": 4, "minimum_clips_per_pair": 1},
        "summary": {
            "notebooks": len(actual_notebooks),
            "implemented_watermark_methods": len(METHODS),
            "completed_pairs": len(METHODS) * len(REMOVERS),
            "deferred_pairs": len(REMOVERS),
            "blocked_pairs": len(REMOVERS),
            "total_pair_cells": (len(METHODS) + len(DEFERRED)) * len(REMOVERS),
        },
        "methods": matrix,
        "failures": failures,
    }
    output = args.output if args.output.is_absolute() else root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    if failures:
        raise SystemExit("Four-inpainter validation failed:\n- " + "\n- ".join(failures))
    print(
        f"Validated {len(actual_notebooks)} notebooks and {len(METHODS) * len(REMOVERS)} "
        f"completed one-clip pairs; RivaGAN is deferred and LVMark is blocked with reasons recorded."
    )


if __name__ == "__main__":
    main()
