#!/usr/bin/env python3
"""Validate and summarize the Task 3 baseline result tree."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from task3_common import sha256, utc_now, write_json


EXPECTED_METHODS = ("propainter", "diffueraser")
EXPECTED_PER_METHOD = 50


def load_metrics(root: Path, baseline: str) -> list[dict]:
    records: list[dict] = []
    for method in EXPECTED_METHODS:
        paths = sorted((root / baseline / method).glob("*/metrics.json"))
        if len(paths) != EXPECTED_PER_METHOD:
            raise RuntimeError(
                f"Expected {EXPECTED_PER_METHOD} {baseline}/{method} results, found {len(paths)}"
            )
        for path in paths:
            record = json.loads(path.read_text())
            if record.get("status") != "complete":
                raise RuntimeError(f"Incomplete result: {path}")
            records.append(record)
    return records


def distribution(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "population_stddev": statistics.pstdev(values),
        "min": min(values),
        "max": max(values),
    }


def group_metric(records: list[dict], getter) -> dict:
    result = {"all": distribution([float(getter(record)) for record in records])}
    for method in EXPECTED_METHODS:
        selected = [
            float(getter(record))
            for record in records
            if record["input"]["method"] == method
        ]
        result[method] = distribution(selected)
    return result


def validate_files(root: Path, records: list[dict]) -> None:
    for record in records:
        baseline = record["baseline"].split("_")[0]
        method = record["input"]["method"]
        clip_id = record["input"]["clip_id"]
        directory = root / baseline / method / clip_id
        for relative, expected in record.get("files", {}).items():
            path = directory / relative
            if not path.is_file() or path.stat().st_size != expected["bytes"]:
                raise RuntimeError(f"Missing or size-mismatched artifact: {path}")
            if sha256(path) != expected["sha256"]:
                raise RuntimeError(f"Checksum mismatch: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    root = args.results_root.resolve()
    videoseal = load_metrics(root, "videoseal")
    safe_sora = load_metrics(root, "safe-sora")
    videoshield = load_metrics(root, "videoshield")
    validate_files(root, videoseal)
    validate_files(root, safe_sora)

    summary = {
        "status": "complete",
        "completed_at": utc_now(),
        "input_count": 100,
        "inputs": {"propainter": 50, "diffueraser": 50},
        "baseline_runs": 300,
        "videoseal": {
            "protocol": "postprocess_embed_then_detect",
            "bit_accuracy": group_metric(videoseal, lambda value: value["bit_accuracy"]),
            "decoded_video_psnr_db": group_metric(
                videoseal, lambda value: value["decoded_fidelity"]["psnr_db"]
            ),
        },
        "safe-sora": {
            "protocol": "postprocess_embed_then_recover",
            "video_psnr_db": group_metric(
                safe_sora, lambda value: value["watermark_embedding_fidelity"]["psnr_db"]
            ),
            "recovered_watermark_psnr_db": group_metric(
                safe_sora, lambda value: value["watermark_recovery_fidelity"]["psnr_db"]
            ),
        },
        "videoshield": {
            "protocol": "unwatermarked_negative_control_detection",
            "interpretation": (
                "Task 1 videos were not generated from VideoShield-marked diffusion noise; "
                "chance-level recovery is expected and is not a robustness score."
            ),
            "reference_watermark_accuracy": group_metric(
                videoshield, lambda value: value["reference_watermark_accuracy"]
            ),
        },
    }
    output = args.output or root / "task3-summary.json"
    write_json(output, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
