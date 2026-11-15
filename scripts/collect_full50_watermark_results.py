#!/usr/bin/env python3
"""Collect per-sample and aggregate metrics for the 11 x 50 watermark study."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from full50_watermark_contract import METHODS, REMOVERS, SAMPLES_PER_METHOD


REVIEW_NAMES = {
    "editguard": "EditGuard",
    "omniguard": "OmniGuard",
    "videoseal": "VideoSeal",
    "safe-sora": "Safe-Sora",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def portable_paths(value: Any, root: Path) -> Any:
    """Replace repository-local absolute paths in committed summaries."""

    if isinstance(value, dict):
        return {key: portable_paths(child, root) for key, child in value.items()}
    if isinstance(value, list):
        return [portable_paths(child, root) for child in value]
    if isinstance(value, str):
        path = Path(value)
        if path.is_absolute() and path.is_relative_to(root):
            return str(path.relative_to(root))
    return value


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return value


def nested(record: dict[str, Any], *paths: tuple[str, ...]) -> float | None:
    for path in paths:
        value: Any = record
        for key in path:
            if not isinstance(value, dict) or key not in value:
                break
            value = value[key]
        else:
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                return float(value)
    return None


def canonical_metrics(record: dict[str, Any], keys: list[str]) -> dict[str, float]:
    candidates = {
        "mean_message_bit_accuracy": (("mean_message_bit_accuracy",),),
        "mean_predicted_tamper_probability": (
            ("mean_predicted_tamper_probability",),
            ("mean_predicted_probability",),
        ),
        "global_localization_iou": (
            ("global_localization_iou",),
            ("global_iou",),
            ("spatial_localization", "iou"),
        ),
        "spatial_localization_iou": (
            ("spatial_localization_iou",),
            ("spatial_localization", "iou"),
            ("global_iou",),
        ),
        "graphical_watermark_recovery_psnr_db": (
            ("graphical_watermark_recovery_psnr_db",),
            ("fidelity", "psnr_db"),
            ("psnr_db",),
        ),
        "mean_absolute_error": (("mean_absolute_error",), ("fidelity", "mean_absolute_error")),
        "video_majority_bit_accuracy": (("video_majority_bit_accuracy",),),
        "frame_bit_accuracy_mean": (("frame_bit_accuracy_mean",),),
        "watermark_accuracy": (("watermark_accuracy",),),
        "frame_detection_rate": (("frame_detection_rate",),),
        "bit_accuracy_authors_protocol": (("bit_accuracy_authors_protocol",),),
        "bit_accuracy": (("bit_accuracy",), ("accuracy",)),
    }
    metrics: dict[str, float] = {}
    for key in keys:
        value = nested(record, *candidates.get(key, ((key,),)))
        if value is not None:
            metrics[key] = value
    return metrics


def review_case(
    review_root: Path,
    review_sample: dict[str, Any],
    method_slug: str,
    remover: str,
) -> tuple[dict[str, Any] | None, str | None]:
    """Load an existing fixed-MOSE detector record without trusting display fields."""

    method_name = REVIEW_NAMES.get(method_slug)
    if method_name is None:
        return None, None
    watermark = review_sample.get("watermarks", {}).get(method_name)
    if not isinstance(watermark, dict):
        return None, None
    metadata_key = "recovery" if method_slug == "safe-sora" else "localization"
    relative = watermark.get("metadata", {}).get(metadata_key)
    if not relative:
        return None, None
    source = review_root / relative
    payload = load_json(source)
    if payload is None or not isinstance(payload.get(remover), dict):
        return None, str(source)
    return payload[remover], str(source)


def generated_case(result_root: Path, remover: str) -> tuple[dict[str, Any] | None, str | None]:
    candidates = (
        result_root / "detection" / "summary.json",
        result_root / "attack" / "detection" / "summary.json",
        result_root / "metrics" / "summary.json",
    )
    for path in candidates:
        payload = load_json(path)
        if payload is None:
            continue
        cases = payload.get("cases", payload)
        if isinstance(cases, dict) and isinstance(cases.get(remover), dict):
            return cases[remover], str(path)
    return None, None


def mose_output_case(
    output_root: Path | None,
    sample_id: str,
    method_slug: str,
    remover: str,
) -> tuple[dict[str, Any] | None, str | None]:
    if output_root is None or method_slug not in REVIEW_NAMES:
        return None, None
    destination = output_root / "clips" / sample_id / "methods" / method_slug
    metadata = destination / ("recovery.json" if method_slug == "safe-sora" else "localization.json")
    payload = load_json(metadata)
    if payload is not None and isinstance(payload.get(remover), dict):
        return payload[remover], str(metadata)
    return None, str(metadata)


def aggregate(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "stddev_population": None, "minimum": None, "maximum": None}
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "stddev_population": statistics.pstdev(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=Path("runs/full50-watermark-evaluation/plan.json"))
    parser.add_argument("--review-manifest", type=Path, default=Path("evidence/mose-task1/review50/manifest.json"))
    parser.add_argument(
        "--mose-output-root",
        type=Path,
        help="Optional live fixed-MOSE output tree; preferred over the frozen review copy when a case exists.",
    )
    parser.add_argument("--output", type=Path, default=Path("evidence/full50-watermark-evaluation-summary.json"))
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    absolute = lambda path: path if path.is_absolute() else root / path
    # Preserve the repository-facing lexical path. On the execution server,
    # ``runs/full50-watermark-evaluation`` is a symlink to the large result
    # tree; resolving it made the committed summary point at a server-only
    # top-level path instead of the portable ``runs/.../plan.json`` file.
    plan_path = absolute(args.plan).absolute()
    review_manifest_path = absolute(args.review_manifest).absolute()
    output = absolute(args.output).resolve()
    mose_output_root = absolute(args.mose_output_root).resolve() if args.mose_output_root else None
    plan = json.loads(plan_path.read_text())
    review_manifest = json.loads(review_manifest_path.read_text())
    review_root = review_manifest_path.parent
    review_by_id = {sample["clip_id"]: sample for sample in review_manifest.get("samples", [])}

    methods: dict[str, Any] = {}
    total_complete = 0
    for method in METHODS:
        plan_method = plan["methods"][method.slug]
        metric_keys = [metric.key for metric in method.metrics]
        rows = []
        aggregate_values = {
            remover: {key: [] for key in metric_keys}
            for remover in REMOVERS
        }
        completed_by_remover = {remover: 0 for remover in REMOVERS}
        for job in plan_method["samples"]:
            sample_id = job["sample_id"]
            row = {"sample_id": sample_id, "input_status": job["status"], "removers": {}}
            for remover in REMOVERS:
                record = None
                evidence = None
                record, evidence = mose_output_case(mose_output_root, sample_id, method.slug, remover)
                if method.slug in REVIEW_NAMES and sample_id in review_by_id:
                    if record is None:
                        record, evidence = review_case(review_root, review_by_id[sample_id], method.slug, remover)
                if record is None:
                    result_root = Path(job["result_root"])
                    result_root = result_root if result_root.is_absolute() else root / result_root
                    record, evidence = generated_case(result_root, remover)
                metrics = canonical_metrics(record or {}, metric_keys)
                complete = len(metrics) == len(metric_keys)
                if complete:
                    completed_by_remover[remover] += 1
                    total_complete += 1
                    for key, value in metrics.items():
                        aggregate_values[remover][key].append(value)
                row["removers"][remover] = {
                    "status": "complete" if complete else "pending",
                    "metrics": metrics,
                    "evidence": evidence,
                }
            rows.append(row)
        aggregates = {
            remover: {key: aggregate(values) for key, values in keyed.items()}
            for remover, keyed in aggregate_values.items()
        }
        methods[method.slug] = {
            "label": method.label,
            "notebook": method.notebook,
            "input_policy": method.input_policy,
            "target_samples": SAMPLES_PER_METHOD,
            "key_metrics": [metric.__dict__ for metric in method.metrics],
            "completed_by_remover": completed_by_remover,
            "aggregates": aggregates,
            "samples": rows,
        }

    expected_cells = len(METHODS) * SAMPLES_PER_METHOD * len(REMOVERS)
    report = {
        "schema_version": 1,
        "created_at": utc_now(),
        "status": "complete" if total_complete == expected_cells else "partial",
        "plan": str(plan_path),
        "plan_sha256": sha256(plan_path),
        "review_manifest": str(review_manifest_path),
        "review_manifest_sha256": sha256(review_manifest_path),
        "requirements": plan["requirements"],
        "completion": {
            "complete_metric_cells": total_complete,
            "expected_metric_cells": expected_cells,
            "pending_metric_cells": expected_cells - total_complete,
        },
        "methods": methods,
    }
    report = portable_paths(report, root)
    atomic_json(output, report)
    print(f"Wrote {total_complete}/{expected_cells} complete metric cells to {output}")
    if args.require_complete and total_complete != expected_cells:
        raise SystemExit(f"Full-50 results are incomplete: {total_complete}/{expected_cells} metric cells")


if __name__ == "__main__":
    main()
