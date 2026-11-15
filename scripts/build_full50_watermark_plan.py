#!/usr/bin/env python3
"""Build the exact 11-method x 50-sample full watermark evaluation plan."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from full50_watermark_contract import METHODS, REMOVERS, SAMPLES_PER_METHOD


LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"
REVIEW_METHOD_NAMES = {
    "editguard": "EditGuard",
    "omniguard": "OmniGuard",
    "videoseal": "VideoSeal",
    "safe-sora": "Safe-Sora",
}

NATIVE_REQUIRED_INPUTS = {
    "videoshield": ("watermark_info",),
    "videomark": ("shift",),
    "sigmark": ("generated_output", "sample_key"),
    "vidsig": (),
}


def require_safe_id(value: Any, label: str) -> str:
    result = str(value or "")
    if not result or result in {".", ".."} or Path(result).name != result:
        raise RuntimeError(f"{label} must be a non-empty filename-safe identifier, found {value!r}")
    return result


def is_materialized_file(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    with path.open("rb") as handle:
        return handle.read(len(LFS_POINTER_PREFIX)) != LFS_POINTER_PREFIX


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def atomic_json(path: Path, value: Any) -> None:
    """Write plan metadata without importing heavyweight video dependencies."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def portable_paths(value: Any, root: Path) -> Any:
    if isinstance(value, dict):
        return {key: portable_paths(child, root) for key, child in value.items()}
    if isinstance(value, list):
        return [portable_paths(child, root) for child in value]
    if isinstance(value, str):
        candidate = Path(value)
        if candidate.is_absolute() and candidate.is_relative_to(root):
            return str(candidate.relative_to(root))
    return value


def parse_native_manifest(value: str) -> tuple[str, Path]:
    slug, separator, raw_path = value.partition("=")
    if not separator or not slug or not raw_path:
        raise argparse.ArgumentTypeError("Use METHOD=manifest.json")
    return slug, Path(raw_path)


def load_review_samples(review_root: Path) -> list[dict[str, Any]]:
    manifest_path = review_root / "manifest.json"
    payload = json.loads(manifest_path.read_text())
    samples = payload.get("samples", [])
    ids = [require_safe_id(sample.get("clip_id"), f"{manifest_path}: clip_id") for sample in samples]
    if len(samples) != SAMPLES_PER_METHOD or len(set(ids)) != SAMPLES_PER_METHOD:
        raise RuntimeError(
            f"Expected {SAMPLES_PER_METHOD} unique review samples in {manifest_path}, found {len(samples)}"
        )
    return samples


def review_path(review_root: Path, relative: str | None, label: str) -> str:
    if not relative:
        raise RuntimeError(f"Missing {label} path in review manifest")
    path = (review_root / relative).resolve()
    if not path.exists():
        raise RuntimeError(f"Missing {label}: {path}")
    return str(path)


def mose_jobs(method, review_root: Path, samples: list[dict[str, Any]], output_root: Path) -> list[dict[str, Any]]:
    jobs = []
    for sample in samples:
        clip_id = require_safe_id(sample["clip_id"], "review clip_id")
        result_root = output_root / "methods" / method.slug / clip_id
        if method.input_policy == "mose_existing":
            record = sample.get("watermarks", {}).get(REVIEW_METHOD_NAMES[method.slug])
            if not isinstance(record, dict) or record.get("status") != "complete":
                raise RuntimeError(f"{clip_id}/{method.slug}: review does not contain a complete watermark input")
            source_video = review_path(review_root, record.get("watermarked"), f"{clip_id}/{method.slug}/watermarked")
            mask_key = "gt_mask"
            source_mask = review_path(review_root, record.get(mask_key), f"{clip_id}/{method.slug}/{mask_key}")
        else:
            source_video = review_path(review_root, sample.get("media", {}).get("original"), f"{clip_id}/original")
            source_mask = review_path(review_root, sample.get("media", {}).get("gt_mask"), f"{clip_id}/gt_mask")
        ready = is_materialized_file(Path(source_video)) and (
            is_materialized_file(Path(source_mask)) or Path(source_mask).is_dir()
        )
        jobs.append(
            {
                "sample_id": clip_id,
                "status": "ready" if ready else "waiting_for_git_lfs",
                "source_kind": method.source_kind,
                "source_video": source_video,
                "source_mask": source_mask,
                "result_root": str(result_root.resolve()),
                "provenance": {
                    "review_manifest": str((review_root / "manifest.json").resolve()),
                    "fixed_selection_index": len(jobs),
                    **(
                        {}
                        if ready
                        else {
                            "reason": (
                                "The review paths exist but at least one input is an unmaterialized Git LFS pointer. "
                                "Run `python3 scripts/prepare_mose_notebooks.py --all` after Git LFS authentication."
                            )
                        }
                    ),
                },
            }
        )
    return jobs


def native_jobs(method, manifest_path: Path | None, output_root: Path) -> list[dict[str, Any]]:
    if manifest_path is None:
        return [
            {
                "sample_id": f"{method.slug}-{index:03d}",
                "status": "waiting_for_native_input",
                "source_kind": method.source_kind,
                "source_video": None,
                "source_mask": None,
                "result_root": str((output_root / "methods" / method.slug / f"{method.slug}-{index:03d}").resolve()),
                "provenance": {
                    "reason": (
                        "Generation-coupled methods require authentic native watermarked videos; "
                        "an existing MOSE clip is not a valid substitute."
                    )
                },
            }
            for index in range(SAMPLES_PER_METHOD)
        ]
    manifest_path = manifest_path.expanduser().resolve()
    payload = json.loads(manifest_path.read_text())
    if payload.get("method") != method.slug:
        raise RuntimeError(f"{manifest_path}: expected method={method.slug!r}")
    samples = payload.get("samples", [])
    ids = [require_safe_id(sample.get("sample_id"), f"{manifest_path}: sample_id") for sample in samples]
    if len(samples) != SAMPLES_PER_METHOD or len(set(ids)) != SAMPLES_PER_METHOD:
        raise RuntimeError(
            f"{manifest_path}: expected {SAMPLES_PER_METHOD} unique native samples, found {len(samples)}"
        )
    jobs = []
    manifest_root = manifest_path.parent
    seen_videos: set[Path] = set()
    for sample in samples:
        sample_id = require_safe_id(sample["sample_id"], f"{manifest_path}: sample_id")
        video = Path(sample["watermarked_video"]).expanduser()
        mask = Path(sample["source_mask"]).expanduser()
        video = (video if video.is_absolute() else manifest_root / video).resolve()
        mask = (mask if mask.is_absolute() else manifest_root / mask).resolve()
        if video in seen_videos:
            raise RuntimeError(f"{manifest_path}:{sample_id}: duplicate native watermarked video: {video}")
        seen_videos.add(video)
        missing = [str(video)] if not is_materialized_file(video) else []
        if not (is_materialized_file(mask) or mask.is_dir()):
            missing.append(str(mask))
        if missing:
            raise RuntimeError(f"{manifest_path}:{sample_id}: missing native input(s): {missing}")
        method_inputs = dict(sample.get("method_inputs", {}))
        missing_inputs = [key for key in NATIVE_REQUIRED_INPUTS[method.slug] if not method_inputs.get(key)]
        if missing_inputs:
            raise RuntimeError(f"{manifest_path}:{sample_id}: missing method_inputs {missing_inputs}")
        for key in ("watermark_info", "shift", "generated_output"):
            if key in method_inputs:
                value = Path(method_inputs[key]).expanduser()
                resolved = (value if value.is_absolute() else manifest_root / value).resolve()
                if not resolved.exists() or (resolved.is_file() and not is_materialized_file(resolved)):
                    raise RuntimeError(
                        f"{manifest_path}:{sample_id}: method_inputs.{key} does not exist: {resolved}"
                    )
                method_inputs[key] = str(resolved)
        provenance = dict(sample.get("provenance", {}))
        missing_provenance = [key for key in ("prompt", "seed", "source_commit") if provenance.get(key) in (None, "")]
        if missing_provenance:
            raise RuntimeError(f"{manifest_path}:{sample_id}: missing provenance {missing_provenance}")
        jobs.append(
            {
                "sample_id": sample_id,
                "status": "ready",
                "source_kind": method.source_kind,
                "source_video": str(video),
                "source_mask": str(mask),
                "result_root": str((output_root / "methods" / method.slug / sample_id).resolve()),
                "provenance": {
                    "native_manifest": str(manifest_path.resolve()),
                    **provenance,
                },
                "method_inputs": method_inputs,
            }
        )
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-root", type=Path, default=Path("evidence/mose-task1/review50"))
    parser.add_argument("--output-root", type=Path, default=Path("runs/full50-watermark-evaluation"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--native-manifest", action="append", default=[], type=parse_native_manifest)
    parser.add_argument(
        "--require-ready",
        action="store_true",
        help="Fail unless all four native-generation manifests are supplied and all 550 jobs are ready.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    review_root = args.review_root if args.review_root.is_absolute() else root / args.review_root
    output_root = args.output_root if args.output_root.is_absolute() else root / args.output_root
    output = args.output or output_root / "plan.json"
    output = output if output.is_absolute() else root / output
    native = dict(args.native_manifest)
    unknown = set(native) - {method.slug for method in METHODS if method.input_policy == "native_manifest"}
    if unknown:
        raise RuntimeError(f"Native manifests supplied for unknown/non-native methods: {sorted(unknown)}")

    review_samples = load_review_samples(review_root.resolve())
    methods: dict[str, Any] = {}
    for item in METHODS:
        if item.input_policy == "native_manifest":
            jobs = native_jobs(item, native.get(item.slug), output_root)
        else:
            jobs = mose_jobs(item, review_root.resolve(), review_samples, output_root)
        methods[item.slug] = {
            "label": item.label,
            "input_policy": item.input_policy,
            "source_kind": item.source_kind,
            "key_metrics": [metric.__dict__ for metric in item.metrics],
            "samples": jobs,
        }

    all_jobs = [job for record in methods.values() for job in record["samples"]]
    ready = sum(job["status"] == "ready" for job in all_jobs)
    status_counts = dict(sorted(Counter(job["status"] for job in all_jobs).items()))
    report = {
        "schema_version": 1,
        "created_at": utc_now(),
        "status": "ready" if ready == len(all_jobs) else "waiting_for_inputs",
        "requirements": {
            "watermark_methods": len(METHODS),
            "samples_per_method": SAMPLES_PER_METHOD,
            "removers": list(REMOVERS),
            "removal_cells": len(METHODS) * SAMPLES_PER_METHOD * len(REMOVERS),
        },
        "summary": {
            "jobs": len(all_jobs),
            "ready_jobs": ready,
            "waiting_jobs": len(all_jobs) - ready,
            "status_counts": status_counts,
        },
        "methods": methods,
    }
    report = portable_paths(report, root)
    atomic_json(output, report)
    print(f"Wrote {len(all_jobs)}-job plan ({ready} ready) to {output}")
    if args.require_ready and ready != len(all_jobs):
        blockers = []
        if status_counts.get("waiting_for_git_lfs"):
            blockers.append(
                "materialize the fixed MOSE Git LFS inputs with "
                "`python3 scripts/prepare_mose_notebooks.py --all`"
            )
        if status_counts.get("waiting_for_native_input"):
            blockers.append(
                "provide exactly 50 authentic native inputs for VideoShield, VideoMark, SIGMark, "
                "and VidSig with `--native-manifest METHOD=file.json`"
            )
        unexpected = sorted(
            key
            for key in status_counts
            if key not in {"ready", "waiting_for_git_lfs", "waiting_for_native_input"}
        )
        if unexpected:
            blockers.append(f"resolve unexpected job states: {unexpected}")
        raise SystemExit("Full-50 plan is not executable yet: " + "; ".join(blockers) + ".")


if __name__ == "__main__":
    main()
