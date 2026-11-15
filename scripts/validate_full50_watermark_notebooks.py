#!/usr/bin/env python3
"""Validate that all 11 notebooks show metrics from one exact full-50 summary."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from full50_watermark_contract import METHODS, REMOVERS, SAMPLES_PER_METHOD


TAG = "full50-key-metrics"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=Path("evidence/full50-watermark-evaluation-summary.json"))
    parser.add_argument("--notebook-dir", type=Path, default=Path("notebooks/mose-watermark-samples"))
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    summary_path = args.summary if args.summary.is_absolute() else root / args.summary
    notebook_dir = args.notebook_dir if args.notebook_dir.is_absolute() else root / args.notebook_dir
    summary = json.loads(summary_path.read_text())
    digest = hashlib.sha256(summary_path.read_bytes()).hexdigest()
    for path_key, hash_key in (("plan", "plan_sha256"), ("review_manifest", "review_manifest_sha256")):
        source = Path(summary[path_key])
        source = source if source.is_absolute() else root / source
        if not source.is_file():
            raise RuntimeError(f"Summary source does not exist: {source}")
        actual = hashlib.sha256(source.read_bytes()).hexdigest()
        if actual != summary[hash_key]:
            raise RuntimeError(f"Summary {path_key} hash is stale: expected {summary[hash_key]}, found {actual}")
    requirements = summary["requirements"]
    if requirements["watermark_methods"] != 11 or requirements["samples_per_method"] != SAMPLES_PER_METHOD:
        raise RuntimeError("Summary does not declare the exact 11 x 50 contract")
    if tuple(requirements["removers"]) != REMOVERS:
        raise RuntimeError("Summary remover order does not match the full-50 contract")

    for method in METHODS:
        record = summary["methods"][method.slug]
        if len(record["samples"]) != SAMPLES_PER_METHOD:
            raise RuntimeError(f"{method.slug}: expected 50 per-sample rows")
        expected_keys = {metric.key for metric in method.metrics}
        if {metric["key"] for metric in record["key_metrics"]} != expected_keys:
            raise RuntimeError(f"{method.slug}: key metric definition drift")
        if args.require_complete and any(record["completed_by_remover"][name] != 50 for name in REMOVERS):
            raise RuntimeError(f"{method.slug}: full-50 metrics are incomplete")
        path = notebook_dir / method.notebook
        notebook = json.loads(path.read_text())
        tagged = [cell for cell in notebook["cells"] if TAG in cell.get("metadata", {}).get("tags", [])]
        if len(tagged) != 2 or {cell["cell_type"] for cell in tagged} != {"markdown", "code"}:
            raise RuntimeError(f"{path}: expected one tagged markdown and one tagged code cell")
        joined = "".join("".join(cell.get("source", [])) for cell in tagged)
        for required in (digest, method.slug, "50", "ProPainter", "E2FGVI-HQ"):
            if required not in joined:
                raise RuntimeError(f"{path}: full-50 cell is missing {required!r}")
    print(f"Validated full-50 metric sections in {len(METHODS)} notebooks against {summary_path}")


if __name__ == "__main__":
    main()
