#!/usr/bin/env python3
"""Statically validate the eight remaining-method execution notebooks."""

from __future__ import annotations

import ast
import json
from pathlib import Path

from remaining_watermark_methods import METHODS


EXPECTED = {
    "rivagan": ("rivagan", "mose_post_hoc"),
    "watermark_anything": ("watermark-anything", "mose_post_hoc"),
    "vine": ("vine", "mose_post_hoc"),
    "videomark": ("videomark", "authors_provided"),
    "sigmark": ("sigmark", "authors_generated"),
    "vidsig": ("vidsig", "authors_generated"),
    "genptw": ("genptw", "mose_post_hoc"),
    "lvmark": ("lvmark", "authors_generated"),
}

REQUIRED_SCRIPTS = (
    "prepare_watermark_removal_sample.py",
    "run_watermark_removal_pair.py",
    "run_prepared_effecterase.py",
    "run_prepared_e2fgvi.py",
    "run_diffueraser_recorded.py",
    "make_native_removal_mask.py",
    "watermark_anything_video.py",
    "vine_video.py",
    "run_videomark_detection.py",
    "stage_sigmark_detection_case.py",
    "run_vidsig_native_sample.py",
    "run_vidsig_detection.py",
    "genptw_video.py",
)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    folder = root / "notebooks/mose-watermark-samples"
    registry = {method.slug: method for method in METHODS}
    failures: list[str] = []

    for script in REQUIRED_SCRIPTS:
        if not (root / "scripts" / script).is_file():
            failures.append(f"missing script: {script}")

    for filename, (slug, source_kind) in EXPECTED.items():
        path = folder / f"{filename}.ipynb"
        if not path.is_file():
            failures.append(f"missing notebook: {path}")
            continue
        try:
            payload = json.loads(path.read_text())
        except Exception as error:
            failures.append(f"invalid JSON {path}: {error}")
            continue
        if payload.get("nbformat") != 4:
            failures.append(f"{path}: nbformat is not 4")
        cells = payload.get("cells", [])
        combined = "\n".join(str(cell.get("source", "")) for cell in cells)
        spec = registry[slug]
        for required in (
            spec.verified_commit,
            "ProPainter",
            "DiffuEraser",
            "EffectErase",
            "E2FGVI-HQ",
            "prepare_watermark_removal_sample.py",
            "run_watermark_removal_pair.py",
            source_kind,
        ):
            if required not in combined:
                failures.append(f"{path}: missing protocol text {required!r}")
        if not spec.existing_video_compatible and "does not fake a MOSE embedding" not in combined:
            failures.append(f"{path}: missing in-generation authenticity warning")
        for index, cell in enumerate(cells):
            if cell.get("cell_type") != "code":
                continue
            try:
                ast.parse(str(cell.get("source", "")))
            except SyntaxError as error:
                failures.append(f"{path}: code cell {index} is invalid: {error}")

    if failures:
        raise SystemExit("Remaining-notebook validation failed:\n- " + "\n- ".join(failures))
    print(
        f"Validated {len(EXPECTED)} notebooks, {len(REQUIRED_SCRIPTS)} shared/adaptor scripts, "
        "all pinned commits, source-kind contracts, and Python code cells."
    )


if __name__ == "__main__":
    main()
