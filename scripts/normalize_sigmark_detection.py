#!/usr/bin/env python3
"""Normalize SIGMark extractor outputs into the shared per-case JSON schema."""

from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def parse_named(value: str) -> tuple[str, Path]:
    name, separator, raw_path = value.partition("=")
    if not separator or not name or not raw_path:
        raise argparse.ArgumentTypeError("Use NAME=/path/to/extractor-output-or-log")
    return name, Path(raw_path)


def numeric(value: Any) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        result = float(value)
        return result if 0.0 <= result <= 1.0 else None
    return None


def values_from_json(value: Any, path: str = "") -> list[tuple[int, str, float]]:
    found: list[tuple[int, str, float]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            current = f"{path}.{key}" if path else key
            normalized = key.lower().replace(" ", "_").replace("-", "_")
            number = numeric(child)
            if number is not None and normalized in {"bit_accuracy", "watermark_bit_accuracy"}:
                found.append((0, current, number))
            elif number is not None and normalized in {"accuracy", "acc"}:
                found.append((1, current, number))
            found.extend(values_from_json(child, current))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(values_from_json(child, f"{path}[{index}]"))
    return found


def candidates(path: Path) -> list[tuple[int, str, float]]:
    found: list[tuple[int, str, float]] = []
    files = [path] if path.is_file() else sorted(path.rglob("*"))
    for file in files:
        if not file.is_file():
            continue
        if file.suffix.lower() == ".json":
            try:
                found.extend(values_from_json(json.loads(file.read_text()), str(file)))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
        elif file.suffix.lower() == ".npz":
            try:
                with np.load(file) as archive:
                    for key in archive.files:
                        normalized_key = key.lower()
                        archive_is_bit_accuracy = "bit_accuracy" in file.stem.lower()
                        if (
                            normalized_key
                            not in {"bit_accuracy", "watermark_bit_accuracy", "accuracy", "acc"}
                            and not archive_is_bit_accuracy
                        ):
                            continue
                        array = np.asarray(archive[key])
                        if array.size == 1:
                            value = numeric(array.reshape(-1)[0].item())
                            if value is not None:
                                priority = 0 if archive_is_bit_accuracy or "bit" in normalized_key else 1
                                found.append((priority, f"{file}:{key}", value))
            except (OSError, ValueError):
                continue
        elif file.suffix.lower() in {".log", ".txt", ".out"}:
            try:
                text = file.read_text(errors="replace")
            except OSError:
                continue
            patterns = (
                (0, r"(?:bit|watermark)[ _-]*accuracy\s*[:=]\s*(0(?:\.\d+)?|1(?:\.0+)?)"),
                (0, r"\bbit\s+acc\s*[:=]\s*(0(?:\.\d+)?|1(?:\.0+)?)"),
                (1, r"\baccuracy\s*[:=]\s*(0(?:\.\d+)?|1(?:\.0+)?)"),
            )
            for priority, pattern in patterns:
                for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                    found.append((priority, f"{file}:log", float(match.group(1))))
    return found


def choose(path: Path) -> tuple[float, str]:
    found = candidates(path)
    if not found:
        raise RuntimeError(f"No scalar bit-accuracy value found under {path}")
    priority = min(item[0] for item in found)
    preferred = [(source, value) for item_priority, source, value in found if item_priority == priority]
    distinct = sorted({round(value, 12) for _source, value in preferred})
    if len(distinct) != 1:
        evidence = ", ".join(f"{source}={value}" for source, value in preferred)
        raise RuntimeError(f"Ambiguous SIGMark bit accuracies under {path}: {evidence}")
    return preferred[-1][1], preferred[-1][0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", required=True, type=parse_named)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    cases = {}
    for name, path in args.case:
        accuracy, evidence = choose(path.resolve())
        cases[name] = {"bit_accuracy": accuracy, "extractor_evidence": evidence}
    payload = {
        "status": "complete",
        "method": "SIGMark",
        "cases": cases,
        "completed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(f"Normalized {len(cases)} SIGMark detector cases to {args.output}")


if __name__ == "__main__":
    main()
