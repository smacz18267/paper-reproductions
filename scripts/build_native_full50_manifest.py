#!/usr/bin/env python3
"""Build and validate one exact 50-sample native-generation manifest."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from mose_watermark_common import (
    atomic_json,
    decode_rgb,
    encode_binary_masks,
    sha256,
    utc_now,
    validate_video,
)


METHODS = ("videoshield", "videomark", "sigmark", "vidsig")


def portable_paths(value: Any, base: Path) -> Any:
    """Write relocatable manifest paths relative to the manifest directory."""
    if isinstance(value, dict):
        return {key: portable_paths(child, base) for key, child in value.items()}
    if isinstance(value, list):
        return [portable_paths(child, base) for child in value]
    if isinstance(value, str):
        candidate = Path(value)
        if candidate.is_absolute():
            return os.path.relpath(candidate, base)
    return value


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return value


def source_commit(source: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(source.resolve()), "rev-parse", "HEAD"], text=True
    ).strip()


def ensure_mask(video: Path, path: Path) -> Path:
    sidecar = path.with_suffix(".json")
    if path.is_file() and sidecar.is_file():
        source = validate_video(video, fps=None)
        validate_video(
            path,
            frames=int(source["frames"]),
            size=(int(source["width"]), int(source["height"])),
            fps=float(source["fps"]),
        )
        return path
    frames, rate = decode_rgb(video)
    height, width = frames[0].shape[:2]
    box_width = max(1, round(width * 0.35))
    box_height = max(1, round(height * 0.35))
    left = (width - box_width) // 2
    top = (height - box_height) // 2
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[top : top + box_height, left : left + box_width] = 255
    path.parent.mkdir(parents=True, exist_ok=True)
    encode_binary_masks(path, [mask for _ in frames], rate)
    atomic_json(
        sidecar,
        {
            "status": "complete",
            "protocol": "static centered rectangle",
            "provenance": (
                "experiment-defined deterministic region-inpainting mask; "
                "not an object annotation supplied by the watermark authors"
            ),
            "video": str(video.resolve()),
            "video_sha256": sha256(video),
            "mask_sha256": sha256(path),
            "frames": len(frames),
            "size": [width, height],
            "box": {"left": left, "top": top, "width": box_width, "height": box_height},
            "created_at": utc_now(),
        },
    )
    return path


def videoshield_samples(native_root: Path, source_root: Path) -> list[dict[str, Any]]:
    records = []
    for metadata_path in sorted((native_root / "videoshield" / "batches").glob("batch-*/generation.json")):
        batch = load_object(metadata_path)
        for index, item in enumerate(batch["samples"]):
            records.append(
                {
                    "video": Path(item["watermarked_video"]),
                    "prompt": item["prompt"],
                    "seed": int(batch["rng_seed"]),
                    "sequence_index": index,
                    "method_inputs": {"watermark_info": item["watermark_info"]},
                }
            )
    if len(records) != 50:
        raise RuntimeError(f"Expected 50 VideoShield generations, found {len(records)}")
    commit = source_commit(source_root / "VideoShield")
    return finalize_samples("videoshield", records, native_root, commit)


def videomark_samples(native_root: Path, source_root: Path) -> list[dict[str, Any]]:
    output = native_root / "videomark"
    generation = load_object(output / "generation.json")
    results_path = output / "videomark" / "modelscope" / "512bit" / "video_results.json"
    results = load_object(results_path)
    prompts = list(generation["prompts"])
    prompt_ids = sorted(
        ((prompt.replace(" ", "_"), prompt) for prompt in prompts),
        key=lambda item: len(item[0]),
        reverse=True,
    )
    records = []
    for sequence_index, key in enumerate(results):
        prompt = next((value for identifier, value in prompt_ids if key.startswith(identifier + "_")), None)
        if prompt is None:
            raise RuntimeError(f"Could not map VideoMark output key to prompt: {key}")
        directory = results_path.parent / key
        records.append(
            {
                "video": directory / "wm.mp4",
                "prompt": prompt,
                "seed": int(generation["rng_seed"]),
                "sequence_index": sequence_index,
                "method_inputs": {"shift": str(directory / "shift_value.npy")},
            }
        )
        if len(records) == 50:
            break
    if len(records) != 50:
        raise RuntimeError(f"Expected 50 VideoMark generations, found {len(records)}")
    commit = source_commit(source_root / "VideoMark")
    return finalize_samples("videomark", records, native_root, commit)


def selected_sigmark_prompts(source: Path) -> list[tuple[str, str]]:
    prompt_root = source / "prompt_set" / "VBench2_aug_prompt"
    selected: list[tuple[str, str]] = []
    for file in sorted(prompt_root.glob("*.txt")):
        dimension = file.stem
        prompts = [line.strip() for line in file.read_text().splitlines() if line.strip()]
        requested = 3
        interval = len(prompts) // requested
        if interval < 1:
            raise RuntimeError(f"Not enough prompts in {file}")
        indices = list(range(0, len(prompts), interval))[:requested]
        for index in indices:
            prompt = prompts[index]
            selected.append((prompt, f"{dimension}/{prompt[:180]}-0"))
    return selected[:50]


def sigmark_samples(native_root: Path, source_root: Path) -> list[dict[str, Any]]:
    source = source_root / "SIGMark-release"
    output = native_root / "sigmark"
    indexed = selected_sigmark_prompts(source)
    if len(indexed) != 50:
        raise RuntimeError(f"Expected 50 indexed SIGMark prompts, found {len(indexed)}")
    records = []
    for sequence_index, (prompt, sample_key) in enumerate(indexed):
        records.append(
            {
                "video": output / f"{sample_key}.mp4",
                "prompt": prompt,
                "seed": 2025,
                "sequence_index": sequence_index,
                "method_inputs": {
                    "generated_output": str(output),
                    "sample_key": sample_key,
                },
            }
        )
    return finalize_samples("sigmark", records, native_root, source_commit(source))


def vidsig_samples(native_root: Path, source_root: Path) -> list[dict[str, Any]]:
    records = []
    for directory in sorted((native_root / "vidsig").glob("vidsig-*")):
        metadata = load_object(directory / "generation.json")
        records.append(
            {
                "video": directory / "watermarked.mp4",
                "prompt": metadata["prompt"],
                "seed": int(metadata["seed"]),
                "sequence_index": int(metadata["prompt_index"]),
                "method_inputs": {},
            }
        )
    if len(records) != 50:
        raise RuntimeError(f"Expected 50 VidSig generations, found {len(records)}")
    return finalize_samples(
        "vidsig", records, native_root, source_commit(source_root / "Video-Signature")
    )


def finalize_samples(
    method: str,
    records: list[dict[str, Any]],
    native_root: Path,
    commit: str,
) -> list[dict[str, Any]]:
    samples = []
    seen: set[Path] = set()
    for index, record in enumerate(records):
        video = Path(record["video"]).resolve()
        if not video.is_file() or video.stat().st_size == 0:
            raise FileNotFoundError(f"Missing native {method} video: {video}")
        if video in seen:
            raise RuntimeError(f"Duplicate native {method} video: {video}")
        seen.add(video)
        sample_id = f"{method}-{index:03d}"
        mask = ensure_mask(video, native_root / "masks" / method / f"{sample_id}.mp4")
        samples.append(
            {
                "sample_id": sample_id,
                "watermarked_video": str(video),
                "source_mask": str(mask.resolve()),
                "method_inputs": record["method_inputs"],
                "provenance": {
                    "prompt": record["prompt"],
                    "seed": record["seed"],
                    "source_commit": commit,
                    "rng_sequence_index": record["sequence_index"],
                    "mask_protocol": "deterministic centered 35% x 35% region",
                },
            }
        )
    if len(samples) != 50:
        raise RuntimeError(f"Expected 50 finalized {method} samples, found {len(samples)}")
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--native-root", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    native_root = args.native_root.resolve()
    source_root = args.source_root.resolve()
    builders = {
        "videoshield": videoshield_samples,
        "videomark": videomark_samples,
        "sigmark": sigmark_samples,
        "vidsig": vidsig_samples,
    }
    samples = builders[args.method](native_root, source_root)
    output = args.output.resolve()
    payload = {
        "schema_version": 1,
        "method": args.method,
        "sample_count": len(samples),
        "mask_protocol": (
            "Every native generation receives the same deterministic centered 35% x 35% "
            "region-inpainting mask. These are not author-provided object annotations."
        ),
        "samples": portable_paths(samples, output.parent),
        "created_at": utc_now(),
    }
    atomic_json(output, payload)
    print(f"Wrote 50-sample {args.method} manifest to {output}")


if __name__ == "__main__":
    main()
