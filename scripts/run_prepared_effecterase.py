#!/usr/bin/env python3
"""Run official EffectErase on one prepared watermark-removal sample."""

from __future__ import annotations

import argparse
import json
import time
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

from mose_watermark_common import atomic_json, decode_rgb, encode_rgb, sha256, utc_now, validate_video
from run_mose_effecterase import chunk_ranges, load_pipeline, processing_size, run_chunk


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--repository-commit", default="bcee0a5da5ef387c2ba39390dc4d579503669fb8")
    parser.add_argument("--device", default="0")
    parser.add_argument("--max-side", type=int, default=832)
    parser.add_argument("--max-model-frames", type=int, default=81)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--cfg", type=float, default=1.0)
    parser.add_argument("--lora-alpha", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--tiled", action="store_true")
    parser.add_argument("--use-teacache", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    run = args.run.resolve()
    sample = json.loads((run / "sample.json").read_text())
    input_video = run / "input-watermarked.mp4"
    mask_video = run / "gt-mask.mp4"
    output_video = run / "effecterase.mp4"
    metadata_path = run / "effecterase.json"
    expected_frames = int(sample["frames"])
    native_size = tuple(int(value) for value in sample["size"])
    fps = float(sample["fps"])

    validate_video(input_video, frames=expected_frames, size=native_size, fps=fps)
    validate_video(mask_video, frames=expected_frames, size=native_size, fps=fps)
    if output_video.is_file() and metadata_path.is_file() and not args.force:
        validate_video(output_video, frames=expected_frames, size=native_size, fps=fps)
        print(f"EffectErase already complete: {output_video}")
        return

    started = time.monotonic()
    frames, rate = decode_rgb(input_video)
    mask_frames, _mask_rate = decode_rgb(mask_video)
    masks = [cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in mask_frames]
    size = processing_size(native_size, args.max_side)
    pipeline, model_record = load_pipeline(args)

    output_frames: list[np.ndarray] = []
    temporal_chunks: list[dict[str, int]] = []
    for chunk_index, (start, end, drop) in enumerate(
        chunk_ranges(expected_frames, args.max_model_frames)
    ):
        generated, padded = run_chunk(
            pipeline,
            frames[start:end],
            masks[start:end],
            size,
            args,
            args.seed + chunk_index,
        )
        restored = [
            frame
            if frame.shape[1::-1] == native_size
            else cv2.resize(frame, native_size, interpolation=cv2.INTER_LANCZOS4)
            for frame in generated
        ]
        output_frames.extend(restored[drop:])
        temporal_chunks.append(
            {"start": start, "end_exclusive": end, "drop_prefix": drop, "padded_frames": padded}
        )
    if len(output_frames) != expected_frames:
        raise RuntimeError(f"EffectErase returned {len(output_frames)} frames; expected {expected_frames}")

    encode_rgb(output_video, output_frames, rate or Fraction(str(fps)), crf=18)
    output_info = validate_video(output_video, frames=expected_frames, size=native_size, fps=fps)
    atomic_json(
        metadata_path,
        {
            "status": "complete",
            "completed_at": utc_now(),
            "watermark_method": sample["watermark_method"],
            "sample_id": sample["sample_id"],
            "remover": "EffectErase",
            "protocol": "official EffectErase inference on the same authentic watermarked video and binary mask",
            "input": {"path": input_video.name, "sha256": sha256(input_video)},
            "mask": {"path": mask_video.name, "sha256": sha256(mask_video)},
            "output": {"path": output_video.name, "sha256": sha256(output_video), "video": output_info},
            "model": model_record,
            "processing_adapter": {
                "processing_size": list(size),
                "reported_size": list(native_size),
                "restoration": "Lanczos resize to the aligned input size",
            },
            "temporal_adapter": {
                "reason": "EffectErase uses Wan 4n+1 latent windows",
                "chunks": temporal_chunks,
                "trimmed_back_to_source_frames": True,
            },
            "settings": {
                "visible_device": args.device,
                "num_inference_steps": args.num_inference_steps,
                "cfg": args.cfg,
                "seed": args.seed,
                "tiled": args.tiled,
                "teacache": args.use_teacache,
            },
            "elapsed_seconds": round(time.monotonic() - started, 3),
        },
    )
    print(f"Completed EffectErase for {sample['watermark_method']}/{sample['sample_id']}")


if __name__ == "__main__":
    main()
