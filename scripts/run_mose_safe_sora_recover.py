#!/usr/bin/env python3
"""Recover Safe-Sora's graphical watermark after each object-removal attack."""

from __future__ import annotations

import argparse
import gc
import math
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from mose_watermark_common import (
    append_jsonl,
    atomic_json,
    decode_rgb,
    load_clips,
    method_dir,
    sha256,
    utc_now,
    validate_video,
)


def recover_half(extractor, revert_order, video: torch.Tensor) -> torch.Tensor:
    watermark_exact = extractor(video).permute(0, 2, 1, 3, 4)
    watermark_exact = watermark_exact.view(1, 4, 8, 4, 16, 8, 16)
    watermark_exact = watermark_exact.permute(0, 1, 2, 3, 5, 4, 6).contiguous()
    watermark_exact = watermark_exact.view(1, 4, 256, 16, 16)

    position_channel = watermark_exact[:, 3:, :, :, :]
    batch = position_channel.size(0)
    bit_prob = ((position_channel + 1) / 2.0).view(batch, 256, 8, -1).mean(-1)
    confidence = (bit_prob - 0.5).abs().mean(-1)
    bits = (bit_prob > 0.5).long()
    weights = (1 << torch.arange(8, device=bits.device)).long()
    positions = (bits * weights).sum(-1).view(-1)
    confidence = confidence.view(-1)

    index_map = torch.full((batch, 256), -1, dtype=torch.long, device=positions.device)
    best_confidence = torch.full((batch, 256), -1.0, device=positions.device)
    for global_index, (position, score) in enumerate(zip(positions, confidence)):
        batch_index = global_index // 256
        if score > best_confidence[batch_index, position]:
            best_confidence[batch_index, position] = score
            index_map[batch_index, position] = global_index

    for batch_index in range(batch):
        all_ids = torch.arange(batch_index * 256, (batch_index + 1) * 256, device=positions.device)
        kept_ids = index_map[batch_index][index_map[batch_index] != -1]
        lost_ids = torch.tensor(
            list(set(all_ids.tolist()) - set(kept_ids.tolist())),
            device=positions.device,
            dtype=torch.long,
        )
        if lost_ids.numel() == 0:
            continue
        vacant = (index_map[batch_index] == -1).nonzero(as_tuple=False).squeeze(1)
        distance = (positions[lost_ids].unsqueeze(1) - vacant.unsqueeze(0)).abs()
        assignments = distance.argmin(dim=1)
        for lost, vacancy_index in zip(lost_ids, assignments):
            vacancy = vacant[vacancy_index]
            if index_map[batch_index, vacancy] == -1:
                index_map[batch_index, vacancy] = lost

    batch, channels, patches, height, width = watermark_exact.size()
    flattened = watermark_exact.permute(0, 2, 1, 3, 4).contiguous().view(batch, patches, -1)
    reordered = revert_order(flattened, index_map)
    watermark = reordered.view(batch, patches, channels, height, width)
    watermark = watermark.permute(0, 2, 1, 3, 4).contiguous()[:, :3]
    watermark = watermark.reshape(1, 3, 16, 16, 16, 16)
    return watermark.permute(0, 1, 2, 4, 3, 5).contiguous().view(1, 3, 256, 256)


def fidelity(reference: np.ndarray, recovered: np.ndarray) -> dict[str, float | None]:
    difference = reference.astype(np.float64) - recovered.astype(np.float64)
    mse = float(np.mean(difference * difference))
    return {
        "mean_absolute_error": float(np.mean(np.abs(difference))),
        "mse": mse,
        "psnr_db": None if mse == 0 else float(20 * math.log10(255.0) - 10 * math.log10(mse)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--watermark", required=True, type=Path)
    parser.add_argument("--only", action="append")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("Safe-Sora recovery requires CUDA")
    clips = load_clips(args.selection)
    if args.only:
        requested = set(args.only)
        clips = [clip for clip in clips if clip.clip_id in requested]
        if {clip.clip_id for clip in clips} != requested:
            raise RuntimeError(f"Unknown --only clip(s): {sorted(requested - {clip.clip_id for clip in clips})}")

    source = args.source.resolve()
    sys.path.insert(0, str(source))
    previous = Path.cwd()
    os.chdir(source)
    try:
        import test as safe_test
    finally:
        os.chdir(previous)
    extractor = safe_test.wm_extractor.eval().cuda()

    watermark = cv2.imread(str(args.watermark.resolve()), cv2.IMREAD_COLOR)
    if watermark is None:
        raise RuntimeError(f"Could not read watermark: {args.watermark}")
    watermark = cv2.cvtColor(cv2.resize(watermark, (256, 256)), cv2.COLOR_BGR2RGB)
    reference = torch.from_numpy(watermark).float().permute(2, 0, 1)[None].cuda() / 255.0
    reference = reference * 2.0 - 1.0
    events = args.output_root / "safe-sora-recovery-events.jsonl"
    failures = 0

    for ordinal, clip in enumerate(clips, 1):
        destination = method_dir(args.output_root, clip.clip_id, "safe-sora")
        summary_path = destination / "recovery.json"
        try:
            outputs = {remover: destination / f"recovery-{remover}.png" for remover in ("propainter", "diffueraser")}
            if summary_path.is_file() and all(path.is_file() for path in outputs.values()) and not args.force:
                print(f"[{ordinal}/50] skip Safe-Sora recovery {clip.clip_id}", flush=True)
                continue

            result: dict[str, object] = {
                "status": "complete",
                "method": "safe-sora",
                "clip_id": clip.clip_id,
                "output_semantics": "graphical watermark recovery; not a spatial tamper mask",
                "started_at": utc_now(),
            }
            started = time.monotonic()
            for remover, output in outputs.items():
                attack = destination / f"{remover}.mp4"
                validate_video(attack, frames=16, size=(512, 320))
                frames, _rate = decode_rgb(attack)
                tensor = torch.from_numpy(np.stack(frames)).float().permute(0, 3, 1, 2)[None].cuda()
                tensor = tensor / 127.5 - 1.0
                with torch.inference_mode():
                    first = recover_half(extractor, safe_test.revert_order, tensor[:, :8])
                    second = recover_half(extractor, safe_test.revert_order, tensor[:, 8:])
                    recovered = torch.clamp((first + second) / 2.0, -1.0, 1.0)
                    composite = torch.cat(
                        [
                            (reference + 1.0) / 2.0,
                            (recovered + 1.0) / 2.0,
                            torch.abs(reference - recovered) * 2.5,
                        ],
                        dim=-1,
                    )
                safe_test.save_image(composite, str(output))
                recovered_rgb = ((recovered[0].permute(1, 2, 0).cpu().numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
                result[remover] = {
                    "image": output.name,
                    "sha256": sha256(output),
                    "fidelity": fidelity(watermark, recovered_rgb),
                }
                del tensor, first, second, recovered, composite
            result["elapsed_seconds"] = round(time.monotonic() - started, 3)
            result["completed_at"] = utc_now()
            atomic_json(summary_path, result)
            append_jsonl(events, result)
            print(f"[{ordinal}/50] complete Safe-Sora recovery {clip.clip_id}", flush=True)
        except Exception as error:
            failures += 1
            append_jsonl(events, {"status": "failed", "clip_id": clip.clip_id, "error": repr(error), "timestamp": utc_now()})
            print(f"[{ordinal}/50] FAILED Safe-Sora recovery {clip.clip_id}: {error!r}", flush=True)
        finally:
            gc.collect()
            torch.cuda.empty_cache()
    if failures:
        raise SystemExit(f"Safe-Sora recovery failed for {failures} clips")


if __name__ == "__main__":
    main()
