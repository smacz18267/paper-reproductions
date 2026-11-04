#!/usr/bin/env python3
"""Embed EditGuard framewise into MOSE clips or localize removal attacks."""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np
import torch

from image_watermark_metrics import (
    effective_attack_mask,
    fidelity,
    localization_metrics,
    mask_video,
    processing_size,
)
from mose_watermark_common import (
    FPS,
    append_jsonl,
    atomic_json,
    decode_rgb,
    encode_binary_masks,
    encode_rgb,
    load_clips,
    method_dir,
    original_video,
    raw_mask_frames,
    read_mask_sequence,
    sha256,
    utc_now,
    validate_video,
)


METHOD = "editguard"
RELEASE_SIZE = (512, 512)
RELEASE_THRESHOLD = 0.2


def choose_clips(args: argparse.Namespace):
    clips = load_clips(args.selection)
    if not args.only:
        return clips
    requested = set(args.only)
    selected = [clip for clip in clips if clip.clip_id in requested]
    missing = requested - {clip.clip_id for clip in selected}
    if missing:
        raise RuntimeError(f"Unknown --only clip(s): {sorted(missing)}")
    return selected


def load_model(source: Path, checkpoint: Path):
    code_root = source.resolve() / "code"
    os.chdir(code_root)
    sys.path.insert(0, str(code_root))
    # BasicSR 1.4.2 imports torchvision's former functional_tensor module.
    # torchvision 0.19 exposes the same functions from functional instead.
    import torchvision.transforms.functional as torchvision_functional

    sys.modules.setdefault(
        "torchvision.transforms.functional_tensor", torchvision_functional
    )
    import options.options as option
    from models import create_model

    opt = option.parse("options/test_editguard.yml", is_train=True)
    opt["dist"] = False
    opt["gpu_ids"] = [0]
    # The experiment supplies external removal attacks. Do not initialize the
    # demo's optional diffusion attack pipelines while loading EditGuard.
    for key in ("sdinpaint", "controlnetinpaint", "sdxl", "repaint"):
        opt[key] = False
    opt = option.dict_to_nonedict(opt)
    model = create_model(opt)
    model.load_test(str(checkpoint.resolve()))
    model.netG.eval()
    return model


def model_input(frame: np.ndarray) -> dict[str, torch.Tensor]:
    # This is the released app's load_image transformation, including its
    # RGB/BGR channel handling and 512x512 spatial protocol.
    image = frame.astype(np.float32) / 255.0
    image = image[:, :, [2, 1, 0]]
    gt = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float()[None, None]
    height, width = frame.shape[:2]
    blue = np.zeros((1, height, width, 3), dtype=np.float32)
    blue[:, :, :, 2] = 1.0
    secret = torch.from_numpy(blue).permute(0, 3, 1, 2)[None, None]
    return {"LQ": secret, "GT": gt}


def fixed_message(path: Path, seed: int) -> tuple[str, np.ndarray]:
    generator = random.Random(seed)
    bits = "".join(str(generator.randint(0, 1)) for _ in range(64))
    if path.exists() and path.read_text().strip() != bits:
        raise RuntimeError(f"Saved EditGuard message does not match seed {seed}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(bits + "\n")
    return bits, np.asarray([int(bit) for bit in bits], dtype=np.float32) - 0.5


def embed(args: argparse.Namespace, model) -> None:
    clips = choose_clips(args)
    bits, message = fixed_message(args.output_root / "editguard-message.txt", args.seed)
    checkpoint = checkpoint_record(args.checkpoint)
    events = args.output_root / "editguard-embed-events.jsonl"
    failures = 0
    for ordinal, clip in enumerate(clips, 1):
        destination = method_dir(args.output_root, clip.clip_id, METHOD)
        watermarked_path = destination / "watermarked.mp4"
        aligned_path = destination / "aligned-original.mp4"
        gt_path = destination / "gt-mask.mp4"
        metrics_path = destination / "embed.json"
        try:
            if watermarked_path.is_file() and metrics_path.is_file() and not args.force:
                validate_video(watermarked_path, frames=clip.frame_count, size=RELEASE_SIZE)
                print(f"[{ordinal}/{len(clips)}] skip EditGuard embed {clip.clip_id}", flush=True)
                continue
            started = time.monotonic()
            source_path = original_video(args.dataset_root, clip.clip_id)
            source_info = validate_video(source_path, frames=clip.frame_count)
            frames, rate = decode_rgb(source_path)
            aligned = [cv2.resize(frame, RELEASE_SIZE, interpolation=cv2.INTER_LANCZOS4) for frame in frames]
            marked: list[np.ndarray] = []
            model.mes = message
            for frame_index, frame in enumerate(aligned, 1):
                model.feed_data(model_input(frame))
                output = model.image_hiding()
                marked.append(np.asarray(output, dtype=np.uint8))
                if frame_index % 10 == 0 or frame_index == len(aligned):
                    print(f"  {clip.clip_id}: embedded {frame_index}/{len(aligned)} frames", flush=True)
            masks = [
                cv2.resize(mask, RELEASE_SIZE, interpolation=cv2.INTER_NEAREST) > 127
                for mask in read_mask_sequence(raw_mask_frames(args.dataset_root, clip.clip_id))
            ]
            encode_rgb(aligned_path, aligned, rate, crf=18)
            encode_rgb(watermarked_path, marked, rate, crf=18)
            encode_binary_masks(gt_path, masks, rate)
            metrics = {
                "status": "complete",
                "method": METHOD,
                "protocol": "framewise_postprocess_original_then_inpaint",
                "clip_id": clip.clip_id,
                "repository_commit": args.repository_commit,
                "checkpoint": checkpoint,
                "seed": args.seed,
                "message": bits,
                "message_bits": len(bits),
                "release_adapter": {
                    "reason": "The released EditGuard app resizes every input image to 512x512.",
                    "resize": list(RELEASE_SIZE),
                    "framewise": True,
                },
                "source": source_info,
                "aligned_source": validate_video(aligned_path, frames=clip.frame_count, size=RELEASE_SIZE),
                "watermarked": validate_video(watermarked_path, frames=clip.frame_count, size=RELEASE_SIZE),
                "gt_mask": validate_video(gt_path, frames=clip.frame_count, size=RELEASE_SIZE),
                "embedding_fidelity": fidelity(aligned, marked),
                "sha256": {
                    "aligned_source": sha256(aligned_path),
                    "watermarked": sha256(watermarked_path),
                    "gt_mask": sha256(gt_path),
                },
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "completed_at": utc_now(),
            }
            atomic_json(metrics_path, metrics)
            append_jsonl(events, metrics)
            print(f"[{ordinal}/{len(clips)}] complete EditGuard embed {clip.clip_id}", flush=True)
        except Exception as error:
            failures += 1
            append_jsonl(events, {"status": "failed", "clip_id": clip.clip_id, "error": repr(error), "timestamp": utc_now()})
            print(f"[{ordinal}/{len(clips)}] FAILED EditGuard embed {clip.clip_id}: {error!r}", flush=True)
    if failures:
        raise SystemExit(f"EditGuard embedding failed for {failures} clips")


def checkpoint_record(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": sha256(path.resolve())}


def localize(args: argparse.Namespace, model) -> None:
    clips = choose_clips(args)
    selected_removers = tuple(args.remover or ("propainter", "diffueraser"))
    bits, _message = fixed_message(args.output_root / "editguard-message.txt", args.seed)
    expected_bits = np.asarray([int(bit) for bit in bits], dtype=np.uint8)
    events = args.output_root / "editguard-localize-events.jsonl"
    failures = 0
    for ordinal, clip in enumerate(clips, 1):
        destination = method_dir(args.output_root, clip.clip_id, METHOD)
        metrics_path = destination / "localization.json"
        try:
            existing: dict[str, object] = {}
            if metrics_path.is_file():
                existing = __import__("json").loads(metrics_path.read_text())
                if not args.force and all(
                    remover in existing
                    and (destination / f"pred-mask-{remover}.mp4").is_file()
                    and (destination / f"effective-gt-mask-{remover}.mp4").is_file()
                    for remover in selected_removers
                ):
                    print(f"[{ordinal}/{len(clips)}] skip EditGuard localization {clip.clip_id}", flush=True)
                    continue
            started = time.monotonic()
            base_masks = mask_video(destination / "gt-mask.mp4")
            results: dict[str, object] = {
                remover: existing[remover]
                for remover in ("propainter", "diffueraser", "effecterase")
                if remover in existing and remover not in selected_removers
            }
            for remover in selected_removers:
                attacked_path = destination / f"{remover}.mp4"
                attacked_info = validate_video(attacked_path, frames=clip.frame_count, size=RELEASE_SIZE)
                frames, rate = decode_rgb(attacked_path)
                predicted: list[np.ndarray] = []
                accuracies: list[float] = []
                for frame_index, frame in enumerate(frames, 1):
                    model.feed_data(model_input(frame))
                    mask, recovered = model.image_recovery(RELEASE_THRESHOLD)
                    predicted.append(np.asarray(mask) > 0)
                    recovered_bits = recovered.detach().cpu().numpy().reshape(-1).astype(np.uint8)
                    accuracies.append(float((recovered_bits == expected_bits).mean()))
                    if frame_index % 10 == 0 or frame_index == len(frames):
                        print(f"  {clip.clip_id}/{remover}: detected {frame_index}/{len(frames)} frames", flush=True)
                inference_size = processing_size(destination / f"{remover}.json", RELEASE_SIZE)
                expected = [
                    effective_attack_mask(mask, remover, RELEASE_SIZE, inference_size)
                    for mask in base_masks
                ]
                pred_path = destination / f"pred-mask-{remover}.mp4"
                gt_path = destination / f"effective-gt-mask-{remover}.mp4"
                encode_binary_masks(pred_path, predicted, rate)
                encode_binary_masks(gt_path, expected, Fraction(FPS, 1))
                results[remover] = {
                    **localization_metrics(predicted, expected),
                    "mean_message_bit_accuracy": float(np.mean(accuracies)),
                    "minimum_message_bit_accuracy": float(np.min(accuracies)),
                    "maximum_message_bit_accuracy": float(np.max(accuracies)),
                    "per_frame_message_bit_accuracy": accuracies,
                    "attacked": attacked_info,
                    "predicted_mask_sha256": sha256(pred_path),
                    "effective_gt_mask_sha256": sha256(gt_path),
                }
            metrics = {
                "status": "complete",
                "method": METHOD,
                "protocol": "method_native_tamper_localization_after_object_removal",
                "clip_id": clip.clip_id,
                "repository_commit": args.repository_commit,
                "detector_residual_threshold": RELEASE_THRESHOLD,
                "message_bits": len(bits),
                "mask_binarization": "any non-zero residual channel after the released 0.2 threshold",
                **results,
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "completed_at": utc_now(),
            }
            atomic_json(metrics_path, metrics)
            append_jsonl(events, metrics)
            print(f"[{ordinal}/{len(clips)}] complete EditGuard localization {clip.clip_id}", flush=True)
        except Exception as error:
            failures += 1
            append_jsonl(events, {"status": "failed", "clip_id": clip.clip_id, "error": repr(error), "timestamp": utc_now()})
            print(f"[{ordinal}/{len(clips)}] FAILED EditGuard localization {clip.clip_id}: {error!r}", flush=True)
    if failures:
        raise SystemExit(f"EditGuard localization failed for {failures} clips")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("embed", "localize"))
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--repository-commit", default="174778fb371640bb31b32fcb9f7bfd440c88766b")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument(
        "--remover",
        action="append",
        choices=("propainter", "diffueraser", "effecterase"),
        help="Localization attack to process; repeat for multiple. Defaults to the original two.",
    )
    parser.add_argument("--only", action="append")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("EditGuard inference requires CUDA")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu))
    model = load_model(args.source, args.checkpoint)
    if args.action == "embed":
        embed(args, model)
    else:
        localize(args, model)


if __name__ == "__main__":
    main()
