#!/usr/bin/env python3
"""Embed OmniGuard framewise into MOSE clips or localize the two removal attacks."""

from __future__ import annotations

import argparse
import os
import sys
import time
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from image_watermark_metrics import (
    effective_attack_mask,
    fidelity,
    localization_metrics,
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


METHOD = "omniguard"
DETECTOR_INPUT_SIZE = 1024
DETECTOR_THRESHOLD = 0.5


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


def load_embedding_model(source: Path, checkpoint: Path):
    source = source.resolve()
    os.chdir(source)
    sys.path.insert(0, str(source))
    import config as config
    from model_invert import Model, init_model

    model = Model().cuda()
    init_model(model)
    model = torch.nn.DataParallel(model, device_ids=[0])
    state = torch.load(str(checkpoint.resolve()), map_location="cpu")
    weights = {
        key: value
        for key, value in state["net"].items()
        if "tmp_var" not in key and "bm" not in key
    }
    model.load_state_dict(weights, strict=False)
    model.eval()
    return model, config


def load_detector(source: Path, checkpoint: Path):
    source = source.resolve()
    os.chdir(source)
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    from iml_vit_model import iml_vit_model

    detector = iml_vit_model()
    state = torch.load(str(checkpoint.resolve()), map_location="cpu")
    detector.load_state_dict(state["model"], strict=True)
    detector = detector.cuda().eval()
    return detector


def hide(model, frame: np.ndarray, secret_image: np.ndarray, dwt) -> np.ndarray:
    cover = torch.from_numpy(frame).permute(2, 0, 1)[None].float().cuda() / 255.0
    height, width = frame.shape[:2]
    secret = cv2.resize(secret_image, (width, height), interpolation=cv2.INTER_LANCZOS4)
    secret = torch.from_numpy(secret).permute(2, 0, 1)[None].float().cuda() / 255.0
    with torch.inference_mode():
        steg, _output_z, _out_temp, _secret_temp = model(dwt(cover), dwt(secret))
    return (steg[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255.0).astype(np.uint8)


def embed(args: argparse.Namespace, model) -> None:
    from modules.Unet_common import DWT

    dwt = DWT()
    secret_path = args.source.resolve() / "bluesky_white2.png"
    secret_bgr = cv2.imread(str(secret_path), cv2.IMREAD_COLOR)
    if secret_bgr is None:
        raise RuntimeError(f"Could not read OmniGuard secret image {secret_path}")
    secret_image = cv2.cvtColor(secret_bgr, cv2.COLOR_BGR2RGB)
    clips = choose_clips(args)
    events = args.output_root / "omniguard-embed-events.jsonl"
    failures = 0
    for ordinal, clip in enumerate(clips, 1):
        destination = method_dir(args.output_root, clip.clip_id, METHOD)
        watermarked_path = destination / "watermarked.mp4"
        metrics_path = destination / "embed.json"
        try:
            source_path = original_video(args.dataset_root, clip.clip_id)
            source_info = validate_video(source_path, frames=clip.frame_count)
            native_size = (int(source_info["width"]), int(source_info["height"]))
            if watermarked_path.is_file() and metrics_path.is_file() and not args.force:
                validate_video(watermarked_path, frames=clip.frame_count, size=native_size)
                print(f"[{ordinal}/{len(clips)}] skip OmniGuard embed {clip.clip_id}", flush=True)
                continue
            started = time.monotonic()
            frames, rate = decode_rgb(source_path)
            marked: list[np.ndarray] = []
            for frame_index, frame in enumerate(frames, 1):
                marked.append(hide(model, frame, secret_image, dwt))
                if frame_index % 10 == 0 or frame_index == len(frames):
                    print(f"  {clip.clip_id}: embedded {frame_index}/{len(frames)} frames", flush=True)
            encode_rgb(watermarked_path, marked, rate, crf=18)
            metrics = {
                "status": "complete",
                "method": METHOD,
                "protocol": "framewise_postprocess_original_then_inpaint",
                "clip_id": clip.clip_id,
                "repository_commit": args.repository_commit,
                "checkpoint": checkpoint_record(args.checkpoint),
                "secret_image": {"path": str(secret_path), "sha256": sha256(secret_path)},
                "release_adapter": {
                    "reason": "OmniGuard is released for images; each source frame is processed independently.",
                    "resize": None,
                    "framewise": True,
                },
                "source": source_info,
                "watermarked": validate_video(watermarked_path, frames=clip.frame_count, size=native_size),
                "embedding_fidelity": fidelity(frames, marked),
                "sha256": sha256(watermarked_path),
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "completed_at": utc_now(),
            }
            atomic_json(metrics_path, metrics)
            append_jsonl(events, metrics)
            print(f"[{ordinal}/{len(clips)}] complete OmniGuard embed {clip.clip_id}", flush=True)
        except Exception as error:
            failures += 1
            append_jsonl(events, {"status": "failed", "clip_id": clip.clip_id, "error": repr(error), "timestamp": utc_now()})
            print(f"[{ordinal}/{len(clips)}] FAILED OmniGuard embed {clip.clip_id}: {error!r}", flush=True)
    if failures:
        raise SystemExit(f"OmniGuard embedding failed for {failures} clips")


def checkpoint_record(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": sha256(path.resolve())}


def normalized_detector_input(tensor: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
    while tensor.shape[-2] > DETECTOR_INPUT_SIZE or tensor.shape[-1] > DETECTOR_INPUT_SIZE:
        tensor = F.interpolate(tensor, scale_factor=0.5, mode="bilinear", align_corners=False)
    height, width = tensor.shape[-2:]
    tensor = F.pad(tensor, (0, DETECTOR_INPUT_SIZE - width, 0, DETECTOR_INPUT_SIZE - height), value=0)
    mean = torch.tensor([0.485, 0.456, 0.406], device=tensor.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=tensor.device).view(1, 3, 1, 1)
    return (tensor - mean) / std, (height, width)


def reveal(model, detector, frame: np.ndarray, dwt, iwt) -> np.ndarray:
    steg = torch.from_numpy(frame).permute(2, 0, 1)[None].float().cuda() / 255.0
    original_size = (frame.shape[1], frame.shape[0])
    with torch.inference_mode():
        output = model(dwt(steg), rev=True)
        artifact = iwt(output.narrow(1, 0, 12))
        artifact_input, detected_size = normalized_detector_input(artifact)
        fuse_input, fuse_size = normalized_detector_input(steg)
        if detected_size != fuse_size:
            raise RuntimeError("OmniGuard artifact and image detector sizes differ")
        height, width = detected_size
        probability = detector(artifact_input, fuse_input)[:, :, :height, :width]
        probability = F.interpolate(
            probability, size=(original_size[1], original_size[0]), mode="bilinear", align_corners=False
        )
    return probability[0, 0].cpu().numpy()


def localize(args: argparse.Namespace, model, detector) -> None:
    from modules.Unet_common import DWT, IWT

    dwt, iwt = DWT(), IWT()
    clips = choose_clips(args)
    events = args.output_root / "omniguard-localize-events.jsonl"
    failures = 0
    for ordinal, clip in enumerate(clips, 1):
        destination = method_dir(args.output_root, clip.clip_id, METHOD)
        metrics_path = destination / "localization.json"
        try:
            embed_record = __import__("json").loads((destination / "embed.json").read_text())
            native_size = (int(embed_record["watermarked"]["width"]), int(embed_record["watermarked"]["height"]))
            if metrics_path.is_file() and not args.force:
                print(f"[{ordinal}/{len(clips)}] skip OmniGuard localization {clip.clip_id}", flush=True)
                continue
            started = time.monotonic()
            base_masks = [mask > 127 for mask in read_mask_sequence(raw_mask_frames(args.dataset_root, clip.clip_id))]
            results: dict[str, object] = {}
            for remover in ("propainter", "diffueraser"):
                attacked_path = destination / f"{remover}.mp4"
                attacked_info = validate_video(attacked_path, frames=clip.frame_count, size=native_size)
                frames, rate = decode_rgb(attacked_path)
                probabilities: list[np.ndarray] = []
                for frame_index, frame in enumerate(frames, 1):
                    probabilities.append(reveal(model, detector, frame, dwt, iwt))
                    if frame_index % 5 == 0 or frame_index == len(frames):
                        print(f"  {clip.clip_id}/{remover}: detected {frame_index}/{len(frames)} frames", flush=True)
                predicted = [probability >= DETECTOR_THRESHOLD for probability in probabilities]
                inference_size = processing_size(destination / f"{remover}.json", native_size)
                expected = [
                    effective_attack_mask(mask, remover, native_size, inference_size)
                    for mask in base_masks
                ]
                pred_path = destination / f"pred-mask-{remover}.mp4"
                gt_path = destination / f"effective-gt-mask-{remover}.mp4"
                encode_binary_masks(pred_path, predicted, rate)
                encode_binary_masks(gt_path, expected, Fraction(FPS, 1))
                results[remover] = {
                    **localization_metrics(predicted, expected),
                    "mean_predicted_probability": float(np.mean([value.mean() for value in probabilities])),
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
                "detector_probability_threshold": DETECTOR_THRESHOLD,
                "detector_input_size": DETECTOR_INPUT_SIZE,
                "detector_checkpoint": checkpoint_record(args.detector_checkpoint),
                **results,
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "completed_at": utc_now(),
            }
            atomic_json(metrics_path, metrics)
            append_jsonl(events, metrics)
            print(f"[{ordinal}/{len(clips)}] complete OmniGuard localization {clip.clip_id}", flush=True)
        except Exception as error:
            failures += 1
            append_jsonl(events, {"status": "failed", "clip_id": clip.clip_id, "error": repr(error), "timestamp": utc_now()})
            print(f"[{ordinal}/{len(clips)}] FAILED OmniGuard localization {clip.clip_id}: {error!r}", flush=True)
    if failures:
        raise SystemExit(f"OmniGuard localization failed for {failures} clips")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("embed", "localize"))
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--detector-checkpoint", type=Path)
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--repository-commit", default="2e3c86b3650cb904d2a8a112c640a78895aa93eb")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--only", action="append")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("OmniGuard inference requires CUDA")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu))
    model, _config = load_embedding_model(args.source, args.checkpoint)
    if args.action == "embed":
        embed(args, model)
    else:
        if args.detector_checkpoint is None:
            raise RuntimeError("--detector-checkpoint is required for localization")
        detector = load_detector(args.source, args.detector_checkpoint)
        localize(args, model, detector)


if __name__ == "__main__":
    main()
