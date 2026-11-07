#!/usr/bin/env python3
"""Decode and validate every video in a generated MOSE review bundle."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2


MEDIA_KEYS = ("original", "gt_mask", "propainter", "diffueraser", "effecterase")
REMOVERS = ("propainter", "diffueraser", "effecterase")


def inspect(path: Path, *, mask: bool = False) -> dict[str, float | int]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames = 0
    foreground_pixels = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames += 1
        if mask:
            foreground_pixels += int((cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) > 127).sum())
    capture.release()
    return {
        "frames": frames,
        "width": width,
        "height": height,
        "fps": fps,
        "bytes": path.stat().st_size,
        "foreground_pixels": foreground_pixels if mask else -1,
    }


def validate_media(
    root: Path,
    relative: str,
    *,
    expected_frames: int,
    expected_size: tuple[int, int],
    used_paths: set[Path],
    label: str,
    mask: bool = False,
    require_foreground: bool = False,
) -> tuple[Path, dict[str, float | int]]:
    path = (root / relative).resolve()
    if root not in path.parents:
        raise RuntimeError(f"Media path leaves review root: {path}")
    if path in used_paths:
        raise RuntimeError(f"Duplicate media file instead of a shared manifest reference: {path}")
    used_paths.add(path)
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"Missing or empty media: {path}")
    info = inspect(path, mask=mask)
    if info["frames"] != expected_frames:
        raise RuntimeError(f"{label}: expected {expected_frames} frames, got {info['frames']}")
    if (info["width"], info["height"]) != expected_size:
        raise RuntimeError(f"{label}: expected {expected_size}, got {(info['width'], info['height'])}")
    if abs(float(info["fps"]) - 25.0) > 0.01:
        raise RuntimeError(f"{label}: expected 25 FPS, got {info['fps']}")
    if require_foreground and int(info["foreground_pixels"]) == 0:
        raise RuntimeError(f"{label}: mask has no foreground pixels")
    return path, info


def mask_iou(predicted: Path, expected: Path) -> float:
    pred_capture = cv2.VideoCapture(str(predicted))
    gt_capture = cv2.VideoCapture(str(expected))
    intersection = 0
    union = 0
    while True:
        pred_ok, pred_frame = pred_capture.read()
        gt_ok, gt_frame = gt_capture.read()
        if pred_ok != gt_ok:
            raise RuntimeError(f"Mask streams have different lengths: {predicted}, {expected}")
        if not pred_ok:
            break
        pred = cv2.cvtColor(pred_frame, cv2.COLOR_BGR2GRAY) > 127
        gt = cv2.cvtColor(gt_frame, cv2.COLOR_BGR2GRAY) > 127
        intersection += int((pred & gt).sum())
        union += int((pred | gt).sum())
    pred_capture.release()
    gt_capture.release()
    return 1.0 if union == 0 else intersection / union


def validate_image(root: Path, relative: str, used_paths: set[Path], label: str) -> dict[str, int]:
    path = (root / relative).resolve()
    if root not in path.parents or path in used_paths:
        raise RuntimeError(f"Invalid or duplicate image path for {label}: {path}")
    used_paths.add(path)
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not decode image for {label}: {path}")
    height, width = image.shape[:2]
    if (width, height) != (768, 256):
        raise RuntimeError(f"{label}: expected 768x256 recovery composite, got {width}x{height}")
    return {"width": width, "height": height, "bytes": path.stat().st_size}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("review_root", type=Path)
    args = parser.parse_args()

    root = args.review_root.resolve()
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    samples = manifest.get("samples", [])
    if len(samples) != 50:
        raise RuntimeError(f"Expected 50 samples, found {len(samples)}")
    clip_ids = [sample["clip_id"] for sample in samples]
    if len(set(clip_ids)) != 50:
        raise RuntimeError("Manifest contains duplicate clip IDs")

    used_paths: set[Path] = set()
    inspected: dict[str, dict[str, dict[str, float | int]]] = {}
    for index, sample in enumerate(samples, 1):
        clip_id = sample["clip_id"]
        expected_frames = int(sample["frame_count"])
        expected_size = (int(sample["width"]), int(sample["height"]))
        inspected[clip_id] = {}
        for key in MEDIA_KEYS:
            relative = sample["media"].get(key)
            if not relative:
                raise RuntimeError(f"{clip_id} has no {key} path")
            _path, info = validate_media(
                root,
                relative,
                expected_frames=expected_frames,
                expected_size=expected_size,
                used_paths=used_paths,
                label=f"{clip_id}/{key}",
                mask=key == "gt_mask",
                require_foreground=key == "gt_mask",
            )
            inspected[clip_id][key] = info

        methods = sample.get("watermarks", {})
        expected_methods = {"VideoSeal", "Safe-Sora", "VideoShield", "EditGuard", "OmniGuard"}
        if set(methods) != expected_methods:
            raise RuntimeError(
                f"{clip_id}: expected all five watermark method records, "
                f"missing={sorted(expected_methods - set(methods))}, "
                f"unexpected={sorted(set(methods) - expected_methods)}"
            )

        videoseal = methods["VideoSeal"]
        if videoseal.get("status") != "complete":
            raise RuntimeError(f"{clip_id}: VideoSeal is not complete")
        if videoseal.get("input_original") != sample["media"]["original"]:
            raise RuntimeError(f"{clip_id}: VideoSeal must share the original video")
        if videoseal.get("gt_mask") != sample["media"]["gt_mask"]:
            raise RuntimeError(f"{clip_id}: VideoSeal must share the raw GT mask")
        inspected[clip_id]["VideoSeal"] = {}
        vs_media = {
            "watermarked": False,
            "propainter": False,
            "diffueraser": False,
            "effecterase": False,
            "gt_mask_propainter": True,
            "gt_mask_diffueraser": True,
            "gt_mask_effecterase": True,
            "pred_mask_propainter": True,
            "pred_mask_diffueraser": True,
            "pred_mask_effecterase": True,
        }
        vs_paths: dict[str, Path] = {}
        for key, is_mask in vs_media.items():
            path, info = validate_media(
                root,
                videoseal[key],
                expected_frames=expected_frames,
                expected_size=expected_size,
                used_paths=used_paths,
                label=f"{clip_id}/VideoSeal/{key}",
                mask=is_mask,
                require_foreground=key.startswith("gt_mask_"),
            )
            vs_paths[key] = path
            inspected[clip_id]["VideoSeal"][key] = info
        for key in ("iou_propainter", "iou_diffueraser", "iou_effecterase"):
            value = videoseal.get(key)
            if not isinstance(value, (int, float)) or not 0.0 <= float(value) <= 1.0:
                raise RuntimeError(f"{clip_id}/VideoSeal/{key}: invalid IoU {value!r}")
        for remover in REMOVERS:
            displayed_iou = mask_iou(
                vs_paths[f"pred_mask_{remover}"],
                vs_paths[f"gt_mask_{remover}"],
            )
            reported_iou = float(videoseal[f"iou_{remover}"])
            if abs(displayed_iou - reported_iou) > 1e-9:
                raise RuntimeError(
                    f"{clip_id}/VideoSeal/{remover}: displayed-mask IoU {displayed_iou} "
                    f"does not match reported IoU {reported_iou}"
                )

        safe_sora = methods["Safe-Sora"]
        if safe_sora.get("status") != "complete":
            raise RuntimeError(f"{clip_id}: Safe-Sora is not complete")
        if any(safe_sora.get(key) is not None for key in (
            "pred_mask_propainter", "pred_mask_diffueraser", "pred_mask_effecterase",
            "iou_propainter", "iou_diffueraser", "iou_effecterase",
        )):
            raise RuntimeError(f"{clip_id}: Safe-Sora must not claim unavailable localization")
        sampling = safe_sora.get("sampling", {})
        safe_frames = int(sampling.get("frames", 0))
        safe_size = tuple(int(value) for value in sampling.get("resize", []))
        if safe_frames != 16 or safe_size != (512, 320):
            raise RuntimeError(f"{clip_id}: unexpected Safe-Sora protocol {safe_frames=} {safe_size=}")
        inspected[clip_id]["Safe-Sora"] = {}
        ss_media = {
            "input_original": False,
            "watermarked": False,
            "gt_mask": True,
            "propainter": False,
            "diffueraser": False,
            "effecterase": False,
        }
        for key, is_mask in ss_media.items():
            _path, info = validate_media(
                root,
                safe_sora[key],
                expected_frames=safe_frames,
                expected_size=safe_size,
                used_paths=used_paths,
                label=f"{clip_id}/Safe-Sora/{key}",
                mask=is_mask,
                require_foreground=is_mask,
            )
            inspected[clip_id]["Safe-Sora"][key] = info
        for remover in REMOVERS:
            key = f"recovery_{remover}"
            inspected[clip_id]["Safe-Sora"][key] = validate_image(
                root,
                safe_sora[key],
                used_paths,
                f"{clip_id}/Safe-Sora/{key}",
            )

        editguard = methods["EditGuard"]
        if editguard.get("status") != "complete":
            raise RuntimeError(f"{clip_id}: EditGuard is not complete")
        edit_size = (512, 512)
        inspected[clip_id]["EditGuard"] = {}
        edit_media = {
            "input_original": False,
            "watermarked": False,
            "gt_mask": True,
            "propainter": False,
            "diffueraser": False,
            "effecterase": False,
            "gt_mask_propainter": True,
            "gt_mask_diffueraser": True,
            "gt_mask_effecterase": True,
            "pred_mask_propainter": True,
            "pred_mask_diffueraser": True,
            "pred_mask_effecterase": True,
        }
        edit_paths: dict[str, Path] = {}
        for key, is_mask in edit_media.items():
            path, info = validate_media(
                root,
                editguard[key],
                expected_frames=expected_frames,
                expected_size=edit_size,
                used_paths=used_paths,
                label=f"{clip_id}/EditGuard/{key}",
                mask=is_mask,
                require_foreground=key in {"gt_mask", "gt_mask_propainter", "gt_mask_diffueraser"},
            )
            edit_paths[key] = path
            inspected[clip_id]["EditGuard"][key] = info
        adapter = editguard.get("release_adapter", {})
        if adapter.get("resize") != [512, 512] or adapter.get("framewise") is not True:
            raise RuntimeError(f"{clip_id}: unexpected EditGuard release adapter {adapter!r}")
        for remover in REMOVERS:
            displayed_iou = mask_iou(
                edit_paths[f"pred_mask_{remover}"],
                edit_paths[f"gt_mask_{remover}"],
            )
            reported_iou = editguard.get(f"iou_{remover}")
            if not isinstance(reported_iou, (int, float)) or not 0.0 <= float(reported_iou) <= 1.0:
                raise RuntimeError(f"{clip_id}/EditGuard/{remover}: invalid IoU {reported_iou!r}")
            if abs(displayed_iou - float(reported_iou)) > 1e-9:
                raise RuntimeError(
                    f"{clip_id}/EditGuard/{remover}: displayed-mask IoU {displayed_iou} "
                    f"does not match reported IoU {reported_iou}"
                )
            bit_accuracy = editguard.get(f"mean_message_bit_accuracy_{remover}")
            if not isinstance(bit_accuracy, (int, float)) or not 0.0 <= float(bit_accuracy) <= 1.0:
                raise RuntimeError(
                    f"{clip_id}/EditGuard/{remover}: invalid message bit accuracy {bit_accuracy!r}"
                )

        omniguard = methods["OmniGuard"]
        if omniguard.get("status") != "complete":
            raise RuntimeError(f"{clip_id}: OmniGuard is not complete")
        if omniguard.get("input_original") != sample["media"]["original"]:
            raise RuntimeError(f"{clip_id}: OmniGuard must share the original video")
        if omniguard.get("gt_mask") != sample["media"]["gt_mask"]:
            raise RuntimeError(f"{clip_id}: OmniGuard must share the raw GT mask")
        inspected[clip_id]["OmniGuard"] = {}
        omni_media = {
            "watermarked": False,
            "propainter": False,
            "diffueraser": False,
            "effecterase": False,
            "gt_mask_propainter": True,
            "gt_mask_diffueraser": True,
            "gt_mask_effecterase": True,
            "pred_mask_propainter": True,
            "pred_mask_diffueraser": True,
            "pred_mask_effecterase": True,
        }
        omni_paths: dict[str, Path] = {}
        for key, is_mask in omni_media.items():
            path, info = validate_media(
                root,
                omniguard[key],
                expected_frames=expected_frames,
                expected_size=expected_size,
                used_paths=used_paths,
                label=f"{clip_id}/OmniGuard/{key}",
                mask=is_mask,
                require_foreground=key.startswith("gt_mask_"),
            )
            omni_paths[key] = path
            inspected[clip_id]["OmniGuard"][key] = info
        adapter = omniguard.get("release_adapter", {})
        if adapter.get("resize") is not None or adapter.get("framewise") is not True:
            raise RuntimeError(f"{clip_id}: unexpected OmniGuard release adapter {adapter!r}")
        for remover in REMOVERS:
            displayed_iou = mask_iou(
                omni_paths[f"pred_mask_{remover}"],
                omni_paths[f"gt_mask_{remover}"],
            )
            reported_iou = omniguard.get(f"iou_{remover}")
            if not isinstance(reported_iou, (int, float)) or not 0.0 <= float(reported_iou) <= 1.0:
                raise RuntimeError(f"{clip_id}/OmniGuard/{remover}: invalid IoU {reported_iou!r}")
            if abs(displayed_iou - float(reported_iou)) > 1e-9:
                raise RuntimeError(
                    f"{clip_id}/OmniGuard/{remover}: displayed-mask IoU {displayed_iou} "
                    f"does not match reported IoU {reported_iou}"
                )

        shield = methods["VideoShield"]
        if shield.get("status") != "not_applicable":
            raise RuntimeError(f"{clip_id}: VideoShield capability status must be not_applicable")
        unavailable = (
            "watermarked", "propainter", "diffueraser", "effecterase", "pred_mask_propainter",
            "pred_mask_diffueraser", "pred_mask_effecterase", "iou_propainter", "iou_diffueraser",
            "iou_effecterase",
        )
        if any(shield.get(key) is not None for key in unavailable):
            raise RuntimeError(f"{clip_id}: VideoShield must not contain fabricated MOSE outputs")
        control = shield.get("negative_control_accuracy")
        if not isinstance(control, (int, float)) or not 0.0 <= float(control) <= 1.0:
            raise RuntimeError(f"{clip_id}: invalid VideoShield negative control {control!r}")
        print(f"[{index}/50] valid {clip_id}", flush=True)

    summary = json.loads((root / "watermark-summary.json").read_text())
    if int(summary.get("sample_count", 0)) != 50:
        raise RuntimeError("Watermark summary does not cover all 50 samples")
    summary_methods = summary.get("methods", {})
    for method_name, list_keys in {
        "videoseal": ("effecterase_iou",),
        "safe-sora": ("effecterase_recovery_psnr_db",),
        "editguard": (
            "embedding_psnr_db",
            "propainter_iou",
            "diffueraser_iou",
            "effecterase_iou",
            "propainter_message_bit_accuracy",
            "diffueraser_message_bit_accuracy",
            "effecterase_message_bit_accuracy",
        ),
        "omniguard": ("embedding_psnr_db", "propainter_iou", "diffueraser_iou", "effecterase_iou"),
    }.items():
        method_summary = summary_methods.get(method_name, {})
        if method_summary.get("complete") != 50:
            raise RuntimeError(f"Summary for {method_name} does not report 50 complete samples")
        for key in list_keys:
            values = method_summary.get(key)
            if not isinstance(values, list) or len(values) != 50:
                raise RuntimeError(f"Summary {method_name}/{key} must contain 50 values")
            if any(not isinstance(value, (int, float)) for value in values):
                raise RuntimeError(f"Summary {method_name}/{key} contains a non-numeric value")
            reported_mean = method_summary.get(f"mean_{key}")
            computed_mean = sum(float(value) for value in values) / len(values)
            if not isinstance(reported_mean, (int, float)) or abs(float(reported_mean) - computed_mean) > 1e-12:
                raise RuntimeError(
                    f"Summary {method_name}/mean_{key} {reported_mean!r} does not match {computed_mean}"
                )
    effecterase_summary = summary.get("effecterase", {})
    if effecterase_summary.get("complete_baseline") != 50:
        raise RuntimeError("EffectErase summary must report 50 complete baseline removals")
    if effecterase_summary.get("complete_watermark_attacks") != 200:
        raise RuntimeError("EffectErase summary must report 200 complete watermark-first removals")
    if effecterase_summary.get("repository") != "https://github.com/FudanCVL/EffectErase":
        raise RuntimeError("EffectErase summary does not identify the official repository")

    flat_media = [
        info
        for sample_info in inspected.values()
        for value in sample_info.values()
        for info in ([value] if "frames" in value else value.values())
    ]
    report = {
        "status": "complete",
        "validated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "samples": len(samples),
        "media_files": len(used_paths),
        "videos": sum("frames" in info for info in flat_media),
        "images": sum("frames" not in info for info in flat_media),
        "decoded_frames": sum(int(info["frames"]) for info in flat_media if "frames" in info),
        "media": inspected,
    }
    (root / "validation.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"Validated {len(samples)} samples, {report['videos']} videos, and {report['images']} images")


if __name__ == "__main__":
    main()
