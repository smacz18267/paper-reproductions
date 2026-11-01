#!/usr/bin/env python3
"""Add full EditGuard and OmniGuard results to the 50-sample MOSE review."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


EXPECTED_CLIPS = 50
REMOVERS = ("propainter", "diffueraser")


def read_complete(path: Path) -> dict:
    if not path.is_file():
        raise RuntimeError(f"Missing metadata: {path}")
    value = json.loads(path.read_text())
    if value.get("status") != "complete":
        raise RuntimeError(f"Result is not complete: {path}")
    return value


def require_files(method_dir: Path, names: list[str]) -> None:
    missing = [name for name in names if not (method_dir / name).is_file()]
    empty = [name for name in names if (method_dir / name).is_file() and (method_dir / name).stat().st_size == 0]
    if missing or empty:
        raise RuntimeError(f"Incomplete result in {method_dir}: {missing=} {empty=}")


def relative(path: Path, root: Path) -> str:
    return str(path.relative_to(root))


def mean(values: list[float]) -> float:
    if not values:
        raise RuntimeError("Cannot average an empty metric list")
    return sum(values) / len(values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-root", required=True, type=Path)
    args = parser.parse_args()

    review_root = args.review_root.resolve()
    manifest_path = review_root / "manifest.json"
    summary_path = review_root / "watermark-summary.json"
    manifest = json.loads(manifest_path.read_text())
    summary = json.loads(summary_path.read_text())
    samples = manifest.get("samples", [])
    if len(samples) != EXPECTED_CLIPS:
        raise RuntimeError(f"Expected {EXPECTED_CLIPS} review samples, found {len(samples)}")

    editguard_summary: dict[str, object] = {
        "complete": 0,
        "embedding_psnr_db": [],
        "propainter_iou": [],
        "diffueraser_iou": [],
        "propainter_message_bit_accuracy": [],
        "diffueraser_message_bit_accuracy": [],
    }
    omniguard_summary: dict[str, object] = {
        "complete": 0,
        "embedding_psnr_db": [],
        "propainter_iou": [],
        "diffueraser_iou": [],
    }

    common_files = [
        "watermarked.mp4",
        "propainter.mp4",
        "diffueraser.mp4",
        "effective-gt-mask-propainter.mp4",
        "effective-gt-mask-diffueraser.mp4",
        "pred-mask-propainter.mp4",
        "pred-mask-diffueraser.mp4",
        "embed.json",
        "propainter.json",
        "diffueraser.json",
        "localization.json",
    ]

    for index, sample in enumerate(samples, 1):
        clip_id = sample["clip_id"]
        methods = sample.setdefault("watermarks", {})

        edit_dir = review_root / "media" / clip_id / "methods" / "editguard"
        require_files(edit_dir, common_files + ["aligned-original.mp4", "gt-mask.mp4"])
        edit_embed = read_complete(edit_dir / "embed.json")
        read_complete(edit_dir / "propainter.json")
        read_complete(edit_dir / "diffueraser.json")
        edit_localization = read_complete(edit_dir / "localization.json")
        methods["EditGuard"] = {
            "status": "complete",
            "input_original": relative(edit_dir / "aligned-original.mp4", review_root),
            "watermarked": relative(edit_dir / "watermarked.mp4", review_root),
            "gt_mask": relative(edit_dir / "gt-mask.mp4", review_root),
            "propainter": relative(edit_dir / "propainter.mp4", review_root),
            "gt_mask_propainter": relative(edit_dir / "effective-gt-mask-propainter.mp4", review_root),
            "pred_mask_propainter": relative(edit_dir / "pred-mask-propainter.mp4", review_root),
            "iou_propainter": edit_localization["propainter"]["global_iou"],
            "mean_message_bit_accuracy_propainter": edit_localization["propainter"]["mean_message_bit_accuracy"],
            "diffueraser": relative(edit_dir / "diffueraser.mp4", review_root),
            "gt_mask_diffueraser": relative(edit_dir / "effective-gt-mask-diffueraser.mp4", review_root),
            "pred_mask_diffueraser": relative(edit_dir / "pred-mask-diffueraser.mp4", review_root),
            "iou_diffueraser": edit_localization["diffueraser"]["global_iou"],
            "mean_message_bit_accuracy_diffueraser": edit_localization["diffueraser"]["mean_message_bit_accuracy"],
            "metadata": {
                "embed": relative(edit_dir / "embed.json", review_root),
                "propainter": relative(edit_dir / "propainter.json", review_root),
                "diffueraser": relative(edit_dir / "diffueraser.json", review_root),
                "localization": relative(edit_dir / "localization.json", review_root),
            },
            "release_adapter": edit_embed["release_adapter"],
            "source": edit_embed["source"],
            "note": (
                "EditGuard's released image app was applied framewise at its official 512x512 size. "
                "Predicted masks use the released residual threshold of 0.2 and may cover the full "
                "frame after attack; IoU uses each remover's effective aligned mask."
            ),
        }
        editguard_summary["complete"] = int(editguard_summary["complete"]) + 1
        cast_lists = editguard_summary
        cast_lists["embedding_psnr_db"].append(edit_embed["embedding_fidelity"]["psnr_db"])
        for remover in REMOVERS:
            cast_lists[f"{remover}_iou"].append(edit_localization[remover]["global_iou"])
            cast_lists[f"{remover}_message_bit_accuracy"].append(
                edit_localization[remover]["mean_message_bit_accuracy"]
            )

        omni_dir = review_root / "media" / clip_id / "methods" / "omniguard"
        require_files(omni_dir, common_files)
        omni_embed = read_complete(omni_dir / "embed.json")
        read_complete(omni_dir / "propainter.json")
        read_complete(omni_dir / "diffueraser.json")
        omni_localization = read_complete(omni_dir / "localization.json")
        methods["OmniGuard"] = {
            "status": "complete",
            "input_original": sample["media"]["original"],
            "watermarked": relative(omni_dir / "watermarked.mp4", review_root),
            "gt_mask": sample["media"]["gt_mask"],
            "propainter": relative(omni_dir / "propainter.mp4", review_root),
            "gt_mask_propainter": relative(omni_dir / "effective-gt-mask-propainter.mp4", review_root),
            "pred_mask_propainter": relative(omni_dir / "pred-mask-propainter.mp4", review_root),
            "iou_propainter": omni_localization["propainter"]["global_iou"],
            "diffueraser": relative(omni_dir / "diffueraser.mp4", review_root),
            "gt_mask_diffueraser": relative(omni_dir / "effective-gt-mask-diffueraser.mp4", review_root),
            "pred_mask_diffueraser": relative(omni_dir / "pred-mask-diffueraser.mp4", review_root),
            "iou_diffueraser": omni_localization["diffueraser"]["global_iou"],
            "metadata": {
                "embed": relative(omni_dir / "embed.json", review_root),
                "propainter": relative(omni_dir / "propainter.json", review_root),
                "diffueraser": relative(omni_dir / "diffueraser.json", review_root),
                "localization": relative(omni_dir / "localization.json", review_root),
            },
            "release_adapter": omni_embed["release_adapter"],
            "source": omni_embed["source"],
            "note": (
                "OmniGuard's released image models were applied independently to every native-size "
                "frame. Predicted masks use the released localization model with a probability "
                "threshold of 0.5; IoU uses each remover's effective aligned mask."
            ),
        }
        omniguard_summary["complete"] = int(omniguard_summary["complete"]) + 1
        omni_lists = omniguard_summary
        omni_lists["embedding_psnr_db"].append(omni_embed["embedding_fidelity"]["psnr_db"])
        for remover in REMOVERS:
            omni_lists[f"{remover}_iou"].append(omni_localization[remover]["global_iou"])

        print(f"[{index}/{EXPECTED_CLIPS}] merged {clip_id}", flush=True)

    for method in (editguard_summary, omniguard_summary):
        for key, values in list(method.items()):
            if isinstance(values, list):
                method[f"mean_{key}"] = mean(values)

    manifest["schema_version"] = 3
    manifest["watermark_protocol"] = "original -> watermark -> object removal -> detection"
    summary["schema_version"] = 2
    summary["sample_count"] = EXPECTED_CLIPS
    summary["methods"]["editguard"] = editguard_summary
    summary["methods"]["omniguard"] = omniguard_summary
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"Merged EditGuard and OmniGuard full-run results into {review_root}", flush=True)


if __name__ == "__main__":
    main()
