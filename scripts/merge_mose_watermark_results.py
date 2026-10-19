#!/usr/bin/env python3
"""Merge corrected watermark-first outputs into the 50-sample MOSE review."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from build_mose_review import html_document


EXPECTED_CLIPS = 50


def read_complete(path: Path) -> dict:
    if not path.is_file():
        raise RuntimeError(f"Missing metadata: {path}")
    value = json.loads(path.read_text())
    if value.get("status") != "complete":
        raise RuntimeError(f"Result is not complete: {path}")
    return value


def copy_file(source: Path, destination: Path) -> str:
    if not source.is_file() or source.stat().st_size == 0:
        raise RuntimeError(f"Missing or empty result: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".part")
    shutil.copy2(source, temporary)
    temporary.replace(destination)
    return str(destination)


def relative(path: Path, root: Path) -> str:
    return str(path.relative_to(root))


def copy_named(source: Path, destination: Path, names: list[str], root: Path) -> dict[str, str]:
    copied: dict[str, str] = {}
    for name in names:
        target = destination / name
        copy_file(source / name, target)
        copied[name] = relative(target, root)
    return copied


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--review-root", required=True, type=Path)
    args = parser.parse_args()

    results_root = args.results_root.resolve()
    review_root = args.review_root.resolve()
    manifest_path = review_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    samples = manifest.get("samples", [])
    if len(samples) != EXPECTED_CLIPS:
        raise RuntimeError(f"Expected {EXPECTED_CLIPS} review samples, found {len(samples)}")

    aggregate = {
        "schema_version": 1,
        "protocol": "original -> watermark -> object removal -> detection",
        "sample_count": EXPECTED_CLIPS,
        "methods": {
            "videoseal": {"complete": 0, "propainter_iou": [], "diffueraser_iou": []},
            "safe-sora": {
                "complete": 0,
                "propainter_recovery_psnr_db": [],
                "diffueraser_recovery_psnr_db": [],
            },
            "videoshield": {"availability_records": 0, "negative_control_accuracy": []},
        },
    }

    for index, sample in enumerate(samples, 1):
        clip_id = sample["clip_id"]
        source_methods = results_root / "clips" / clip_id / "methods"
        media_methods = review_root / "media" / clip_id / "methods"

        vs_source = source_methods / "videoseal"
        vs_target = media_methods / "videoseal"
        vs_embed = read_complete(vs_source / "embed.json")
        vs_propainter = read_complete(vs_source / "propainter.json")
        vs_diffueraser = read_complete(vs_source / "diffueraser.json")
        vs_localization = read_complete(vs_source / "localization.json")
        vs_files = copy_named(
            vs_source,
            vs_target,
            [
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
            ],
            review_root,
        )
        sample.setdefault("watermarks", {})["VideoSeal"] = {
            "status": "complete",
            "input_original": sample["media"]["original"],
            "watermarked": vs_files["watermarked.mp4"],
            "gt_mask": sample["media"]["gt_mask"],
            "propainter": vs_files["propainter.mp4"],
            "gt_mask_propainter": vs_files["effective-gt-mask-propainter.mp4"],
            "pred_mask_propainter": vs_files["pred-mask-propainter.mp4"],
            "iou_propainter": vs_localization["propainter"]["global_iou"],
            "diffueraser": vs_files["diffueraser.mp4"],
            "gt_mask_diffueraser": vs_files["effective-gt-mask-diffueraser.mp4"],
            "pred_mask_diffueraser": vs_files["pred-mask-diffueraser.mp4"],
            "iou_diffueraser": vs_localization["diffueraser"]["global_iou"],
            "metadata": {
                "embed": vs_files["embed.json"],
                "propainter": vs_files["propainter.json"],
                "diffueraser": vs_files["diffueraser.json"],
                "localization": vs_files["localization.json"],
            },
            "note": (
                "VideoSeal was embedded into the original before either removal method. "
                "The released checkpoint has a global presence detector, so the displayed "
                "predicted masks are a documented derived 4x4 patch-grid localization; they "
                "are not a native pixelwise detector output. IoU uses each remover's effective mask."
            ),
            "source": vs_embed["source"],
        }
        aggregate["methods"]["videoseal"]["complete"] += 1
        aggregate["methods"]["videoseal"]["propainter_iou"].append(vs_localization["propainter"]["global_iou"])
        aggregate["methods"]["videoseal"]["diffueraser_iou"].append(vs_localization["diffueraser"]["global_iou"])

        ss_source = source_methods / "safe-sora"
        ss_target = media_methods / "safe-sora"
        ss_embed = read_complete(ss_source / "embed.json")
        read_complete(ss_source / "propainter.json")
        read_complete(ss_source / "diffueraser.json")
        ss_recovery = read_complete(ss_source / "recovery.json")
        ss_files = copy_named(
            ss_source,
            ss_target,
            [
                "sampled-original.mp4",
                "watermarked.mp4",
                "gt-mask.mp4",
                "propainter.mp4",
                "diffueraser.mp4",
                "recovery-propainter.png",
                "recovery-diffueraser.png",
                "embed.json",
                "propainter.json",
                "diffueraser.json",
                "recovery.json",
            ],
            review_root,
        )
        sample["watermarks"]["Safe-Sora"] = {
            "status": "complete",
            "input_original": ss_files["sampled-original.mp4"],
            "watermarked": ss_files["watermarked.mp4"],
            "gt_mask": ss_files["gt-mask.mp4"],
            "propainter": ss_files["propainter.mp4"],
            "gt_mask_propainter": None,
            "pred_mask_propainter": None,
            "iou_propainter": None,
            "diffueraser": ss_files["diffueraser.mp4"],
            "gt_mask_diffueraser": None,
            "pred_mask_diffueraser": None,
            "iou_diffueraser": None,
            "recovery_propainter": ss_files["recovery-propainter.png"],
            "recovery_diffueraser": ss_files["recovery-diffueraser.png"],
            "metadata": {
                "embed": ss_files["embed.json"],
                "propainter": ss_files["propainter.json"],
                "diffueraser": ss_files["diffueraser.json"],
                "recovery": ss_files["recovery.json"],
            },
            "note": (
                "Safe-Sora uses its official 16-frame 512x320 protocol. The original, mask, "
                "watermarked video, and both removal results are aligned to those sampled frames. "
                "Its released method recovers a graphical watermark but does not output a spatial "
                "tamper mask, so Pred mask and IoU are not applicable."
            ),
            "sampling": ss_embed["sampling"],
        }
        aggregate["methods"]["safe-sora"]["complete"] += 1
        for remover in ("propainter", "diffueraser"):
            value = ss_recovery[remover]["fidelity"]["psnr_db"]
            if value is not None:
                aggregate["methods"]["safe-sora"][f"{remover}_recovery_psnr_db"].append(value)

        shield_source = source_methods / "videoshield"
        shield_target = media_methods / "videoshield"
        shield = read_complete(shield_source / "availability.json")
        shield_files = copy_named(shield_source, shield_target, ["availability.json"], review_root)
        sample["watermarks"]["VideoShield"] = {
            "status": "not_applicable",
            "input_original": sample["media"]["original"],
            "watermarked": None,
            "gt_mask": sample["media"]["gt_mask"],
            "propainter": None,
            "gt_mask_propainter": None,
            "pred_mask_propainter": None,
            "iou_propainter": None,
            "diffueraser": None,
            "gt_mask_diffueraser": None,
            "pred_mask_diffueraser": None,
            "iou_diffueraser": None,
            "negative_control_accuracy": shield["reference_watermark_accuracy"],
            "metadata": {"availability": shield_files["availability.json"]},
            "note": (
                f"Not applicable: {shield['reason']} The shown score is an explicitly labelled "
                "unwatermarked negative control, not an embedding, robustness, or localization score."
            ),
        }
        aggregate["methods"]["videoshield"]["availability_records"] += 1
        aggregate["methods"]["videoshield"]["negative_control_accuracy"].append(shield["reference_watermark_accuracy"])
        print(f"[{index}/{EXPECTED_CLIPS}] merged {clip_id}", flush=True)

    for key in ("propainter_iou", "diffueraser_iou"):
        values = aggregate["methods"]["videoseal"][key]
        aggregate["methods"]["videoseal"][f"mean_{key}"] = sum(values) / len(values)
    values = aggregate["methods"]["videoshield"]["negative_control_accuracy"]
    aggregate["methods"]["videoshield"]["mean_negative_control_accuracy"] = sum(values) / len(values)
    for remover in ("propainter", "diffueraser"):
        values = aggregate["methods"]["safe-sora"][f"{remover}_recovery_psnr_db"]
        aggregate["methods"]["safe-sora"][f"mean_{remover}_recovery_psnr_db"] = (
            sum(values) / len(values) if values else None
        )

    manifest["schema_version"] = 2
    manifest["watermark_protocol"] = "original -> watermark -> object removal -> detection"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (review_root / "watermark-summary.json").write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n")
    (review_root / "index.html").write_text(html_document())
    print(f"Merged all watermark results into {review_root}", flush=True)


if __name__ == "__main__":
    main()
