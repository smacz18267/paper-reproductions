#!/usr/bin/env python3
"""Add full EffectErase baseline and watermark-first results to the MOSE review."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_mose_review import html_document


EXPECTED_CLIPS = 50


def read_complete(path: Path) -> dict:
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"Missing or empty metadata: {path}")
    value = json.loads(path.read_text())
    if value.get("status") != "complete":
        raise RuntimeError(f"Incomplete result: {path}")
    return value


def require(path: Path) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"Missing or empty result: {path}")


def relative(path: Path, root: Path) -> str:
    return str(path.relative_to(root))


def mean(values: list[float]) -> float:
    if not values:
        raise RuntimeError("Cannot average an empty list")
    return sum(values) / len(values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-root", required=True, type=Path)
    args = parser.parse_args()
    root = args.review_root.resolve()
    manifest_path = root / "manifest.json"
    summary_path = root / "watermark-summary.json"
    manifest = json.loads(manifest_path.read_text())
    summary = json.loads(summary_path.read_text())
    samples = manifest.get("samples", [])
    if len(samples) != EXPECTED_CLIPS:
        raise RuntimeError(f"Expected {EXPECTED_CLIPS} samples, found {len(samples)}")

    method_metrics: dict[str, dict[str, list[float]]] = {
        "videoseal": {"effecterase_iou": []},
        "safe-sora": {"effecterase_recovery_psnr_db": []},
        "editguard": {
            "effecterase_iou": [],
            "effecterase_message_bit_accuracy": [],
        },
        "omniguard": {"effecterase_iou": []},
    }
    baseline_runtimes: list[float] = []

    for index, sample in enumerate(samples, 1):
        clip_id = sample["clip_id"]
        media_root = root / "media" / clip_id
        baseline = media_root / "baseline"
        baseline_video = baseline / "effecterase.mp4"
        baseline_json = baseline / "effecterase.json"
        require(baseline_video)
        baseline_record = read_complete(baseline_json)
        sample["media"]["effecterase"] = relative(baseline_video, root)
        sample.setdefault("media_metadata", {})["effecterase"] = relative(baseline_json, root)
        baseline_runtimes.append(float(baseline_record["elapsed_seconds"]))

        methods = sample["watermarks"]
        for label, slug in (
            ("VideoSeal", "videoseal"),
            ("Safe-Sora", "safe-sora"),
            ("EditGuard", "editguard"),
            ("OmniGuard", "omniguard"),
        ):
            method = methods[label]
            directory = media_root / "methods" / slug
            attack_video = directory / "effecterase.mp4"
            attack_json = directory / "effecterase.json"
            require(attack_video)
            read_complete(attack_json)
            method["effecterase"] = relative(attack_video, root)
            method["metadata"]["effecterase"] = relative(attack_json, root)

            if slug == "safe-sora":
                method["note"] = (
                    "Safe-Sora uses its official 16-frame 512x320 protocol. The original, mask, "
                    "watermarked video, and all three removal results are aligned to those sampled "
                    "frames. Its released method recovers a graphical watermark but does not output "
                    "a spatial tamper mask, so Pred mask and IoU are not applicable."
                )
                recovery_path = directory / "recovery-effecterase.png"
                require(recovery_path)
                recovery = read_complete(directory / "recovery.json")
                method["gt_mask_effecterase"] = None
                method["pred_mask_effecterase"] = None
                method["iou_effecterase"] = None
                method["recovery_effecterase"] = relative(recovery_path, root)
                psnr = recovery["effecterase"]["fidelity"]["psnr_db"]
                if psnr is not None:
                    method_metrics[slug]["effecterase_recovery_psnr_db"].append(float(psnr))
                continue

            if slug == "videoseal":
                method["note"] = (
                    "VideoSeal was embedded into the original before each removal method. The "
                    "released checkpoint has a global presence detector, so the displayed predicted "
                    "masks are a documented derived 4x4 patch-grid localization; they are not a "
                    "native pixelwise detector output. IoU uses each remover's effective mask."
                )

            localization = read_complete(directory / "localization.json")
            pred = directory / "pred-mask-effecterase.mp4"
            effective = directory / "effective-gt-mask-effecterase.mp4"
            require(pred)
            require(effective)
            method["gt_mask_effecterase"] = relative(effective, root)
            method["pred_mask_effecterase"] = relative(pred, root)
            method["iou_effecterase"] = localization["effecterase"]["global_iou"]
            method_metrics[slug]["effecterase_iou"].append(
                float(localization["effecterase"]["global_iou"])
            )
            if slug == "editguard":
                accuracy = localization["effecterase"]["mean_message_bit_accuracy"]
                method["mean_message_bit_accuracy_effecterase"] = accuracy
                method_metrics[slug]["effecterase_message_bit_accuracy"].append(float(accuracy))

        shield = methods["VideoShield"]
        shield["effecterase"] = None
        shield["gt_mask_effecterase"] = None
        shield["pred_mask_effecterase"] = None
        shield["iou_effecterase"] = None

        print(f"[{index}/{EXPECTED_CLIPS}] merged EffectErase {clip_id}", flush=True)

    for slug, metrics in method_metrics.items():
        target = summary["methods"][slug]
        for key, values in metrics.items():
            if len(values) != EXPECTED_CLIPS:
                raise RuntimeError(f"Expected 50 {slug}/{key} values, found {len(values)}")
            target[key] = values
            target[f"mean_{key}"] = mean(values)

    summary["effecterase"] = {
        "complete_baseline": EXPECTED_CLIPS,
        "complete_watermark_attacks": EXPECTED_CLIPS * 4,
        "repository": "https://github.com/FudanCVL/EffectErase",
        "repository_commit": "bcee0a5da5ef387c2ba39390dc4d579503669fb8",
        "baseline_elapsed_seconds": baseline_runtimes,
        "mean_baseline_elapsed_seconds": mean(baseline_runtimes),
        "protocol": "official Wan remover, aspect-preserving <=832px processing, 4n+1 windows",
    }
    summary["schema_version"] = max(3, int(summary.get("schema_version", 0)))
    manifest["schema_version"] = max(4, int(manifest.get("schema_version", 0)))
    manifest["removers"] = ["ProPainter", "DiffuEraser", "EffectErase"]
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (root / "index.html").write_text(html_document())
    print(f"Merged full EffectErase results into {root}", flush=True)


if __name__ == "__main__":
    main()
