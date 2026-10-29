#!/usr/bin/env python3
"""Build one MOSE notebook for each watermark method."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_SAMPLES = ("02b72b8d", "0ab75563", "0b02886a")
NOTEBOOK_FORMAT = 4
NOTEBOOK_MINOR = 5


def markdown(source: str) -> dict[str, Any]:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.strip() + "\n",
    }


def code(source: str) -> dict[str, Any]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.strip() + "\n",
    }


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return value


def fmt(value: float | int | None, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, int):
        return f"{value:,}"
    return f"{value:.{digits}f}"


def seconds(value: float | int | None) -> str:
    return "N/A" if value is None else f"{float(value):.3f} s"


def link(relative_path: str, label: str | None = None) -> str:
    return f"[{label or Path(relative_path).name}](../../evidence/mose-task1/review50/{relative_path})"


def video_cell(items: list[tuple[str, str | None, str]]) -> dict[str, Any]:
    return code("show_videos(" + repr(items) + ")")


def image_cell(items: list[tuple[str, str, str]]) -> dict[str, Any]:
    return code("show_images(" + repr(items) + ")")


def json_cell(paths: list[str]) -> dict[str, Any]:
    return code("show_metadata(" + repr(paths) + ")")


def notebook(cells: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3",
            },
        },
        "nbformat": NOTEBOOK_FORMAT,
        "nbformat_minor": NOTEBOOK_MINOR,
    }


def setup_markdown() -> dict[str, Any]:
    return markdown(
        """
## Notebook setup

After a fresh clone, run `python3 scripts/prepare_mose_notebooks.py` from the repository root. This downloads and verifies the 48 videos shared by the three method notebooks. Run the next cell once to enable the video, image, download, and metadata viewers. If Git LFS has left a pointer file in place of a video, the viewer reports the preparation command instead of displaying a broken player.
"""
    )


def setup_cell() -> dict[str, Any]:
    return code(
        '''import base64
from pathlib import Path
import html
import json
from IPython.display import HTML, Markdown, display

LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"

def find_repo_root():
    for start in (Path.cwd().resolve(), *Path.cwd().resolve().parents):
        if (start / "evidence/mose-task1/review50/manifest.json").is_file():
            return start
    raise FileNotFoundError("Run this notebook from inside the paper-reproductions checkout.")

REPO_ROOT = find_repo_root()
REVIEW_ROOT = REPO_ROOT / "evidence/mose-task1/review50"

def artifact_data_url(source, mime_type):
    if not source.is_file():
        raise FileNotFoundError(source)
    payload = source.read_bytes()
    if payload.startswith(LFS_POINTER_PREFIX):
        raise RuntimeError(
            f"{source} is a Git LFS pointer, not the real media file. "
            "From the repository root run: python3 scripts/prepare_mose_notebooks.py"
        )
    return f"data:{mime_type};base64,{base64.b64encode(payload).decode('ascii')}"

def show_videos(items, columns=3):
    cards = []
    for title, relative, explanation in items:
        if relative is None:
            cards.append(
                f"<section class='media-card'><h4>{html.escape(title)}</h4>"
                f"<p><strong>Not applicable.</strong></p><p>{html.escape(explanation)}</p></section>"
            )
            continue
        source = REVIEW_ROOT / relative
        data_url = artifact_data_url(source, "video/mp4")
        cards.append(
            "<section class='media-card'>"
            f"<h4>{html.escape(title)}</h4>"
            f"<video controls muted loop playsinline preload='metadata' src='{data_url}'></video>"
            f"<p>{html.escape(explanation)}</p>"
            f"<p><a href='{data_url}' download='{html.escape(source.name)}'>Download this video</a></p>"
            "</section>"
        )
    display(HTML(
        f"<style>.media-grid{{display:grid;grid-template-columns:repeat({columns},minmax(220px,1fr));gap:16px}}"
        ".media-card{border:1px solid #d0d7de;border-radius:8px;padding:12px;background:#fff;color:#24292f}"
        ".media-card video{width:100%;background:#000}.media-card h4{margin:0 0 8px}.media-card p{font-size:13px}</style>"
        f"<div class='media-grid'>{''.join(cards)}</div>"
    ))

def show_images(items, columns=2):
    cards = []
    for title, relative, explanation in items:
        source = REVIEW_ROOT / relative
        data_url = artifact_data_url(source, "image/png")
        cards.append(
            "<section class='media-card'>"
            f"<h4>{html.escape(title)}</h4>"
            f"<img style='width:100%' src='{data_url}' alt='{html.escape(title)}'>"
            f"<p>{html.escape(explanation)}</p>"
            f"<p><a href='{data_url}' download='{html.escape(source.name)}'>Download this image</a></p>"
            "</section>"
        )
    display(HTML(
        f"<div class='media-grid' style='grid-template-columns:repeat({columns},minmax(260px,1fr))'>"
        f"{''.join(cards)}</div>"
    ))

def show_metadata(paths):
    for relative in paths:
        path = REVIEW_ROOT / relative
        payload = json.loads(path.read_text())
        display(Markdown(f"#### `{relative}`"))
        print(json.dumps(payload, indent=2))

print(f"Loaded evidence from {REVIEW_ROOT}")'''
    )


def baseline_markdown(record: dict[str, Any]) -> dict[str, Any]:
    media = record["media"]
    duration = record["frame_count"] / 25.0
    return markdown(
        f"""
### Source and unwatermarked removal reference

The source has {record['frame_count']} frames at {record['width']} x {record['height']}, 25 FPS, and a duration of about {duration:.2f} seconds. The raw mask marks the object selected for removal. The two unwatermarked outputs show how ProPainter and DiffuEraser behave before watermarking is introduced.

Direct links: {link(media['original'], 'original')} | {link(media['gt_mask'], 'raw mask')} | {link(media['propainter'], 'ProPainter reference')} | {link(media['diffueraser'], 'DiffuEraser reference')}
"""
    )


def baseline_videos(record: dict[str, Any]) -> dict[str, Any]:
    media = record["media"]
    return video_cell(
        [
            ("Original MOSE video", media["original"], "Source video before watermarking or removal."),
            ("Raw selected-object mask", media["gt_mask"], "White pixels identify the object selected for removal."),
            ("ProPainter without watermark", media["propainter"], "Unwatermarked ProPainter removal reference."),
            ("DiffuEraser without watermark", media["diffueraser"], "Unwatermarked DiffuEraser removal reference."),
        ]
    )


def validation_cell(method_label: str) -> dict[str, Any]:
    return markdown(
        f"""
## Validation and provenance

- The committed validator decoded the videos, checked frame counts, dimensions, and FPS, and verified the required method fields.
- Aggregate and per-file validation: [validation.json](../../evidence/mose-task1/review50/validation.json)
- Aggregate results: [watermark-summary.json](../../evidence/mose-task1/review50/watermark-summary.json)
- Experimental contract: [PROTOCOL.md](../../evidence/mose-task1/PROTOCOL.md)
- Batch summary: [README.md](../../evidence/mose-task1/README.md)

The displayed values come from the committed {method_label} evidence records.
"""
    )


def build_videoseal(records: list[dict[str, Any]], review_root: Path) -> dict[str, Any]:
    cells: list[dict[str, Any]] = [
        markdown(
            """
# VideoSeal results for three MOSE samples

VideoSeal watermark embedding, object removal, and derived spatial localization are shown for three fixed MOSE samples.

```text
original -> VideoSeal embedding -> watermarked video -> ProPainter or DiffuEraser -> derived localization
```

VideoSeal can watermark an existing video. The released checkpoint provides a global watermark-presence detector rather than a native pixel-level tamper map. The evaluation therefore divides each frame into a 4 x 4 crop grid and derives an absence mask. White predicted pixels indicate crops where the detector did not find the watermark. The result must be described as a derived crop-grid proxy, not native VideoSeal localization.
"""
        ),
        markdown(
            """
## Metric guide

| Term | Meaning | Interpretation |
| --- | --- | --- |
| Global pixel IoU | Total intersection divided by total union across all frames and pixels. | 1 is perfect overlap. 0 means no overlap. This is the primary localization result. |
| Mean frame IoU | IoU calculated for each frame and then averaged. | Empty ground-truth and prediction masks score 1, so this value can look high when many frames have no target. |
| Effective GT mask | The removal mask after the remover's actual preprocessing. | This is the correct reference mask for IoU. |
| Predicted mask | The derived 4 x 4 crop-grid absence prediction. | White indicates predicted tampering. Black indicates no predicted tampering. |
| Intersection pixels | Pixels marked by both the prediction and effective ground truth. | Used in the numerator of global IoU. |
| Union pixels | Pixels marked by either mask. | Used in the denominator of global IoU. |
| Runtime | Wall-clock time recorded on the experiment hardware. | Useful for this run, not a hardware-independent benchmark. |
| SHA-256 | File fingerprint stored in the JSON record. | Confirms exact file identity. It is not a quality metric. |
"""
        ),
        setup_markdown(),
        setup_cell(),
    ]

    for index, record in enumerate(records, 1):
        sample_id = record["clip_id"]
        method = record["watermarks"]["VideoSeal"]
        method_dir = review_root / "media" / sample_id / "methods/videoseal"
        embed = read_json(method_dir / "embed.json")
        prop = read_json(method_dir / "propainter.json")
        diff = read_json(method_dir / "diffueraser.json")
        localization = read_json(method_dir / "localization.json")
        prop_loc = localization["propainter"]
        diff_loc = localization["diffueraser"]

        frame_ious = prop_loc.get("per_frame_iou", [])
        empty_match_frames = sum(value == 1.0 for value in frame_ious)
        empty_note = ""
        if prop_loc["global_iou"] == 0 and prop_loc["mean_frame_iou"] > 0:
            empty_note = (
                f" The mean frame IoU is raised by {empty_match_frames} frames where both masks are empty. "
                "Those empty frames score 1 even though the detector did not localize the edited region."
            )

        cells.extend(
            [
                markdown(f"## {index}. Sample `{sample_id}`"),
                baseline_markdown(record),
                baseline_videos(record),
                markdown(
                    f"""
### VideoSeal results

VideoSeal embedded a {embed['message_bits']}-bit message into {embed['watermarked']['frames']} frames. Embedding took {seconds(embed.get('elapsed_seconds'))}. The detector threshold was `{localization['threshold']}`. A positive detector score means watermark present. A score at or below the threshold becomes predicted tampering.

| Metric | ProPainter | DiffuEraser |
| --- | ---: | ---: |
| Removal runtime | {seconds(prop.get('elapsed_seconds'))} | {seconds(diff.get('elapsed_seconds'))} |
| Frames | {fmt(prop['output']['frames'])} | {fmt(diff['output']['frames'])} |
| Output dimensions | {prop['output']['width']} x {prop['output']['height']} | {diff['output']['width']} x {diff['output']['height']} |
| Global pixel IoU | **{fmt(prop_loc['global_iou'])}** | **{fmt(diff_loc['global_iou'])}** |
| Mean frame IoU | {fmt(prop_loc['mean_frame_iou'])} | {fmt(diff_loc['mean_frame_iou'])} |
| Minimum frame IoU | {fmt(prop_loc['minimum_frame_iou'])} | {fmt(diff_loc['minimum_frame_iou'])} |
| Maximum frame IoU | {fmt(prop_loc['maximum_frame_iou'])} | {fmt(diff_loc['maximum_frame_iou'])} |
| Intersection pixels | {fmt(prop_loc['intersection_pixels'])} | {fmt(diff_loc['intersection_pixels'])} |
| Union pixels | {fmt(prop_loc['union_pixels'])} | {fmt(diff_loc['union_pixels'])} |

Localization for both attacks took {seconds(localization.get('elapsed_seconds'))} in total.

**Interpretation:** The derived proxy is evaluated against the effective mask used by each remover. A global IoU of 0 means that no predicted tamper pixels overlapped the actual removal region.{empty_note}
"""
                ),
                video_cell(
                    [
                        ("VideoSeal watermarked", method["watermarked"], "Watermarked video before object removal."),
                        ("ProPainter attacked video", method["propainter"], "ProPainter output from the VideoSeal-watermarked input."),
                        ("ProPainter effective GT", method["gt_mask_propainter"], "Mask after ProPainter preprocessing. This is its IoU reference."),
                        ("ProPainter predicted mask", method["pred_mask_propainter"], "Derived VideoSeal crop-grid prediction after ProPainter."),
                        ("DiffuEraser attacked video", method["diffueraser"], "DiffuEraser output from the VideoSeal-watermarked input."),
                        ("DiffuEraser effective GT", method["gt_mask_diffueraser"], "Mask after DiffuEraser preprocessing. This is its IoU reference."),
                        ("DiffuEraser predicted mask", method["pred_mask_diffueraser"], "Derived VideoSeal crop-grid prediction after DiffuEraser."),
                    ]
                ),
                markdown(
                    "### Run records\n\nThe next cell prints the embedding, removal, and localization JSON records. The localization record includes per-frame IoU values and file fingerprints."
                ),
                json_cell(list(method["metadata"].values())),
            ]
        )

    cells.append(validation_cell("VideoSeal"))
    return notebook(cells)


def build_safe_sora(records: list[dict[str, Any]], review_root: Path) -> dict[str, Any]:
    cells: list[dict[str, Any]] = [
        markdown(
            """
# Safe-Sora results for three MOSE samples

Safe-Sora graphical watermark embedding, object removal, and watermark recovery are shown for three fixed MOSE samples.

```text
sampled original -> Safe-Sora embedding -> watermarked video -> ProPainter or DiffuEraser -> graphical watermark recovery
```

Safe-Sora reconstructs a graphical watermark image. It does not produce a spatial tamper mask, so predicted-mask and IoU fields are not applicable.
"""
        ),
        markdown(
            """
## Metric guide

| Term | Meaning | Interpretation |
| --- | --- | --- |
| PSNR | Logarithmic similarity derived from mean squared error. | Higher is better. Values around 8 dB indicate weak graphical recovery. Values above 40 dB indicate small pixel differences. |
| MSE | Mean squared 8-bit pixel error. | Lower is better. Large errors receive extra weight because they are squared. |
| MAE | Mean absolute 8-bit pixel error. | Lower is better. It is the average absolute change per channel value. |
| VAE reconstruction fidelity | Difference between the selected input and its VAE reconstruction. | Separates ordinary VAE change from the additional watermark change. |
| Embedding fidelity | Difference between the VAE cover and watermarked output. | High PSNR and low errors mean the embedded watermark causes small pixel changes. |
| Recovery image | Reference watermark, recovered watermark, and amplified error shown side by side. | It is not an object mask and cannot be used for IoU. |
| Runtime | Wall-clock time recorded on the experiment hardware. | Useful for this run, not a hardware-independent benchmark. |
| SHA-256 | File fingerprint stored in the JSON record. | Confirms exact file identity. It is not a quality metric. |
"""
        ),
        setup_markdown(),
        setup_cell(),
    ]

    for index, record in enumerate(records, 1):
        sample_id = record["clip_id"]
        method = record["watermarks"]["Safe-Sora"]
        method_dir = review_root / "media" / sample_id / "methods/safe-sora"
        embed = read_json(method_dir / "embed.json")
        prop = read_json(method_dir / "propainter.json")
        diff = read_json(method_dir / "diffueraser.json")
        recovery = read_json(method_dir / "recovery.json")
        vae = embed["input_to_vae_reconstruction"]
        embedding = embed["watermark_embedding_fidelity"]
        prop_recovery = recovery["propainter"]["fidelity"]
        diff_recovery = recovery["diffueraser"]["fidelity"]

        if prop_recovery["psnr_db"] > diff_recovery["psnr_db"]:
            better = "ProPainter"
            gap = prop_recovery["psnr_db"] - diff_recovery["psnr_db"]
        else:
            better = "DiffuEraser"
            gap = diff_recovery["psnr_db"] - prop_recovery["psnr_db"]

        adapter = diff.get("compatibility_adapter")
        adapter_text = "No temporal compatibility adapter was needed."
        if adapter:
            adapter_text = (
                f"DiffuEraser requires at least 22 frames. The final frame and mask were repeated to "
                f"{adapter['padded_frames']} frames during inference, then the output was trimmed back to "
                f"{adapter['reported_frames']} genuine frames."
            )

        indices = ", ".join(str(value) for value in embed["sampling"]["indices"])
        cells.extend(
            [
                markdown(f"## {index}. Sample `{sample_id}`"),
                baseline_markdown(record),
                baseline_videos(record),
                markdown(
                    f"""
### Safe-Sora results

Safe-Sora selected 16 frames at indices `{indices}` and resized them to 512 x 320. Embedding took {seconds(embed.get('elapsed_seconds'))}.

{adapter_text}

#### Embedding fidelity

| Comparison | PSNR | MSE | MAE |
| --- | ---: | ---: | ---: |
| Selected input to VAE cover | {fmt(vae['psnr_db'])} dB | {fmt(vae['mse'])} | {fmt(vae['mean_absolute_error'])} |
| VAE cover to watermarked output | **{fmt(embedding['psnr_db'])} dB** | {fmt(embedding['mse'])} | {fmt(embedding['mean_absolute_error'])} |

The first row measures ordinary VAE reconstruction change. The second row isolates the additional visible change caused by watermark embedding.

#### Removal and recovery

| Metric | ProPainter | DiffuEraser |
| --- | ---: | ---: |
| Removal runtime | {seconds(prop.get('elapsed_seconds'))} | {seconds(diff.get('elapsed_seconds'))} |
| Reported output frames | {fmt(prop['output']['frames'])} | {fmt(diff['output']['frames'])} |
| Recovery PSNR | **{fmt(prop_recovery['psnr_db'])} dB** | **{fmt(diff_recovery['psnr_db'])} dB** |
| Recovery MSE | {fmt(prop_recovery['mse'])} | {fmt(diff_recovery['mse'])} |
| Recovery MAE | {fmt(prop_recovery['mean_absolute_error'])} | {fmt(diff_recovery['mean_absolute_error'])} |
| Spatial predicted mask and IoU | N/A | N/A |

Recovery for both attacks took {seconds(recovery.get('elapsed_seconds'))} in total. {better} has the higher recovery PSNR by {gap:.4f} dB for this sample. This comparison concerns watermark recovery, not the visual quality of the inpainted video.
"""
                ),
                video_cell(
                    [
                        ("Safe-Sora sampled original", method["input_original"], "Aligned 16-frame source used by Safe-Sora."),
                        ("Safe-Sora aligned mask", method["gt_mask"], "Mask sampled and resized to match the 16 input frames."),
                        ("Safe-Sora watermarked", method["watermarked"], "Graphical-watermarked video before object removal."),
                        ("Safe-Sora with ProPainter", method["propainter"], "ProPainter output after Safe-Sora embedding."),
                        ("Safe-Sora with DiffuEraser", method["diffueraser"], "DiffuEraser output after Safe-Sora embedding. Temporary padding is not shown."),
                    ]
                ),
                markdown(
                    "### Recovered graphical watermarks\n\nEach image shows the reference watermark on the left, the recovered watermark in the middle, and amplified pixel error on the right. These are not object masks."
                ),
                image_cell(
                    [
                        ("Recovery after ProPainter", method["recovery_propainter"], f"Recovery PSNR: {prop_recovery['psnr_db']:.4f} dB."),
                        ("Recovery after DiffuEraser", method["recovery_diffueraser"], f"Recovery PSNR: {diff_recovery['psnr_db']:.4f} dB."),
                    ]
                ),
                markdown(
                    "### Run records\n\nThe next cell prints the embedding, removal, and recovery JSON records, including sampled frame indices, compatibility details, fidelity metrics, and file fingerprints."
                ),
                json_cell(list(method["metadata"].values())),
            ]
        )

    cells.append(validation_cell("Safe-Sora"))
    return notebook(cells)


def build_videoshield(records: list[dict[str, Any]], review_root: Path) -> dict[str, Any]:
    cells: list[dict[str, Any]] = [
        markdown(
            """
# VideoShield records for three MOSE samples

VideoShield embeds its watermark in a diffusion generator's initial noise. The MOSE clips already exist, so an authentic VideoShield watermark cannot be added by post-processing them.

Each sample therefore contains an unwatermarked negative-control detection record. There is no VideoShield-watermarked video, removal output, predicted mask, or IoU for these clips. Those fields are not applicable because watermark embedding never occurred.
"""
        ),
        markdown(
            """
## Metric guide

| Term | Meaning | Interpretation |
| --- | --- | --- |
| Negative-control bit accuracy | Fraction of reference watermark bits matched in an unwatermarked input. | About 0.5 is chance for balanced binary bits. It is not an embedding or robustness score. |
| DDIM inversion steps | Number of reverse-diffusion steps used by the detector control. | A configuration value, not a quality metric. |
| Not applicable | The experiment cannot be performed faithfully for an existing MOSE clip. | It is different from failure and different from a score of zero. |
| Runtime | Wall-clock time recorded on the experiment hardware. | Useful for this run, not a hardware-independent benchmark. |
| SHA-256 | File fingerprint stored in the JSON record. | Confirms exact file identity. It is not a quality metric. |
"""
        ),
        setup_markdown(),
        setup_cell(),
    ]

    for index, record in enumerate(records, 1):
        sample_id = record["clip_id"]
        method = record["watermarks"]["VideoShield"]
        availability_path = review_root / method["metadata"]["availability"]
        availability = read_json(availability_path)
        accuracy = availability["reference_watermark_accuracy"]
        chance_delta = abs(accuracy - 0.5) * 100
        sampling = availability["sampling"]

        cells.extend(
            [
                markdown(
                    f"""
## {index}. Sample `{sample_id}`

The source has {record['frame_count']} frames at {record['width']} x {record['height']} and 25 FPS. The original and raw mask are shown as context. They are not VideoShield outputs.

The negative control sampled {sampling['frames']} frames, resized them to {sampling['resize'][0]} x {sampling['resize'][1]}, and used {availability['ddim_inversion_steps']} DDIM inversion steps. The reference watermark was not embedded into the clip.

| Metric or configuration | Value | Interpretation |
| --- | ---: | --- |
| Negative-control bit accuracy | **{accuracy:.4f}** ({accuracy * 100:.2f}%) | {chance_delta:.2f} percentage points from the 50% chance level. |
| Sampled frames | {sampling['frames']} | Evenly selected control frames from the source. |
| Detector resize | {sampling['resize'][0]} x {sampling['resize'][1]} | Resolution used by the detector control. |
| DDIM inversion steps | {availability['ddim_inversion_steps']} | Detector configuration. |
| Control runtime | {seconds(availability.get('elapsed_seconds'))} | Runtime for the negative control. |
| Watermarked video | N/A | No authentic VideoShield embedding exists for this clip. |
| ProPainter and DiffuEraser outputs | N/A | No watermarked input exists to attack. |
| Predicted tamper mask and IoU | N/A | No valid removal experiment exists for this clip. |

**Interpretation:** A value near 0.5 is expected for an unwatermarked input. This control does not measure watermark robustness or localization.
"""
                ),
                video_cell(
                    [
                        ("Original MOSE video", method["input_original"], "Unwatermarked source used for the negative control."),
                        ("Raw selected-object mask", method["gt_mask"], "Removal region shown only as source context."),
                        ("VideoShield watermarked and removal outputs", None, "Authentic VideoShield embedding requires diffusion generation, so these outputs do not exist for this clip."),
                    ]
                ),
                markdown(
                    "### Availability and control record\n\nThe next cell prints the exact sampling configuration, control result, runtime, interpretation, and explicit not-applicable fields."
                ),
                json_cell([method["metadata"]["availability"]]),
            ]
        )

    cells.append(validation_cell("VideoShield"))
    return notebook(cells)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-root", type=Path, default=Path("evidence/mose-task1/review50"))
    parser.add_argument("--output", type=Path, default=Path("notebooks/mose-watermark-samples"))
    parser.add_argument("--sample", action="append", dest="samples")
    args = parser.parse_args()

    review_root = args.review_root.resolve()
    output = args.output.resolve()
    manifest = read_json(review_root / "manifest.json")
    records_by_id = {item["clip_id"]: item for item in manifest["samples"]}
    samples = tuple(args.samples or DEFAULT_SAMPLES)
    missing = [sample for sample in samples if sample not in records_by_id]
    if missing:
        raise SystemExit(f"Unknown sample IDs: {', '.join(missing)}")

    records = [records_by_id[sample_id] for sample_id in samples]
    output.mkdir(parents=True, exist_ok=True)
    outputs = {
        "videoseal.ipynb": build_videoseal(records, review_root),
        "safe_sora.ipynb": build_safe_sora(records, review_root),
        "videoshield.ipynb": build_videoshield(records, review_root),
    }
    for filename, document in outputs.items():
        destination = output / filename
        destination.write_text(json.dumps(document, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(destination)


if __name__ == "__main__":
    main()
