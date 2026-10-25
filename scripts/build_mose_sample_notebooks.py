#!/usr/bin/env python3
"""Build three self-contained MOSE watermark comparison notebooks.

The notebooks do not duplicate the large media bundle.  They display and link
the already committed files under ``evidence/mose-task1/review50`` and explain
the different output semantics of VideoSeal, Safe-Sora, and VideoShield.
"""

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


def describe_artifact(path: Path) -> str:
    name = path.name
    parent = path.parent.name
    if "methods" in path.parts and name == "propainter.mp4":
        return f"ProPainter attack/removal output for {parent}."
    if "methods" in path.parts and name == "diffueraser.mp4":
        return f"DiffuEraser attack/removal output for {parent}."
    descriptions = {
        "original.mp4": "Full MOSE source clip shared by the baseline and compatible watermark methods.",
        "gt_mask.mp4": "Raw selected-object mask; white pixels identify the object chosen for removal.",
        "propainter.mp4": "Object-removal output produced from the unwatermarked source clip.",
        "diffueraser.mp4": "Object-removal output produced from the unwatermarked source clip.",
        "sampled-original.mp4": "The 16 frames sampled and resized for Safe-Sora's released protocol.",
        "gt-mask.mp4": "Safe-Sora-aligned 16-frame object mask.",
        "watermarked.mp4": f"Watermarked input produced by {parent} before either removal attack.",
        "effective-gt-mask-propainter.mp4": "Ground-truth region after ProPainter's actual mask preprocessing.",
        "effective-gt-mask-diffueraser.mp4": "Ground-truth region after DiffuEraser's actual mask preprocessing.",
        "pred-mask-propainter.mp4": "VideoSeal's derived patch-grid predicted tamper mask after ProPainter.",
        "pred-mask-diffueraser.mp4": "VideoSeal's derived patch-grid predicted tamper mask after DiffuEraser.",
        "recovery-propainter.png": "Safe-Sora graphical-watermark recovery after ProPainter.",
        "recovery-diffueraser.png": "Safe-Sora graphical-watermark recovery after DiffuEraser.",
        "embed.json": "Embedding settings, runtime, media properties, checksums, and fidelity metrics.",
        "localization.json": "VideoSeal localization mode, masks, IoU values, and per-frame measurements.",
        "recovery.json": "Safe-Sora recovery metrics and recovery-image checksums.",
        "availability.json": "VideoShield capability/availability record and negative-control result.",
    }
    if name in descriptions:
        return descriptions[name]
    if name == "propainter.json":
        return f"{parent} + ProPainter run settings, runtime, output properties, and checksum."
    if name == "diffueraser.json":
        return f"{parent} + DiffuEraser run settings, runtime, output properties, and checksum."
    return "Committed experiment artifact."


def video_inventory(sample_root: Path, review_root: Path) -> str:
    rows = ["| Group | Video | What it shows | Size |", "| --- | --- | --- | ---: |"]
    for path in sorted(sample_root.rglob("*.mp4")):
        relative = path.relative_to(review_root).as_posix()
        group = "baseline" if "methods" not in path.parts else path.parent.name
        size_mib = path.stat().st_size / (1024 * 1024)
        rows.append(
            f"| {group} | {link(relative)} | {describe_artifact(path)} | {size_mib:.2f} MiB |"
        )
    return "\n".join(rows)


def artifact_inventory(sample_root: Path, review_root: Path) -> str:
    rows = ["| Artifact | Type | Description |", "| --- | --- | --- |"]
    for path in sorted(sample_root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(review_root).as_posix()
        suffix = path.suffix.lstrip(".").upper()
        rows.append(f"| {link(relative)} | {suffix} | {describe_artifact(path)} |")
    return "\n".join(rows)


def video_cell(items: list[tuple[str, str, str]]) -> dict[str, Any]:
    return code("show_videos(" + repr(items) + ")")


def image_cell(items: list[tuple[str, str, str]]) -> dict[str, Any]:
    return code("show_images(" + repr(items) + ")")


def json_cell(paths: list[str]) -> dict[str, Any]:
    return code("show_metadata(" + repr(paths) + ")")


def metric_glossary() -> str:
    return r"""
## Metric guide

The methods do not expose the same kind of signal, so their numbers must not be treated as interchangeable.

| Term | Meaning | Direction and interpretation |
| --- | --- | --- |
| **Global pixel IoU** | `sum(predicted AND ground truth) / sum(predicted OR ground truth)` over every pixel in every frame. The ground truth is the effective mask actually consumed by the remover. | 1 is perfect overlap; 0 means no overlap. This is the primary spatial-localization number. |
| **Mean frame IoU** | IoU is calculated separately for every frame, then averaged. A frame where both masks are empty is conventionally scored 1. | Useful for finding temporal behavior, but it can look high when many frames contain no target. It is not interchangeable with global IoU. |
| **Minimum/maximum frame IoU** | Worst and best individual-frame IoU. | Shows the range across time; it does not describe total pixel overlap. |
| **Intersection pixels** | Number of pixels classified as tampered by both the predicted mask and effective ground truth. | Larger is not automatically better because object size varies; interpret together with union and IoU. |
| **Union pixels** | Number of pixels classified as tampered by either mask. | The denominator of global IoU. |
| **PSNR (dB)** | Logarithmic fidelity derived from mean squared error for 8-bit images. | Higher is more similar. Identical images have infinite PSNR. Values around 8 dB indicate a visually weak reconstruction; values above 40 dB indicate very small pixel differences. |
| **MSE** | Mean squared 8-bit pixel error. Large errors receive extra weight because they are squared. | Lower is better; 0 is identical. |
| **MAE** | Mean absolute 8-bit pixel error. | Lower is better; it is the average absolute change per channel value. |
| **Bit accuracy** | Fraction of binary watermark bits recovered correctly. | 1 is perfect. For a balanced binary watermark, about 0.5 is chance. |
| **Elapsed seconds** | Wall-clock runtime recorded for that stage on the experiment GPU. | Useful for this run, not a hardware-independent benchmark. |
| **Frames / FPS / dimensions** | Output length, playback rate, and width × height. | Used to prove that compared artifacts are aligned and decodable. |
| **Mask dilation iterations** | Number of times the selected-object mask was expanded before inpainting. | A configuration value, not a quality score. It gives the remover context around object boundaries. |
| **DDIM inversion steps** | Number of reverse-diffusion steps used by the VideoShield detector control. | A detector configuration value, not a quality score. |
| **SHA-256** | Content fingerprint stored in metadata. | An integrity check: matching hashes mean the exact recorded file is present. It is not a model metric. |
| **Complete / not applicable** | `complete` means the recorded stage produced and validated its expected artifacts. `not applicable` means the method cannot perform that experiment faithfully. | `Not applicable` is deliberately different from failure or a score of zero. |

### Mask terminology

- **Raw mask:** the original MOSE annotation chosen for object removal; white means “remove this region.”
- **Effective GT mask:** the mask after the remover's real resize, erosion, and/or dilation path. This is the correct IoU reference.
- **Predicted mask:** where a watermark detector says tampering occurred. For VideoSeal here it is a documented **derived 4 × 4 crop-grid proxy**, not a native pixelwise output from the released checkpoint.
"""


def setup_cell(sample_id: str) -> dict[str, Any]:
    return code(
        f'''from pathlib import Path
import html
import json
from IPython.display import HTML, Image, Markdown, display

SAMPLE_ID = "{sample_id}"
# Jupyter's /files route serves repository files when the server is started
# from the repository root. Static Markdown links separately use GitHub-safe
# ../../evidence/... paths.
BROWSER_PREFIX = "/files/evidence/mose-task1/review50/"

def find_repo_root():
    for start in (Path.cwd().resolve(), *Path.cwd().resolve().parents):
        if (start / "evidence/mose-task1/review50/manifest.json").is_file():
            return start
    raise FileNotFoundError("Run this notebook from inside the paper-reproductions checkout.")

REPO_ROOT = find_repo_root()
REVIEW_ROOT = REPO_ROOT / "evidence/mose-task1/review50"
SAMPLE_ROOT = REVIEW_ROOT / "media" / SAMPLE_ID
MANIFEST = json.loads((REVIEW_ROOT / "manifest.json").read_text())
SAMPLE = next(item for item in MANIFEST["samples"] if item["clip_id"] == SAMPLE_ID)

def browser_url(relative):
    return BROWSER_PREFIX + str(relative).replace("\\\\", "/")

def show_videos(items, columns=3):
    cards = []
    for title, relative, explanation in items:
        if relative is None:
            cards.append(f"<section class='media-card'><h4>{{html.escape(title)}}</h4><p><strong>Not applicable.</strong></p><p>{{html.escape(explanation)}}</p></section>")
            continue
        source = REVIEW_ROOT / relative
        if not source.is_file():
            raise FileNotFoundError(source)
        cards.append(
            "<section class='media-card'>"
            f"<h4>{{html.escape(title)}}</h4>"
            f"<video controls muted loop playsinline preload='metadata' src='{{html.escape(browser_url(relative))}}'></video>"
            f"<p>{{html.escape(explanation)}}</p>"
            f"<p><a href='{{html.escape(browser_url(relative))}}'>Open video directly</a></p>"
            "</section>"
        )
    display(HTML(
        f"<style>.media-grid{{{{display:grid;grid-template-columns:repeat({{columns}},minmax(220px,1fr));gap:16px}}}}"
        ".media-card{{border:1px solid #d0d7de;border-radius:8px;padding:12px;background:#fff;color:#24292f}}"
        ".media-card video{{width:100%;background:#000}}.media-card h4{{margin:0 0 8px}}.media-card p{{font-size:13px}}</style>"
        f"<div class='media-grid'>{{''.join(cards)}}</div>"
    ))

def show_images(items, columns=2):
    cards = []
    for title, relative, explanation in items:
        source = REVIEW_ROOT / relative
        if not source.is_file():
            raise FileNotFoundError(source)
        cards.append(
            "<section class='media-card'>"
            f"<h4>{{html.escape(title)}}</h4>"
            f"<img style='width:100%' src='{{html.escape(browser_url(relative))}}' alt='{{html.escape(title)}}'>"
            f"<p>{{html.escape(explanation)}}</p>"
            f"<p><a href='{{html.escape(browser_url(relative))}}'>Open image directly</a></p>"
            "</section>"
        )
    display(HTML(f"<div class='media-grid' style='grid-template-columns:repeat({{columns}},minmax(260px,1fr))'>{{''.join(cards)}}</div>"))

def show_metadata(paths):
    for relative in paths:
        path = REVIEW_ROOT / relative
        payload = json.loads(path.read_text())
        display(Markdown(f"#### `{{relative}}`"))
        print(json.dumps(payload, indent=2))

print(f"Loaded sample {{SAMPLE_ID}} from {{SAMPLE_ROOT}}")'''
    )


def build_notebook(record: dict[str, Any], review_root: Path) -> dict[str, Any]:
    sample_id = record["clip_id"]
    sample_root = review_root / "media" / sample_id
    fps = 25.0
    duration = record["frame_count"] / fps
    watermarks = record["watermarks"]
    videoseal = watermarks["VideoSeal"]
    safe_sora = watermarks["Safe-Sora"]
    videoshield = watermarks["VideoShield"]

    vs_dir = sample_root / "methods/videoseal"
    ss_dir = sample_root / "methods/safe-sora"
    shield_dir = sample_root / "methods/videoshield"
    vs_embed = read_json(vs_dir / "embed.json")
    vs_prop = read_json(vs_dir / "propainter.json")
    vs_diff = read_json(vs_dir / "diffueraser.json")
    vs_loc = read_json(vs_dir / "localization.json")
    ss_embed = read_json(ss_dir / "embed.json")
    ss_prop = read_json(ss_dir / "propainter.json")
    ss_diff = read_json(ss_dir / "diffueraser.json")
    ss_recovery = read_json(ss_dir / "recovery.json")
    shield = read_json(shield_dir / "availability.json")

    base_media = record["media"]
    vs_prop_loc = vs_loc["propainter"]
    vs_diff_loc = vs_loc["diffueraser"]
    ss_embed_fidelity = ss_embed["watermark_embedding_fidelity"]
    ss_vae_fidelity = ss_embed["input_to_vae_reconstruction"]
    ss_prop_recovery = ss_recovery["propainter"]["fidelity"]
    ss_diff_recovery = ss_recovery["diffueraser"]["fidelity"]
    negative_accuracy = shield["reference_watermark_accuracy"]

    if ss_prop_recovery["psnr_db"] > ss_diff_recovery["psnr_db"]:
        better_recovery = "ProPainter"
        recovery_gap = ss_prop_recovery["psnr_db"] - ss_diff_recovery["psnr_db"]
    else:
        better_recovery = "DiffuEraser"
        recovery_gap = ss_diff_recovery["psnr_db"] - ss_prop_recovery["psnr_db"]

    empty_frame_note = ""
    if (
        vs_prop_loc["global_iou"] == 0
        and vs_prop_loc["mean_frame_iou"] > 0
        and vs_prop_loc["intersection_pixels"] == 0
    ):
        empty_frame_note = (
            " The non-zero mean frame IoU comes from frames where both the predicted and effective-GT "
            "masks are empty; those frames score 1 individually. Global IoU still correctly reports zero "
            "overlap across the complete clip."
        )

    cells: list[dict[str, Any]] = []
    cells.append(
        markdown(
            f"""
# MOSE sample `{sample_id}` — complete watermark/removal comparison

This notebook is one self-contained case study. It presents every video and result artifact committed for MOSE clip `{sample_id}` across the unwatermarked baseline and the three requested watermark methods: **VideoSeal**, **Safe-Sora**, and **VideoShield**.

The experiment order is always:

```text
original → watermark embedding → watermarked video → ProPainter or DiffuEraser → detector/recovery
```

That ordering matters: the removal method attacks a video that was already watermarked. The unwatermarked baseline is shown separately so visual removal quality can be inspected without conflating it with watermark behavior.

> **Important capability result:** VideoShield cannot faithfully add its watermark to an existing MOSE video because it embeds into a diffusion generator's initial noise. The repository therefore contains a labelled negative control—not fabricated watermarked, ProPainter, DiffuEraser, mask, or IoU outputs—for this method/sample pair.

All media remains in the validated GitHub evidence bundle under `evidence/mose-task1/review50/media/{sample_id}`. The notebook references those files instead of duplicating them.
"""
        )
    )
    cells.append(
        markdown(
            f"""
## At-a-glance status

| Section | Status | What is available |
| --- | --- | --- |
| Unwatermarked baseline | Complete | Original, raw mask, ProPainter output, DiffuEraser output |
| VideoSeal | Complete | Watermarked input, both removal attacks, attack-specific effective GT masks, both derived predicted masks, localization metrics |
| Safe-Sora | Complete | Aligned sampled original, aligned mask, watermarked input, both removal attacks, two graphical-watermark recovery images, fidelity metrics |
| VideoShield | Not applicable for watermarking this MOSE clip | Shared original/mask plus an explicit negative-control and availability record |

**Source properties:** {record['frame_count']} frames, {record['width']} × {record['height']}, {fps:.0f} FPS, approximately {duration:.2f} seconds.
"""
        )
    )
    cells.append(markdown(metric_glossary()))
    cells.append(markdown("## Notebook setup\n\nRun this cell once. It locates the repository and defines the inline video, image, and metadata viewers used below. Direct artifact links still work when reading the notebook without executing it."))
    cells.append(setup_cell(sample_id))

    cells.append(
        markdown(
            f"""
## 1. Unwatermarked removal baseline

This section answers a visual question: **can each inpainting method remove the selected object when no watermark is involved?**

- The original is the full MOSE clip.
- The raw mask shows the selected object in white.
- ProPainter and DiffuEraser each receive that selected object, with their recorded mask preprocessing.
- These two outputs are context only; they are not watermark-robustness results.

Direct links: {link(base_media['original'], 'original')} · {link(base_media['gt_mask'], 'raw mask')} · {link(base_media['propainter'], 'ProPainter baseline')} · {link(base_media['diffueraser'], 'DiffuEraser baseline')}
"""
        )
    )
    cells.append(
        video_cell(
            [
                ("Original MOSE video", base_media["original"], "The source clip before watermarking or removal."),
                ("Raw selected-object mask", base_media["gt_mask"], "White pixels identify the object selected for removal."),
                ("ProPainter, no watermark", base_media["propainter"], "Baseline ProPainter removal on the unwatermarked source."),
                ("DiffuEraser, no watermark", base_media["diffueraser"], "Baseline DiffuEraser removal on the unwatermarked source."),
            ]
        )
    )

    cells.append(
        markdown(
            f"""
## 2. VideoSeal → removal → derived localization

VideoSeal post-processes an existing video, so it can follow the required protocol exactly. It embeds a {vs_embed['message_bits']}-bit message into all {vs_embed['watermarked']['frames']} frames. ProPainter and DiffuEraser then remove the selected object from that watermarked input.

The released checkpoint exposes a global watermark-presence detector rather than a native pixelwise tamper map. This experiment therefore derives a clearly labelled **4 × 4 crop-grid absence mask**. White predicted-mask pixels mean the crop proxy judged the watermark absent there. These masks and IoUs must not be described as native VideoSeal localization.

### Recorded VideoSeal metrics

| Metric | ProPainter attack | DiffuEraser attack |
| --- | ---: | ---: |
| Removal runtime | {seconds(vs_prop.get('elapsed_seconds'))} | {seconds(vs_diff.get('elapsed_seconds'))} |
| Frames | {fmt(vs_prop['output']['frames'])} | {fmt(vs_diff['output']['frames'])} |
| Output dimensions | {vs_prop['output']['width']} × {vs_prop['output']['height']} | {vs_diff['output']['width']} × {vs_diff['output']['height']} |
| Global pixel IoU | **{fmt(vs_prop_loc['global_iou'])}** | **{fmt(vs_diff_loc['global_iou'])}** |
| Mean frame IoU | {fmt(vs_prop_loc['mean_frame_iou'])} | {fmt(vs_diff_loc['mean_frame_iou'])} |
| Minimum frame IoU | {fmt(vs_prop_loc['minimum_frame_iou'])} | {fmt(vs_diff_loc['minimum_frame_iou'])} |
| Maximum frame IoU | {fmt(vs_prop_loc['maximum_frame_iou'])} | {fmt(vs_diff_loc['maximum_frame_iou'])} |
| Intersection pixels | {fmt(vs_prop_loc['intersection_pixels'])} | {fmt(vs_diff_loc['intersection_pixels'])} |
| Union pixels | {fmt(vs_prop_loc['union_pixels'])} | {fmt(vs_diff_loc['union_pixels'])} |

Embedding runtime was {seconds(vs_embed.get('elapsed_seconds'))}; localization runtime for both attacks together was {seconds(vs_loc.get('elapsed_seconds'))}. The detection threshold was `{vs_loc['threshold']}`: positive detector score means watermark present, and a score at or below the threshold becomes predicted tampering.

### Interpretation

Both global IoUs are {fmt(vs_prop_loc['global_iou'])}. The derived detector predicted no pixels overlapping the actual removed region, so the crop-grid proxy did not spatially localize the attack in this sample. This does **not** by itself prove that the watermark was absent everywhere; it says the documented spatial proxy failed to overlap the effective ground truth.{empty_frame_note}
"""
        )
    )
    cells.append(
        video_cell(
            [
                ("VideoSeal input original", videoseal["input_original"], "The exact source that receives the VideoSeal message."),
                ("VideoSeal watermarked", videoseal["watermarked"], "The watermarked video before either removal attack."),
                ("Raw source mask", videoseal["gt_mask"], "Human-readable MOSE annotation before remover-specific preprocessing."),
                ("ProPainter attacked video", videoseal["propainter"], "ProPainter output after attacking the VideoSeal-watermarked input."),
                ("ProPainter effective GT", videoseal["gt_mask_propainter"], "Mask after ProPainter's resize and eight dilation iterations; this is its IoU reference."),
                ("ProPainter predicted mask", videoseal["pred_mask_propainter"], "Derived 4 × 4 VideoSeal absence-grid prediction after ProPainter."),
                ("DiffuEraser attacked video", videoseal["diffueraser"], "DiffuEraser output after attacking the VideoSeal-watermarked input."),
                ("DiffuEraser effective GT", videoseal["gt_mask_diffueraser"], "Mask after DiffuEraser's erosion/dilation path; this is its IoU reference."),
                ("DiffuEraser predicted mask", videoseal["pred_mask_diffueraser"], "Derived 4 × 4 VideoSeal absence-grid prediction after DiffuEraser."),
            ]
        )
    )
    cells.append(
        markdown(
            "### VideoSeal run records\n\nThe next cell prints the exact committed embedding, attack, and localization JSON. It includes per-frame IoUs and SHA-256 fingerprints in addition to the summarized values above."
        )
    )
    cells.append(json_cell(list(videoseal["metadata"].values())))

    adapter = ss_diff.get("compatibility_adapter")
    adapter_text = (
        f"DiffuEraser requires at least 22 frames, so the final frame and mask were repeated to "
        f"{adapter['padded_frames']} frames only during inference and the output was trimmed back to "
        f"{adapter['reported_frames']} genuine frames."
        if adapter
        else "No temporal compatibility adapter was needed."
    )
    cells.append(
        markdown(
            f"""
## 3. Safe-Sora → removal → graphical-watermark recovery

Safe-Sora is evaluated under its released 16-frame, 512 × 320 protocol. It samples 16 aligned frames from the MOSE clip, passes them through the VAE path, embeds a **graphical watermark**, and then applies each remover. Its released method reconstructs a watermark image; it does not produce a spatial tamper mask. Therefore predicted-mask and IoU fields are truthfully **not applicable**.

{adapter_text}

### Embedding fidelity

| Comparison | PSNR | MSE | MAE | Meaning |
| --- | ---: | ---: | ---: | --- |
| Selected input → VAE cover reconstruction | {fmt(ss_vae_fidelity['psnr_db'])} dB | {fmt(ss_vae_fidelity['mse'])} | {fmt(ss_vae_fidelity['mean_absolute_error'])} | Measures ordinary VAE reconstruction change before the watermark difference is isolated. |
| VAE cover → watermarked output | **{fmt(ss_embed_fidelity['psnr_db'])} dB** | {fmt(ss_embed_fidelity['mse'])} | {fmt(ss_embed_fidelity['mean_absolute_error'])} | Measures the additional visible pixel change attributable to embedding. |

The embedding PSNR is high and the errors are low, meaning the watermarked video is pixelwise close to Safe-Sora's cover reconstruction. Embedding took {seconds(ss_embed.get('elapsed_seconds'))}.

### Removal and watermark-recovery metrics

| Metric | ProPainter attack | DiffuEraser attack |
| --- | ---: | ---: |
| Removal runtime | {seconds(ss_prop.get('elapsed_seconds'))} | {seconds(ss_diff.get('elapsed_seconds'))} |
| Reported output frames | {fmt(ss_prop['output']['frames'])} | {fmt(ss_diff['output']['frames'])} |
| Recovery PSNR | **{fmt(ss_prop_recovery['psnr_db'])} dB** | **{fmt(ss_diff_recovery['psnr_db'])} dB** |
| Recovery MSE | {fmt(ss_prop_recovery['mse'])} | {fmt(ss_diff_recovery['mse'])} |
| Recovery MAE | {fmt(ss_prop_recovery['mean_absolute_error'])} | {fmt(ss_diff_recovery['mean_absolute_error'])} |
| Spatial predicted mask / IoU | N/A | N/A |

Graphical recovery for both attacks took {seconds(ss_recovery.get('elapsed_seconds'))} in total. {better_recovery} has the higher recovery PSNR by {recovery_gap:.4f} dB for this sample, but both recovery PSNR values are around 8 dB, indicating a weak reconstruction of the original graphical watermark. This is a watermark-recovery result—not a judgment of which inpainted video looks better.
"""
        )
    )
    cells.append(
        video_cell(
            [
                ("Safe-Sora sampled original", safe_sora["input_original"], "The aligned 16-frame, 512 × 320 source used by Safe-Sora."),
                ("Safe-Sora watermarked", safe_sora["watermarked"], "The graphical-watermarked input before removal."),
                ("Safe-Sora aligned mask", safe_sora["gt_mask"], "The source removal mask sampled and resized to the same 16 frames."),
                ("Safe-Sora + ProPainter", safe_sora["propainter"], "ProPainter removal applied after Safe-Sora embedding."),
                ("Safe-Sora + DiffuEraser", safe_sora["diffueraser"], "DiffuEraser removal applied after Safe-Sora embedding; temporary padding is not shown."),
            ]
        )
    )
    cells.append(
        markdown(
            "### Recovered graphical watermarks\n\nEach image contains the recovery visualization produced from the attacked video. These images are not object masks and must not be used to calculate IoU."
        )
    )
    cells.append(
        image_cell(
            [
                ("Recovery after ProPainter", safe_sora["recovery_propainter"], f"Recovery PSNR: {ss_prop_recovery['psnr_db']:.4f} dB."),
                ("Recovery after DiffuEraser", safe_sora["recovery_diffueraser"], f"Recovery PSNR: {ss_diff_recovery['psnr_db']:.4f} dB."),
            ]
        )
    )
    cells.append(
        markdown(
            "### Safe-Sora run records\n\nThe next cell prints the committed embedding, removal, and recovery JSON, including sampled frame indices, the compatibility adapter, all fidelity values, and checksums."
        )
    )
    cells.append(json_cell(list(safe_sora["metadata"].values())))

    chance_delta = abs(negative_accuracy - 0.5) * 100
    cells.append(
        markdown(
            f"""
## 4. VideoShield capability record and negative control

VideoShield's watermark is injected into a diffusion model's initial generation noise. This MOSE clip already exists, so VideoShield cannot add its authentic watermark by post-processing the clip. Consequently there is no truthful VideoShield-watermarked video to send to ProPainter or DiffuEraser, and there can be no attack-specific predicted mask or IoU for this sample.

The repository still records an explicit control: 16 frames were sampled, resized to 256 × 256, inverted for {shield['ddim_inversion_steps']} DDIM steps, and compared with a reference watermark **without embedding that watermark into the clip**.

| Metric/configuration | Value | Interpretation |
| --- | ---: | --- |
| Negative-control bit accuracy | **{negative_accuracy:.4f}** ({negative_accuracy * 100:.2f}%) | {chance_delta:.2f} percentage points from the 50% chance level. This is expected for an unwatermarked input and is not robustness evidence. |
| Sampled frames | {shield['sampling']['frames']} | Evenly selected control frames from the source clip. |
| Detector resize | {shield['sampling']['resize'][0]} × {shield['sampling']['resize'][1]} | Resolution used by the control detector. |
| DDIM inversion steps | {shield['ddim_inversion_steps']} | Reverse-diffusion configuration, not a quality metric. |
| Control runtime | {seconds(shield.get('elapsed_seconds'))} | Wall-clock time for this recorded control. |
| Watermarked / ProPainter / DiffuEraser / predicted mask / IoU | **Not applicable** | Missing by design because a faithful VideoShield watermark was never embedded in this MOSE clip. |

The original and raw mask below are the shared control context. They are not VideoShield outputs.

Direct link: {link(videoshield['metadata']['availability'], 'VideoShield availability/control JSON')}
"""
        )
    )
    cells.append(
        video_cell(
            [
                ("Shared original (control input)", videoshield["input_original"], "Unwatermarked MOSE source used only for the VideoShield negative control."),
                ("Shared raw mask (context)", videoshield["gt_mask"], "Selected removal region shown for context; no VideoShield attack was performed."),
                ("VideoShield watermarked/attacked outputs", None, "Authentic VideoShield embedding requires diffusion generation, so no faithful MOSE watermarked, ProPainter, or DiffuEraser videos exist."),
            ]
        )
    )
    cells.append(json_cell([videoshield["metadata"]["availability"]]))

    cells.append(
        markdown(
            f"""
## 5. Cross-method reading of this sample

| Method | What was actually measured | ProPainter result | DiffuEraser result | Correct conclusion |
| --- | --- | ---: | ---: | --- |
| VideoSeal | Derived spatial localization against remover-specific effective GT | Global IoU {vs_prop_loc['global_iou']:.4f} | Global IoU {vs_diff_loc['global_iou']:.4f} | The documented crop-grid proxy did not overlap the removed region. |
| Safe-Sora | Graphical-watermark recovery fidelity | PSNR {ss_prop_recovery['psnr_db']:.4f} dB | PSNR {ss_diff_recovery['psnr_db']:.4f} dB | Both attacks substantially degraded graphical recovery; no spatial IoU is available. |
| VideoShield | Unwatermarked negative-control bit accuracy | N/A | N/A | Accuracy {negative_accuracy:.4f} is a control near chance, not an embedding/removal result. |

### What can and cannot be compared

- You **can** compare ProPainter and DiffuEraser within VideoSeal using global IoU because they use the same derived-localization procedure and attack-specific effective GT masks.
- You **can** compare the two Safe-Sora recovery PSNR values because both reconstruct the same graphical watermark.
- You **cannot** compare Safe-Sora PSNR numerically with VideoSeal IoU; they measure different outputs.
- You **cannot** rank VideoShield against the other two on this MOSE sample. Its negative control is included to document the capability boundary, not to fill missing experiment cells with misleading values.
- None of these watermark metrics alone determines which inpainted video looks best. That remains a visual removal-quality judgment supported by the baseline and attacked videos above.
"""
        )
    )

    cells.append(markdown("## 6. Every committed video for this sample\n\nThis table is generated from the repository tree. It is the complete MP4 inventory for this sample, including shared baseline media and all compatible watermark/removal outputs.\n\n" + video_inventory(sample_root, review_root)))
    cells.append(markdown("## 7. Complete artifact inventory\n\nEvery committed file for this sample is linked below, including videos, recovery images, metrics, settings, adapters, and integrity hashes.\n\n" + artifact_inventory(sample_root, review_root)))
    cells.append(
        markdown(
            f"""
## 8. Validation and provenance

- The final bundle validator decoded all committed videos, checked frame count, dimensions and FPS, rejected empty effective GT masks, enforced explicit `not applicable` fields, and independently recomputed every displayed VideoSeal global IoU.
- Aggregate and per-file validation: [validation.json](../../evidence/mose-task1/review50/validation.json)
- Aggregate method results: [watermark-summary.json](../../evidence/mose-task1/review50/watermark-summary.json)
- Exact experimental contract: [PROTOCOL.md](../../evidence/mose-task1/PROTOCOL.md)
- Batch result summary: [README.md](../../evidence/mose-task1/README.md)

This notebook adds no new experimental claims. It reorganizes the already committed and validated evidence for sample `{sample_id}` into one readable case study.
"""
        )
    )

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-root", type=Path, default=Path("evidence/mose-task1/review50"))
    parser.add_argument("--output", type=Path, default=Path("notebooks/mose-watermark-samples"))
    parser.add_argument("--sample", action="append", dest="samples")
    args = parser.parse_args()

    review_root = args.review_root.resolve()
    output = args.output.resolve()
    manifest = read_json(review_root / "manifest.json")
    records = {item["clip_id"]: item for item in manifest["samples"]}
    samples = tuple(args.samples or DEFAULT_SAMPLES)
    missing = [sample for sample in samples if sample not in records]
    if missing:
        raise SystemExit(f"Unknown sample IDs: {', '.join(missing)}")
    output.mkdir(parents=True, exist_ok=True)

    for sample_id in samples:
        notebook = build_notebook(records[sample_id], review_root)
        destination = output / f"sample_{sample_id}.ipynb"
        destination.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(destination)


if __name__ == "__main__":
    main()
