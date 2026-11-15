#!/usr/bin/env python3
"""Authoritative 50-sample contract for the 11 runnable watermark methods."""

from __future__ import annotations

from dataclasses import dataclass


SAMPLES_PER_METHOD = 50
REMOVERS = ("propainter", "diffueraser", "effecterase", "e2fgvi")
REMOVER_LABELS = {
    "propainter": "ProPainter",
    "diffueraser": "DiffuEraser",
    "effecterase": "EffectErase",
    "e2fgvi": "E2FGVI-HQ",
}


@dataclass(frozen=True)
class MetricSpec:
    key: str
    label: str
    unit: str = "decimal"


@dataclass(frozen=True)
class Full50Method:
    slug: str
    label: str
    notebook: str
    input_policy: str
    source_kind: str
    metrics: tuple[MetricSpec, ...]


METHODS: tuple[Full50Method, ...] = (
    Full50Method(
        "editguard", "EditGuard", "editguard.ipynb", "mose_existing", "mose_framewise",
        (
            MetricSpec("mean_message_bit_accuracy", "Mean 64-bit recovery accuracy", "fraction"),
            MetricSpec("global_localization_iou", "Global localization IoU", "fraction"),
        ),
    ),
    Full50Method(
        "omniguard", "OmniGuard", "omniguard.ipynb", "mose_existing", "mose_framewise",
        (
            MetricSpec("mean_predicted_tamper_probability", "Mean tamper probability", "fraction"),
            MetricSpec("global_localization_iou", "Global localization IoU", "fraction"),
        ),
    ),
    Full50Method(
        "watermark-anything", "Watermark Anything", "watermark_anything.ipynb", "mose_embed", "mose_framewise",
        (
            MetricSpec("video_majority_bit_accuracy", "Video-majority bit accuracy", "fraction"),
            MetricSpec("frame_bit_accuracy_mean", "Mean frame bit accuracy", "fraction"),
            MetricSpec("spatial_localization_iou", "Spatial localization IoU", "fraction"),
        ),
    ),
    Full50Method(
        "vine", "VINE-R", "vine.ipynb", "mose_embed", "mose_framewise",
        (
            MetricSpec("video_majority_bit_accuracy", "Video-majority bit accuracy", "fraction"),
            MetricSpec("frame_bit_accuracy_mean", "Mean frame bit accuracy", "fraction"),
        ),
    ),
    Full50Method(
        "genptw", "GenPTW", "genptw.ipynb", "mose_embed", "mose_framewise",
        (
            MetricSpec("video_majority_bit_accuracy", "Video-majority bit accuracy", "fraction"),
            MetricSpec("frame_bit_accuracy_mean", "Mean frame bit accuracy", "fraction"),
            MetricSpec("spatial_localization_iou", "Spatial localization IoU", "fraction"),
        ),
    ),
    Full50Method(
        "videoseal", "VideoSeal", "videoseal.ipynb", "mose_existing", "mose_post_hoc",
        (MetricSpec("global_localization_iou", "Derived crop-grid localization IoU", "fraction"),),
    ),
    Full50Method(
        "safe-sora", "Safe-Sora", "safe_sora.ipynb", "mose_existing", "mose_framewise",
        (
            MetricSpec("graphical_watermark_recovery_psnr_db", "Graphical recovery PSNR", "db"),
            MetricSpec("mean_absolute_error", "Graphical recovery MAE", "decimal"),
        ),
    ),
    Full50Method(
        "videoshield", "VideoShield", "videoshield.ipynb", "native_manifest", "authors_generated",
        (
            MetricSpec("watermark_accuracy", "Watermark accuracy", "fraction"),
            MetricSpec("spatial_localization_iou", "Spatial localization IoU", "fraction"),
        ),
    ),
    Full50Method(
        "videomark", "VideoMark", "videomark.ipynb", "native_manifest", "authors_generated",
        (
            MetricSpec("frame_detection_rate", "Frame detection rate", "fraction"),
            MetricSpec("bit_accuracy_authors_protocol", "Authors-protocol bit accuracy", "fraction"),
        ),
    ),
    Full50Method(
        "sigmark", "SIGMark", "sigmark.ipynb", "native_manifest", "authors_generated",
        (MetricSpec("bit_accuracy", "Bit accuracy", "fraction"),),
    ),
    Full50Method(
        "vidsig", "VidSig / Video Signature", "vidsig.ipynb", "native_manifest", "authors_generated",
        (
            MetricSpec("video_majority_bit_accuracy", "Video-majority bit accuracy", "fraction"),
            MetricSpec("frame_bit_accuracy_mean", "Mean frame bit accuracy", "fraction"),
        ),
    ),
)

METHOD_BY_SLUG = {method.slug: method for method in METHODS}


def method(slug: str) -> Full50Method:
    try:
        return METHOD_BY_SLUG[slug]
    except KeyError as error:
        choices = ", ".join(METHOD_BY_SLUG)
        raise ValueError(f"Unknown full-50 method {slug!r}; choose one of: {choices}") from error


def validate_contract() -> None:
    if len(METHODS) != 11 or len(METHOD_BY_SLUG) != 11:
        raise RuntimeError("The full-50 contract must contain exactly 11 unique watermark methods")
    if len(REMOVERS) != 4:
        raise RuntimeError("The full-50 contract must contain exactly four removers")
    for item in METHODS:
        if item.input_policy not in {"mose_existing", "mose_embed", "native_manifest"}:
            raise RuntimeError(f"Invalid input policy for {item.slug}: {item.input_policy}")
        if not item.metrics:
            raise RuntimeError(f"No key metrics declared for {item.slug}")


validate_contract()
