#!/usr/bin/env python3
"""Integrate the four-inpainter comparison consistently into all 13 notebooks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks/mose-watermark-samples"
TAG = "equal-four-inpainter-comparison"
LEGACY_TAG = "four-inpainter-recorded-status"

REMOVERS = (
    ("ProPainter", "propainter"),
    ("DiffuEraser", "diffueraser"),
    ("EffectErase", "effecterase"),
    ("E2FGVI-HQ", "e2fgvi"),
)

CORE = {
    "editguard": ROOT / "evidence/mose-task1/review50/media/02b72b8d/methods/editguard",
    "omniguard": ROOT / "evidence/mose-task1/review50/media/02b72b8d/methods/omniguard",
    "safe_sora": ROOT / "evidence/mose-task1/review50/media/02b72b8d/methods/safe-sora",
    "videoseal": ROOT / "evidence/mose-task1/review50/media/02b72b8d/methods/videoseal",
}

REMAINING = {
    "watermark_anything": ROOT / "runs/remaining-watermarks/watermark-anything/02b72b8d/attack",
    "vine": ROOT / "runs/remaining-watermarks/vine/02b72b8d/attack",
    "genptw": ROOT / "runs/remaining-watermarks/genptw/02b72b8d/attack",
    "videomark": ROOT / "runs/remaining-watermarks/videomark/a_red_panda_eating_leaves/attack",
    "sigmark": ROOT / "runs/remaining-watermarks/sigmark/native-vbench2/attack",
    "vidsig": ROOT / "runs/remaining-watermarks/vidsig/red-panda-seed42/attack",
}


def source(value: str) -> str:
    return value.strip() + "\n"


def markdown(value: str) -> dict[str, Any]:
    return {"cell_type": "markdown", "metadata": {"tags": [TAG]}, "source": source(value)}


def code(value: str) -> dict[str, Any]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"tags": [TAG]},
        "outputs": [],
        "source": source(value),
    }


def relative(path: Path) -> str:
    return "../../" + str(path.relative_to(ROOT))


def cell_source(cell: dict[str, Any]) -> str:
    value = cell.get("source", "")
    return "".join(value) if isinstance(value, list) else value


def output_record(metadata: dict[str, Any]) -> dict[str, Any]:
    output = metadata.get("output", {})
    return output.get("video", output)


def format_number(value: Any, *, digits: int = 3) -> str:
    if value is None:
        return "not recorded"
    return f"{float(value):.{digits}f}"


def completion_markdown(method: str, attack: Path) -> str:
    rows = []
    for label, stem in REMOVERS:
        video = attack / f"{stem}.mp4"
        metadata = attack / f"{stem}.json"
        status = "complete" if video.is_file() and metadata.is_file() else "missing"
        if status == "complete":
            record = json.loads(metadata.read_text())
            output = output_record(record)
            runtime = format_number(record.get("elapsed_seconds"))
            frames = output.get("frames", "not recorded")
            width = output.get("width")
            height = output.get("height")
            dimensions = f"{width} × {height}" if width is not None and height is not None else "not recorded"
            fps = format_number(output.get("fps"))
            size = output.get("bytes")
            size_text = f"{int(size):,}" if size is not None else "not recorded"
            evidence = f"[video]({relative(video)}) · [JSON]({relative(metadata)})"
        else:
            runtime = frames = dimensions = fps = size_text = evidence = "—"
        rows.append(
            f"| {label} | {status} | {runtime} | {frames} | {dimensions} | "
            f"{fps} | {size_text} | {evidence} |"
        )
    return f"""
## Canonical one-clip comparison: all four inpainters

This is the canonical comparison for the minimum one-clip requirement. All four removers use the same frozen watermarked input and aligned binary mask. Every remover is presented in the same order, with the same output-level fields and the same video/JSON evidence. Method-specific resizing, padding, and mask dilation remain recorded in JSON rather than being silently normalized.

| Inpainting method | Status | Runtime (s) | Frames | Dimensions | FPS | Bytes | Evidence |
|---|---:|---:|---:|---:|---:|---:|---|
{chr(10).join(rows)}

Watermark method: **{method}**. Runtime is shown only when the original run record captured it; `not recorded` is not replaced with an estimate. Watermark-specific localization or recovery metrics are downstream evaluations and are kept separate from this equal inpainting comparison.
"""


def display_cell(attack: Path) -> str:
    rel = attack.relative_to(ROOT)
    return f'''from pathlib import Path
import json
from IPython.display import JSON, Markdown, Video, display

def find_repo_root():
    for candidate in (Path.cwd().resolve(), *Path.cwd().resolve().parents):
        if (candidate / "scripts/finalize_four_inpainter_notebooks.py").is_file():
            return candidate
    raise FileNotFoundError("Start Jupyter inside the paper-reproductions-task1 checkout")

ROOT = find_repo_root()
ATTACK = ROOT / {str(rel)!r}
for name in ("input-watermarked.mp4", "watermarked.mp4", "gt-mask.mp4"):
    path = ATTACK / name
    if path.is_file():
        print(name)
        display(Video(str(path), embed=True, html_attributes="controls muted loop"))

REMOVERS = (
    ("ProPainter", "propainter"),
    ("DiffuEraser", "diffueraser"),
    ("EffectErase", "effecterase"),
    ("E2FGVI-HQ", "e2fgvi"),
)
for label, stem in REMOVERS:
    display(Markdown(f"### {{label}}"))
    video = ATTACK / f"{{stem}}.mp4"
    metadata = ATTACK / f"{{stem}}.json"
    if not video.is_file() or not metadata.is_file():
        raise FileNotFoundError(f"Missing {{label}} evidence in {{ATTACK}}")
    display(Video(str(video), embed=True, html_attributes="controls muted loop"))
    display(JSON(json.loads(metadata.read_text())))

for name in ("sample.json", "removal-pair.json"):
    metadata = ATTACK / name
    if metadata.is_file():
        print(name)
        display(JSON(json.loads(metadata.read_text())))'''


def comparison_cells(method: str, attack: Path) -> list[dict[str, Any]]:
    return [markdown(completion_markdown(method, attack)), code(display_cell(attack))]


def status_markdown(method: str, status: str, reason: str) -> str:
    rows = "\n".join(f"| {label} | {status} | {reason} |" for label, _stem in REMOVERS)
    return f"""
## Canonical one-clip comparison: all four inpainters

The watermark input cannot be prepared, so no remover is selectively presented as if it had run. All four inpainters carry the same explicit status.

| Inpainting method | Status | Reason |
|---|---:|---|
{rows}

Watermark method: **{method}**.
"""


def insertion_index(cells: list[dict[str, Any]]) -> int:
    for index, cell in enumerate(cells):
        if cell.get("cell_type") == "code" and "def find_repo_root" in cell_source(cell):
            return index + 1
    return 1


def remove_generated_cells(document: dict[str, Any]) -> None:
    document["cells"] = [
        cell
        for cell in document["cells"]
        if not ({TAG, LEGACY_TAG} & set(cell.get("metadata", {}).get("tags", [])))
    ]


def integrate_record(path: Path, method: str, attack: Path, *, preface: str | None = None) -> None:
    document = json.loads(path.read_text())
    remove_generated_cells(document)
    generated = comparison_cells(method, attack)
    if preface:
        generated.insert(0, markdown(preface))
    index = insertion_index(document["cells"])
    document["cells"][index:index] = generated
    path.write_text(json.dumps(document, indent=1, ensure_ascii=False) + "\n")


def integrate_status(path: Path, method: str, status: str, reason: str) -> None:
    document = json.loads(path.read_text())
    remove_generated_cells(document)
    index = insertion_index(document["cells"])
    document["cells"][index:index] = [markdown(status_markdown(method, status, reason))]
    path.write_text(json.dumps(document, indent=1, ensure_ascii=False) + "\n")


def add_e2fgvi_to_legacy_tables(path: Path, attack: Path) -> None:
    """Keep old multi-sample tables honest while making the fourth column visible."""
    document = json.loads(path.read_text())
    metadata = json.loads((attack / "e2fgvi.json").read_text())
    output = output_record(metadata)
    first_values = {
        "removal runtime": f"{float(metadata['elapsed_seconds']):.3f} s",
        "frames": str(output["frames"]),
        "reported output frames": str(output["frames"]),
        "output dimensions": f"{output['width']} × {output['height']}",
    }
    table_cells = [
        cell
        for cell in document["cells"]
        if cell.get("cell_type") == "markdown"
        and any(
            line.strip().startswith("| Metric | ProPainter | DiffuEraser | EffectErase |")
            for line in cell_source(cell).splitlines()
        )
    ]
    # EditGuard and OmniGuard have an aggregate 50-sample table before the three
    # per-sample tables. The other core notebooks start directly with sample 1.
    sample_occurrence = 2 if len(table_cells) == 4 else 1
    occurrence = 0
    changed = False
    for cell in table_cells:
        text = cell_source(cell)
        occurrence += 1
        replacement = []
        if occurrence < sample_occurrence:
            note = (
                "> **Four-inpainter scope:** this is a historical 50-sample aggregate for the first "
                "three removers. E2FGVI-HQ was run for the required one-clip matrix, so an aggregate "
                "value is not comparable and is not fabricated."
            )
        elif occurrence == sample_occurrence:
            note = (
                "> **Four-inpainter scope:** E2FGVI-HQ is shown in this table. Its common inpainting "
                "metrics come from the recorded one-clip run; downstream watermark "
                "localization/recovery was not rerun and is marked `not evaluated`."
            )
        else:
            note = (
                "> **Four-inpainter scope:** only the first fixed sample was required and run for "
                "E2FGVI-HQ; this additional sample is marked `not run`."
            )
        inserted_note = False
        skip_note_blank = False
        in_target_table = False
        text = text.replace(
            "The localization record contains results for all three attacks.",
            "The historical localization record contains results for ProPainter, DiffuEraser, "
            "and EffectErase. E2FGVI-HQ downstream localization was not run.",
        ).replace(
            "The recovery record contains all three attacks.",
            "The historical recovery record contains results for ProPainter, DiffuEraser, and "
            "EffectErase. E2FGVI-HQ downstream recovery was not run.",
        )
        for line in text.splitlines():
            if line.startswith("> **Four-inpainter scope:**"):
                skip_note_blank = True
                continue
            if skip_note_blank and not line.strip():
                skip_note_blank = False
                continue
            skip_note_blank = False
            if line.strip().startswith("| Metric | ProPainter | DiffuEraser | EffectErase |"):
                if not inserted_note:
                    replacement.extend([note, ""])
                    inserted_note = True
                replacement.append("| Metric | ProPainter | DiffuEraser | EffectErase | E2FGVI-HQ |")
                in_target_table = True
            elif in_target_table and line.strip().startswith("| --- | ---:"):
                parts = [part.strip() for part in line.strip().strip("|").split("|")][:4]
                replacement.append("| " + " | ".join(parts + ["---:"]) + " |")
            elif in_target_table and line.strip().startswith("|"):
                parts = [part.strip() for part in line.strip().strip("|").split("|")]
                if len(parts) >= 4:
                    parts = parts[:4]
                    metric = parts[0].lower()
                    if occurrence < sample_occurrence:
                        value = "not comparable"
                    elif occurrence == sample_occurrence:
                        value = first_values.get(metric, "not evaluated")
                    else:
                        value = "not run"
                    replacement.append("| " + " | ".join(parts + [value]) + " |")
                else:
                    replacement.append(line)
            else:
                if in_target_table and line.strip():
                    in_target_table = False
                replacement.append(line)
        cell["source"] = "\n".join(replacement).rstrip() + "\n"
        changed = True
    if changed:
        path.write_text(json.dumps(document, indent=1, ensure_ascii=False) + "\n")


def main() -> None:
    for notebook_name, evidence in CORE.items():
        path = NOTEBOOKS / f"{notebook_name}.ipynb"
        integrate_record(path, notebook_name.replace("_", " ").title(), evidence)
        add_e2fgvi_to_legacy_tables(path, evidence)
    for notebook_name, attack in REMAINING.items():
        integrate_record(NOTEBOOKS / f"{notebook_name}.ipynb", notebook_name.replace("_", " ").title(), attack)

    shield_attack = ROOT / "runs/remaining-watermarks/videoshield/modelscope-red-panda/attack"
    integrate_record(
        NOTEBOOKS / "videoshield.ipynb",
        "VideoShield",
        shield_attack,
        preface="""
## Native VideoShield input

VideoShield is generation-coupled, so MOSE is not used for the canonical comparison. The equal four-inpainter section uses the authors' recorded 16-frame, 256×256 ModelScope red-panda output, its saved watermark information, and the same spatial tamper mask for every remover.
""",
    )
    integrate_status(
        NOTEBOOKS / "rivagan.ipynb",
        "RivaGAN",
        "deferred",
        "deferred by user; no official pretrained model was released",
    )
    integrate_status(
        NOTEBOOKS / "lvmark.ipynb",
        "LVMark",
        "blocked",
        "official weight server unavailable and required release prompt inputs are missing",
    )
    print(f"Finalized equal four-inpainter presentation in 13 notebooks under {NOTEBOOKS}")


if __name__ == "__main__":
    main()
