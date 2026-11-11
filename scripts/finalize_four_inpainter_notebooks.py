#!/usr/bin/env python3
"""Attach recorded four-inpainter evidence and build the VideoShield notebook."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks/mose-watermark-samples"
TAG = "four-inpainter-recorded-status"

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


def completion_markdown(method: str, attack: Path) -> str:
    rows = []
    for label, stem in (
        ("ProPainter", "propainter"),
        ("DiffuEraser", "diffueraser"),
        ("EffectErase", "effecterase"),
        ("E2FGVI-HQ", "e2fgvi"),
    ):
        video = attack / f"{stem}.mp4"
        metadata = attack / f"{stem}.json"
        status = "complete" if video.is_file() and metadata.is_file() else "missing"
        evidence = f"[video]({relative(video)}) | [JSON]({relative(metadata)})" if status == "complete" else "—"
        rows.append(f"| {label} | {status} | {evidence} |")
    return f"""
## Recorded one-clip four-inpainter completion

All four removers use the same frozen watermarked input and aligned binary mask for this sample. Method-specific resizing, padding, and mask dilation are recorded in the adjacent JSON rather than silently normalized.

| Inpainting method | Status | Evidence |
|---|---:|---|
{chr(10).join(rows)}

Watermark method: **{method}**.
"""


def display_cell(attack: Path) -> str:
    rel = attack.relative_to(ROOT)
    return f'''from pathlib import Path
import json
from IPython.display import JSON, Video, display

def find_repo_root():
    for candidate in (Path.cwd().resolve(), *Path.cwd().resolve().parents):
        if (candidate / "scripts/finalize_four_inpainter_notebooks.py").is_file():
            return candidate
    raise FileNotFoundError("Start Jupyter inside the paper-reproductions-task1 checkout")

ROOT = find_repo_root()
ATTACK = ROOT / {str(rel)!r}
for name in ("input-watermarked.mp4", "watermarked.mp4", "gt-mask.mp4", "propainter.mp4", "diffueraser.mp4", "effecterase.mp4", "e2fgvi.mp4"):
    path = ATTACK / name
    if path.is_file():
        print(name)
        display(Video(str(path), embed=True, html_attributes="controls muted loop"))
for name in ("sample.json", "removal-pair.json", "effecterase.json", "e2fgvi.json"):
    path = ATTACK / name
    if path.is_file():
        print(name)
        display(JSON(json.loads(path.read_text())))'''


def append_record(path: Path, method: str, attack: Path) -> None:
    document = json.loads(path.read_text())
    document["cells"] = [
        cell for cell in document["cells"] if TAG not in cell.get("metadata", {}).get("tags", [])
    ]
    document["cells"].extend([markdown(completion_markdown(method, attack)), code(display_cell(attack))])
    path.write_text(json.dumps(document, indent=1, ensure_ascii=False) + "\n")


def append_video_shield_record(path: Path, attack: Path) -> None:
    document = json.loads(path.read_text())
    document["cells"] = [
        cell for cell in document["cells"] if TAG not in cell.get("metadata", {}).get("tags", [])
    ]
    document["cells"].extend(
        [
            markdown(
                """
## Native VideoShield sample: four inpainting methods

VideoShield is generation-coupled, so MOSE is not used. This notebook uses the authors' recorded 16-frame, 256×256 ModelScope red-panda output, its saved watermark information, and its spatial tamper mask. The same authentic watermarked clip and mask are passed to ProPainter, DiffuEraser, EffectErase, and E2FGVI-HQ.

Official repository: [Shield-LM/VideoShield](https://github.com/Shield-LM/VideoShield)

Inspected commit: `a61efa73abb15d30f50ee535c3301cea7dec3075`
"""
            ),
            markdown(completion_markdown("VideoShield", attack)),
            code(display_cell(attack)),
        ]
    )
    path.write_text(json.dumps(document, indent=1, ensure_ascii=False) + "\n")


def main() -> None:
    for notebook_name, evidence in CORE.items():
        append_record(NOTEBOOKS / f"{notebook_name}.ipynb", notebook_name.replace("_", " ").title(), evidence)
    for notebook_name, attack in REMAINING.items():
        append_record(NOTEBOOKS / f"{notebook_name}.ipynb", notebook_name.replace("_", " ").title(), attack)

    shield_attack = ROOT / "runs/remaining-watermarks/videoshield/modelscope-red-panda/attack"
    append_video_shield_record(NOTEBOOKS / "videoshield.ipynb", shield_attack)
    print(f"Finalized 11 implemented notebooks in {NOTEBOOKS}")


if __name__ == "__main__":
    main()
