#!/usr/bin/env python3
"""Insert validated full-50 aggregate and per-sample metric views into notebooks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from full50_watermark_contract import METHODS, REMOVERS, REMOVER_LABELS


TAG = "full50-key-metrics"
CANONICAL_TAG = "equal-four-inpainter-comparison"


def fmt(value: float | None, unit: str) -> str:
    if value is None:
        return "pending"
    suffix = " dB" if unit == "db" else ""
    return f"{value:.4f}{suffix}"


def markdown_cell(method, record: dict, summary_path: Path, digest: str) -> dict:
    metrics = {item["key"]: item for item in record["key_metrics"]}
    lines = [
        "## Full-50 evaluation: key metrics",
        "",
        (
            "This section is generated from the machine-readable full-50 summary. "
            "It reports the required 50 samples and all four removers; incomplete GPU runs are shown as "
            "`pending`, never as zero."
        ),
        "",
        f"Summary: `{summary_path.as_posix()}`  ",
        f"SHA-256: `{digest}`",
        "",
        "| Remover | Complete | Key metric aggregates (mean ± population SD; range) |",
        "|---|---:|---|",
    ]
    for remover in REMOVERS:
        pieces = []
        for key, spec in metrics.items():
            aggregate = record["aggregates"][remover][key]
            if aggregate["count"] == 0:
                rendered = "pending"
            else:
                rendered = (
                    f"{fmt(aggregate['mean'], spec['unit'])} ± "
                    f"{fmt(aggregate['stddev_population'], spec['unit'])}; "
                    f"[{fmt(aggregate['minimum'], spec['unit'])}, {fmt(aggregate['maximum'], spec['unit'])}]"
                )
            pieces.append(f"{spec['label']}: {rendered}")
        count = record["completed_by_remover"][remover]
        lines.append(f"| {REMOVER_LABELS[remover]} | {count}/50 | {'<br>'.join(pieces)} |")
    lines.extend(
        [
            "",
            (
                "The next cell displays the complete per-sample table directly from the same summary and "
                "checks its hash before showing values."
            ),
        ]
    )
    return {
        "cell_type": "markdown",
        "metadata": {"tags": [TAG]},
        "source": [line + "\n" for line in lines],
    }


def code_cell(method, digest: str) -> dict:
    source = f'''from pathlib import Path
import hashlib
import html
import json
from IPython.display import HTML, display

FULL50_METHOD = {method.slug!r}
FULL50_SUMMARY_SHA256 = {digest!r}
FULL50_SUMMARY_RELATIVE = Path("evidence/full50-watermark-evaluation-summary.json")

def find_full50_summary():
    starts = [Path.cwd(), Path.cwd().resolve()]
    for start in starts:
        for parent in (start, *start.parents):
            candidate = parent / FULL50_SUMMARY_RELATIVE
            if candidate.is_file():
                return candidate
    raise FileNotFoundError(f"Could not find {{FULL50_SUMMARY_RELATIVE}} from {{Path.cwd()}}")

summary_path = find_full50_summary()
actual_hash = hashlib.sha256(summary_path.read_bytes()).hexdigest()
if actual_hash != FULL50_SUMMARY_SHA256:
    raise RuntimeError(
        f"Full-50 summary changed (expected {{FULL50_SUMMARY_SHA256}}, found {{actual_hash}}). "
        "Run scripts/update_full50_watermark_notebooks.py before trusting this table."
    )
full50 = json.loads(summary_path.read_text())["methods"][FULL50_METHOD]
metric_labels = {{item["key"]: item["label"] for item in full50["key_metrics"]}}
headers = ["sample", "remover", "status", *metric_labels.values(), "evidence"]
rows = []
for sample in full50["samples"]:
    for remover, result in sample["removers"].items():
        values = [
            sample["sample_id"], remover, result["status"],
            *[result["metrics"].get(key) for key in metric_labels],
            result.get("evidence"),
        ]
        cells = "".join(
            f"<td>{{html.escape('' if value is None else f'{{value:.6g}}' if isinstance(value, float) else str(value))}}</td>"
            for value in values
        )
        rows.append(f"<tr>{{cells}}</tr>")
display(HTML(
    "<div style='max-height:560px;overflow:auto'><table>"
    + "<thead><tr>" + "".join(f"<th>{{html.escape(value)}}</th>" for value in headers) + "</tr></thead>"
    + "<tbody>" + "".join(rows) + "</tbody></table></div>"
))
'''
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"tags": [TAG]},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=Path("evidence/full50-watermark-evaluation-summary.json"))
    parser.add_argument("--notebook-dir", type=Path, default=Path("notebooks/mose-watermark-samples"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    summary_path = args.summary if args.summary.is_absolute() else root / args.summary
    notebook_dir = args.notebook_dir if args.notebook_dir.is_absolute() else root / args.notebook_dir
    summary = json.loads(summary_path.read_text())
    digest = hashlib.sha256(summary_path.read_bytes()).hexdigest()
    display_path = summary_path.relative_to(root) if summary_path.is_relative_to(root) else summary_path

    for method in METHODS:
        path = notebook_dir / method.notebook
        notebook = json.loads(path.read_text())
        cells = [cell for cell in notebook["cells"] if TAG not in cell.get("metadata", {}).get("tags", [])]
        canonical = [
            index
            for index, cell in enumerate(cells)
            if CANONICAL_TAG in cell.get("metadata", {}).get("tags", [])
        ]
        # Keep the pre-existing canonical comparison within its validated
        # near-the-beginning position, then place the expanded full-50 view
        # immediately after the entire generated comparison block.
        insert_at = max(canonical) + 1 if canonical else (1 if cells else 0)
        cells[insert_at:insert_at] = [
            markdown_cell(method, summary["methods"][method.slug], display_path, digest),
            code_cell(method, digest),
        ]
        notebook["cells"] = cells
        path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
        print(f"Updated {path}")


if __name__ == "__main__":
    main()
