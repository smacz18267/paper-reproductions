#!/usr/bin/env python3
"""Build a browsable 50-clip MOSE removal-review bundle.

The builder combines the exact MOSE JPEG sequences with the previously generated
ProPainter and DiffuEraser videos into the review bundle committed for inspection.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path


FPS = 25
EXPECTED_CLIPS = 50
MAX_SIDE = 960


def target_size(width: int, height: int) -> tuple[int, int]:
    longest = max(width, height)
    if longest > MAX_SIDE:
        ratio = longest / MAX_SIDE
        width = int(width / ratio)
        height = int(height / ratio)
    return min(MAX_SIDE, math.ceil(width / 16) * 16), min(MAX_SIDE, math.ceil(height / 16) * 16)


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def write_member(archive: tarfile.TarFile, member_name: str, destination: Path) -> None:
    try:
        member = archive.getmember(member_name)
    except KeyError as error:
        raise RuntimeError(f"Archive is missing {member_name}") from error
    if not member.isfile():
        raise RuntimeError(f"Archive member is not a file: {member_name}")
    source = archive.extractfile(member)
    if source is None:
        raise RuntimeError(f"Could not read {member_name}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    with source, temporary.open("wb") as output:
        shutil.copyfileobj(source, output)
    temporary.replace(destination)


def extract_mask_frames(
    archive: tarfile.TarFile,
    clip_id: str,
    expected: int,
    destination: Path,
) -> list[Path]:
    prefix = f"binary_masks/{clip_id}/"
    members = sorted(
        (
            member for member in archive.getmembers()
            if member.isfile() and member.name.startswith(prefix) and member.name.endswith(".png")
        ),
        key=lambda member: member.name,
    )
    if len(members) != expected:
        raise RuntimeError(f"Expected {expected} masks for {clip_id}, found {len(members)}")
    paths: list[Path] = []
    for member in members:
        path = destination / Path(member.name).name
        source = archive.extractfile(member)
        if source is None:
            raise RuntimeError(f"Could not read {member.name}")
        with source, path.open("wb") as output:
            shutil.copyfileobj(source, output)
        paths.append(path)
    return paths


def validate_frames(directory: Path, suffix: str, expected: int, clip_id: str) -> list[Path]:
    frames = sorted(directory.glob(f"*.{suffix}"))
    if len(frames) != expected:
        raise RuntimeError(
            f"Expected {expected} {suffix} frames for {clip_id} in {directory}, found {len(frames)}"
        )
    expected_stems = [f"{index:05d}" for index in range(expected)]
    if [frame.stem for frame in frames] != expected_stems:
        raise RuntimeError(f"Frames for {clip_id} are not a complete zero-based sequence")
    return frames


def encode_sequence(
    ffmpeg: Path,
    source: Path,
    extension: str,
    destination: Path,
    width: int,
    height: int,
    *,
    mask: bool,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".part.mp4")
    filter_text = f"scale={width}:{height}:flags={'neighbor' if mask else 'lanczos'}"
    command = [
        str(ffmpeg), "-hide_banner", "-loglevel", "error", "-y",
        "-framerate", str(FPS), "-start_number", "0",
        "-i", str(source / f"%05d.{extension}"),
        "-vf", filter_text,
        "-an", "-c:v", "libx264", "-preset", "medium",
        "-crf", "0" if mask else "18", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(temporary),
    ]
    run(command)
    temporary.replace(destination)


def transcode_video(ffmpeg: Path, source: Path, destination: Path) -> None:
    """Convert an existing result to a browser-safe H.264 MP4."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".part.mp4")
    command = [
        str(ffmpeg), "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(source), "-an", "-c:v", "libx264", "-preset", "medium",
        "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(temporary),
    ]
    run(command)
    temporary.replace(destination)


def html_document() -> str:
    return r'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MOSE 50-sample removal review</title>
  <style>
    :root { color-scheme: light dark; font-family: ui-sans-serif, system-ui, sans-serif; }
    body { margin: 0; background: Canvas; color: CanvasText; }
    main { max-width: 1500px; margin: auto; padding: 20px; }
    header, .controls, .review, .metric-row { display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }
    header { justify-content: space-between; }
    button, select, textarea { font: inherit; }
    button, select { min-height: 40px; padding: 7px 12px; }
    .counter { font-variant-numeric: tabular-nums; }
    .videos { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; margin: 16px 0; }
    .method-videos { grid-template-columns: repeat(3, minmax(0, 1fr)); }
    figure { margin: 0; padding: 12px; border: 1px solid color-mix(in srgb, CanvasText 20%, transparent); border-radius: 10px; }
    figcaption { margin-bottom: 8px; font-weight: 600; }
    video { width: 100%; background: #000; aspect-ratio: 16 / 9; object-fit: contain; }
    img.recovery { width: 100%; background: #000; aspect-ratio: 3 / 1; object-fit: contain; }
    .review { align-items: flex-start; }
    .review label { padding: 10px 0; }
    textarea { width: min(100%, 760px); min-height: 72px; padding: 8px; }
    .watermark { margin-top: 24px; padding-top: 18px; border-top: 1px solid color-mix(in srgb, CanvasText 20%, transparent); }
    .method-note { max-width: 1100px; line-height: 1.45; }
    .hidden { display: none; }
    .status { min-height: 1.4em; }
    @media (max-width: 1000px) { .method-videos { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
    @media (max-width: 800px) { .videos, .method-videos { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
<main>
  <header>
    <div>
      <h1>MOSE removal review</h1>
      <div class="counter" id="counter" aria-live="polite"></div>
    </div>
    <div class="controls">
      <button id="previous" type="button">Previous</button>
      <select id="sample" aria-label="Sample"></select>
      <button id="next" type="button">Next</button>
      <button id="play" type="button">Play all</button>
      <button id="export" type="button">Export review CSV</button>
    </div>
  </header>

  <section class="videos" aria-label="Removal comparison">
    <figure><figcaption>Original MOSE video</figcaption><video id="original" controls muted playsinline></video></figure>
    <figure><figcaption>GT object mask (white = removed)</figcaption><video id="gt-mask" controls muted playsinline></video></figure>
    <figure><figcaption>ProPainter</figcaption><video id="propainter" controls muted playsinline></video></figure>
    <figure><figcaption>DiffuEraser</figcaption><video id="diffueraser" controls muted playsinline></video></figure>
  </section>

  <section class="review" aria-label="Manual review">
    <label><input id="propainter-pass" type="checkbox"> ProPainter removed the object satisfactorily</label>
    <label><input id="diffueraser-pass" type="checkbox"> DiffuEraser removed the object satisfactorily</label>
    <label for="notes">Notes</label>
    <textarea id="notes"></textarea>
  </section>

  <section class="watermark hidden" id="watermark-section" aria-label="Watermark localization">
    <h2>Watermark-before-removal experiment</h2>
    <div class="controls">
      <label for="watermark-method">Watermark method</label>
      <select id="watermark-method"></select>
    </div>
    <p class="method-note" id="method-note"></p>
    <div class="metric-row" id="iou"></div>
    <div class="videos method-videos">
      <figure><figcaption>Method input — original</figcaption><video id="method-original" controls muted playsinline></video></figure>
      <figure><figcaption>Watermarked video</figcaption><video id="watermarked" controls muted playsinline></video></figure>
      <figure><figcaption>Selected/aligned GT mask</figcaption><video id="method-gt-mask" controls muted playsinline></video></figure>
      <figure><figcaption>ProPainter after watermarking</figcaption><video id="method-propainter" controls muted playsinline></video></figure>
      <figure><figcaption>ProPainter effective GT mask</figcaption><video id="gt-propainter" controls muted playsinline></video></figure>
      <figure><figcaption>ProPainter predicted tamper mask</figcaption><video id="pred-propainter" controls muted playsinline></video></figure>
      <figure><figcaption>DiffuEraser after watermarking</figcaption><video id="method-diffueraser" controls muted playsinline></video></figure>
      <figure><figcaption>DiffuEraser effective GT mask</figcaption><video id="gt-diffueraser" controls muted playsinline></video></figure>
      <figure><figcaption>DiffuEraser predicted tamper mask</figcaption><video id="pred-diffueraser" controls muted playsinline></video></figure>
      <figure><figcaption>Safe-Sora recovery — ProPainter</figcaption><img class="recovery" id="recovery-propainter" alt="Safe-Sora original, recovered watermark, and amplified error after ProPainter"></figure>
      <figure><figcaption>Safe-Sora recovery — DiffuEraser</figcaption><img class="recovery" id="recovery-diffueraser" alt="Safe-Sora original, recovered watermark, and amplified error after DiffuEraser"></figure>
    </div>
  </section>
  <p class="status" id="status" aria-live="polite"></p>
</main>
<script>
const state = { manifest: null, index: 0, reviews: {}, method: null };
const ids = ['original','gt-mask','propainter','diffueraser','method-original','watermarked','method-gt-mask','method-propainter','gt-propainter','pred-propainter','method-diffueraser','gt-diffueraser','pred-diffueraser'];
const videos = () => ids.map(id => document.getElementById(id)).filter(v => v.getAttribute('src') && !v.closest('figure').classList.contains('hidden'));
const reviewKey = 'mose-review50-v1';

function saveReview() {
  const clip = state.manifest.samples[state.index].clip_id;
  state.reviews[clip] = {
    propainter_removed: document.getElementById('propainter-pass').checked,
    diffueraser_removed: document.getElementById('diffueraser-pass').checked,
    notes: document.getElementById('notes').value
  };
  localStorage.setItem(reviewKey, JSON.stringify(state.reviews));
}

function mediaPath(path) { return path ? new URL(path, document.baseURI).href : ''; }
function setVideo(id, path) {
  const element = document.getElementById(id);
  element.pause();
  const card = element.closest('figure');
  card.classList.toggle('hidden', !path);
  if (path) element.src = mediaPath(path);
  else element.removeAttribute('src');
  element.load();
}

function setImage(id, path) {
  const element = document.getElementById(id);
  const card = element.closest('figure');
  card.classList.toggle('hidden', !path);
  if (path) element.src = mediaPath(path);
  else element.removeAttribute('src');
}

function render() {
  const sample = state.manifest.samples[state.index];
  document.getElementById('sample').value = String(state.index);
  document.getElementById('counter').textContent = `Sample ${state.index + 1} of ${state.manifest.samples.length} · ${sample.clip_id} · ${sample.frame_count} frames`;
  setVideo('original', sample.media.original);
  setVideo('gt-mask', sample.media.gt_mask);
  setVideo('propainter', sample.media.propainter);
  setVideo('diffueraser', sample.media.diffueraser);
  const review = state.reviews[sample.clip_id] || {};
  document.getElementById('propainter-pass').checked = Boolean(review.propainter_removed);
  document.getElementById('diffueraser-pass').checked = Boolean(review.diffueraser_removed);
  document.getElementById('notes').value = review.notes || '';
  renderWatermark(sample);
}

function renderWatermark(sample) {
  const section = document.getElementById('watermark-section');
  const methods = sample.watermarks || {};
  const names = Object.keys(methods);
  section.classList.toggle('hidden', names.length === 0);
  if (!names.length) return;
  if (!state.method || !methods[state.method]) state.method = names[0];
  const picker = document.getElementById('watermark-method');
  picker.innerHTML = names.map(name => `<option value="${name}">${name}</option>`).join('');
  picker.value = state.method;
  const result = methods[state.method];
  setVideo('method-original', result.input_original);
  setVideo('watermarked', result.watermarked);
  setVideo('method-gt-mask', result.gt_mask);
  setVideo('method-propainter', result.propainter);
  setVideo('gt-propainter', result.gt_mask_propainter);
  setVideo('pred-propainter', result.pred_mask_propainter);
  setVideo('method-diffueraser', result.diffueraser);
  setVideo('gt-diffueraser', result.gt_mask_diffueraser);
  setVideo('pred-diffueraser', result.pred_mask_diffueraser);
  setImage('recovery-propainter', result.recovery_propainter);
  setImage('recovery-diffueraser', result.recovery_diffueraser);
  const fmt = value => value == null ? 'not available' : Number(value).toFixed(4);
  const control = result.negative_control_accuracy == null ? '' : ` · Unwatermarked negative-control accuracy: ${fmt(result.negative_control_accuracy)}`;
  document.getElementById('iou').textContent = `IoU — ProPainter: ${fmt(result.iou_propainter)} · DiffuEraser: ${fmt(result.iou_diffueraser)}${control}`;
  document.getElementById('method-note').textContent = result.note || '';
}

function move(delta) {
  saveReview();
  state.index = (state.index + delta + state.manifest.samples.length) % state.manifest.samples.length;
  render();
}

function exportCsv() {
  saveReview();
  const quote = value => `"${String(value ?? '').replaceAll('"', '""')}"`;
  const rows = [['clip_id','propainter_removed','diffueraser_removed','notes']];
  for (const sample of state.manifest.samples) {
    const review = state.reviews[sample.clip_id] || {};
    rows.push([sample.clip_id, Boolean(review.propainter_removed), Boolean(review.diffueraser_removed), review.notes || '']);
  }
  const blob = new Blob([rows.map(row => row.map(quote).join(',')).join('\n') + '\n'], {type:'text/csv'});
  const link = document.createElement('a');
  link.href = URL.createObjectURL(blob); link.download = 'mose-review50.csv'; link.click();
  URL.revokeObjectURL(link.href);
}

async function init() {
  try {
    const response = await fetch('manifest.json');
    if (!response.ok) throw new Error(`manifest.json: ${response.status}`);
    state.manifest = await response.json();
    state.reviews = JSON.parse(localStorage.getItem(reviewKey) || '{}');
    const picker = document.getElementById('sample');
    state.manifest.samples.forEach((sample, index) => picker.add(new Option(`${String(index + 1).padStart(2,'0')} — ${sample.clip_id}`, index)));
    picker.addEventListener('change', event => { saveReview(); state.index = Number(event.target.value); render(); });
    document.getElementById('previous').addEventListener('click', () => move(-1));
    document.getElementById('next').addEventListener('click', () => move(1));
    document.getElementById('export').addEventListener('click', exportCsv);
    document.getElementById('watermark-method').addEventListener('change', event => { state.method = event.target.value; renderWatermark(state.manifest.samples[state.index]); });
    document.getElementById('play').addEventListener('click', async event => {
      const active = videos(); const playing = active.some(video => !video.paused);
      if (playing) { active.forEach(video => video.pause()); event.target.textContent = 'Play all'; }
      else { active.forEach(video => { video.currentTime = 0; }); await Promise.allSettled(active.map(video => video.play())); event.target.textContent = 'Pause all'; }
    });
    document.getElementById('original').addEventListener('timeupdate', event => {
      if (event.target.paused) return;
      videos().slice(1).forEach(video => { if (Math.abs(video.currentTime - event.target.currentTime) > 0.08) video.currentTime = event.target.currentTime; });
    });
    ['propainter-pass','diffueraser-pass','notes'].forEach(id => document.getElementById(id).addEventListener('change', saveReview));
    render();
  } catch (error) {
    document.getElementById('status').textContent = `Could not load the review bundle: ${error.message}. Start a local web server in this folder; do not open index.html directly.`;
  }
}
init();
</script>
</body>
</html>
'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--mask-metadata", required=True, type=Path)
    parser.add_argument("--source-frames", required=True, type=Path)
    parser.add_argument("--results-archive", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--ffmpeg", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    for required in (args.selection, args.mask_metadata, args.results_archive):
        if not required.exists():
            raise RuntimeError(f"Required path does not exist: {required}")
    if args.ffmpeg is None:
        executable = shutil.which("ffmpeg")
        if executable is None:
            try:
                import imageio_ffmpeg
            except ImportError as error:
                raise RuntimeError("Pass --ffmpeg or install imageio-ffmpeg") from error
            executable = imageio_ffmpeg.get_ffmpeg_exe()
        args.ffmpeg = Path(executable)
    if not args.ffmpeg.is_file():
        raise RuntimeError(f"FFmpeg executable does not exist: {args.ffmpeg}")
    selection = json.loads(args.selection.read_text())
    clips = selection.get("clips", [])
    if len(clips) != EXPECTED_CLIPS:
        raise RuntimeError(f"Expected {EXPECTED_CLIPS} selected clips, found {len(clips)}")
    mask_metadata = json.loads(args.mask_metadata.read_text())
    metadata = {item["video_id"]: item for item in mask_metadata.get("clips", [])}
    if set(metadata) != {item["video_id"] for item in clips}:
        raise RuntimeError("Selection and mask metadata contain different clip IDs")

    output = args.output.resolve()
    media_root = output / "media"
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "dataset": "MOSEv1 train",
        "fps": FPS,
        "sample_count": EXPECTED_CLIPS,
        "samples": [],
    }

    with tarfile.open(args.results_archive, "r") as archive, tempfile.TemporaryDirectory(prefix="mose-review-") as temporary_text:
        temporary = Path(temporary_text)
        for index, clip in enumerate(clips, 1):
            clip_id = clip["video_id"]
            expected = int(clip["frame_count"])
            frames = args.source_frames / clip_id
            width, height = target_size(int(metadata[clip_id]["width"]), int(metadata[clip_id]["height"]))
            clip_media = media_root / clip_id
            clip_media.mkdir(parents=True, exist_ok=True)
            paths = {
                "original": clip_media / "original.mp4",
                "gt_mask": clip_media / "gt_mask.mp4",
                "propainter": clip_media / "propainter.mp4",
                "diffueraser": clip_media / "diffueraser.mp4",
            }

            print(f"[{index}/{EXPECTED_CLIPS}] {clip_id}", flush=True)
            if args.force or not paths["propainter"].is_file():
                write_member(archive, f"results50/propainter/{clip_id}/inpaint_out.mp4", paths["propainter"])
            if args.force or not paths["diffueraser"].is_file():
                diffueraser_source = temporary / f"{clip_id}-diffueraser.mp4"
                write_member(
                    archive,
                    f"results50/diffueraser/{clip_id}/diffueraser_result.mp4",
                    diffueraser_source,
                )
                transcode_video(args.ffmpeg, diffueraser_source, paths["diffueraser"])
            if args.force or not paths["original"].is_file():
                validate_frames(frames, "jpg", expected, clip_id)
                encode_sequence(args.ffmpeg, frames, "jpg", paths["original"], width, height, mask=False)
            if args.force or not paths["gt_mask"].is_file():
                mask_dir = temporary / clip_id
                mask_dir.mkdir()
                extract_mask_frames(archive, clip_id, expected, mask_dir)
                validate_frames(mask_dir, "png", expected, clip_id)
                encode_sequence(args.ffmpeg, mask_dir, "png", paths["gt_mask"], width, height, mask=True)

            manifest["samples"].append({
                "clip_id": clip_id,
                "frame_count": expected,
                "width": width,
                "height": height,
                "media": {name: str(path.relative_to(output)) for name, path in paths.items()},
                "watermarks": {},
            })

    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "index.html").write_text(html_document())
    with (output / "review.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("clip_id", "propainter_removed", "diffueraser_removed", "notes"))
        writer.writerows((sample["clip_id"], "", "", "") for sample in manifest["samples"])
    print(f"Review bundle ready: {output / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
