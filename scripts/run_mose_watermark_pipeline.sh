#!/usr/bin/env bash
set -euo pipefail

GPU="${GPU:-1}"
SOURCE_ROOT="${WATERMARK_SOURCE_ROOT:-${ROOT:-}}"
: "${SOURCE_ROOT:?Set WATERMARK_SOURCE_ROOT to the directory containing the method checkouts}"
DATASET="${MOSE_DATASET_ROOT:?Set MOSE_DATASET_ROOT to the extracted MOSEv1 directory}"
RUNNER="${WATERMARK_SCRIPT_ROOT:-$(cd "$(dirname "$0")" && pwd)}"
OUTPUT="${MOSE_OUTPUT_ROOT:-$DATASET/watermark-review50}"
SELECTION="${MOSE_SELECTION:-$DATASET/selection50.json}"

run() {
  printf '[%s] RUN %q\n' "$(date -u +%FT%TZ)" "$1"
  CUDA_VISIBLE_DEVICES="$GPU" "$@"
}

run_inpainting() {
  local method="$1"
  local remover="$2"
  if ! run "$SOURCE_ROOT/${3}/.venv/bin/python" -u "$RUNNER/run_mose_watermarked_inpainting.py" \
    --selection "$SELECTION" --dataset-root "$DATASET" --output-root "$OUTPUT" \
    --watermark-method "$method" --remover "$remover" \
    --propainter "$SOURCE_ROOT/ProPainter" --diffueraser "$SOURCE_ROOT/DiffuEraser" \
    --propainter-subvideo-length 40; then
    printf '[%s] RETRY %s/%s with 20-frame memory window\n' "$(date -u +%FT%TZ)" "$method" "$remover"
    if ! run "$SOURCE_ROOT/${3}/.venv/bin/python" -u "$RUNNER/run_mose_watermarked_inpainting.py" \
      --selection "$SELECTION" --dataset-root "$DATASET" --output-root "$OUTPUT" \
      --watermark-method "$method" --remover "$remover" \
      --propainter "$SOURCE_ROOT/ProPainter" --diffueraser "$SOURCE_ROOT/DiffuEraser" \
      --propainter-subvideo-length 20; then
      printf '[%s] RETRY %s/%s with 10-frame memory window\n' "$(date -u +%FT%TZ)" "$method" "$remover"
      run "$SOURCE_ROOT/${3}/.venv/bin/python" -u "$RUNNER/run_mose_watermarked_inpainting.py" \
        --selection "$SELECTION" --dataset-root "$DATASET" --output-root "$OUTPUT" \
        --watermark-method "$method" --remover "$remover" \
        --propainter "$SOURCE_ROOT/ProPainter" --diffueraser "$SOURCE_ROOT/DiffuEraser" \
        --propainter-subvideo-length 10 \
        --processing-max-side 800
    fi
  fi
}

run "$SOURCE_ROOT/VideoSeal/.venv/bin/python" -u "$RUNNER/run_mose_videoseal_embed.py" \
  --source "$SOURCE_ROOT/VideoSeal" --selection "$SELECTION" --dataset-root "$DATASET" \
  --output-root "$OUTPUT" --chunk-size 1

run "$SOURCE_ROOT/Safe-Sora/.venv/bin/python" -u "$RUNNER/run_mose_safe_sora_embed.py" \
  --source "$SOURCE_ROOT/Safe-Sora" --selection "$SELECTION" --dataset-root "$DATASET" \
  --output-root "$OUTPUT" --watermark "$SOURCE_ROOT/Safe-Sora/demo_watermark.png"

run "$SOURCE_ROOT/VideoShield/.venv/bin/python" -u "$RUNNER/run_mose_videoshield_control.py" \
  --source "$SOURCE_ROOT/VideoShield" --model-path "$SOURCE_ROOT/VideoShield/models/modelscope" \
  --selection "$SELECTION" --dataset-root "$DATASET" --output-root "$OUTPUT"

for method in videoseal safe-sora; do
  run_inpainting "$method" propainter ProPainter
  run_inpainting "$method" diffueraser DiffuEraser
done

run "$SOURCE_ROOT/Safe-Sora/.venv/bin/python" -u "$RUNNER/run_mose_safe_sora_recover.py" \
  --source "$SOURCE_ROOT/Safe-Sora" --selection "$SELECTION" --output-root "$OUTPUT" \
  --watermark "$SOURCE_ROOT/Safe-Sora/demo_watermark.png"

if ! run "$SOURCE_ROOT/VideoSeal/.venv/bin/python" -u "$RUNNER/run_mose_videoseal_localize.py" \
  --source "$SOURCE_ROOT/VideoSeal" --selection "$SELECTION" --dataset-root "$DATASET" \
  --output-root "$OUTPUT" --chunk-size 2 --grid-size 4; then
  run "$SOURCE_ROOT/VideoSeal/.venv/bin/python" -u "$RUNNER/run_mose_videoseal_localize.py" \
    --source "$SOURCE_ROOT/VideoSeal" --selection "$SELECTION" --dataset-root "$DATASET" \
    --output-root "$OUTPUT" --chunk-size 1 --grid-size 4
fi

# The pilot localization predated the boolean-mask encoding fix. Recompute it
# explicitly; subsequent clips were encoded with the corrected shared helper.
run "$SOURCE_ROOT/VideoSeal/.venv/bin/python" -u "$RUNNER/run_mose_videoseal_localize.py" \
  --source "$SOURCE_ROOT/VideoSeal" --selection "$SELECTION" --dataset-root "$DATASET" \
  --output-root "$OUTPUT" --chunk-size 1 --grid-size 4 --only 02b72b8d --force

printf '[%s] PIPELINE_COMPLETE\n' "$(date -u +%FT%TZ)"
