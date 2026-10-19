#!/usr/bin/env bash
set -euo pipefail

GPU="${GPU:-1}"
ROOT="${ROOT:-/home/smayan/research}"
DATASET="$ROOT/datasets/MOSEv1"
RUNNER="$ROOT/mose50-corrected/scripts"
OUTPUT="$DATASET/watermark-review50"
SELECTION="$DATASET/selection50.json"

run() {
  printf '[%s] RUN %q\n' "$(date -u +%FT%TZ)" "$1"
  CUDA_VISIBLE_DEVICES="$GPU" "$@"
}

run_inpainting() {
  local method="$1"
  local remover="$2"
  if ! run "$ROOT/${3}/.venv/bin/python" -u "$RUNNER/run_mose_watermarked_inpainting.py" \
    --selection "$SELECTION" --dataset-root "$DATASET" --output-root "$OUTPUT" \
    --watermark-method "$method" --remover "$remover" --propainter-subvideo-length 40; then
    printf '[%s] RETRY %s/%s with 20-frame memory window\n' "$(date -u +%FT%TZ)" "$method" "$remover"
    if ! run "$ROOT/${3}/.venv/bin/python" -u "$RUNNER/run_mose_watermarked_inpainting.py" \
      --selection "$SELECTION" --dataset-root "$DATASET" --output-root "$OUTPUT" \
      --watermark-method "$method" --remover "$remover" --propainter-subvideo-length 20; then
      printf '[%s] RETRY %s/%s with 10-frame memory window\n' "$(date -u +%FT%TZ)" "$method" "$remover"
      run "$ROOT/${3}/.venv/bin/python" -u "$RUNNER/run_mose_watermarked_inpainting.py" \
        --selection "$SELECTION" --dataset-root "$DATASET" --output-root "$OUTPUT" \
        --watermark-method "$method" --remover "$remover" --propainter-subvideo-length 10 \
        --processing-max-side 800
    fi
  fi
}

run "$ROOT/VideoSeal/.venv/bin/python" -u "$RUNNER/run_mose_videoseal_embed.py" \
  --source "$ROOT/VideoSeal" --selection "$SELECTION" --dataset-root "$DATASET" \
  --output-root "$OUTPUT" --chunk-size 1

run "$ROOT/Safe-Sora/.venv/bin/python" -u "$RUNNER/run_mose_safe_sora_embed.py" \
  --source "$ROOT/Safe-Sora" --selection "$SELECTION" --dataset-root "$DATASET" \
  --output-root "$OUTPUT" --watermark "$ROOT/Safe-Sora/demo_watermark.png"

run "$ROOT/VideoShield/.venv/bin/python" -u "$RUNNER/run_mose_videoshield_control.py" \
  --source "$ROOT/VideoShield" --model-path "$ROOT/VideoShield/models/modelscope" \
  --selection "$SELECTION" --dataset-root "$DATASET" --output-root "$OUTPUT"

for method in videoseal safe-sora; do
  run_inpainting "$method" propainter ProPainter
  run_inpainting "$method" diffueraser DiffuEraser
done

run "$ROOT/Safe-Sora/.venv/bin/python" -u "$RUNNER/run_mose_safe_sora_recover.py" \
  --source "$ROOT/Safe-Sora" --selection "$SELECTION" --output-root "$OUTPUT" \
  --watermark "$ROOT/Safe-Sora/demo_watermark.png"

if ! run "$ROOT/VideoSeal/.venv/bin/python" -u "$RUNNER/run_mose_videoseal_localize.py" \
  --source "$ROOT/VideoSeal" --selection "$SELECTION" --dataset-root "$DATASET" \
  --output-root "$OUTPUT" --chunk-size 2 --grid-size 4; then
  run "$ROOT/VideoSeal/.venv/bin/python" -u "$RUNNER/run_mose_videoseal_localize.py" \
    --source "$ROOT/VideoSeal" --selection "$SELECTION" --dataset-root "$DATASET" \
    --output-root "$OUTPUT" --chunk-size 1 --grid-size 4
fi

# The pilot localization predated the boolean-mask encoding fix. Recompute it
# explicitly; subsequent clips were encoded with the corrected shared helper.
run "$ROOT/VideoSeal/.venv/bin/python" -u "$RUNNER/run_mose_videoseal_localize.py" \
  --source "$ROOT/VideoSeal" --selection "$SELECTION" --dataset-root "$DATASET" \
  --output-root "$OUTPUT" --chunk-size 1 --grid-size 4 --only 02b72b8d --force

printf '[%s] PIPELINE_COMPLETE\n' "$(date -u +%FT%TZ)"
