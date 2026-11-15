#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
VIDEOSHIELD=${1:-"$HOME/research/VideoShield"}
DIFFUERASER=${2:-"$HOME/research/DiffuEraser"}
SAMPLE_RUN=${3:-"$ROOT/evidence/videoshield/recorded/20261013T115409Z-1323688"}
GPU=${4:-0}
RUN=${5:-"$ROOT/evidence/videoshield/removal-verification/$(date -u +%Y%m%dT%H%M%SZ)-$$"}

VIDEOSHIELD=$(cd -- "$VIDEOSHIELD" && pwd)
DIFFUERASER=$(cd -- "$DIFFUERASER" && pwd)
SAMPLE_RUN=$(cd -- "$SAMPLE_RUN" && pwd)
VIDEOSHIELD_PYTHON="$VIDEOSHIELD/.venv/bin/python"
DIFFUERASER_PYTHON="$DIFFUERASER/.venv/bin/python"
SAMPLE="$SAMPLE_RUN/results/modelscope/a_red_panda_eating_leaves"
MASKS="$SAMPLE/wm_spatial_tampered/frames_mask_gt"
WORK="$RUN/work"

for required in \
  "$VIDEOSHIELD_PYTHON" \
  "$VIDEOSHIELD/models/modelscope/model_index.json" \
  "$DIFFUERASER_PYTHON" \
  "$DIFFUERASER/model_downloads.json" \
  "$SAMPLE/wm.mp4" \
  "$SAMPLE/wm_info.bin" \
  "$MASKS/00.png"; do
  if [ ! -s "$required" ]; then
    echo "Missing required input: $required" >&2
    exit 1
  fi
done

mkdir -p "$RUN/logs" "$WORK"
on_error() {
  local code=$1
  trap - ERR INT TERM
  printf '%s\n' "$code" > "$RUN/exit_code.txt"
  date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN/finished_at.txt"
  echo "VideoShield removal verification failed (exit $code): $RUN" >&2
  exit "$code"
}
trap 'on_error $?' ERR
trap 'on_error 130' INT
trap 'on_error 143' TERM

date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN/started_at.txt"
printf '%s\n' "$GPU" > "$RUN/selected_gpu.txt"
nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total,memory.used --format=csv > "$RUN/gpu.csv"
git -C "$VIDEOSHIELD" rev-parse HEAD > "$RUN/videoshield-source-commit.txt"
git -C "$DIFFUERASER" rev-parse HEAD > "$RUN/diffueraser-source-commit.txt"
"$VIDEOSHIELD_PYTHON" -m pip freeze > "$RUN/videoshield-environment.freeze.txt"
"$DIFFUERASER_PYTHON" -m pip freeze > "$RUN/diffueraser-environment.freeze.txt"

"$DIFFUERASER_PYTHON" "$ROOT/scripts/prepare_videoshield_removal_sample.py" prepare \
  --source-video "$SAMPLE/wm.mp4" \
  --source-mask-frames "$MASKS" \
  --source-watermark-info "$SAMPLE/wm_info.bin" \
  --output "$RUN"

DIFFUERASER_COMMAND=(
  "$DIFFUERASER_PYTHON" -u "$ROOT/scripts/run_diffueraser_recorded.py"
  --source "$DIFFUERASER" --seed 2025
  --input_video "$RUN/padded-input.mp4"
  --input_mask "$RUN/padded-mask.mp4"
  --video_length 3
  --mask_dilation_iter 8
  --max_img_size 960
  --ref_stride 10
  --neighbor_length 10
  --subvideo_length 40
  --save_path "$WORK/diffueraser"
  --base_model_path "$DIFFUERASER/weights/stable-diffusion-v1-5"
  --vae_path "$DIFFUERASER/weights/sd-vae-ft-mse"
  --diffueraser_path "$DIFFUERASER/weights/diffuEraser"
  --propainter_model_dir "$DIFFUERASER/weights/propainter"
)
{
  printf 'cd %q\n' "$DIFFUERASER"
  printf 'CUDA_VISIBLE_DEVICES=%q ' "$GPU"
  printf '%q ' "${DIFFUERASER_COMMAND[@]}"
  printf '\n'
} > "$RUN/diffueraser-command.sh"

mkdir -p "$WORK/diffueraser"
cd "$DIFFUERASER"
CUDA_VISIBLE_DEVICES="$GPU" "${DIFFUERASER_COMMAND[@]}" 2>&1 | tee "$RUN/logs/diffueraser.log"

"$DIFFUERASER_PYTHON" "$ROOT/scripts/prepare_videoshield_removal_sample.py" finalize \
  --source-video "$WORK/diffueraser/propainter_prior.mp4" \
  --sample-json "$RUN/sample.json" \
  --output-video "$RUN/propainter.mp4" \
  --output-frames "$RUN/propainter-frames"
"$DIFFUERASER_PYTHON" "$ROOT/scripts/prepare_videoshield_removal_sample.py" finalize \
  --source-video "$WORK/diffueraser/diffueraser_result.mp4" \
  --sample-json "$RUN/sample.json" \
  --output-video "$RUN/diffueraser.mp4" \
  --output-frames "$RUN/diffueraser-frames"

cd "$VIDEOSHIELD"
CUDA_VISIBLE_DEVICES="$GPU" "$VIDEOSHIELD_PYTHON" -u \
  "$ROOT/scripts/run_videoshield_removal_detection.py" \
  --source "$VIDEOSHIELD" \
  --model-path "$VIDEOSHIELD/models/modelscope" \
  --watermark-info "$RUN/wm_info.bin" \
  --gt-mask-frames "$RUN/gt-mask-frames" \
  --output "$RUN/detection" \
  --case "original=$RUN/input-watermarked.mp4" \
  --case "propainter=$RUN/propainter.mp4" \
  --case "diffueraser=$RUN/diffueraser.mp4" \
  2>&1 | tee "$RUN/logs/videoshield-detection.log"

"$DIFFUERASER_PYTHON" "$ROOT/scripts/videoshield_removal_artifacts.py" "$RUN" \
  2>&1 | tee "$RUN/artifact-checks.log"
date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN/finished_at.txt"
printf '0\n' > "$RUN/exit_code.txt"
printf 'VideoShield sample was processed by ProPainter and DiffuEraser and detected successfully.\n' > "$RUN/SUCCESS.txt"
echo "VideoShield removal verification complete: $RUN"
