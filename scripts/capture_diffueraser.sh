#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
SOURCE=${1:-"$HOME/research/DiffuEraser"}
GPU=${2:-1}
SOURCE=$(cd -- "$SOURCE" && pwd)
PYTHON="$SOURCE/.venv/bin/python"
PIN=$(tr -d '\r\n' < "$ROOT/papers/diffueraser/upstream_commit.txt")
if [ ! -x "$PYTHON" ] || [ ! -f "$SOURCE/run_diffueraser.py" ] || [ ! -s "$SOURCE/model_downloads.json" ]; then
  echo "Finish setup_diffueraser.sh first. Expected checkout: $SOURCE" >&2
  exit 1
fi
if [ "$(git -C "$SOURCE" rev-parse HEAD)" != "$PIN" ]; then
  echo "The DiffuEraser checkout is not at the pinned source commit." >&2
  exit 1
fi
if [ -n "$(git -C "$SOURCE" status --porcelain --untracked-files=no)" ]; then
  echo "The DiffuEraser source has local edits. Preserve them and use a clean pinned checkout for this run." >&2
  exit 1
fi
RUN_ID=$(date -u +%Y%m%dT%H%M%SZ)-$$
RUN="$ROOT/evidence/diffueraser/recorded/$RUN_ID"
mkdir -p "$RUN"
on_error() {
  local code=$1
  trap - ERR INT TERM
  printf '%s\n' "$code" > "$RUN/exit_code.txt"
  date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN/finished_at.txt"
  echo "DiffuEraser failed (exit $code). Partial evidence: $RUN" >&2
  exit "$code"
}
trap 'on_error $?' ERR
trap 'on_error 130' INT
trap 'on_error 143' TERM
date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN/started_at.txt"
printf '%s\n' "$GPU" > "$RUN/selected_gpu.txt"
VIDEO="$SOURCE/examples/example3/video.mp4"
MASK="$SOURCE/examples/example3/mask.mp4"
cp "$VIDEO" "$RUN/input_video.mp4"
cp "$MASK" "$RUN/input_mask.mp4"
cp "$SOURCE/model_downloads.json" "$RUN/model_downloads.json"
command=("$PYTHON" -u "$ROOT/scripts/run_diffueraser_recorded.py"
  --source "$SOURCE" --seed 2025
  --input_video "$RUN/input_video.mp4" --input_mask "$RUN/input_mask.mp4"
  --save_path "$RUN" --video_length 10 --max_img_size 960 --mask_dilation_iter 8
  --ref_stride 10 --neighbor_length 10 --subvideo_length 50
  --base_model_path "$SOURCE/weights/stable-diffusion-v1-5"
  --vae_path "$SOURCE/weights/sd-vae-ft-mse"
  --diffueraser_path "$SOURCE/weights/diffuEraser"
  --propainter_model_dir "$SOURCE/weights/propainter")
{
  printf 'cd %q\n' "$SOURCE"
  printf 'CUDA_VISIBLE_DEVICES=%q ' "$GPU"
  printf '%q ' "${command[@]}"
  printf '\n'
} > "$RUN/command.sh"
cd "$SOURCE"
git rev-parse HEAD > "$RUN/source_commit.txt"
git diff --binary HEAD -- > "$RUN/source_changes.patch"
git status --porcelain --untracked-files=no > "$RUN/source_status.txt"
git -C "$ROOT" rev-parse HEAD > "$RUN/reproduction_commit.txt"
git -C "$ROOT" diff --binary HEAD -- scripts papers/diffueraser/environment-constraints.txt > "$RUN/reproduction_changes.patch"
"$PYTHON" --version > "$RUN/python_version.txt" 2>&1
"$PYTHON" -m pip freeze > "$RUN/environment.freeze.txt"
"$PYTHON" -m pip check > "$RUN/dependency_checks.log" 2>&1
nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total,memory.used --format=csv > "$RUN/gpu.csv"
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" - <<'PY' > "$RUN/torch_runtime.txt"
import torch, torchvision, diffusers, transformers
assert torch.cuda.is_available(), "The selected CUDA GPU is unavailable"
print("torch", torch.__version__, "torchvision", torchvision.__version__)
print("diffusers", diffusers.__version__, "transformers", transformers.__version__)
print("CUDA runtime", torch.version.cuda)
print("Visible GPU", torch.cuda.get_device_name(0))
PY
"$PYTHON" "$ROOT/scripts/diffueraser_artifacts.py" "$RUN" --source "$SOURCE" --preflight 2>&1 | tee "$RUN/input_checks.log"
if CUDA_VISIBLE_DEVICES="$GPU" "${command[@]}" 2>&1 | tee "$RUN/run.log"; then
  statuses=("${PIPESTATUS[@]}")
else
  statuses=("${PIPESTATUS[@]}")
fi
printf '%s\n' "${statuses[0]}" > "$RUN/inference_exit_code.txt"
if [ "${statuses[0]}" -ne 0 ]; then on_error "${statuses[0]}"; fi
if [ "${statuses[1]}" -ne 0 ]; then on_error "${statuses[1]}"; fi
"$PYTHON" "$ROOT/scripts/diffueraser_artifacts.py" "$RUN" --source "$SOURCE" 2>&1 | tee "$RUN/artifact_checks.log"
date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN/finished_at.txt"
printf '0\n' > "$RUN/exit_code.txt"
printf 'Inference finished; final output and preserved ProPainter prior decoded and verified.\n' > "$RUN/SUCCESS.txt"
echo "DiffuEraser output and execution evidence: $RUN"
