#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
SOURCE=${1:-"$HOME/research/VideoSeal"}
GPU=${2:-1}
SOURCE=$(cd -- "$SOURCE" && pwd)
PYTHON="$SOURCE/.venv/bin/python"
PIN=$(tr -d '\r\n' < "$ROOT/papers/videoseal/upstream_commit.txt")
INPUT="$SOURCE/assets/videos/1.mp4"

if [ ! -x "$PYTHON" ] || [ ! -s "$INPUT" ] || [ ! -s "$SOURCE/model_downloads.json" ]; then
  echo "Finish setup_videoseal.sh first. Expected checkout: $SOURCE" >&2
  exit 1
fi
if [ "$(git -C "$SOURCE" rev-parse HEAD)" != "$PIN" ] || [ -n "$(git -C "$SOURCE" status --porcelain --untracked-files=no)" ]; then
  echo "VideoSeal must be a clean tracked checkout at the pinned commit." >&2
  exit 1
fi

RUN_ID=$(date -u +%Y%m%dT%H%M%SZ)-$$
RUN="$ROOT/evidence/videoseal/recorded/$RUN_ID"
mkdir -p "$RUN"
on_error() {
  local code=$1
  trap - ERR INT TERM
  printf '%s\n' "$code" > "$RUN/exit_code.txt"
  date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN/finished_at.txt"
  echo "VideoSeal failed (exit $code). Partial evidence: $RUN" >&2
  exit "$code"
}
trap 'on_error $?' ERR
trap 'on_error 130' INT
trap 'on_error 143' TERM

date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN/started_at.txt"
printf '%s\n' "$GPU" > "$RUN/selected_gpu.txt"
cp "$INPUT" "$RUN/input.mp4"
cp "$SOURCE/model_downloads.json" "$RUN/model_downloads.json"
command=("$PYTHON" -u "$ROOT/scripts/run_videoseal_recorded.py"
  --input "$RUN/input.mp4" --output "$RUN/watermarked.mp4"
  --metrics "$RUN/metrics.json" --message "$RUN/message.txt"
  --seed 2025 --chunk-size 16)
{
  printf 'cd %q\n' "$SOURCE"
  printf 'CUDA_VISIBLE_DEVICES=%q ' "$GPU"
  printf '%q ' "${command[@]}"
  printf '\n'
} > "$RUN/command.sh"

git -C "$SOURCE" rev-parse HEAD > "$RUN/source_commit.txt"
git -C "$SOURCE" diff --binary HEAD -- > "$RUN/source_changes.patch"
git -C "$SOURCE" status --porcelain --untracked-files=no > "$RUN/source_status.txt"
git -C "$ROOT" rev-parse HEAD > "$RUN/reproduction_commit.txt"
git -C "$ROOT" diff --binary HEAD -- scripts papers/videoseal > "$RUN/reproduction_changes.patch"
"$PYTHON" --version > "$RUN/python_version.txt" 2>&1
"$PYTHON" -m pip freeze > "$RUN/environment.freeze.txt"
"$PYTHON" -m pip check > "$RUN/dependency_checks.log" 2>&1
nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total,memory.used --format=csv > "$RUN/gpu.csv"
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" - <<'PY' > "$RUN/torch_runtime.txt"
import torch, torchvision, videoseal
assert torch.cuda.is_available()
print("torch", torch.__version__, "torchvision", torchvision.__version__)
print("videoseal", videoseal.__version__, "CUDA runtime", torch.version.cuda)
print("Visible GPU", torch.cuda.get_device_name(0))
PY

cd "$SOURCE"
CUDA_VISIBLE_DEVICES="$GPU" "${command[@]}" 2>&1 | tee "$RUN/run.log"
printf '0\n' > "$RUN/inference_exit_code.txt"
"$PYTHON" "$ROOT/scripts/videoseal_artifacts.py" "$RUN" 2>&1 | tee "$RUN/artifact_checks.log"
date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN/finished_at.txt"
printf '0\n' > "$RUN/exit_code.txt"
printf 'VideoSeal embedding, extraction and decoded artifact validation passed.\n' > "$RUN/SUCCESS.txt"
echo "VideoSeal output and execution evidence: $RUN"
