#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
SOURCE=${1:-"$HOME/research/ProPainter"}
GPU=${2:-1}
PYTHON="$SOURCE/.venv/bin/python"
if [ ! -x "$PYTHON" ]; then
  echo "Expected the tested environment at $PYTHON" >&2
  exit 1
fi
if [ ! -f "$SOURCE/inference_propainter.py" ]; then
  echo "ProPainter checkout not found: $SOURCE" >&2
  exit 1
fi
RUN_ID=$(date -u +%Y%m%dT%H%M%SZ)-$$
RUN="$ROOT/evidence/propainter/recorded/$RUN_ID"
mkdir -p "$RUN"
cd "$SOURCE"
git rev-parse HEAD > "$RUN/source_commit.txt"
git diff --binary HEAD -- > "$RUN/source_changes.patch"
git status --porcelain --untracked-files=no > "$RUN/source_status.txt"
"$PYTHON" -m pip freeze > "$RUN/environment.freeze.txt"
"$PYTHON" --version > "$RUN/python_version.txt" 2>&1
nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total,memory.used --format=csv > "$RUN/gpu.csv"
printf '%s\n' "$GPU" > "$RUN/selected_gpu.txt"
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" -c 'import torch, torchvision; print("torch", torch.__version__); print("torchvision", torchvision.__version__); print("CUDA runtime", torch.version.cuda); assert torch.cuda.is_available(); print("Visible GPU", torch.cuda.get_device_name(0))' > "$RUN/torch_runtime.txt"
date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN/started_at.txt"

run_case() {
  local case_name=$1
  local video=$2
  local mask=$3
  local dest="$RUN/$case_name"
  mkdir -p "$dest"
  local -a command=("$PYTHON" -u inference_propainter.py --video "$video" --mask "$mask" --height 240 --width 432 --fp16 --output "$dest/videos")
  {
    printf 'cd %q\n' "$SOURCE"
    printf 'CUDA_VISIBLE_DEVICES=%q ' "$GPU"
    printf '%q ' "${command[@]}"
    printf '\n'
  } > "$dest/command.sh"
  date -u +%Y-%m-%dT%H:%M:%SZ > "$dest/started_at.txt"
  CUDA_VISIBLE_DEVICES="$GPU" "${command[@]}" 2>&1 | tee "$dest/run.log"
  printf '0\n' > "$dest/exit_code.txt"
  date -u +%Y-%m-%dT%H:%M:%SZ > "$dest/finished_at.txt"
}

run_case video-completion inputs/video_completion/running_car.mp4 inputs/video_completion/mask_square.png
run_case object-removal inputs/object_removal/bmx-trees inputs/object_removal/bmx-trees_mask
date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN/finished_at.txt"
"$PYTHON" "$ROOT/scripts/describe_results.py" "$RUN" --source "$SOURCE"
printf 'Both inference commands completed successfully.\n' > "$RUN/SUCCESS.txt"
echo "Run evidence saved to: $RUN"
echo "The file hashes and videos confirm which artifacts this run produced."
