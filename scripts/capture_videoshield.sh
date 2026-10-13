#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
SOURCE=${1:-"$HOME/research/VideoShield"}
GPU=${2:-1}
SOURCE=$(cd -- "$SOURCE" && pwd)
PYTHON="$SOURCE/.venv/bin/python"
PIN=$(tr -d '\r\n' < "$ROOT/papers/videoshield/upstream_commit.txt")
MODEL="$SOURCE/models/modelscope"

if [ ! -x "$PYTHON" ] || [ ! -s "$MODEL/model_index.json" ] || [ ! -s "$SOURCE/model_downloads.json" ]; then
  echo "Finish setup_videoshield.sh first. Expected checkout: $SOURCE" >&2
  exit 1
fi
if [ "$(git -C "$SOURCE" rev-parse HEAD)" != "$PIN" ] || [ -n "$(git -C "$SOURCE" status --porcelain --untracked-files=no)" ]; then
  echo "VideoShield must be a clean tracked checkout at the pinned commit." >&2
  exit 1
fi

RUN_ID=$(date -u +%Y%m%dT%H%M%SZ)-$$
RUN="$ROOT/evidence/videoshield/recorded/$RUN_ID"
mkdir -p "$RUN"
on_error() {
  local code=$1
  trap - ERR INT TERM
  printf '%s\n' "$code" > "$RUN/exit_code.txt"
  date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN/finished_at.txt"
  echo "VideoShield failed (exit $code). Partial evidence: $RUN" >&2
  exit "$code"
}
trap 'on_error $?' ERR
trap 'on_error 130' INT
trap 'on_error 143' TERM

date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN/started_at.txt"
printf '%s\n' "$GPU" > "$RUN/selected_gpu.txt"
printf '%s\n' 'a red panda eating leaves' > "$RUN/prompt.txt"
cp "$SOURCE/model_downloads.json" "$RUN/model_downloads.json"
FRAMES="$RUN/results/modelscope/a_red_panda_eating_leaves/wm/frames"
embed=("$PYTHON" -u "$ROOT/scripts/run_videoshield_stage.py" embed
  --source "$SOURCE" --model-path "$MODEL" --run-dir "$RUN" --seed 2025)
temporal=("$PYTHON" -u "$ROOT/scripts/run_videoshield_stage.py" temporal
  --source "$SOURCE" --model-path "$MODEL" --run-dir "$RUN" --frames-dir "$FRAMES" --seed 2025)
spatial=("$PYTHON" -u "$ROOT/scripts/run_videoshield_stage.py" spatial
  --source "$SOURCE" --model-path "$MODEL" --run-dir "$RUN" --frames-dir "$FRAMES" --seed 2025)
{
  printf 'cd %q\n' "$SOURCE"
  for command_name in embed temporal spatial; do
    printf 'CUDA_VISIBLE_DEVICES=%q ' "$GPU"
    if [ "$command_name" = embed ]; then printf '%q ' "${embed[@]}"; fi
    if [ "$command_name" = temporal ]; then printf '%q ' "${temporal[@]}"; fi
    if [ "$command_name" = spatial ]; then printf '%q ' "${spatial[@]}"; fi
    printf '\n'
  done
} > "$RUN/command.sh"

git -C "$SOURCE" rev-parse HEAD > "$RUN/source_commit.txt"
git -C "$SOURCE" diff --binary HEAD -- > "$RUN/source_changes.patch"
git -C "$SOURCE" status --porcelain --untracked-files=no > "$RUN/source_status.txt"
git -C "$ROOT" rev-parse HEAD > "$RUN/reproduction_commit.txt"
git -C "$ROOT" diff --binary HEAD -- scripts papers/videoshield > "$RUN/reproduction_changes.patch"
"$PYTHON" --version > "$RUN/python_version.txt" 2>&1
"$PYTHON" -m pip freeze > "$RUN/environment.freeze.txt"
"$PYTHON" -m pip check > "$RUN/dependency_checks.log" 2>&1
nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total,memory.used --format=csv > "$RUN/gpu.csv"
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" - <<'PY' > "$RUN/torch_runtime.txt"
import torch, torchvision, diffusers, transformers
assert torch.cuda.is_available()
print("torch", torch.__version__, "torchvision", torchvision.__version__)
print("diffusers", diffusers.__version__, "transformers", transformers.__version__)
print("CUDA runtime", torch.version.cuda, "GPU", torch.cuda.get_device_name(0))
PY

cd "$SOURCE"
CUDA_VISIBLE_DEVICES="$GPU" "${embed[@]}" 2>&1 | tee "$RUN/embed.log"
CUDA_VISIBLE_DEVICES="$GPU" "${temporal[@]}" 2>&1 | tee "$RUN/temporal.log"
CUDA_VISIBLE_DEVICES="$GPU" "${spatial[@]}" 2>&1 | tee "$RUN/spatial.log"
printf '0\n' > "$RUN/inference_exit_code.txt"
"$PYTHON" "$ROOT/scripts/videoshield_artifacts.py" "$RUN" 2>&1 | tee "$RUN/artifact_checks.log"
date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN/finished_at.txt"
printf '0\n' > "$RUN/exit_code.txt"
printf 'VideoShield embedding, extraction and tamper-localization validation passed.\n' > "$RUN/SUCCESS.txt"
echo "VideoShield outputs and execution evidence: $RUN"
