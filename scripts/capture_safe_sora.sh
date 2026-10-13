#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
SOURCE=${1:-"$HOME/research/Safe-Sora"}
GPU=${2:-1}
SOURCE=$(cd -- "$SOURCE" && pwd)
PYTHON="$SOURCE/.venv/bin/python"
PIN=$(tr -d '\r\n' < "$ROOT/papers/safe-sora/upstream_commit.txt")

for path in "$PYTHON" "$SOURCE/ckpt/model.ckpt" "$SOURCE/ckpt/model_latest.pth" "$SOURCE/ckpt/vae.pth" "$SOURCE/ckpt/model_3dcnn_crf24.pth" "$SOURCE/demo_watermark.png"; do
  if [ ! -s "$path" ]; then
    echo "Missing required Safe-Sora file: $path. Finish setup_safe_sora.sh first." >&2
    exit 1
  fi
done
if [ "$(git -C "$SOURCE" rev-parse HEAD)" != "$PIN" ] || [ -n "$(git -C "$SOURCE" status --porcelain --untracked-files=no)" ]; then
  echo "Safe-Sora must be a clean tracked checkout at the pinned commit." >&2
  exit 1
fi

RUN_ID=$(date -u +%Y%m%dT%H%M%SZ)-$$
RUN="$ROOT/evidence/safe-sora/recorded/$RUN_ID"
WORK="$RUN/work"
mkdir -p "$WORK"
cleanup() {
  if [ -L "$WORK/ckpt" ]; then unlink "$WORK/ckpt"; fi
}
on_error() {
  local code=$1
  trap - ERR INT TERM
  cleanup
  printf '%s\n' "$code" > "$RUN/exit_code.txt"
  date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN/finished_at.txt"
  echo "Safe-Sora failed (exit $code). Partial evidence: $RUN" >&2
  exit "$code"
}
trap cleanup EXIT
trap 'on_error $?' ERR
trap 'on_error 130' INT
trap 'on_error 143' TERM

date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN/started_at.txt"
printf '%s\n' "$GPU" > "$RUN/selected_gpu.txt"
cp "$SOURCE/prompt.txt" "$WORK/prompt.txt"
cp "$SOURCE/demo_watermark.png" "$WORK/watermark.png"
cp "$SOURCE/model_sha256.txt" "$RUN/model_sha256.txt"
ln -s "$SOURCE/ckpt" "$WORK/ckpt"
command=("$PYTHON" -u "$SOURCE/scripts/evaluation/inference.py"
  --seed 123 --mode base --ckpt_path "$SOURCE/ckpt/model.ckpt"
  --config "$SOURCE/configs/inference_t2v_512_v2.0.yaml"
  --savedir "$WORK/results" --n_samples 1 --bs 1 --height 320 --width 512
  --unconditional_guidance_scale 12.0 --ddim_steps 50 --ddim_eta 1.0
  --prompt_file "$WORK/prompt.txt" --fps 28 --watermark_path "$WORK/watermark.png")
{
  printf 'cd %q\n' "$WORK"
  printf 'CUDA_VISIBLE_DEVICES=%q ' "$GPU"
  printf '%q ' "${command[@]}"
  printf '\n'
} > "$RUN/command.sh"

git -C "$SOURCE" rev-parse HEAD > "$RUN/source_commit.txt"
git -C "$SOURCE" diff --binary HEAD -- > "$RUN/source_changes.patch"
git -C "$SOURCE" status --porcelain --untracked-files=no > "$RUN/source_status.txt"
git -C "$ROOT" rev-parse HEAD > "$RUN/reproduction_commit.txt"
git -C "$ROOT" diff --binary HEAD -- scripts papers/safe-sora > "$RUN/reproduction_changes.patch"
"$PYTHON" --version > "$RUN/python_version.txt" 2>&1
"$PYTHON" -m pip freeze > "$RUN/environment.freeze.txt"
"$PYTHON" -m pip check > "$RUN/dependency_checks.log" 2>&1
nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total,memory.used --format=csv > "$RUN/gpu.csv"
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" - <<'PY' > "$RUN/torch_runtime.txt"
import torch, torchvision
from mamba_ssm import Mamba
assert torch.cuda.is_available()
print("torch", torch.__version__, "torchvision", torchvision.__version__)
print("CUDA runtime", torch.version.cuda, "GPU", torch.cuda.get_device_name(0))
PY

cd "$WORK"
CUDA_VISIBLE_DEVICES="$GPU" "${command[@]}" 2>&1 | tee "$RUN/run.log"
printf '0\n' > "$RUN/inference_exit_code.txt"
cleanup
"$PYTHON" "$ROOT/scripts/safe_sora_artifacts.py" "$RUN" 2>&1 | tee "$RUN/artifact_checks.log"
date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN/finished_at.txt"
printf '0\n' > "$RUN/exit_code.txt"
printf 'Safe-Sora graphical watermark embedding, recovery and artifact validation passed.\n' > "$RUN/SUCCESS.txt"
echo "Safe-Sora outputs and execution evidence: $RUN"
