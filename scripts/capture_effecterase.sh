#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
SOURCE=${1:-"$HOME/research/EffectErase"}
GPU=${2:-1}
SOURCE=$(cd -- "$SOURCE" && pwd)
PYTHON="$SOURCE/.venv/bin/python"
MODEL_DIR="$SOURCE/Wan-AI/Wan2.1-Fun-1.3B-InP"
VIDEO="$SOURCE/demo/FG_BG/WILD_ENV006_00042.mp4"
MASK="$SOURCE/demo/MASK/WILD_ENV006_00042.mp4"
if [ ! -x "$PYTHON" ] || [ ! -f "$SOURCE/examples/remove_wan/infer_remove_wan.py" ]; then
  echo "Run setup_effecterase.sh first. Expected EffectErase at $SOURCE" >&2
  exit 1
fi
for path in "$VIDEO" "$MASK" "$SOURCE/EffectErase.ckpt" "$MODEL_DIR/diffusion_pytorch_model.safetensors" "$MODEL_DIR/Wan2.1_VAE.pth" "$MODEL_DIR/models_t5_umt5-xxl-enc-bf16.pth" "$MODEL_DIR/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" "$MODEL_DIR/google/umt5-xxl/tokenizer_config.json"; do
  if [ ! -s "$path" ]; then
    echo "Missing required file: $path. Finish setup_effecterase.sh first." >&2
    exit 1
  fi
done

RUN_ID=$(date -u +%Y%m%dT%H%M%SZ)-$$
RUN="$ROOT/evidence/effecterase/recorded/$RUN_ID"
mkdir -p "$RUN"
on_error() {
  local code=$1
  trap - ERR
  printf '%s\n' "$code" > "$RUN/exit_code.txt"
  date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN/finished_at.txt"
  echo "EffectErase failed (exit $code). Partial evidence: $RUN" >&2
  exit "$code"
}
trap 'on_error $?' ERR
trap 'on_error 130' INT
trap 'on_error 143' TERM
date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN/started_at.txt"
printf '%s\n' "$GPU" > "$RUN/selected_gpu.txt"
# These are the authors' default settings: 81 frames, 832x480, 50 steps,
# seed 2025, CFG 1, BF16, no TeaCache and no additional VAE tiling.
command=("$PYTHON" -u examples/remove_wan/infer_remove_wan.py
  --fg_bg_path "$VIDEO" --mask_path "$MASK" --output_path "$RUN/removed.mp4"
  --num_frames 81 --frame_interval 1 --height 480 --width 832
  --seed 2025 --cfg 1.0 --lora_alpha 1.0 --num_inference_steps 50
  --text_encoder_path "$MODEL_DIR/models_t5_umt5-xxl-enc-bf16.pth"
  --vae_path "$MODEL_DIR/Wan2.1_VAE.pth"
  --dit_path "$MODEL_DIR/diffusion_pytorch_model.safetensors"
  --image_encoder_path "$MODEL_DIR/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
  --pretrained_lora_path "$SOURCE/EffectErase.ckpt")
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
"$PYTHON" --version > "$RUN/python_version.txt" 2>&1
"$PYTHON" -m pip freeze > "$RUN/environment.freeze.txt"
nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total,memory.used --format=csv > "$RUN/gpu.csv"
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" - <<'PY' > "$RUN/torch_runtime.txt"
import torch, torchvision, diffsynth
assert torch.cuda.is_available(), "CUDA is unavailable"
assert torch.cuda.is_bf16_supported(), "BF16 is unsupported"
print("torch", torch.__version__)
print("torchvision", torchvision.__version__)
print("CUDA runtime", torch.version.cuda)
print("Visible GPU", torch.cuda.get_device_name(0))
print("DiffSynth source", diffsynth.__file__)
PY
if [ -f "$SOURCE/model_downloads.json" ]; then
  cp "$SOURCE/model_downloads.json" "$RUN/model_downloads.json"
fi
if [ -f "$SOURCE/setup_dependency_checks.json" ]; then
  cp "$SOURCE/setup_dependency_checks.json" "$RUN/setup_dependency_checks.json"
fi
"$PYTHON" "$ROOT/scripts/effecterase_artifacts.py" "$RUN" --source "$SOURCE" --preflight 2>&1 | tee "$RUN/input_checks.log"
CUDA_VISIBLE_DEVICES="$GPU" "${command[@]}" 2>&1 | tee "$RUN/run.log"
printf '0\n' > "$RUN/inference_exit_code.txt"
"$PYTHON" "$ROOT/scripts/effecterase_artifacts.py" "$RUN" --source "$SOURCE" 2>&1 | tee "$RUN/artifact_checks.log"
date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN/finished_at.txt"
printf '0\n' > "$RUN/exit_code.txt"
printf 'Inference completed and all 81 output frames were decoded and verified.\n' > "$RUN/SUCCESS.txt"
echo "EffectErase completed. Video and execution evidence: $RUN"
