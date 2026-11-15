#!/usr/bin/env bash
set -Eeuo pipefail

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 STAGED_REPOSITORY OUTPUT_ARCHIVE" >&2
  exit 2
fi
if [ -z "${SLURM_JOB_ID:-}" ]; then
  echo "Run this script inside a Slurm GPU allocation." >&2
  exit 2
fi

STAGE=$(cd -- "$1" && pwd)
OUTPUT_ARCHIVE=$2
JOB_ROOT="/tmp/${USER}-videoshield-removal-${SLURM_JOB_ID}"
IMAGE="$JOB_ROOT/pytorch-2.3.1-cuda12.1.sif"
VIDEOSHIELD="$JOB_ROOT/VideoShield"
DIFFUERASER="$JOB_ROOT/DiffuEraser"
RUN="$JOB_ROOT/output"
VIDEOSHIELD_PIN=$(tr -d '\r\n' < "$STAGE/papers/videoshield/upstream_commit.txt")
DIFFUERASER_PIN=$(tr -d '\r\n' < "$STAGE/papers/diffueraser/upstream_commit.txt")

mkdir -p "$JOB_ROOT/apptainer-cache" "$RUN"
export APPTAINER_CACHEDIR="$JOB_ROOT/apptainer-cache"
export HF_HOME="$JOB_ROOT/huggingface-cache"
export PIP_CACHE_DIR="$JOB_ROOT/pip-cache"

nvidia-smi
df -h /tmp
apptainer pull "$IMAGE" docker://pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime
git clone --no-checkout https://github.com/hurunyi/VideoShield.git "$VIDEOSHIELD"
git -C "$VIDEOSHIELD" checkout --detach "$VIDEOSHIELD_PIN"
git clone --no-checkout https://github.com/lixiaowen-xw/DiffuEraser.git "$DIFFUERASER"
git -C "$DIFFUERASER" checkout --detach "$DIFFUERASER_PIN"

container() {
  apptainer exec --nv \
    --bind "$STAGE:$STAGE,$JOB_ROOT:$JOB_ROOT" \
    --env "HF_HOME=$HF_HOME,PIP_CACHE_DIR=$PIP_CACHE_DIR" \
    "$IMAGE" "$@"
}

# The container already carries the exact torch/torchvision/torchaudio CUDA
# builds. System-site-packages avoids downloading those wheels twice while the
# pinned setup scripts install and validate every remaining dependency.
container python -m venv --system-site-packages "$VIDEOSHIELD/.venv"
container env PYTHON_BOOTSTRAP=python bash "$STAGE/scripts/setup_videoshield.sh" "$VIDEOSHIELD" 0
container python -m venv --system-site-packages "$DIFFUERASER/.venv"
container bash "$STAGE/scripts/setup_diffueraser.sh" "$DIFFUERASER" 0

container bash "$STAGE/scripts/capture_videoshield_removal_verification.sh" \
  "$VIDEOSHIELD" \
  "$DIFFUERASER" \
  "$STAGE/evidence/videoshield/recorded/20261013T115409Z-1323688" \
  0 \
  "$RUN"

tar --exclude='./work' -czf "$OUTPUT_ARCHIVE" -C "$RUN" .
echo "Archived VideoShield removal verification to $OUTPUT_ARCHIVE"
