#!/usr/bin/env bash
set -euo pipefail

SCRIPT_ROOT="/home/smayan/research/mose50-corrected/scripts"
DATASET_ROOT="/home/smayan/research/datasets/MOSEv1"
OUTPUT_ROOT="${DATASET_ROOT}/watermark-review50"
SELECTION="${DATASET_ROOT}/selection50.json"
PYTHON="/home/smayan/research/WatermarkMethods/.venv/bin/python"
EDITGUARD="/home/smayan/research/EditGuard"
OMNIGUARD="/home/smayan/research/OmniGuard"
LOCK_FILE="${OUTPUT_ROOT}/image-methods-full50.lock"
STATUS_FILE="${OUTPUT_ROOT}/image-methods-full50.status"
GPU=0

exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
    echo "Another full-50 image-method batch already holds ${LOCK_FILE}." >&2
    exit 1
fi

export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
cd "${SCRIPT_ROOT}"

CURRENT_STAGE="initializing"

status() {
    printf '%s\t%s\n' "$(date --iso-8601=seconds)" "$1" | tee "${STATUS_FILE}"
}

on_exit() {
    code=$?
    if [[ ${code} -eq 0 ]]; then
        status "COMPLETE"
    else
        status "FAILED stage=${CURRENT_STAGE} exit=${code}"
    fi
}
trap on_exit EXIT

run_stage() {
    CURRENT_STAGE="$1"
    shift
    status "RUNNING stage=${CURRENT_STAGE}"
    "$@"
    status "FINISHED stage=${CURRENT_STAGE}"
}

wait_for_gpu_memory() {
    local required_mb="$1"
    local free_mb
    CURRENT_STAGE="waiting-for-gpu-memory-${required_mb}MB"
    while true; do
        free_mb="$(nvidia-smi --id="${GPU}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
        if [[ "${free_mb}" -ge "${required_mb}" ]]; then
            status "GPU_READY free_mb=${free_mb} required_mb=${required_mb}"
            return
        fi
        status "WAITING_GPU free_mb=${free_mb} required_mb=${required_mb}"
        sleep 60
    done
}

run_stage "editguard-embed" \
    "${PYTHON}" -u run_mose_editguard.py embed \
    --source "${EDITGUARD}" \
    --checkpoint "${EDITGUARD}/checkpoints/clean.pth" \
    --selection "${SELECTION}" \
    --dataset-root "${DATASET_ROOT}" \
    --output-root "${OUTPUT_ROOT}"

run_stage "omniguard-embed" \
    "${PYTHON}" -u run_mose_omniguard.py embed \
    --source "${OMNIGUARD}" \
    --checkpoint "${OMNIGUARD}/checkpoint/model_checkpoint_01500.pt" \
    --selection "${SELECTION}" \
    --dataset-root "${DATASET_ROOT}" \
    --output-root "${OUTPUT_ROOT}"

# Preserve the same native-resolution removal protocol used for the three
# notebook samples. Wait for enough free VRAM instead of silently reducing it.
for method in editguard omniguard; do
    for remover in propainter diffueraser; do
        wait_for_gpu_memory 22000
        run_stage "${method}-${remover}" \
            "${PYTHON}" -u run_mose_watermarked_inpainting.py \
            --selection "${SELECTION}" \
            --dataset-root "${DATASET_ROOT}" \
            --output-root "${OUTPUT_ROOT}" \
            --watermark-method "${method}" \
            --remover "${remover}"
    done
done

run_stage "editguard-localize" \
    "${PYTHON}" -u run_mose_editguard.py localize \
    --source "${EDITGUARD}" \
    --checkpoint "${EDITGUARD}/checkpoints/clean.pth" \
    --selection "${SELECTION}" \
    --dataset-root "${DATASET_ROOT}" \
    --output-root "${OUTPUT_ROOT}"

run_stage "omniguard-localize" \
    "${PYTHON}" -u run_mose_omniguard.py localize \
    --source "${OMNIGUARD}" \
    --checkpoint "${OMNIGUARD}/checkpoint/model_checkpoint_01500.pt" \
    --detector-checkpoint "${OMNIGUARD}/checkpoint/checkpoint-175.pth" \
    --selection "${SELECTION}" \
    --dataset-root "${DATASET_ROOT}" \
    --output-root "${OUTPUT_ROOT}"
