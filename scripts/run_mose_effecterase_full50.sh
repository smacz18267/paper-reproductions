#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/smayan/research"
SCRIPT_ROOT="${ROOT}/mose50-corrected/scripts"
DATASET_ROOT="${ROOT}/datasets/MOSEv1"
OUTPUT_ROOT="${DATASET_ROOT}/watermark-review50"
SELECTION="${DATASET_ROOT}/selection50.json"
EFFECTERASE="${ROOT}/EffectErase"
PYTHON="${EFFECTERASE}/.venv/bin/python"
GPU=0
LOCK_FILE="${OUTPUT_ROOT}/effecterase-full50.lock"
STATUS_FILE="${OUTPUT_ROOT}/effecterase-full50.status"
PREVIOUS_STATUS="${OUTPUT_ROOT}/image-methods-full50.status"

exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
    echo "Another EffectErase full-50 batch already holds ${LOCK_FILE}." >&2
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

wait_for_previous_batch() {
    CURRENT_STAGE="waiting-for-image-methods-full50"
    while true; do
        previous="$(tail -n 1 "${PREVIOUS_STATUS}" 2>/dev/null || true)"
        if [[ "${previous}" == *$'\tCOMPLETE' ]]; then
            status "PREVIOUS_BATCH_COMPLETE"
            return
        fi
        if [[ "${previous}" == *$'\tFAILED'* ]]; then
            echo "The prerequisite image-method batch failed: ${previous}" >&2
            return 1
        fi
        status "WAITING_PREVIOUS_BATCH ${previous}"
        sleep 60
    done
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

effecterase() {
    local method="$1"
    wait_for_gpu_memory 28000
    run_stage "${method}-effecterase" \
        "${PYTHON}" -u run_mose_effecterase.py \
        --source "${EFFECTERASE}" \
        --selection "${SELECTION}" \
        --dataset-root "${DATASET_ROOT}" \
        --output-root "${OUTPUT_ROOT}" \
        --watermark-method "${method}" \
        --max-side 832 \
        --max-model-frames 81 \
        --num-inference-steps 50 \
        --tiled
}

wait_for_previous_batch

# Fail fast on one representative native-resolution clip before committing to
# the five complete 50-clip passes. The subsequent baseline pass skips it.
wait_for_gpu_memory 28000
run_stage "pilot-baseline-effecterase" \
    "${PYTHON}" -u run_mose_effecterase.py \
    --source "${EFFECTERASE}" \
    --selection "${SELECTION}" \
    --dataset-root "${DATASET_ROOT}" \
    --output-root "${OUTPUT_ROOT}" \
    --watermark-method baseline \
    --max-side 832 \
    --max-model-frames 81 \
    --num-inference-steps 50 \
    --tiled \
    --only 02b72b8d

for method in baseline videoseal safe-sora editguard omniguard; do
    effecterase "${method}"
done

wait_for_gpu_memory 10000
run_stage "videoseal-effecterase-localization" \
    "${ROOT}/VideoSeal/.venv/bin/python" -u run_mose_videoseal_localize.py \
    --source "${ROOT}/VideoSeal" \
    --selection "${SELECTION}" \
    --dataset-root "${DATASET_ROOT}" \
    --output-root "${OUTPUT_ROOT}" \
    --chunk-size 2 \
    --grid-size 4 \
    --remover effecterase

wait_for_gpu_memory 10000
run_stage "safe-sora-effecterase-recovery" \
    "${ROOT}/Safe-Sora/.venv/bin/python" -u run_mose_safe_sora_recover.py \
    --source "${ROOT}/Safe-Sora" \
    --selection "${SELECTION}" \
    --output-root "${OUTPUT_ROOT}" \
    --watermark "${ROOT}/Safe-Sora/demo_watermark.png" \
    --remover effecterase

wait_for_gpu_memory 10000
run_stage "editguard-effecterase-localization" \
    "${ROOT}/WatermarkMethods/.venv/bin/python" -u run_mose_editguard.py localize \
    --source "${ROOT}/EditGuard" \
    --checkpoint "${ROOT}/EditGuard/checkpoints/clean.pth" \
    --selection "${SELECTION}" \
    --dataset-root "${DATASET_ROOT}" \
    --output-root "${OUTPUT_ROOT}" \
    --remover effecterase

wait_for_gpu_memory 10000
run_stage "omniguard-effecterase-localization" \
    "${ROOT}/WatermarkMethods/.venv/bin/python" -u run_mose_omniguard.py localize \
    --source "${ROOT}/OmniGuard" \
    --checkpoint "${ROOT}/OmniGuard/checkpoint/model_checkpoint_01500.pt" \
    --detector-checkpoint "${ROOT}/OmniGuard/checkpoint/checkpoint-175.pth" \
    --selection "${SELECTION}" \
    --dataset-root "${DATASET_ROOT}" \
    --output-root "${OUTPUT_ROOT}" \
    --remover effecterase
