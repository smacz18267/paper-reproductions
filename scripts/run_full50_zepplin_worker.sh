#!/usr/bin/env bash
set -Eeuo pipefail

GPU=${1:?Pass physical GPU 0 or 1}
STAGE=${2:-/mnt/data_fast/smayan/research/full50-watermark-20261112}
if [[ "$GPU" != 0 && "$GPU" != 1 ]]; then
  echo "GPU must be 0 or 1" >&2
  exit 2
fi

LOG_ROOT="$STAGE/logs"
STATE_ROOT="$STAGE/state"
CONFIG="$STAGE/config/full50-watermark.zepplin.json"
PYTHON=/mnt/data_fast/smayan/research/DiffuEraser/.venv/bin/python
mkdir -p "$LOG_ROOT" "$STATE_ROOT"
exec 9>"$STATE_ROOT/gpu-$GPU.lock"
if ! flock -n 9; then
  echo "A full-50 worker already owns GPU $GPU" >&2
  exit 3
fi

export HF_HOME=/mnt/data_fast/smayan/research/.cache/huggingface
export HF_HUB_DISABLE_TELEMETRY=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=$GPU

allowed_now() {
  local hour
  hour=$((10#$(date +%H)))
  [[ "$hour" -ge 18 || "$hour" -lt 9 ]]
}

gpu_unused() {
  if nvidia-smi --id="$GPU" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
    | awk 'NF {found=1} END {exit(found ? 0 : 1)}'; then
    return 1
  fi
  nvidia-smi --id="$GPU" --query-gpu=memory.free,utilization.gpu \
    --format=csv,noheader,nounits 2>/dev/null \
    | awk -F, '{gsub(/ /, "", $1); gsub(/ /, "", $2); ok=($1 >= 47000 && $2 <= 10)} END {exit(ok ? 0 : 1)}'
}

wait_for_window_and_gpu() {
  while ! allowed_now; do
    printf '%s waiting for the 18:00-09:00 Zepplin GPU window\n' "$(date -Is)"
    sleep 60
  done
  while ! gpu_unused; do
    printf '%s waiting because physical GPU %s has a compute process\n' "$(date -Is)" "$GPU"
    sleep 60
    while ! allowed_now; do
      printf '%s waiting for the 18:00-09:00 Zepplin GPU window\n' "$(date -Is)"
      sleep 60
    done
  done
  printf '%s physical GPU %s is unused; checking stability\n' "$(date -Is)" "$GPU"
  sleep 30
  allowed_now && gpu_unused
}

seconds_until_cutoff() {
  local now cutoff
  now=$(date +%s)
  if [[ $((10#$(date +%H))) -ge 18 ]]; then
    cutoff=$(date -d 'tomorrow 09:00' +%s)
  else
    cutoff=$(date -d 'today 09:00' +%s)
  fi
  echo $((cutoff - now))
}

run_stage() {
  local marker=$1 log=$2
  shift 2
  while [[ ! -s "$marker" ]]; do
    wait_for_window_and_gpu || continue
    local remaining status
    remaining=$(seconds_until_cutoff)
    if [[ "$remaining" -le 60 ]]; then
      sleep 60
      continue
    fi
    printf '%s starting %s on physical GPU %s with %s seconds left in window\n' \
      "$(date -Is)" "$(basename "$marker")" "$GPU" "$remaining" | tee -a "$log"
    set +e
    timeout --signal=TERM --kill-after=120 "$remaining" "$@" >>"$log" 2>&1
    status=$?
    set -e
    if [[ "$status" -eq 0 ]]; then
      date -Is >"$marker"
      printf '%s completed %s\n' "$(date -Is)" "$(basename "$marker")" | tee -a "$log"
    elif [[ "$status" -eq 124 || "$status" -eq 137 ]]; then
      printf '%s paused %s at the 09:00 cutoff; it will resume next window\n' \
        "$(date -Is)" "$(basename "$marker")" | tee -a "$log"
    else
      printf '%s failed %s with status %s; see %s\n' \
        "$(date -Is)" "$(basename "$marker")" "$status" "$log" | tee -a "$log" >&2
      return "$status"
    fi
  done
}

cd "$STAGE"
if [[ "$GPU" == 0 ]]; then
  run_stage "$STATE_ROOT/fixed-e2fgvi.complete" "$LOG_ROOT/fixed-e2fgvi.log" \
    "$PYTHON" scripts/run_mose_e2fgvi_full50.py \
    --selection selection50.json --dataset-root mose-dataset --output-root fixed-output \
    --e2fgvi /mnt/data_fast/smayan/research/E2FGVI \
    --checkpoint /mnt/data_fast/smayan/research/E2FGVI/release_model/E2FGVI-HQ-CVPR22.pth \
    --e2fgvi-python /mnt/data_fast/smayan/research/DiffuEraser/.venv/bin/python --device 0
  run_stage "$STATE_ROOT/fixed-detection.complete" "$LOG_ROOT/fixed-detection.log" \
    "$PYTHON" scripts/run_mose_full50_detection.py --config "$CONFIG"
  run_stage "$STATE_ROOT/genptw.complete" "$LOG_ROOT/genptw.log" \
    "$PYTHON" scripts/run_full50_watermark_jobs.py --config "$CONFIG" --method genptw
  run_stage "$STATE_ROOT/sigmark-generation.complete" "$LOG_ROOT/sigmark-generation.log" \
    "$PYTHON" scripts/run_native_full50_generation.py --method sigmark \
    --stage "$STAGE" --source-root /mnt/data_fast/smayan/research
else
  run_stage "$STATE_ROOT/wam-vine.complete" "$LOG_ROOT/wam-vine.log" \
    "$PYTHON" scripts/run_full50_watermark_jobs.py --config "$CONFIG" \
    --method watermark-anything --method vine
  run_stage "$STATE_ROOT/videoshield-generation.complete" "$LOG_ROOT/videoshield-generation.log" \
    "$PYTHON" scripts/run_native_full50_generation.py --method videoshield \
    --stage "$STAGE" --source-root /mnt/data_fast/smayan/research
  run_stage "$STATE_ROOT/videomark-generation.complete" "$LOG_ROOT/videomark-generation.log" \
    "$PYTHON" scripts/run_native_full50_generation.py --method videomark \
    --stage "$STAGE" --source-root /mnt/data_fast/smayan/research
  run_stage "$STATE_ROOT/vidsig-generation.complete" "$LOG_ROOT/vidsig-generation.log" \
    "$PYTHON" scripts/run_native_full50_generation.py --method vidsig \
    --stage "$STAGE" --source-root /mnt/data_fast/smayan/research
fi

while [[ ! -s "$STATE_ROOT/native-plan.ready" ]]; do
  printf '%s GPU %s queue waiting for all four native manifests\n' "$(date -Is)" "$GPU"
  sleep 60
done

if [[ "$GPU" == 0 ]]; then
  run_stage "$STATE_ROOT/native-sigmark.complete" "$LOG_ROOT/native-sigmark.log" \
    "$PYTHON" scripts/run_full50_watermark_jobs.py --config "$CONFIG" --method sigmark
else
  run_stage "$STATE_ROOT/native-other.complete" "$LOG_ROOT/native-other.log" \
    "$PYTHON" scripts/run_full50_watermark_jobs.py --config "$CONFIG" \
    --method videoshield --method videomark --method vidsig
fi

if [[ "$GPU" == 0 ]]; then
  while [[ ! -s "$STATE_ROOT/native-other.complete" ]]; do
    printf '%s waiting for the GPU 1 native evaluation before final collection\n' "$(date -Is)"
    sleep 60
  done
  if [[ ! -s "$STATE_ROOT/full50-finalized.complete" ]]; then
    "$PYTHON" scripts/collect_full50_watermark_results.py \
      --mose-output-root fixed-output --require-complete >>"$LOG_ROOT/finalize.log" 2>&1
    "$PYTHON" scripts/update_full50_watermark_notebooks.py >>"$LOG_ROOT/finalize.log" 2>&1
    "$PYTHON" scripts/validate_full50_watermark_notebooks.py \
      --require-complete >>"$LOG_ROOT/finalize.log" 2>&1
    date -Is >"$STATE_ROOT/full50-finalized.complete"
  fi
fi

date -Is >"$STATE_ROOT/gpu-$GPU.complete"
echo "Full-50 GPU $GPU queue complete."
