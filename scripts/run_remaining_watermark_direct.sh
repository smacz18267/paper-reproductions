#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=${1:-/home/smayan/research/remaining-watermark-experiments}
SOURCE_ROOT=${2:-/home/smayan/research}
METHOD=${3:?Pass one of: watermark-anything, vine, videomark, sigmark, vidsig, genptw}
SCRIPTS="$ROOT/scripts"
DIFFUERASER="$SOURCE_ROOT/DiffuEraser"
DIFFUERASER_PYTHON="$DIFFUERASER/.venv/bin/python"
MOSE_VIDEO="$ROOT/evidence/mose-task1/review50/media/02b72b8d/original.mp4"
MOSE_MASK="$ROOT/evidence/mose-task1/review50/media/02b72b8d/gt_mask.mp4"

export HF_HOME=${HF_HOME:-$SOURCE_ROOT/.cache/huggingface}
export HF_HUB_DISABLE_TELEMETRY=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$ROOT/runs" "$ROOT/artifacts" "$ROOT/logs"

wait_for_ready() {
  local method=$1
  while [ ! -s "$ROOT/logs/$method-environment.ready" ] || [ ! -s "$ROOT/logs/$method-models.ready" ]; do
    local env_running=0 model_running=0
    if [ -s "$ROOT/logs/environment-manager.pid" ] && kill -0 "$(cat "$ROOT/logs/environment-manager.pid")" 2>/dev/null; then
      env_running=1
    fi
    if [ -s "$ROOT/logs/model-manager.pid" ] && kill -0 "$(cat "$ROOT/logs/model-manager.pid")" 2>/dev/null; then
      model_running=1
    fi
    if [ ! -s "$ROOT/logs/$method-environment.ready" ] && [ "$env_running" -eq 0 ]; then
      echo "$method environment setup did not pass; see $ROOT/logs/$method-environment.log" >&2
      return 1
    fi
    if [ ! -s "$ROOT/logs/$method-models.ready" ] && [ "$model_running" -eq 0 ]; then
      echo "$method model preparation did not pass; see $ROOT/logs/$method-models.log" >&2
      return 1
    fi
    printf '%s waiting for %s environment/models\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$method" >&2
    sleep 30
  done
}

gpu_snapshot() {
  nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits
}

gpu_has_compute_processes() {
  local gpu=$1
  nvidia-smi --id="$gpu" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
    | awk 'NF {found=1} END {exit(found ? 0 : 1)}'
}

gpu_meets_requirement() {
  local gpu=$1 required=$2
  if gpu_has_compute_processes "$gpu"; then
    return 1
  fi
  gpu_snapshot | awk -F, -v gpu="$gpu" -v required="$required" '
    {gsub(/ /, "", $1); gsub(/ /, "", $2); gsub(/ /, "", $3)}
    $1 == gpu && $2 >= required && $3 <= 10 {found=1}
    END {exit(found ? 0 : 1)}'
}

wait_for_allowed_start() {
  local method=$1 cutoff hour weekday
  weekday=$((10#$(date +%u)))
  if [ "$weekday" -ge 6 ]; then
    return 0
  fi
  cutoff=8
  if [ "$method" = sigmark ]; then
    cutoff=5
  fi
  while true; do
    hour=$((10#$(date +%H)))
    if [ "$hour" -ge 18 ] || [ "$hour" -lt "$cutoff" ]; then
      return 0
    fi
    printf '%s waiting for Zepplin after-hours start window for %s\n' \
      "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$method" >&2
    sleep 300
  done
}

wait_for_gpu() {
  local required=$1 method=$2
  while true; do
    wait_for_allowed_start "$method"
    local candidate candidates
    candidates=$(gpu_snapshot | awk -F, -v required="$required" '
      {gsub(/ /, "", $1); gsub(/ /, "", $2); gsub(/ /, "", $3)}
      $2 >= required && $3 <= 10 {print $1, $2}' | sort -k2,2nr | awk '{print $1}')
    candidate=""
    for available_gpu in $candidates; do
      if ! gpu_has_compute_processes "$available_gpu"; then
        candidate=$available_gpu
        break
      fi
    done
    if [ -n "$candidate" ]; then
      printf '%s candidate GPU %s is unused and has at least %s MiB free; verifying stability\n' \
        "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$candidate" "$required" >&2
      sleep 30
      wait_for_allowed_start "$method"
      if gpu_meets_requirement "$candidate" "$required"; then
        printf '%s\n' "$candidate"
        return 0
      fi
    fi
    printf '%s waiting for a completely unused GPU with %s MiB free\n' \
      "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$required" >&2
    gpu_snapshot >&2
    nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
      --format=csv,noheader,nounits >&2 || true
    sleep 60
  done
}

prepare_attack() {
  local run=$1 method=$2 sample_id=$3 sample_source=$4 source_kind=$5 video=$6 mask=$7
  local attack="$run/attack"
  mkdir -p "$run"
  if [ ! -s "$attack/sample.json" ]; then
    "$DIFFUERASER_PYTHON" "$SCRIPTS/prepare_watermark_removal_sample.py" prepare \
      --source-video "$video" --source-mask "$mask" --output "$attack" \
      --watermark-method "$method" --sample-id "$sample_id" \
      --sample-source "$sample_source" --source-kind "$source_kind"
  fi
  if [ ! -s "$attack/removal-pair.json" ]; then
    if [ -s "$attack/work/removal-pair/propainter_prior.mp4" ] && \
       [ -s "$attack/work/removal-pair/diffueraser_result.mp4" ]; then
      "$DIFFUERASER_PYTHON" "$SCRIPTS/run_watermark_removal_pair.py" \
        --run "$attack" --diffueraser "$DIFFUERASER" \
        --diffueraser-python "$DIFFUERASER_PYTHON" --device "$GPU_ID" --seed 2025 \
        --finalize-existing
    else
      "$DIFFUERASER_PYTHON" "$SCRIPTS/run_watermark_removal_pair.py" \
        --run "$attack" --diffueraser "$DIFFUERASER" \
        --diffueraser-python "$DIFFUERASER_PYTHON" --device "$GPU_ID" --seed 2025 --force
    fi
  fi
  test -s "$attack/propainter.mp4"
  test -s "$attack/diffueraser.mp4"
}

archive_run() {
  local method=$1 run=$2
  local manifest="$run/artifact-sha256.txt"
  find "$run" -type f ! -path '*/work/*' ! -name artifact-sha256.txt -print0 \
    | sort -z | xargs -0 sha256sum > "$manifest"
  tar --exclude='./attack/work' -czf "$ROOT/artifacts/$method.tar.gz" -C "$run" .
  sha256sum "$ROOT/artifacts/$method.tar.gz" > "$ROOT/artifacts/$method.tar.gz.sha256"
  date -u +'%Y-%m-%dT%H:%M:%SZ' > "$ROOT/logs/$method-experiment.complete"
}

run_wam() {
  local repository="$SOURCE_ROOT/watermark-anything"
  local python="$repository/.venv/bin/python"
  local run="$ROOT/runs/watermark-anything/02b72b8d"
  local watermarked="$run/watermarked.mp4"
  mkdir -p "$run"
  if [ ! -s "$watermarked" ]; then
    "$python" "$SCRIPTS/watermark_anything_video.py" embed \
      --source "$repository" --checkpoint "$repository/checkpoints/wam_mit.pth" \
      --params "$repository/checkpoints/params.json" --input-video "$MOSE_VIDEO" \
      --output-video "$watermarked"
  fi
  prepare_attack "$run" 'Watermark Anything' 02b72b8d 'MOSE fixed sample' mose_post_hoc \
    "$watermarked" "$MOSE_MASK"
  if [ ! -s "$run/detection/summary.json" ]; then
    "$python" "$SCRIPTS/watermark_anything_video.py" detect \
      --source "$repository" --checkpoint "$repository/checkpoints/wam_mit.pth" \
      --params "$repository/checkpoints/params.json" --gt-mask "$run/attack/gt-mask.mp4" \
      --output "$run/detection" \
      --case "original=$run/attack/input-watermarked.mp4" \
      --case "propainter=$run/attack/propainter.mp4" \
      --case "diffueraser=$run/attack/diffueraser.mp4"
  fi
  archive_run watermark-anything "$run"
}

run_vine() {
  local repository="$SOURCE_ROOT/VINE"
  local python="$repository/.venv/bin/python"
  local run="$ROOT/runs/vine/02b72b8d"
  local watermarked="$run/watermarked.mp4"
  local mask="$run/aligned-mask.mp4"
  mkdir -p "$run"
  if [ ! -s "$watermarked" ] || [ ! -s "$mask" ]; then
    "$python" "$SCRIPTS/vine_video.py" embed --source "$repository" \
      --input-video "$MOSE_VIDEO" --input-mask "$MOSE_MASK" \
      --output-video "$watermarked" --output-mask "$mask" \
      --encoder-model Shilin-LU/VINE-R-Enc --message 'Hello World!'
  fi
  prepare_attack "$run" VINE 02b72b8d 'MOSE fixed sample after upstream center crop' mose_post_hoc \
    "$watermarked" "$mask"
  if [ ! -s "$run/detection/summary.json" ]; then
    "$python" "$SCRIPTS/vine_video.py" detect --source "$repository" \
      --decoder-model Shilin-LU/VINE-R-Dec --message 'Hello World!' \
      --output "$run/detection" \
      --case "original=$run/attack/input-watermarked.mp4" \
      --case "propainter=$run/attack/propainter.mp4" \
      --case "diffueraser=$run/attack/diffueraser.mp4"
  fi
  archive_run vine "$run"
}

run_videomark() {
  local repository="$SOURCE_ROOT/VideoMark"
  local python="$repository/.venv/bin/python"
  local run="$ROOT/runs/videomark/a_red_panda_eating_leaves"
  local sample="$repository/result/videomark/modelscope/512bit/a_red_panda_eating_leaves/wm.mp4"
  local shift="$repository/result/videomark/modelscope/512bit/a_red_panda_eating_leaves/shift_value.npy"
  local keys="$repository/keys/64_64_512bit.pkl"
  local mask="$run/aligned-mask.mp4"
  mkdir -p "$run"
  if [ ! -s "$mask" ]; then
    "$DIFFUERASER_PYTHON" "$SCRIPTS/make_native_removal_mask.py" center-box \
      --video "$sample" --output-video "$mask" --output-frames "$run/aligned-mask-frames" \
      --width-fraction 0.35 --height-fraction 0.35
  fi
  prepare_attack "$run" VideoMark a_red_panda_eating_leaves \
    "VideoMark authors' committed ModelScope 512-bit sample" authors_provided "$sample" "$mask"
  if [ ! -s "$run/detection/summary.json" ]; then
    "$python" "$SCRIPTS/run_videomark_detection.py" \
      --source "$repository" --model-path "$ROOT/models/modelscope-text-to-video-ms-1.7b" \
      --keys "$keys" --shift "$shift" --output "$run/detection" \
      --case "original=$run/attack/input-watermarked.mp4" \
      --case "propainter=$run/attack/propainter.mp4" \
      --case "diffueraser=$run/attack/diffueraser.mp4"
  fi
  archive_run videomark "$run"
}

run_vidsig() {
  local repository="$SOURCE_ROOT/Video-Signature"
  local python="$repository/.venv/bin/python"
  local run="$ROOT/runs/vidsig/red-panda-seed42"
  local native="$run/native-generation"
  local watermarked="$native/watermarked.mp4"
  local mask="$run/aligned-mask.mp4"
  local vae="$repository/ckpts/vae/modelscope/checkpoint.pth"
  local decoder="$repository/ckpts/msg_decoder/dec_48b_whit.torchscript.pt"
  mkdir -p "$run"
  if [ ! -s "$watermarked" ]; then
    "$python" "$SCRIPTS/run_vidsig_native_sample.py" --source "$repository" \
      --vae-checkpoint "$vae" --message-decoder "$decoder" --output "$native" \
      --model-id "$ROOT/models/modelscope-text-to-video-ms-1.7b" \
      --prompt-file "$repository/prompt.txt" --prompt-index 0 --seed 42
  fi
  if [ ! -s "$mask" ]; then
    "$DIFFUERASER_PYTHON" "$SCRIPTS/make_native_removal_mask.py" center-box \
      --video "$watermarked" --output-video "$mask" --output-frames "$run/aligned-mask-frames"
  fi
  prepare_attack "$run" VidSig red-panda-prompt-0-seed-42 \
    "Video Signature authors' prompt, key, checkpoints, and ModelScope pipeline" \
    authors_generated "$watermarked" "$mask"
  if [ ! -s "$run/detection/summary.json" ]; then
    "$python" "$SCRIPTS/run_vidsig_detection.py" --source "$repository" \
      --decoder "$decoder" --output "$run/detection" \
      --case "original=$run/attack/input-watermarked.mp4" \
      --case "propainter=$run/attack/propainter.mp4" \
      --case "diffueraser=$run/attack/diffueraser.mp4"
  fi
  archive_run vidsig "$run"
}

run_genptw() {
  local repository="$SOURCE_ROOT/GenPTW"
  local python="$repository/.venv/bin/python"
  local run="$ROOT/runs/genptw/02b72b8d"
  local watermarked="$run/watermarked.mp4"
  local mask="$run/aligned-mask.mp4"
  local checkpoint="$repository/Checkpoint"
  local vae="$ROOT/models/stable-diffusion-2-base/vae"
  local convnext="$ROOT/models/convnext_tiny-983f1562.pth"
  mkdir -p "$run"
  if [ ! -s "$watermarked" ] || [ ! -s "$mask" ]; then
    "$python" "$SCRIPTS/genptw_video.py" embed --source "$repository" \
      --checkpoint "$checkpoint" --pretrained-vae "$vae" \
      --convnext-checkpoint "$convnext" --input-video "$MOSE_VIDEO" \
      --input-mask "$MOSE_MASK" --output-video "$watermarked" --output-mask "$mask" --seed 2026
  fi
  prepare_attack "$run" GenPTW 02b72b8d \
    'MOSE fixed sample, GenPTW 512x512 image adapter' mose_post_hoc "$watermarked" "$mask"
  if [ ! -s "$run/detection/summary.json" ]; then
    "$python" "$SCRIPTS/genptw_video.py" detect --source "$repository" \
      --checkpoint "$checkpoint" --pretrained-vae "$vae" \
      --convnext-checkpoint "$convnext" --gt-mask "$run/attack/gt-mask.mp4" \
      --output "$run/detection" \
      --case "original=$run/attack/input-watermarked.mp4" \
      --case "propainter=$run/attack/propainter.mp4" \
      --case "diffueraser=$run/attack/diffueraser.mp4"
  fi
  archive_run genptw "$run"
}

run_sigmark() {
  local repository="$SOURCE_ROOT/SIGMark-release"
  local python="$repository/.venv/bin/python"
  local torchrun="$repository/.venv/bin/torchrun"
  local run="$ROOT/runs/sigmark/native-vbench2"
  local native="$run/native-generation"
  local mask="$run/aligned-mask.mp4"
  mkdir -p "$native"
  local watermarked
  watermarked=$(find "$native" -type f -name '*.mp4' -print -quit)
  if [ -z "$watermarked" ]; then
    (cd "$repository" && "$torchrun" --nproc_per_node=1 main.py \
      --mode gen --model_base_path "$ROOT/models" --model_name HunyuanVideo-community \
      --prompt_set VBench2_aug --watermark_method sigmark --quant_text_encoder 1 \
      --ch_factor 2 --hw_factor 8 --fr_factor 1 --batch_size 1 \
      --num_prompts_per_dimension 1 --num_videos_per_prompt 1 \
      --num_prompts_diversity 1 --num_videos_per_prompt_diversity 1 \
      --small_scale_test 1 --output_path "$native")
    watermarked=$(find "$native" -type f -name '*.mp4' -print -quit)
  fi
  test -n "$watermarked"
  local sample_key
  sample_key=$(realpath --relative-to="$native" "$watermarked")
  sample_key=${sample_key%.mp4}
  if [ ! -s "$mask" ]; then
    "$DIFFUERASER_PYTHON" "$SCRIPTS/make_native_removal_mask.py" center-box \
      --video "$watermarked" --output-video "$mask" --output-frames "$run/aligned-mask-frames"
  fi
  prepare_attack "$run" SIGMark first-VBench2-aug-small-scale \
    "SIGMark authors' VBench2 prompt set and native HunyuanVideo pipeline" \
    authors_generated "$watermarked" "$mask"
  local name video staged
  for name in original propainter diffueraser; do
    case "$name" in
      original) video="$run/attack/input-watermarked.mp4" ;;
      propainter) video="$run/attack/propainter.mp4" ;;
      diffueraser) video="$run/attack/diffueraser.mp4" ;;
    esac
    staged="$run/detection/$name"
    if [ ! -s "$staged/staged-case.json" ]; then
      "$DIFFUERASER_PYTHON" "$SCRIPTS/stage_sigmark_detection_case.py" \
        --generated-output "$native" --sample-key "$sample_key" \
        --replacement-video "$video" --destination "$staged"
    fi
    if [ ! -s "$staged/extraction.complete" ]; then
      (cd "$repository" && "$torchrun" --nproc_per_node=1 main.py \
        --mode extract --model_base_path "$ROOT/models" --model_name HunyuanVideo-community \
        --prompt_set VBench2_aug --watermark_method sigmark --quant_text_encoder 1 \
        --ch_factor 2 --hw_factor 8 --fr_factor 1 --batch_size 1 \
        --num_prompts_per_dimension 1 --num_videos_per_prompt 1 \
        --num_prompts_diversity 1 --num_videos_per_prompt_diversity 1 \
        --small_scale_test 1 --output_path "$staged")
      date -u +'%Y-%m-%dT%H:%M:%SZ' > "$staged/extraction.complete"
    fi
  done
  archive_run sigmark "$run"
}

wait_for_ready "$METHOD"
required_memory=28000
if [ "$METHOD" = sigmark ]; then
  required_memory=47000
fi
GPU_ID=$(wait_for_gpu "$required_memory" "$METHOD")
export CUDA_VISIBLE_DEVICES=$GPU_ID
printf '%s selected physical GPU %s for %s\n' \
  "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$GPU_ID" "$METHOD"
nvidia-smi --id="$GPU_ID" --query-gpu=index,name,uuid,memory.total,memory.free,utilization.gpu --format=csv

case "$METHOD" in
  watermark-anything) run_wam ;;
  vine) run_vine ;;
  videomark) run_videomark ;;
  vidsig) run_vidsig ;;
  genptw) run_genptw ;;
  sigmark) run_sigmark ;;
  *) echo "Unknown method: $METHOD" >&2; exit 2 ;;
esac

echo "$METHOD experiment completed and archived."
