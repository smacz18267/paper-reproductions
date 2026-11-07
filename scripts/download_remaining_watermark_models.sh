#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=${1:-/home/smayan/research/remaining-watermark-experiments}
SOURCE_ROOT=${2:-/home/smayan/research}
METHOD=${3:?Pass one of: watermark-anything, vine, videomark, sigmark, vidsig, genptw}
DOWNLOADS="$ROOT/downloads"
MODELS="$ROOT/models"
DOWNLOADER="$ROOT/.download-venv/bin/python"
SAFE_EXTRACT="$ROOT/scripts/safe_extract_archive.py"

export HF_HOME=${HF_HOME:-$SOURCE_ROOT/.cache/huggingface}
export HF_HUB_DISABLE_TELEMETRY=1
mkdir -p "$DOWNLOADS" "$MODELS" "$HF_HOME" "$ROOT/logs"

if [ ! -x "$DOWNLOADER" ]; then
  python3.12 -m venv "$ROOT/.download-venv"
  "$DOWNLOADER" -m pip install --upgrade pip setuptools wheel
  "$DOWNLOADER" -m pip install --upgrade gdown huggingface_hub
fi

snapshot() {
  local repo_id=$1
  local destination=${2:-}
  local allow_pattern=${3:-}
  REPO_ID=$repo_id DESTINATION=$destination ALLOW_PATTERN=$allow_pattern "$DOWNLOADER" - <<'PY'
import os
from huggingface_hub import snapshot_download

kwargs = {"repo_id": os.environ["REPO_ID"]}
destination = os.environ.get("DESTINATION")
pattern = os.environ.get("ALLOW_PATTERN")
if destination:
    kwargs["local_dir"] = destination
if pattern:
    kwargs["allow_patterns"] = [pattern]
path = snapshot_download(**kwargs)
print(path)
PY
}

download_drive_archive() {
  local url=$1
  local archive=$2
  local extracted=$3
  if [ ! -s "$archive" ]; then
    # Current gdown accepts Google Drive share URLs directly.  Older releases
    # exposed a --fuzzy switch, but keeping that switch breaks on gdown 6.x.
    "$DOWNLOADER" -m gdown "$url" -O "$archive"
  fi
  if [ ! -f "$extracted/.complete" ]; then
    if [ -d "$extracted" ] && [ -n "$(find "$extracted" -mindepth 1 -maxdepth 1 -print -quit)" ]; then
      echo "Partial extraction exists without marker: $extracted" >&2
      return 1
    fi
    "$DOWNLOADER" "$SAFE_EXTRACT" "$archive" "$extracted"
    touch "$extracted/.complete"
  fi
}

find_one() {
  local root=$1
  local name=$2
  local result
  result=$(find "$root" -type f -name "$name" -print -quit)
  if [ -z "$result" ]; then
    echo "Could not find $name under $root" >&2
    return 1
  fi
  printf '%s\n' "$result"
}

setup_wam_models() {
  test -s "$SOURCE_ROOT/watermark-anything/checkpoints/wam_mit.pth"
  test -s "$SOURCE_ROOT/watermark-anything/checkpoints/params.json"
}

setup_vine_models() {
  snapshot Shilin-LU/VINE-R-Enc
  snapshot Shilin-LU/VINE-R-Dec
}

setup_videomark_models() {
  snapshot damo-vilab/text-to-video-ms-1.7b "$MODELS/modelscope-text-to-video-ms-1.7b"
  test -s "$MODELS/modelscope-text-to-video-ms-1.7b/model_index.json"
  test -s "$SOURCE_ROOT/VideoMark/result/videomark/modelscope/512bit/a_red_panda_eating_leaves/wm.mp4"
  test -s "$SOURCE_ROOT/VideoMark/result/videomark/modelscope/512bit/a_red_panda_eating_leaves/shift_value.npy"
  test -s "$SOURCE_ROOT/VideoMark/keys/64_64_512bit.pkl"
}

setup_sigmark_models() {
  snapshot hunyuanvideo-community/HunyuanVideo "$MODELS/HunyuanVideo-community"
  test -s "$MODELS/HunyuanVideo-community/model_index.json"
}

setup_vidsig_models() {
  local archive="$DOWNLOADS/vidsig-checkpoints.archive"
  local extracted="$DOWNLOADS/vidsig-checkpoints"
  download_drive_archive \
    'https://drive.google.com/file/d/1XFyzeX6T0iHgcxSN_DxvjLFy1EXZye-Q/view?usp=drive_link' \
    "$archive" "$extracted"
  local vae decoder
  vae=$(find "$extracted" -type f -path '*/vae/modelscope/checkpoint.pth' -print -quit)
  if [ -z "$vae" ]; then
    vae=$(find_one "$extracted" checkpoint.pth)
  fi
  decoder=$(find_one "$extracted" dec_48b_whit.torchscript.pt)
  install -D -m 0644 "$vae" "$SOURCE_ROOT/Video-Signature/ckpts/vae/modelscope/checkpoint.pth"
  install -D -m 0644 "$decoder" "$SOURCE_ROOT/Video-Signature/ckpts/msg_decoder/dec_48b_whit.torchscript.pt"
  test -s "$MODELS/modelscope-text-to-video-ms-1.7b/model_index.json"
}

setup_genptw_models() {
  local archive="$DOWNLOADS/genptw-checkpoint.archive"
  local extracted="$DOWNLOADS/genptw-checkpoint"
  download_drive_archive \
    'https://drive.google.com/file/d/1nC85Jc0B6K5ycqRHN0NFWVQP2jSLHsoT/view?usp=drive_link' \
    "$archive" "$extracted"
  for name in msg_decoder.pth localizer.pth diffusion_pytorch_model.safetensors; do
    source_file=$(find_one "$extracted" "$name")
    install -D -m 0644 "$source_file" "$SOURCE_ROOT/GenPTW/Checkpoint/$name"
  done
  snapshot Manojb/stable-diffusion-2-base "$MODELS/stable-diffusion-2-base" 'vae/*'
  if [ ! -s "$MODELS/convnext_tiny-983f1562.pth" ]; then
    curl -fL --retry 5 \
      https://download.pytorch.org/models/convnext_tiny-983f1562.pth \
      -o "$MODELS/convnext_tiny-983f1562.pth"
  fi
  test -s "$MODELS/stable-diffusion-2-base/vae/config.json"
}

case "$METHOD" in
  watermark-anything) setup_wam_models ;;
  vine) setup_vine_models ;;
  videomark) setup_videomark_models ;;
  sigmark) setup_sigmark_models ;;
  vidsig) setup_vidsig_models ;;
  genptw) setup_genptw_models ;;
  *) echo "Unknown method: $METHOD" >&2; exit 2 ;;
esac

date -u +'%Y-%m-%dT%H:%M:%SZ' > "$ROOT/logs/$METHOD-models.ready"
echo "$METHOD model preparation passed."
