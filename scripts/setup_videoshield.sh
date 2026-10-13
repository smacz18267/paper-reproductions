#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
SOURCE=${1:-"$HOME/research/VideoShield"}
GPU=${2:-1}
PIN=$(tr -d '\r\n' < "$ROOT/papers/videoshield/upstream_commit.txt")
REQUIREMENTS="$ROOT/papers/videoshield/environment-requirements.txt"

command -v nvidia-smi >/dev/null
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv
mkdir -p -- "$(dirname -- "$SOURCE")"
if [ ! -e "$SOURCE" ]; then
  git clone --no-checkout https://github.com/hurunyi/VideoShield.git "$SOURCE"
  git -C "$SOURCE" checkout --detach "$PIN"
fi
SOURCE=$(cd -- "$SOURCE" && pwd)
if [ ! -d "$SOURCE/.git" ] || [ "$(git -C "$SOURCE" rev-parse HEAD)" != "$PIN" ]; then
  echo "Expected the pinned VideoShield checkout at $SOURCE. Use a new directory to preserve existing work." >&2
  exit 1
fi
if [ -n "$(git -C "$SOURCE" status --porcelain --untracked-files=no)" ]; then
  echo "VideoShield has tracked source changes. Preserve them and use a clean pinned checkout." >&2
  exit 1
fi

cd "$SOURCE"
if [ ! -x .venv/bin/python ]; then
  "${PYTHON_BOOTSTRAP:-python3}" -m venv .venv
fi
PYTHON="$SOURCE/.venv/bin/python"
"$PYTHON" - <<'PY'
import sys
if not (3, 9) <= sys.version_info[:2] <= (3, 12):
    raise RuntimeError("This VideoShield environment requires Python 3.9–3.12")
print("Python:", sys.version)
PY
"$PYTHON" -m pip install --upgrade pip setuptools wheel
"$PYTHON" -m pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
"$PYTHON" -m pip install -r "$REQUIREMENTS"
"$PYTHON" -m pip check
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" - <<'PY'
import torch
from diffusers import DDIMInverseScheduler, TextToVideoSDPipeline
from watermark import VideoShield
assert torch.cuda.is_available(), "The selected CUDA GPU is unavailable"
print("torch:", torch.__version__, "CUDA runtime:", torch.version.cuda)
print("Visible GPU:", torch.cuda.get_device_name(0))
print("VideoShield and Diffusers imports passed.")
PY

echo "Downloading the pinned ModelScope text-to-video generator. Reruns reuse the recorded revision."
"$PYTHON" - <<'PY'
import json
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download
repo_id = "ali-vilab/text-to-video-ms-1.7b"
manifest_path = Path("model_downloads.json")
if manifest_path.exists():
    manifest = json.loads(manifest_path.read_text())
else:
    manifest = {"repo_id": repo_id, "revision": HfApi().model_info(repo_id).sha}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
snapshot_download(
    repo_id=manifest["repo_id"],
    revision=manifest["revision"],
    local_dir="models/modelscope",
)
PY
if [ ! -s "$SOURCE/models/modelscope/model_index.json" ]; then
  echo "Incomplete ModelScope download." >&2
  exit 1
fi
"$PYTHON" -m pip freeze > "$ROOT/papers/videoshield/environment.freeze.txt"
df -h "$SOURCE"
echo "Setup complete. Run: bash $ROOT/scripts/capture_videoshield.sh $SOURCE $GPU"
