#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
SOURCE=${1:-"$HOME/research/VideoSeal"}
GPU=${2:-1}
PIN=$(tr -d '\r\n' < "$ROOT/papers/videoseal/upstream_commit.txt")
REQUIREMENTS="$ROOT/papers/videoseal/environment-requirements.txt"

command -v nvidia-smi >/dev/null
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv
mkdir -p -- "$(dirname -- "$SOURCE")"
if [ ! -e "$SOURCE" ]; then
  git clone --no-checkout https://github.com/facebookresearch/videoseal.git "$SOURCE"
  git -C "$SOURCE" checkout --detach "$PIN"
fi
SOURCE=$(cd -- "$SOURCE" && pwd)
if [ ! -d "$SOURCE/.git" ] || [ "$(git -C "$SOURCE" rev-parse HEAD)" != "$PIN" ]; then
  echo "Expected the pinned VideoSeal checkout at $SOURCE. Use a new directory to preserve existing work." >&2
  exit 1
fi
if [ -n "$(git -C "$SOURCE" status --porcelain --untracked-files=no)" ]; then
  echo "VideoSeal has tracked source changes. Preserve them and use a clean pinned checkout." >&2
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
    raise RuntimeError("VideoSeal requires Python 3.9–3.12")
print("Python:", sys.version)
PY
"$PYTHON" -m pip install --upgrade pip setuptools wheel
"$PYTHON" -m pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
"$PYTHON" -m pip install -r "$REQUIREMENTS"
"$PYTHON" -m pip install --no-deps -e .
"$PYTHON" -m pip check

CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" - <<'PY'
import torch
import videoseal
assert torch.cuda.is_available(), "The selected CUDA GPU is unavailable"
model = videoseal.load("videoseal")
model.eval().to("cuda")
message = model.get_random_msg()
print("torch:", torch.__version__, "CUDA runtime:", torch.version.cuda)
print("Visible GPU:", torch.cuda.get_device_name(0))
print("VideoSeal v1.0 loaded; payload bits:", message.shape[-1])
PY

CHECKPOINT="$SOURCE/ckpts/videoseal_y_256b_img.pth"
if [ ! -s "$CHECKPOINT" ]; then
  echo "VideoSeal checkpoint was not downloaded to the expected path: $CHECKPOINT" >&2
  exit 1
fi
"$PYTHON" - "$CHECKPOINT" "$SOURCE/model_downloads.json" <<'PY'
import hashlib, json, sys
from pathlib import Path
path, output = map(Path, sys.argv[1:])
digest = hashlib.sha256(path.read_bytes()).hexdigest()
output.write_text(json.dumps({
    "model": "videoseal_1.0",
    "url": "https://dl.fbaipublicfiles.com/videoseal/y_256b_img.pth",
    "file": str(path),
    "bytes": path.stat().st_size,
    "sha256": digest,
}, indent=2) + "\n")
PY
"$PYTHON" -m pip freeze > "$ROOT/papers/videoseal/environment.freeze.txt"
df -h "$SOURCE"
echo "Setup complete. Run: bash $ROOT/scripts/capture_videoseal.sh $SOURCE $GPU"
