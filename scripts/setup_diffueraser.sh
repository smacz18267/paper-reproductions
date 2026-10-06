#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
SOURCE=${1:-"$HOME/research/DiffuEraser"}
GPU=${2:-1}
PIN=$(tr -d '\r\n' < "$ROOT/papers/diffueraser/upstream_commit.txt")
CONSTRAINTS="$ROOT/papers/diffueraser/environment-constraints.txt"
command -v nvidia-smi >/dev/null
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv
mkdir -p -- "$(dirname -- "$SOURCE")"
if [ ! -e "$SOURCE" ]; then
  git clone --no-checkout https://github.com/lixiaowen-xw/DiffuEraser.git "$SOURCE"
  git -C "$SOURCE" checkout --detach "$PIN"
fi
SOURCE=$(cd -- "$SOURCE" && pwd)
if [ ! -d "$SOURCE/.git" ] || [ "$(git -C "$SOURCE" rev-parse HEAD)" != "$PIN" ]; then
  echo "Expected the pinned DiffuEraser checkout at $SOURCE. Use a new directory to preserve existing work." >&2
  exit 1
fi
if [ -n "$(git -C "$SOURCE" status --porcelain --untracked-files=no)" ]; then
  echo "DiffuEraser has local source changes. Keep them and use a new directory for setup." >&2
  exit 1
fi
cd "$SOURCE"
if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv
fi
PYTHON="$SOURCE/.venv/bin/python"
"$PYTHON" - <<'PY'
import sys
if not (3, 9) <= sys.version_info[:2] <= (3, 12):
    raise RuntimeError("Use Python 3.9–3.12 for this pinned environment")
print("Python:", sys.version)
PY
"$PYTHON" -m pip install --upgrade pip setuptools wheel
"$PYTHON" -m pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
"$PYTHON" -m pip install -r requirements.txt -c "$CONSTRAINTS"
"$PYTHON" -m pip check
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" - <<'PY'
import torch, torchvision
from diffueraser.diffueraser import DiffuEraser
from propainter.inference import Propainter
assert torch.cuda.is_available(), "The selected GPU is unavailable"
# Exercise the CUDA extension used by ProPainter, not just torch import.
x = torch.ones(1, 1, 4, 4, device="cuda")
offset = torch.zeros(1, 18, 4, 4, device="cuda")
weight = torch.ones(1, 1, 3, 3, device="cuda")
y = torchvision.ops.deform_conv2d(x, offset, weight, padding=1)
assert y.shape == x.shape
torch.cuda.synchronize()
print("torch:", torch.__version__, "torchvision:", torchvision.__version__)
print("CUDA runtime:", torch.version.cuda, "GPU:", torch.cuda.get_device_name(0))
print("Upstream imports and CUDA deformable convolution passed.")
PY
df -h "$SOURCE"
echo "Downloading the DiffuEraser, SD 1.5, VAE and 2-step PCM inference weights. Reruns reuse the recorded revisions."
"$PYTHON" "$ROOT/scripts/download_diffueraser_models.py" --source "$SOURCE"
SNAPSHOT="$ROOT/papers/diffueraser/source"
if [ ! -e "$SNAPSHOT" ]; then
  mkdir -p "$SNAPSHOT"
  git archive "$PIN" | tar -x -C "$SNAPSHOT"
fi
if [ ! -s "$SNAPSHOT/run_diffueraser.py" ]; then
  echo "Incomplete source snapshot: $SNAPSHOT" >&2
  exit 1
fi
cp "$ROOT/papers/diffueraser/ProPainter-LICENSE" "$SNAPSHOT/propainter/LICENSE"
"$PYTHON" -m pip freeze > "$ROOT/papers/diffueraser/environment.freeze.txt"
echo "Setup complete. Run: bash $ROOT/scripts/capture_diffueraser.sh $SOURCE $GPU"
