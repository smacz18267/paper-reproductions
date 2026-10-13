#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
SOURCE=${1:-"$HOME/research/Safe-Sora"}
GPU=${2:-1}
PIN=$(tr -d '\r\n' < "$ROOT/papers/safe-sora/upstream_commit.txt")
REQUIREMENTS="$ROOT/papers/safe-sora/environment-requirements.txt"

command -v nvidia-smi >/dev/null
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv
mkdir -p -- "$(dirname -- "$SOURCE")"
if [ ! -e "$SOURCE" ]; then
  git clone --no-checkout https://github.com/Zihan-Su/Safe-Sora.git "$SOURCE"
  git -C "$SOURCE" checkout --detach "$PIN"
fi
SOURCE=$(cd -- "$SOURCE" && pwd)
if [ ! -d "$SOURCE/.git" ] || [ "$(git -C "$SOURCE" rev-parse HEAD)" != "$PIN" ]; then
  echo "Expected the pinned Safe-Sora checkout at $SOURCE. Use a new directory to preserve existing work." >&2
  exit 1
fi
if [ -n "$(git -C "$SOURCE" status --porcelain --untracked-files=no)" ]; then
  echo "Safe-Sora has tracked source changes. Preserve them and use a clean pinned checkout." >&2
  exit 1
fi

cd "$SOURCE"
if [ ! -x .venv/bin/python ]; then
  if [ -n "${PYTHON_BOOTSTRAP:-}" ]; then
    "$PYTHON_BOOTSTRAP" -m venv .venv
  elif command -v python3.10 >/dev/null; then
    python3.10 -m venv .venv
  else
    python3 -m venv .venv
  fi
fi
PYTHON="$SOURCE/.venv/bin/python"
"$PYTHON" - <<'PY'
import sys
if not (3, 9) <= sys.version_info[:2] <= (3, 11):
    raise RuntimeError("Safe-Sora's PyTorch 2.1 environment requires Python 3.9–3.11. Set PYTHON_BOOTSTRAP to a compatible interpreter.")
print("Python:", sys.version)
PY
"$PYTHON" -m pip install --upgrade pip setuptools wheel
"$PYTHON" -m pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
"$PYTHON" -m pip install packaging==23.2 ninja==1.11.1.1 gdown==5.2.0
"$PYTHON" -m pip install -r "$REQUIREMENTS"

DOWNLOADS="$SOURCE/downloads"
mkdir -p "$DOWNLOADS"
download_drive() {
  local url=$1
  local destination=$2
  if [ ! -s "$destination" ]; then
    "$PYTHON" -m gdown --fuzzy "$url" -O "$destination.part"
    mv "$destination.part" "$destination"
  fi
}

download_drive 'https://drive.google.com/file/d/1ciKIafSQFbulL3xvQjARYPxxnvnpXGaI/view?usp=sharing' "$DOWNLOADS/checkpoints.archive"
download_drive 'https://drive.google.com/file/d/1fnm67gq_0xfTWpTG08ByKXlK4wfG39Qz/view?usp=sharing' "$DOWNLOADS/mamba.archive"
download_drive 'https://drive.google.com/file/d/1RIR4i9MyCfgnmeXrjhQOBR0bkIV-VDFT/view?usp=sharing' "$DOWNLOADS/causal-conv1d.archive"

"$PYTHON" "$ROOT/scripts/safe_extract_archive.py" "$DOWNLOADS/checkpoints.archive" "$DOWNLOADS/checkpoints"
"$PYTHON" "$ROOT/scripts/safe_extract_archive.py" "$DOWNLOADS/mamba.archive" "$DOWNLOADS/mamba"
"$PYTHON" "$ROOT/scripts/safe_extract_archive.py" "$DOWNLOADS/causal-conv1d.archive" "$DOWNLOADS/causal-conv1d"

CKPT_DIR=$("$PYTHON" - "$DOWNLOADS/checkpoints" <<'PY'
import sys
from pathlib import Path
root = Path(sys.argv[1])
required = ("model.ckpt", "model_latest.pth", "vae.pth", "model_3dcnn_crf24.pth")
for candidate in root.rglob("model_latest.pth"):
    parent = candidate.parent
    if all((parent / name).is_file() for name in required):
        print(parent.resolve())
        break
else:
    raise RuntimeError(f"Could not find the complete Safe-Sora checkpoint directory under {root}")
PY
)
if [ ! -e "$SOURCE/ckpt" ]; then
  ln -s "$CKPT_DIR" "$SOURCE/ckpt"
fi
if [ "$(cd "$SOURCE/ckpt" && pwd)" != "$CKPT_DIR" ]; then
  echo "Existing $SOURCE/ckpt does not point to the downloaded checkpoint set." >&2
  exit 1
fi

CAUSAL_SETUP=$(find "$DOWNLOADS/causal-conv1d" -type f -name setup.py -print -quit)
MAMBA_SETUP=$(find "$DOWNLOADS/mamba" -type f -name setup.py -print -quit)
if [ -z "$CAUSAL_SETUP" ] || [ -z "$MAMBA_SETUP" ]; then
  echo "Could not locate setup.py in the supplied CUDA extension archives." >&2
  exit 1
fi
MAX_JOBS=${SAFE_SORA_BUILD_JOBS:-8} "$PYTHON" -m pip install --no-build-isolation "$(dirname -- "$CAUSAL_SETUP")"
MAX_JOBS=${SAFE_SORA_BUILD_JOBS:-8} "$PYTHON" -m pip install --no-build-isolation "$(dirname -- "$MAMBA_SETUP")"
"$PYTHON" -m pip check

"$PYTHON" - "$SOURCE/fig/0_secret_combined.png" "$SOURCE/demo_watermark.png" <<'PY'
import sys
from pathlib import Path
from PIL import Image
source, output = map(Path, sys.argv[1:])
with Image.open(source) as image:
    if image.size != (768, 256):
        raise RuntimeError(f"Unexpected bundled comparison size: {image.size}")
    image.convert("RGB").crop((0, 0, 256, 256)).save(output)
print("Demonstration watermark:", output)
PY

CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" - <<'PY'
import torch
from mamba_ssm import Mamba
from custom_modules import EmbeddingNet, RevealNet
assert torch.cuda.is_available(), "The selected CUDA GPU is unavailable"
print("torch:", torch.__version__, "CUDA runtime:", torch.version.cuda)
print("Visible GPU:", torch.cuda.get_device_name(0))
print("Safe-Sora, Mamba and CUDA extension imports passed.")
PY
sha256sum "$SOURCE/ckpt/model.ckpt" "$SOURCE/ckpt/model_latest.pth" "$SOURCE/ckpt/vae.pth" "$SOURCE/ckpt/model_3dcnn_crf24.pth" > "$SOURCE/model_sha256.txt"
"$PYTHON" -m pip freeze > "$ROOT/papers/safe-sora/environment.freeze.txt"
df -h "$SOURCE"
echo "Setup complete. Run: bash $ROOT/scripts/capture_safe_sora.sh $SOURCE $GPU"
