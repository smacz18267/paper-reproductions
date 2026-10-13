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

if [ -n "${SAFE_SORA_CUDA_HOME:-}" ]; then
  CUDA_HOME=$SAFE_SORA_CUDA_HOME
elif [ -n "${CUDA_HOME:-}" ] && [ -x "$CUDA_HOME/bin/nvcc" ]; then
  CUDA_HOME=$CUDA_HOME
else
  CUDA_HOME=
  for candidate in /usr/local/cuda-12.2 /usr/local/cuda-11.8 /usr/local/cuda; do
    if [ -x "$candidate/bin/nvcc" ]; then
      CUDA_HOME=$candidate
      break
    fi
  done
fi
if [ -z "$CUDA_HOME" ] || [ ! -x "$CUDA_HOME/bin/nvcc" ]; then
  echo "Safe-Sora requires a CUDA toolkit with nvcc. Set SAFE_SORA_CUDA_HOME to its directory." >&2
  exit 1
fi
export CUDA_HOME
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

CUDA_VERSION=$("$CUDA_HOME/bin/nvcc" --version | sed -n 's/.*release \([0-9][0-9.]*\).*/\1/p' | head -n 1)
case "$CUDA_VERSION" in
  12.*)
    TORCH_INDEX_URL=${SAFE_SORA_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}
    EXPECTED_TORCH_CUDA=12.1
    HOST_COMPILER_VERSIONS=(12 11 10)
    ;;
  11.8*)
    TORCH_INDEX_URL=${SAFE_SORA_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu118}
    EXPECTED_TORCH_CUDA=11.8
    HOST_COMPILER_VERSIONS=(11 10 9)
    ;;
  *)
    echo "Unsupported CUDA toolkit $CUDA_VERSION. Use CUDA 11.8 or CUDA 12.x." >&2
    exit 1
    ;;
esac

if [ -n "${SAFE_SORA_CC:-}" ] || [ -n "${SAFE_SORA_CXX:-}" ]; then
  if [ -z "${SAFE_SORA_CC:-}" ] || [ -z "${SAFE_SORA_CXX:-}" ]; then
    echo "Set both SAFE_SORA_CC and SAFE_SORA_CXX." >&2
    exit 1
  fi
  CC=$SAFE_SORA_CC
  CXX=$SAFE_SORA_CXX
else
  CC=
  CXX=
  for version in "${HOST_COMPILER_VERSIONS[@]}"; do
    if command -v "gcc-$version" >/dev/null 2>&1 && command -v "g++-$version" >/dev/null 2>&1; then
      CC=$(command -v "gcc-$version")
      CXX=$(command -v "g++-$version")
      break
    fi
  done
fi
if [ -z "$CC" ] || [ -z "$CXX" ]; then
  echo "No CUDA-compatible GCC/G++ pair found. Set SAFE_SORA_CC and SAFE_SORA_CXX." >&2
  exit 1
fi
export CC CXX
export CUDAHOSTCXX=${CUDAHOSTCXX:-$CXX}
echo "CUDA toolkit: $CUDA_VERSION at $CUDA_HOME"
echo "CUDA host compiler: $CC / $CXX"

"$PYTHON" -m pip install "pip==24.0" "setuptools==75.6.0" "wheel==0.45.1"
if ! "$PYTHON" -c "import torch; raise SystemExit(0 if torch.__version__.startswith('2.1.1') and torch.version.cuda == '$EXPECTED_TORCH_CUDA' else 1)" >/dev/null 2>&1; then
  "$PYTHON" -m pip install --force-reinstall --no-cache-dir \
    torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 \
    --index-url "$TORCH_INDEX_URL"
fi
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
if [ "$(cd "$SOURCE/ckpt" && pwd -P)" != "$CKPT_DIR" ]; then
  echo "Existing $SOURCE/ckpt does not point to the downloaded checkpoint set." >&2
  exit 1
fi

CAUSAL_SETUP=$(find "$DOWNLOADS/causal-conv1d" -type f -name setup.py -print -quit)
MAMBA_SETUP=$(find "$DOWNLOADS/mamba" -type f -name setup.py -print -quit)
if [ -z "$CAUSAL_SETUP" ] || [ -z "$MAMBA_SETUP" ]; then
  echo "Could not locate setup.py in the supplied CUDA extension archives." >&2
  exit 1
fi
CAUSAL_CONV1D_FORCE_BUILD=TRUE MAX_JOBS=${SAFE_SORA_BUILD_JOBS:-4} \
  "$PYTHON" -m pip install --force-reinstall --no-build-isolation --no-deps --no-cache-dir \
  "$(dirname -- "$CAUSAL_SETUP")"
MAMBA_FORCE_BUILD=TRUE MAX_JOBS=${SAFE_SORA_BUILD_JOBS:-4} \
  "$PYTHON" -m pip install --force-reinstall --no-build-isolation --no-deps --no-cache-dir \
  "$(dirname -- "$MAMBA_SETUP")"
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
import inspect
import numpy
import torch
import causal_conv1d_cuda
import selective_scan_cuda
from mamba_ssm import Mamba
from custom_modules import EmbeddingNet, RevealNet
assert int(numpy.__version__.split(".", 1)[0]) < 2, "Safe-Sora requires NumPy 1.x"
assert "bimamba_type" in inspect.signature(Mamba.__init__).parameters, "The supplied modified Mamba was not installed"
assert torch.cuda.is_available(), "The selected CUDA GPU is unavailable"
print("torch:", torch.__version__, "CUDA runtime:", torch.version.cuda)
print("Visible GPU:", torch.cuda.get_device_name(0))
print("Safe-Sora, Mamba and CUDA extension imports passed.")
PY
sha256sum "$SOURCE/ckpt/model.ckpt" "$SOURCE/ckpt/model_latest.pth" "$SOURCE/ckpt/vae.pth" "$SOURCE/ckpt/model_3dcnn_crf24.pth" > "$SOURCE/model_sha256.txt"
"$PYTHON" -m pip freeze > "$ROOT/papers/safe-sora/environment.freeze.txt"
df -h "$SOURCE"
echo "Setup complete. Run: bash $ROOT/scripts/capture_safe_sora.sh $SOURCE $GPU"
