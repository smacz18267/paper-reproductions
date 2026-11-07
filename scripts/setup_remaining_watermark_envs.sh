#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=${1:-/home/smayan/research/remaining-watermark-experiments}
SOURCE_ROOT=${2:-/home/smayan/research}
METHOD=${3:?Pass one of: watermark-anything, vine, videomark, sigmark, vidsig, genptw}

export PIP_CACHE_DIR=${PIP_CACHE_DIR:-$SOURCE_ROOT/.cache/pip}
export HF_HOME=${HF_HOME:-$SOURCE_ROOT/.cache/huggingface}
mkdir -p "$PIP_CACHE_DIR" "$HF_HOME" "$ROOT/logs"

make_venv() {
  local repository=$1
  if [ ! -x "$repository/.venv/bin/python" ]; then
    python3.12 -m venv "$repository/.venv"
  fi
  "$repository/.venv/bin/python" -m pip install --upgrade pip setuptools wheel
}

install_torch() {
  local python=$1
  local torch_version=$2
  local vision_version=$3
  local cuda_index=$4
  "$python" -m pip install \
    "torch==$torch_version" "torchvision==$vision_version" \
    --index-url "https://download.pytorch.org/whl/$cuda_index"
}

validate_cuda_runtime() {
  local python=$1
  "$python" - <<'PY'
import torch, torchvision
print("torch", torch.__version__, "torchvision", torchvision.__version__)
print("CUDA runtime", torch.version.cuda, "CUDA visible", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable in the method environment")
PY
}

setup_wam() {
  local repository="$SOURCE_ROOT/watermark-anything"
  make_venv "$repository"
  local python="$repository/.venv/bin/python"
  install_torch "$python" 2.5.1 0.20.1 cu124
  "$python" -m pip install -r "$repository/requirements.txt" av==14.0.1
  mkdir -p "$repository/checkpoints"
  if [ ! -s "$repository/checkpoints/wam_mit.pth" ]; then
    curl -fL --retry 5 \
      https://dl.fbaipublicfiles.com/watermark_anything/wam_mit.pth \
      -o "$repository/checkpoints/wam_mit.pth"
  fi
  test -s "$repository/checkpoints/params.json"
  (cd "$repository" && "$python" - <<'PY'
from pathlib import Path
from notebooks.inference_utils import load_model_from_checkpoint
model = load_model_from_checkpoint("checkpoints/params.json", "checkpoints/wam_mit.pth")
print(type(model).__name__, Path("checkpoints/wam_mit.pth").stat().st_size)
PY
  )
  validate_cuda_runtime "$python"
  "$python" -m pip check
}

setup_vine() {
  local repository="$SOURCE_ROOT/VINE"
  make_venv "$repository"
  local python="$repository/.venv/bin/python"
  install_torch "$python" 2.5.1 0.20.1 cu124
  "$python" -m pip install \
    accelerate==0.30.0 transformers==4.43.3 peft==0.11.1 \
    huggingface-hub==0.23.4 safetensors==0.4.5 tokenizers==0.19.1 \
    einops==0.8.0 numpy==1.26.4 open-clip-torch==2.26.1 \
    opencv-python==4.10.0.84 pillow==10.4.0 scipy==1.11.4 \
    timm==1.0.11 tqdm==4.67.1 sentencepiece==0.2.0 av==14.0.1 \
    ftfy regex requests filelock packaging importlib-metadata
  "$python" -m pip install -e "$repository/diffusers" --no-deps
  # The top-level VINE project metadata pins its complete training/demo stack
  # (including Python-3.10-era Torch and optional UI packages).  Installing it
  # with --no-deps makes `pip check` report those unused extras as missing.  The
  # released inference code is imported directly from this pinned checkout,
  # while its bundled Diffusers fork is installed normally above.
  "$python" -m pip uninstall -y vine >/dev/null 2>&1 || true
  "$python" - <<PY
import sys
sys.path.insert(0, "$repository")
from vine.src.vine_turbo import VINE_Turbo
from vine.src.stega_encoder_decoder import CustomConvNeXt
print(VINE_Turbo.__name__, CustomConvNeXt.__name__)
PY
  validate_cuda_runtime "$python"
  "$python" -m pip check
}

setup_videomark() {
  local repository="$SOURCE_ROOT/VideoMark"
  make_venv "$repository"
  local python="$repository/.venv/bin/python"
  install_torch "$python" 2.5.1 0.20.1 cu124
  "$python" -m pip install \
    diffusers==0.30.3 transformers==4.49.0 accelerate==1.3.0 \
    huggingface-hub==0.28.1 safetensors==0.5.2 \
    numpy==1.26.4 scipy==1.11.3 opencv-python-headless==4.11.0.86 \
    av==14.2.0 galois==0.4.4 ldpc==2.2.8 \
    python-Levenshtein==0.27.1 pycryptodome==3.20.0 tqdm==4.67.1
  (cd "$repository" && "$python" - <<'PY'
from diffusers import TextToVideoSDPipeline
from diffusers.schedulers import DDIMInverseScheduler
from src.prc import Decode, Detect
print(TextToVideoSDPipeline.__name__, DDIMInverseScheduler.__name__, Decode.__name__, Detect.__name__)
PY
  )
  validate_cuda_runtime "$python"
  "$python" -m pip check
}

setup_sigmark() {
  local repository="$SOURCE_ROOT/SIGMark-release"
  make_venv "$repository"
  local python="$repository/.venv/bin/python"
  # The release defaults every configuration to an image-prompt directory,
  # although its HunyuanVideo T2V configurations do not consume image prompts.
  # Make the release's already-supported None path the default. I2V callers can
  # still pass --image_prompt_dir explicitly.
  "$python" - "$repository/main.py" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text()
old = 'parser.add_argument("--image_prompt_dir", type=str, default="./prompt_set/VBench2_aug_img_prompt")'
new = 'parser.add_argument("--image_prompt_dir", type=str, default=None)'
if old in text:
    path.write_text(text.replace(old, new, 1))
elif new not in text:
    raise SystemExit("SIGMark image-prompt default no longer matches the validated release")
PY
  install_torch "$python" 2.5.1 0.20.1 cu124
  # The release specifies Torch 2.5 but gives open-ended lower bounds for its
  # Hugging Face stack.  Diffusers 0.42+ now calls torch.accelerator, an API
  # absent from Torch 2.5.  These versions satisfy the release bounds and are
  # validated together with the Hunyuan pipeline and quantization class below.
  "$python" -m pip install \
    diffusers==0.38.0 transformers==4.52.4 accelerate==1.10.1 \
    bitsandbytes==0.47.0 huggingface-hub==0.36.0 tokenizers==0.21.4 \
    safetensors==0.8.0 \
    'opencv-python-headless>=4.12' 'imageio>=2.37.0' 'imageio-ffmpeg>=0.6.0' \
    'scipy>=1.15' 'pycryptodome>=3.23' 'galois>=0.4.4' 'ldpc>=2.2.8' \
    'Levenshtein>=0.27.1' av==14.2.0 sentencepiece==0.2.0 protobuf==5.29.5
  (cd "$repository" && "$python" - <<'PY'
from diffusers import HunyuanVideoPipeline
from diffusers.quantizers import PipelineQuantizationConfig
from watermarks import SIGMarkWatermark
print(HunyuanVideoPipeline.__name__, PipelineQuantizationConfig.__name__, SIGMarkWatermark.__name__)
PY
  )
  validate_cuda_runtime "$python"
  "$python" -m pip check
}

setup_vidsig() {
  local repository="$SOURCE_ROOT/Video-Signature"
  make_venv "$repository"
  local python="$repository/.venv/bin/python"
  install_torch "$python" 2.7.1 0.22.1 cu128
  # The release pins NumPy 2.3.2 and OpenCV 4.12 together, but OpenCV 4.12
  # requires NumPy <2.3 on Python 3.12.  Keep the authors' remaining pins and
  # use the newest NumPy branch that satisfies their OpenCV wheel.
  grep -Ev '^(torch|torchvision|numpy)==' "$repository/requirements.txt" > "$ROOT/logs/vidsig-requirements-py312.txt"
  "$python" -m pip install -r "$ROOT/logs/vidsig-requirements-py312.txt" numpy==2.2.6
  "$python" -m pip install \
    transformers==4.52.4 accelerate==1.8.1 safetensors==0.5.3 \
    imageio-ffmpeg==0.6.0 av==14.2.0 gdown==5.2.0
  (cd "$repository" && "$python" - <<'PY'
from diffusers import TextToVideoSDPipeline
from src.utils import data_utils
print(TextToVideoSDPipeline.__name__, data_utils.__name__)
PY
  )
  validate_cuda_runtime "$python"
  "$python" -m pip check
}

setup_genptw() {
  local repository="$SOURCE_ROOT/GenPTW"
  make_venv "$repository"
  local python="$repository/.venv/bin/python"
  install_torch "$python" 2.7.0 0.22.0 cu128
  "$python" -m pip install \
    diffusers==0.38.0 transformers==5.10.2 accelerate==1.13.0 \
    huggingface-hub==1.8.0 safetensors==0.8.0 \
    timm==1.0.27 kornia==0.8.3 einops==0.8.2 \
    opencv-python==4.11.0.86 pillow==10.4.0 numpy==2.2.6 \
    scipy==1.15.3 scikit-image==0.25.2 scikit-learn==1.6.1 \
    av==14.2.0 gdown==5.2.0
  (cd "$repository" && "$python" - <<'PY'
from aaai_final_adapter import Decoder, inject_wmadapter
from Localizer.model import Localizer
print(Decoder.__name__, inject_wmadapter.__name__, Localizer.__name__)
PY
  )
  validate_cuda_runtime "$python"
  "$python" -m pip check
}

case "$METHOD" in
  watermark-anything) setup_wam ;;
  vine) setup_vine ;;
  videomark) setup_videomark ;;
  sigmark) setup_sigmark ;;
  vidsig) setup_vidsig ;;
  genptw) setup_genptw ;;
  *) echo "Unknown method: $METHOD" >&2; exit 2 ;;
esac

date -u +'%Y-%m-%dT%H:%M:%SZ' > "$ROOT/logs/$METHOD-environment.ready"
echo "$METHOD environment setup and validation passed."
