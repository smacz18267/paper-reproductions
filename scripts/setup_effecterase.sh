#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
SOURCE=${1:-"$HOME/research/EffectErase"}
GPU=${2:-1}
PIN=$(tr -d '\r\n' < "$ROOT/papers/effecterase/upstream_commit.txt")
CONSTRAINTS="$ROOT/papers/effecterase/environment-constraints.txt"

command -v nvidia-smi >/dev/null
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv
mkdir -p -- "$(dirname -- "$SOURCE")"
if [ ! -e "$SOURCE" ]; then
  git clone --no-checkout https://github.com/FudanCVL/EffectErase.git "$SOURCE"
  git -C "$SOURCE" checkout --detach "$PIN"
fi
SOURCE=$(cd -- "$SOURCE" && pwd)
if [ ! -d "$SOURCE/.git" ] || [ "$(git -C "$SOURCE" rev-parse HEAD)" != "$PIN" ]; then
  echo "Expected the pinned EffectErase checkout at $SOURCE. Use a new source directory if it already contains other work." >&2
  exit 1
fi
if [ -n "$(git -C "$SOURCE" status --porcelain --untracked-files=no)" ]; then
  echo "EffectErase has local source changes. Keep those changes and use a new directory for setup." >&2
  exit 1
fi

SNAPSHOT="$ROOT/papers/effecterase/source"
if [ ! -e "$SNAPSHOT" ]; then
  mkdir -p "$SNAPSHOT"
  git -C "$SOURCE" archive "$PIN" diffsynth examples/remove_wan script demo/FG_BG demo/MASK README.md LICENSE requirements.txt setup.py | tar -x -C "$SNAPSHOT"
fi
if [ ! -f "$SNAPSHOT/examples/remove_wan/infer_remove_wan.py" ]; then
  echo "Incomplete source snapshot: $SNAPSHOT" >&2
  exit 1
fi

cd "$SOURCE"
if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv
fi
PYTHON="$SOURCE/.venv/bin/python"
"$PYTHON" -m pip install --upgrade pip setuptools wheel
"$PYTHON" -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
"$PYTHON" -m pip install -c "$CONSTRAINTS" -e . 'huggingface-hub>=0.34,<1'
"$PYTHON" -m pip check
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" - <<'PY'
import torch, torchvision
from diffsynth import ModelManager, WanRemovePipeline
assert torch.cuda.is_available(), "CUDA is unavailable in this environment"
assert torch.cuda.is_bf16_supported(), "EffectErase requires BF16 support"
print("torch:", torch.__version__, "torchvision:", torchvision.__version__)
print("CUDA runtime:", torch.version.cuda, "visible GPU:", torch.cuda.get_device_name(0))
print("EffectErase imports and CUDA checks passed.")
PY

df -h "$SOURCE"
echo "Downloading the Wan base model (about 20 GB) and EffectErase checkpoint. Downloads resume when rerun."
"$PYTHON" - <<'PY'
import json
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download, hf_hub_download
api = HfApi()
manifest_path = Path("model_downloads.json")
if manifest_path.exists():
    manifest = json.loads(manifest_path.read_text())
else:
    manifest = {
        "base_model": {"repo_id": "alibaba-pai/Wan2.1-Fun-1.3B-InP"},
        "effecterase": {"repo_id": "FudanCVL/EffectErase"},
    }
    for entry in manifest.values():
        entry["revision"] = api.model_info(entry["repo_id"]).sha
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
base = manifest["base_model"]
snapshot_download(base["repo_id"], revision=base["revision"], local_dir="Wan-AI/Wan2.1-Fun-1.3B-InP")
effect = manifest["effecterase"]
hf_hub_download(effect["repo_id"], "EffectErase.ckpt", revision=effect["revision"], local_dir=".")
PY
"$PYTHON" -m pip freeze > "$ROOT/papers/effecterase/environment.freeze.txt"
echo "Setup complete. Run: bash $ROOT/scripts/capture_effecterase.sh $SOURCE $GPU"
