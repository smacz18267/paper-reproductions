#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
SOURCE=${1:-"$HOME/research/ProPainter"}
TARGET="$ROOT/papers/propainter/source"

if [ ! -f "$SOURCE/inference_propainter.py" ]; then
  echo "ProPainter checkout not found: $SOURCE" >&2
  exit 1
fi
if [ -e "$TARGET" ]; then
  echo "Source snapshot already exists: $TARGET. No files overwritten." >&2
  exit 1
fi
if ! git -C "$SOURCE" diff --quiet HEAD --; then
  echo "The original checkout has tracked changes. Record those changes before importing a committed snapshot." >&2
  exit 1
fi
ARCHIVE=$(mktemp)
trap 'rm -f "$ARCHIVE"' EXIT
git -C "$SOURCE" archive --format=tar -o "$ARCHIVE" HEAD
mkdir -p "$TARGET"
tar -xf "$ARCHIVE" -C "$TARGET"
git -C "$SOURCE" rev-parse HEAD > "$ROOT/papers/propainter/upstream_commit.txt"
if [ -x "$SOURCE/.venv/bin/python" ]; then
  "$SOURCE/.venv/bin/python" -m pip freeze > "$ROOT/papers/propainter/environment.freeze.txt"
fi
echo "Imported committed source and its original license to $TARGET"
echo "The running environment and downloaded weights stay in the original checkout."
