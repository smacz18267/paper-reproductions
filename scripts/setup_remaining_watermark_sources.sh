#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=${1:-/home/smayan/research/remaining-watermark-experiments}
SOURCE_ROOT=${2:-/home/smayan/research}
LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR"

clone_exact() {
  local name=$1
  local url=$2
  local commit=$3
  local destination="$SOURCE_ROOT/$name"
  local created=0

  if [ ! -e "$destination" ]; then
    git clone --filter=blob:none --no-checkout "$url" "$destination"
    created=1
  fi
  if [ ! -d "$destination/.git" ]; then
    echo "$destination exists but is not a Git checkout" >&2
    return 1
  fi
  if [ "$created" -eq 0 ] && [ -n "$(git -C "$destination" status --porcelain --untracked-files=no)" ]; then
    echo "$destination has tracked local changes; refusing to overwrite them" >&2
    return 1
  fi
  if ! git -C "$destination" cat-file -e "$commit^{commit}" 2>/dev/null; then
    git -C "$destination" fetch --depth=1 origin "$commit"
  fi
  if [ "$created" -eq 1 ]; then
    git -C "$destination" checkout --detach --force "$commit"
  else
    git -C "$destination" checkout --detach "$commit"
  fi
  local actual
  actual=$(git -C "$destination" rev-parse HEAD)
  if [ "$actual" != "$commit" ]; then
    echo "$name checkout mismatch: expected $commit, found $actual" >&2
    return 1
  fi
  printf '%s\t%s\t%s\n' "$name" "$actual" "$destination"
}

{
  clone_exact watermark-anything https://github.com/facebookresearch/watermark-anything.git 2c08af04d037d5667c02f6ddebbda9ff04581c3e
  clone_exact VINE https://github.com/Shilin-LU/VINE.git d72f11a93f8de9bc45ef61202b9c8dda2a70cc77
  clone_exact VideoMark https://github.com/KYRIE-LI11/VideoMark.git 9f8d78b73ab9f9f055651b1b4f37d68bdb05e7be
  clone_exact SIGMark-release https://github.com/JeremyZhao1998/SIGMark-release.git 3713243f2e002cb21bbc3da896094d39b2b528c9
  clone_exact Video-Signature https://github.com/hardenyu21/Video-Signature.git 9d219c47670c30fdaaccd17b24e7783c976a8c66
  clone_exact GenPTW https://github.com/GanZhenliang/GenPTW.git 8c22fc04dcf9d896eaeabd034d09918588e82cb3
} | tee "$LOG_DIR/source-checkouts.tsv"

test "$(wc -l < "$LOG_DIR/source-checkouts.tsv")" -eq 6
echo "Verified six pinned watermark source checkouts."
