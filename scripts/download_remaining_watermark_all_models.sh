#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=${1:-/home/smayan/research/remaining-watermark-experiments}
SOURCE_ROOT=${2:-/home/smayan/research}
DOWNLOADER="$ROOT/scripts/download_remaining_watermark_models.sh"
STATUS="$ROOT/logs/model-status.tsv"
LOCK="$ROOT/logs/model-download.lock"

exec 9>"$LOCK"
if ! flock -n 9; then
  echo "Model preparation is already running." >&2
  exit 3
fi

: > "$STATUS"
failed=0
for method in watermark-anything vine videomark vidsig genptw sigmark; do
  log="$ROOT/logs/$method-models.log"
  started=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
  if bash "$DOWNLOADER" "$ROOT" "$SOURCE_ROOT" "$method" > "$log" 2>&1; then
    state=complete
  else
    state=failed
    failed=$((failed + 1))
  fi
  finished=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
  printf '%s\t%s\t%s\t%s\t%s\n' "$method" "$state" "$started" "$finished" "$log" | tee -a "$STATUS"
done

if [ "$failed" -ne 0 ]; then
  echo "$failed method model preparation step(s) failed; inspect $STATUS" >&2
  exit 1
fi
echo "All six method model preparations passed."
