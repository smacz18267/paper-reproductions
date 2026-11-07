#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=${1:-/home/smayan/research/remaining-watermark-experiments}
SOURCE_ROOT=${2:-/home/smayan/research}
RUNNER="$ROOT/scripts/run_remaining_watermark_direct.sh"
STATUS="$ROOT/logs/experiment-status.tsv"
LOCK="$ROOT/logs/experiment-manager.lock"

exec 9>"$LOCK"
if ! flock -n 9; then
  echo "Experiment manager is already running." >&2
  exit 3
fi

: > "$STATUS"
failed=0
for method in watermark-anything vine videomark vidsig genptw sigmark; do
  log="$ROOT/logs/$method-experiment.log"
  started=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
  if bash "$RUNNER" "$ROOT" "$SOURCE_ROOT" "$method" > "$log" 2>&1; then
    state=complete
  else
    state=failed
    failed=$((failed + 1))
  fi
  finished=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
  printf '%s\t%s\t%s\t%s\t%s\n' "$method" "$state" "$started" "$finished" "$log" | tee -a "$STATUS"
done

if [ "$failed" -ne 0 ]; then
  echo "$failed method experiment(s) failed; inspect $STATUS" >&2
  exit 1
fi
echo "All six watermark-removal experiments completed."
