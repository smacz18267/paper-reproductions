#!/usr/bin/env bash
# Validate an existing successful inference without rerunning either model.
set -Eeuo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
SOURCE=${1:-"$HOME/research/DiffuEraser"}
if [ "$#" -lt 2 ]; then
  echo "Usage: bash scripts/finalize_diffueraser.sh SOURCE EXISTING_RUN_DIRECTORY" >&2
  exit 1
fi
SOURCE=$(cd -- "$SOURCE" && pwd)
RUN=$(cd -- "$2" && pwd)
PYTHON="$SOURCE/.venv/bin/python"
case "$RUN" in
  "$ROOT/evidence/diffueraser/recorded/"*) ;;
  *) echo "Expected an existing DiffuEraser evidence directory in this repository." >&2; exit 1 ;;
esac
if [ ! -x "$PYTHON" ] || [ ! -f "$RUN/inference_exit_code.txt" ] || [ "$(tr -d '\r\n' < "$RUN/inference_exit_code.txt")" != 0 ]; then
  echo "Finalization requires a saved inference exit code of 0 and the original environment." >&2
  exit 1
fi
for name in input_video.mp4 input_mask.mp4 diffueraser_result.mp4 propainter_prior.mp4 model_downloads.json; do
  if [ ! -s "$RUN/$name" ]; then echo "Missing required inference artifact: $RUN/$name" >&2; exit 1; fi
done
ATTEMPT="$RUN/validation_attempts/$(date -u +%Y%m%dT%H%M%SZ)-$$"
mkdir -p "$ATTEMPT"
# Keep the original failure status and validation log before any regeneration.
for name in exit_code.txt finished_at.txt artifact_checks.log SUCCESS.txt; do
  if [ -f "$RUN/$name" ]; then cp "$RUN/$name" "$ATTEMPT/previous_$name"; fi
done
on_error() {
  local code=$1
  trap - ERR INT TERM
  printf '%s\n' "$code" > "$ATTEMPT/exit_code.txt"
  date -u +%Y-%m-%dT%H:%M:%SZ > "$ATTEMPT/finished_at.txt"
  echo "Validation failed (exit $code). Existing videos were retained. Details: $ATTEMPT" >&2
  exit "$code"
}
trap 'on_error $?' ERR
trap 'on_error 130' INT
trap 'on_error 143' TERM
date -u +%Y-%m-%dT%H:%M:%SZ > "$ATTEMPT/started_at.txt"
git -C "$ROOT" rev-parse HEAD > "$ATTEMPT/reproduction_commit.txt"
git -C "$ROOT" diff --binary HEAD -- scripts > "$ATTEMPT/reproduction_changes.patch"
cp "$ROOT/scripts/diffueraser_artifacts.py" "$ATTEMPT/validator.py"
command=("$PYTHON" -u "$ROOT/scripts/diffueraser_artifacts.py" "$RUN" --source "$SOURCE")
printf '%q ' "${command[@]}" > "$ATTEMPT/command.sh"
printf '\n' >> "$ATTEMPT/command.sh"
"$PYTHON" - "$SOURCE" "$RUN" <<'PY'
import json, sys
from pathlib import Path
source, run = map(Path, sys.argv[1:])
if json.loads((source / 'model_downloads.json').read_text()) != json.loads((run / 'model_downloads.json').read_text()):
    raise RuntimeError('The source model manifest differs from the original run; preserve its original model revisions')
PY
"${command[@]}" 2>&1 | tee "$ATTEMPT/validation.log"
cp "$ATTEMPT/validation.log" "$RUN/artifact_checks.log"
date -u +%Y-%m-%dT%H:%M:%SZ > "$ATTEMPT/finished_at.txt"
printf '0\n' > "$ATTEMPT/exit_code.txt"
cp "$ATTEMPT/finished_at.txt" "$RUN/finished_at.txt"
printf '0\n' > "$RUN/exit_code.txt"
printf 'Inference finished; existing final output and preserved ProPainter prior decoded and verified.\n' > "$RUN/SUCCESS.txt"
echo "Existing DiffuEraser run finalized. Output: $RUN/diffueraser_result.mp4"
echo "Comparison: $RUN/comparison.png"
