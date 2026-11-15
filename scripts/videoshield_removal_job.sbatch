#!/usr/bin/env bash
#SBATCH --job-name=videoshield-remove
#SBATCH --partition=h100-plus
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=videoshield-removal-%j.log

set -Eeuo pipefail
STAGE="$HOME/videoshield-removal-stage"
OUTPUT="$HOME/videoshield-removal-${SLURM_JOB_ID}.tar.gz"
bash "$STAGE/scripts/run_videoshield_removal_slurm.sh" "$STAGE" "$OUTPUT"
