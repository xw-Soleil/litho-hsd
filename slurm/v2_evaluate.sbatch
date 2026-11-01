#!/bin/bash
#SBATCH -J dtco-v2-eval
#SBATCH -p gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH -o logs/%x-%j.out
# No GPU requested: evaluation reads saved predictions and embeddings only.
set -euo pipefail
W="$HOME/work/dtco"
V="$W/v2"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PYTHONUNBUFFERED=1
export MPLCONFIGDIR="/tmp/$USER/dtco-v2-matplotlib-$SLURM_JOB_ID"
P="$W/venv/bin/python"
"$P" "$V/code/evaluate_v2.py" --protocol "$V/protocol/manifest.csv" \
  --cache "$W/cache" --runs "$V/runs" --out "$V/results"
"$P" "$V/code/report_v2.py" --out "$V/results"
echo "EVALUATION_COMPLETE $(date -Is)"
