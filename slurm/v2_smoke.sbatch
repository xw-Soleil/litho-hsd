#!/bin/bash
#SBATCH -J dtco-v2-smoke
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH -o logs/%x-%j.out
set -euo pipefail
W="$HOME/work/dtco"
V="$W/v2"
export DTCO_CACHE="$W/cache" DTCO_RUNS="$V/smoke"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PYTHONUNBUFFERED=1
cd "$V/code"
for mode in binary proto sup3; do
  "$W/venv/bin/python" train_v2.py --mode "$mode" --rep densm --seed 0 \
    --protocol "$V/protocol/smoke.csv" --steps 2 --tta 1 --checkpoint-every 1 --inference-bs 128
done
echo SMOKE_COMPLETE
