#!/bin/bash
#SBATCH -J dtco-v2
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:50:00
#SBATCH -o logs/%x-%j.out
# Submit from ~/work/dtco/v2. Each experiment has its own resumable checkpoint.
set -euo pipefail
W="$HOME/work/dtco"
V="$W/v2"
S="${SLURM_TMPDIR:-/tmp/$USER/dtco-v2-$SLURM_JOB_ID}"
P="$W/venv/bin/python"
mkdir -p "$S/cache" "$V/runs" "$V/logs" "$V/results"
rsync -a "$W/cache/" "$S/cache/"
export DTCO_CACHE="$S/cache" DTCO_RUNS="$V/runs"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export PYTHONUNBUFFERED=1 MPLCONFIGDIR="$S/matplotlib"
cd "$V/code"
run() {
    local mode="$1" rep="$2" seed="$3" holdout="${4:-}"
    local name="${mode}_${rep}_s${seed}"
    local args=(--mode "$mode" --rep "$rep" --seed "$seed" --protocol "$V/protocol/manifest.csv")
    if [[ -n "$holdout" ]]; then
        name="${name}_ho${holdout}"
        args+=(--holdout "$holdout")
    fi
    echo "START $name $(date -Is)"
    "$P" train_v2.py "${args[@]}" > "$V/logs/$name.log" 2>&1
    tail -3 "$V/logs/$name.log"
    echo "END $name $(date -Is)"
}
for seed in 0 1 2; do
    for mode in binary proto sup3; do run "$mode" densm "$seed"; done
done
for seed in 0 1 2; do
    for mode in binary proto; do run "$mode" densm "$seed" iccad1; done
done
for rep in dens squish; do
    for mode in binary proto sup3; do run "$mode" "$rep" 0; done
done
echo "ALL_TRAINING_COMPLETE $(date -Is)"
