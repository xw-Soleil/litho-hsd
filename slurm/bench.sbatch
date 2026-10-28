#!/bin/bash
#SBATCH -J dtco-bench
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH -o logs/%x-%j.out
set -euo pipefail
W=$HOME/work/dtco
S=/tmp/$USER/dtco
mkdir -p $S/cache
rsync -a $W/cache/ $S/cache/
export DTCO_CACHE=$S/cache DTCO_RUNS=$W/runs
cd $W/code
$W/venv/bin/python bench_cuda.py
echo BENCH_DONE
