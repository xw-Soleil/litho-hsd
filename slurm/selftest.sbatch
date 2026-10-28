#!/bin/bash
#SBATCH -J dtco-selftest
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:15:00
#SBATCH -o logs/%x-%j.out
set -euo pipefail
W=$HOME/work/dtco
S=/tmp/$USER/dtco
mkdir -p $S/cache
until $W/venv/bin/python -c "import torch, matplotlib" 2>/dev/null; do echo "$(date) waiting for torch..."; sleep 20; done
rsync -a $W/cache/ $S/cache/
export DTCO_CACHE=$S/cache DTCO_RUNS=$W/runs
cd $W/code
$W/venv/bin/python -c "import torch;print(torch.__version__, torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
$W/venv/bin/python selftest.py
echo SELFTEST_DONE
