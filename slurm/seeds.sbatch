#!/bin/bash
#SBATCH -J dtco-seeds
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH -o logs/%x-%j.out
# Sequential experiment list (student QOS: 1 running job, 4 h wall).
# Each step is skipped if its output already exists -> just resubmit to resume.
set -uo pipefail
W=$HOME/work/dtco
S=/tmp/$USER/dtco
mkdir -p $S/cache $W/runs
rsync -a $W/cache/ $S/cache/
export DTCO_CACHE=$S/cache DTCO_RUNS=$W/runs
cd $W/code
P=$W/venv/bin/python
run() { local marker=$1; shift
  if [ -f $W/runs/$marker ]; then echo "skip $marker"; return; fi
  echo "=== $(date '+%H:%M:%S') START $*"
  local log=$W/logs/run_$(dirname $marker).log
  "$@" > $log 2>&1; local rc=$?
  grep -v -i warning $log | tail -3
  echo "=== $(date '+%H:%M:%S') END exit=$rc $marker"; }
PRO="--steps 1500 --episodes 6"

# training-seed robustness of the cross-node (28nm -> 32nm) result
for SEED in 1 2; do
  run binary_dens_s${SEED}_hoiccad1/pred.npz $P train_binary.py --rep dens --seed $SEED --holdout iccad1
  run proto_dens_s${SEED}_hoiccad1/emb.npz   $P train_proto.py  --rep dens --seed $SEED --holdout iccad1 $PRO
done
echo SEEDS_DONE
