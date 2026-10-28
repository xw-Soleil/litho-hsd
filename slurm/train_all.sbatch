#!/bin/bash
#SBATCH -J dtco-train
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:55:00
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

# 1. AMP sanity check + 3rd ensemble member
run binary_dens_s2/pred.npz        $P train_binary.py --rep dens --seed 2
# 2. does rendering the train clips like the test clips (88px black border) help?
run binary_densm_s0/pred.npz       $P train_binary.py --rep densm --seed 0
# 3. leave-one-design-out (28nm designs): conventional CNN vs meta-learned ProtoNet
for B in iccad2 iccad3 iccad4 iccad5; do
  run binary_dens_s0_ho$B/pred.npz $P train_binary.py --rep dens --seed 0 --holdout $B
  run proto_dens_s0_ho$B/emb.npz   $P train_proto.py  --rep dens --seed 0 --holdout $B $PRO
done
# 4. seed variance for scheme 1, and the masked rendering for scheme 1
run proto_dens_s1/emb.npz          $P train_proto.py --rep dens --seed 1 $PRO
run proto_densm_s0/emb.npz         $P train_proto.py --rep densm --seed 0 $PRO
# 5. representation ablation for scheme 1 / option A
run proto_squish_s0/emb.npz        $P train_proto.py --rep squish --seed 0 $PRO
run sup3_squish_s0/pred.npz        $P train_sup3.py  --rep squish --seed 0
run binary_densm_s1/pred.npz       $P train_binary.py --rep densm --seed 1
echo TRAIN_DONE
