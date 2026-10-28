#!/bin/bash
#SBATCH -J dtco-prep
#SBATCH -p gpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH -o logs/%x-%j.out
# unzip on node-local NVMe, build the compact cache there, copy the cache files back to shared home
set -euo pipefail
W=$HOME/work/dtco
S=/tmp/$USER/dtco
mkdir -p $S $W/cache $W/results
until $W/venv/bin/python -c "import numpy, PIL, pandas" 2>/dev/null; do echo "$(date) waiting for venv packages..."; sleep 20; done
cd $S
[ -d data/iccad-official ] || unzip -q -o $W/iccad-official.zip -x '__MACOSX/*' '*.DS_Store' -d data
echo "files: $(find data -type f | wc -l)"
$W/venv/bin/python -I $W/code/preprocess.py $S/data/iccad-official $S/cache
$W/venv/bin/python -I $W/code/prep_masked.py $S/data/iccad-official $S/cache
cp $S/cache/* $W/cache/
md5sum $W/cache/meta.csv
$W/venv/bin/python -I $W/code/audit.py $W/cache $W/results
echo PREP_DONE
