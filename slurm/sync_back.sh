#!/bin/zsh
# Pull finished runs (predictions / embeddings / logs; model weights optional) from the cluster into local runs/.
# usage: slurm/sync_back.sh [--models]
set -euo pipefail
cd "$(dirname "$0")/.."
EXCL=(--exclude '*.pt')
[[ "${1:-}" == "--models" ]] && EXCL=()
rsync -avz --progress "${EXCL[@]}" xlab01:~/work/dtco/runs/ runs/
rsync -avz xlab01:~/work/dtco/logs/ runs/cluster_logs/
ls runs
