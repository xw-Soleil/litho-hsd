#!/bin/zsh
# Sync the isolated v2 experiments. Does not replace the historical results.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p runs/v2 results/v2
rsync -az --exclude '*.pt' xlab01:~/work/dtco/v2/runs/ runs/v2/
rsync -az xlab01:~/work/dtco/v2/results/ results/v2/
if [[ "${1:-}" == "--models" ]]; then
  rsync -az --include '*/' --include 'model.pt' --exclude '*' xlab01:~/work/dtco/v2/runs/ runs/v2/
fi
