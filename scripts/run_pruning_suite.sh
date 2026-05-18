#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/prune_50.yaml}"
CHECKPOINT="${CHECKPOINT:-runs/contrastive-sft-0p2b/latest}"

python scripts/prune.py --config "${CONFIG}" --method magnitude --checkpoint "${CHECKPOINT}" --output-dir runs/pruned-magnitude-50
python scripts/prune.py --config "${CONFIG}" --method 2of4 --checkpoint "${CHECKPOINT}" --output-dir runs/pruned-nvidia-2of4-50
python scripts/prune.py --config "${CONFIG}" --method wanda --checkpoint "${CHECKPOINT}" --output-dir runs/pruned-wanda-50
python scripts/prune.py --config "${CONFIG}" --method gradient --checkpoint "${CHECKPOINT}" --output-dir runs/pruned-gradient-50
