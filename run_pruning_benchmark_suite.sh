#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

CONFIG_PATH="${CONFIG_PATH:-configs/pruning_benchmark.yaml}"

python scripts/run_pruning_benchmark.py --config "${CONFIG_PATH}"
