#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

REGULAR_CONFIG="${REGULAR_CONFIG:-configs/pruning_benchmark_regular_sft.yaml}"
CONTRASTIVE_CONFIG="${CONTRASTIVE_CONFIG:-configs/pruning_benchmark_contrastive_sft.yaml}"
SKIP_REGULAR="${SKIP_REGULAR:-0}"
SKIP_CONTRASTIVE="${SKIP_CONTRASTIVE:-0}"

run_config() {
  local label="$1"
  local config_path="$2"
  echo
  echo "=== ${label} pruning benchmark ==="
  echo "config: ${config_path}"
  MODE=generic CONFIG_PATH="${config_path}" ./scripts/run_pruning_benchmark_8way.sh "${config_path}"
}

if [[ "${SKIP_REGULAR}" != "1" ]]; then
  run_config "Regular SFT" "${REGULAR_CONFIG}"
fi

if [[ "${SKIP_CONTRASTIVE}" != "1" ]]; then
  run_config "Contrastive SFT" "${CONTRASTIVE_CONFIG}"
fi

echo
echo "Done."
echo "Regular SFT summary:      runs/pruning-benchmark-regular-sft-0p2b/benchmark_summary.csv"
echo "Contrastive SFT summary:  runs/pruning-benchmark-contrastive-sft-0p2b/benchmark_summary.csv"
