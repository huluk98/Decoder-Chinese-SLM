#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

CONFIG_PATH="${CONFIG_PATH:-configs/qwen25_instruct_pruning_benchmark.yaml}"
PYTHON_BIN="${PYTHON:-python}"
KEEP_GOING="${KEEP_GOING:-1}"

ARGS=(--config "${CONFIG_PATH}")
if [[ "${KEEP_GOING}" == "1" ]]; then
  ARGS+=(--continue-on-error)
else
  ARGS+=(--stop-on-error)
fi

echo "Qwen2.5-Instruct 8-GPU pruning benchmark"
echo "  config: ${CONFIG_PATH}"
echo "  CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "  keep going after method failure: ${KEEP_GOING}"

"${PYTHON_BIN}" scripts/run_qwen25_instruct_pruning_benchmark.py \
  "${ARGS[@]}"
