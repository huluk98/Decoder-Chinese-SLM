#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=WARN
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8

CONFIG_PATH="${CONFIG_PATH:-configs/qwen25_instruct_pruning_benchmark.yaml}"

python scripts/run_qwen25_instruct_pruning_benchmark.py \
  --config "${CONFIG_PATH}"
