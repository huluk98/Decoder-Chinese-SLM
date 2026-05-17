#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/h20_7gpu_llama_0p2b_deepspeed.yaml}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-configs/accelerate_h20_7gpu.yaml}"
LAUNCHER="${LAUNCHER:-accelerate}"
NPROC=7

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,2,3,4,5,6,7}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

case "${LAUNCHER}" in
  accelerate)
    exec accelerate launch --config_file "${ACCELERATE_CONFIG}" scripts/train.py --config "${CONFIG}" "$@"
    ;;
  torchrun)
    exec torchrun --standalone --nproc_per_node="${NPROC}" scripts/train.py --config "${CONFIG}" "$@"
    ;;
  deepspeed)
    exec deepspeed --num_gpus="${NPROC}" scripts/train.py --config "${CONFIG}" "$@"
    ;;
  *)
    echo "Unknown LAUNCHER=${LAUNCHER}. Use accelerate, torchrun, or deepspeed." >&2
    exit 2
    ;;
esac
