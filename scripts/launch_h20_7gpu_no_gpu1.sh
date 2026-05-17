#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/h20_7gpu_llama_0p2b_deepspeed.yaml}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-configs/accelerate_h20_7gpu.yaml}"
LAUNCHER="${LAUNCHER:-accelerate}"
PYTHON="${PYTHON:-python}"
PACK_TOKENS="${PACK_TOKENS:-auto}"
NPROC=7

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,2,3,4,5,6,7}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

count_visible_gpus() {
  local devices="${CUDA_VISIBLE_DEVICES}"
  local count=0
  local IFS=','
  read -ra ids <<< "${devices}"
  for id in "${ids[@]}"; do
    id="${id//[[:space:]]/}"
    if [[ -n "${id}" ]]; then
      count=$((count + 1))
    fi
  done
  echo "${count}"
}

VISIBLE_GPU_COUNT="$(count_visible_gpus)"
if [[ "${VISIBLE_GPU_COUNT}" -ne "${NPROC}" ]]; then
  echo "Expected ${NPROC} visible GPUs for the 7-GPU H20 launcher, got ${VISIBLE_GPU_COUNT}: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" >&2
  exit 2
fi

if [[ "${LAUNCHER}" == "accelerate" ]]; then
  ACCELERATE_NPROC="$(awk -F: '/^num_processes:/ {gsub(/[[:space:]]/, "", $2); print $2}' "${ACCELERATE_CONFIG}")"
  if [[ "${ACCELERATE_NPROC}" != "${NPROC}" ]]; then
    echo "Expected ${ACCELERATE_CONFIG} to set num_processes: ${NPROC}, got ${ACCELERATE_NPROC:-unset}" >&2
    exit 2
  fi
fi

echo "[launch] 7-GPU H20 run | launcher=${LAUNCHER} | config=${CONFIG} | CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

if [[ "${PACK_TOKENS}" != "0" && "${PACK_TOKENS}" != "false" ]]; then
  echo "[launch] Ensuring packed token ids exist before GPU training. Set PACK_TOKENS=0 to skip."
  "${PYTHON}" scripts/pack_tokens.py --config "${CONFIG}"
fi

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
