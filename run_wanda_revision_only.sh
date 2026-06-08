#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

usage() {
  cat <<'EOF'
Usage:
  bash run_wanda_revision_only.sh
  bash run_wanda_revision_only.sh REGULAR_SFT_CHECKPOINT CONTRASTIVE_SFT_CHECKPOINT

Runs only the original one-shot WANDA pruning/eval rows for the current revision:
  - regular SFT: 30% and 50%
  - contrastive SFT: 30% and 50%

This does not retrain. With no checkpoint arguments, it uses:
  runs/revision-original-four-one-shot/training/base_sft_5ep/final
  runs/revision-original-four-one-shot/training/contrastive_sft_5ep/final

Environment overrides:
  PYTHON                         default: python3, then python
  RUN_ROOT                       default: runs/revision-original-four-one-shot
  RESULTS_JSON                   default: results/scenic_wanda_only_results.json
  BENCHMARK_FILE                 default: data/benchmarks/iot_instruction_benchmark_200.json
  SFT_EVAL_FILE                  default: data/scenic/SCENIC_full_training_dataset.json
  CONTRASTIVE_EVAL_FILE          default: data/scenic/SCENIC_full_training_dataset.json
  SPARSITY_LEVELS                default: "0.3 0.5"
  WANDA_GRANULARITY              default: row
  RUN_DENSE_BASELINE             default: 0, so only WANDA rows are rerun
  CUDA_VISIBLE_DEVICES           default: 0,1,2,3,4,5,6,7
  NPROC_PER_NODE                 default: 8
  EXPECTED_GPU_COUNT             default: 8
  ALLOW_H20_WORLD_SIZE_MISMATCH  set to 1 only for deliberate debug runs

Example:
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  NPROC_PER_NODE=8 \
  DTYPE=fp16 \
  bash run_wanda_revision_only.sh
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -ne 0 && $# -ne 2 ]]; then
  usage
  exit 2
fi

python_bin="${PYTHON:-}"
if [[ -z "${python_bin}" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    python_bin="python3"
  else
    python_bin="python"
  fi
fi
if [[ "${python_bin}" != */* ]]; then
  python_bin="$(command -v "${python_bin}")"
fi

export PYTHON="${python_bin}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export EXPECTED_GPU_COUNT="${EXPECTED_GPU_COUNT:-8}"
export SYMPY_GROUND_TYPES="${SYMPY_GROUND_TYPES:-python}"
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
export TORCH_COMPILE_DISABLE="${TORCH_COMPILE_DISABLE:-1}"
export ACCELERATE_DYNAMO_BACKEND="${ACCELERATE_DYNAMO_BACKEND:-no}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

visible_gpu_count() {
  local visible="${CUDA_VISIBLE_DEVICES:-}"
  local count=0
  local gpu_id
  if [[ -z "${visible}" ]]; then
    echo 0
    return
  fi
  IFS=',' read -r -a gpu_ids <<< "${visible}"
  for gpu_id in "${gpu_ids[@]}"; do
    gpu_id="${gpu_id//[[:space:]]/}"
    if [[ -n "${gpu_id}" ]]; then
      count=$((count + 1))
    fi
  done
  echo "${count}"
}

VISIBLE_GPU_COUNT="$(visible_gpu_count)"
if [[ "${ALLOW_H20_WORLD_SIZE_MISMATCH:-0}" != "1" ]]; then
  if [[ "${VISIBLE_GPU_COUNT}" -ne "${EXPECTED_GPU_COUNT}" ]]; then
    echo "Expected ${EXPECTED_GPU_COUNT} visible GPUs, got ${VISIBLE_GPU_COUNT}: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" >&2
    echo "Set CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 or ALLOW_H20_WORLD_SIZE_MISMATCH=1 for a deliberate debug run." >&2
    exit 2
  fi
  if [[ "${NPROC_PER_NODE}" -ne "${EXPECTED_GPU_COUNT}" ]]; then
    echo "Expected NPROC_PER_NODE=${EXPECTED_GPU_COUNT}, got ${NPROC_PER_NODE}." >&2
    echo "Set NPROC_PER_NODE=8 or ALLOW_H20_WORLD_SIZE_MISMATCH=1 for a deliberate debug run." >&2
    exit 2
  fi
fi
if [[ "${VISIBLE_GPU_COUNT}" -gt 0 && "${NPROC_PER_NODE}" -gt "${VISIBLE_GPU_COUNT}" ]]; then
  echo "NPROC_PER_NODE=${NPROC_PER_NODE} cannot exceed visible GPU count ${VISIBLE_GPU_COUNT}: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" >&2
  exit 2
fi

export RUN_ROOT="${RUN_ROOT:-runs/revision-original-four-one-shot}"
if [[ $# -eq 2 ]]; then
  export REGULAR_OUTPUT_DIR="$(dirname "$1")"
  export CONTRASTIVE_OUTPUT_DIR="$(dirname "$2")"
else
  export REGULAR_OUTPUT_DIR="${REGULAR_OUTPUT_DIR:-${RUN_ROOT}/training/base_sft_5ep}"
  export CONTRASTIVE_OUTPUT_DIR="${CONTRASTIVE_OUTPUT_DIR:-${RUN_ROOT}/training/contrastive_sft_5ep}"
fi

for checkpoint in "${REGULAR_OUTPUT_DIR}/final" "${CONTRASTIVE_OUTPUT_DIR}/final"; do
  if [[ ! -d "${checkpoint}" ]]; then
    echo "Checkpoint directory not found: ${checkpoint}" >&2
    exit 2
  fi
done

export BASE_MODEL="${BASE_MODEL:-${REGULAR_OUTPUT_DIR}/final}"
export ORIGINAL_MODEL="${ORIGINAL_MODEL:-${BASE_MODEL}}"
export CONTRASTIVE_BASE_MODEL="${CONTRASTIVE_BASE_MODEL:-${REGULAR_OUTPUT_DIR}/final}"
export TRAIN_REGULAR_SFT=0
export TRAIN_CONTRASTIVE_SFT=0
export TRAIN_ONLY=0
export RUN_ORIGINAL_DECODER_EVAL=0
export RUN_PRUNING_BENCHMARKS=1
export RUN_DENSE_BASELINE="${RUN_DENSE_BASELINE:-0}"
export METHODS="${METHODS:-wanda}"
export SPARSITY_LEVELS="${SPARSITY_LEVELS:-0.3 0.5}"
export WANDA_GRANULARITY="${WANDA_GRANULARITY:-row}"
export BENCHMARK_FILE="${BENCHMARK_FILE:-data/benchmarks/iot_instruction_benchmark_200.json}"
export SFT_EVAL_FILE="${SFT_EVAL_FILE:-data/scenic/SCENIC_full_training_dataset.json}"
export CONTRASTIVE_EVAL_FILE="${CONTRASTIVE_EVAL_FILE:-data/scenic/SCENIC_full_training_dataset.json}"
export TOP_K_EXACT_MATCH="${TOP_K_EXACT_MATCH:-5}"
export COMPARISON_MODE="${COMPARISON_MODE:-whitespace}"
export MAX_NEW_TOKEN_HIT_RATE_THRESHOLD="${MAX_NEW_TOKEN_HIT_RATE_THRESHOLD:-1.01}"
export RESULTS_JSON="${RESULTS_JSON:-results/scenic_wanda_only_results.json}"

echo "== WANDA revision only =="
echo "regular checkpoint:     ${REGULAR_OUTPUT_DIR}/final"
echo "contrastive checkpoint: ${CONTRASTIVE_OUTPUT_DIR}/final"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "visible_gpu_count=${VISIBLE_GPU_COUNT}"
echo "NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "WANDA_GRANULARITY=${WANDA_GRANULARITY}"
echo "RUN_DENSE_BASELINE=${RUN_DENSE_BASELINE}"
echo "SPARSITY_LEVELS=${SPARSITY_LEVELS}"
echo "RESULTS_JSON=${RESULTS_JSON}"
echo

exec "${python_bin}" run_5epoch_sft_contrastive_one_shot_pruning.py
