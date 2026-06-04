#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

usage() {
  cat <<'EOF'
Run dense eval plus one-shot decoder pruning for a trained checkpoint.

Usage:
  bash run_trained_model_pruning_eval.sh /path/to/trained_model
  bash run_trained_model_pruning_eval.sh /path/to/trained_model /path/to/eval.json

Defaults:
  DATA_FILE=data/scenic/SCENIC_full_training_dataset.json
  CALIBRATION_FILE=$DATA_FILE
  METHODS="magnitude wanda gradient 2of4"
  SPARSITY=0.5
  SPARSITY_DENOMINATOR=whole_model
  GRANULARITY=layer
  PRUNING_SCOPE=transformer_linears
  INCLUDE_LM_HEAD=false

Common overrides:
  DATA_FILE=/path/to/eval.json
  CALIBRATION_FILE=/path/to/calibration.json
  OUTPUT_DIR=runs/my-pruning-check
  METHODS="magnitude wanda gradient"
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  NPROC=8
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

MODEL_PATH="${MODEL_PATH:-${1:-}}"
if [[ -z "${MODEL_PATH}" ]]; then
  usage
  exit 2
fi

DATA_FILE="${DATA_FILE:-${EVAL_FILE:-${2:-data/scenic/SCENIC_full_training_dataset.json}}}"
CALIBRATION_FILE="${CALIBRATION_FILE:-${DATA_FILE}}"
METHODS="${METHODS:-magnitude wanda gradient 2of4}"
MAX_LENGTH="${MAX_LENGTH:-128}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
TOP_K_EXACT_MATCH="${TOP_K_EXACT_MATCH:-5}"
SPARSITY="${SPARSITY:-0.5}"
SPARSITY_DENOMINATOR="${SPARSITY_DENOMINATOR:-whole_model}"
GRANULARITY="${GRANULARITY:-layer}"
PRUNING_SCOPE="${PRUNING_SCOPE:-transformer_linears}"
INCLUDE_LM_HEAD="${INCLUDE_LM_HEAD:-false}"
CALIBRATION_BATCHES="${CALIBRATION_BATCHES:-128}"
PRUNE_BATCH_SIZE="${PRUNE_BATCH_SIZE:-2}"
NPROC="${NPROC:-8}"
DTYPE="${DTYPE:-bf16}"
BENCHMARK_RUNS="${BENCHMARK_RUNS:-1}"
COMPARISON_MODE="${COMPARISON_MODE:-whitespace}"

if [[ -z "${OUTPUT_DIR:-}" ]]; then
  STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
  OUTPUT_DIR="runs/trained-model-pruning-eval-${STAMP}"
fi

export MODEL_PATH
export DATA_FILE
export CALIBRATION_FILE
export METHODS
export MAX_LENGTH
export EVAL_BATCH_SIZE
export MAX_NEW_TOKENS
export TOP_K_EXACT_MATCH
export SPARSITY
export SPARSITY_DENOMINATOR
export GRANULARITY
export PRUNING_SCOPE
export INCLUDE_LM_HEAD
export CALIBRATION_BATCHES
export PRUNE_BATCH_SIZE
export NPROC
export DTYPE
export BENCHMARK_RUNS
export COMPARISON_MODE
export OUTPUT_DIR

echo "Trained-model pruning eval"
echo "  model:       ${MODEL_PATH}"
echo "  eval data:   ${DATA_FILE}"
echo "  calibration: ${CALIBRATION_FILE}"
echo "  methods:     ${METHODS}"
echo "  output:      ${OUTPUT_DIR}"
echo

bash scripts/run_sft_pruning_eval.sh "${MODEL_PATH}" "${DATA_FILE}"
