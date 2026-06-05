#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

usage() {
  cat <<'EOF'
Usage:
  bash run_linear50_iot_pruning.sh [BASE_MODEL_PATH] [extra python args]
  bash run_linear50_iot_pruning.sh one-shot [BASE_MODEL_PATH] [extra python args]
  bash run_linear50_iot_pruning.sh retune [BASE_MODEL_PATH] [extra python args]

Runs the 5-epoch regular SFT and 5-epoch contrastive SFT training pipeline,
then evaluates dense, one-shot pruned, and fixed-mask EOS-retuned checkpoints
on the IoT benchmark with EM1 and EM5 reporting.

Defaults:
  METHODS="magnitude wanda taylor 2of4"
  SPARSITY=0.5
  PRUNING_SCOPE=transformer_linears
  SPARSITY_DENOMINATOR=prunable
  GRANULARITY=global
  INCLUDE_LM_HEAD=0
  BENCHMARK_FILE=data/benchmarks/iot_instruction_benchmark_200.json
  EOS_RETUNE=1
  EOS_LOSS_WEIGHT=5.0

Override any default with an environment variable, for example:
  PYTHON=/path/to/env/bin/python RUN_ROOT=runs/iot-linear50 bash run_linear50_iot_pruning.sh /path/to/base_model
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

export METHODS="${METHODS:-magnitude wanda taylor 2of4}"
export SPARSITY="${SPARSITY:-0.5}"
export PRUNING_SCOPE="${PRUNING_SCOPE:-transformer_linears}"
export SPARSITY_DENOMINATOR="${SPARSITY_DENOMINATOR:-prunable}"
export GRANULARITY="${GRANULARITY:-global}"
export INCLUDE_LM_HEAD="${INCLUDE_LM_HEAD:-0}"
export BENCHMARK_FILE="${BENCHMARK_FILE:-data/benchmarks/iot_instruction_benchmark_200.json}"
export TOP_K_EXACT_MATCH="${TOP_K_EXACT_MATCH:-5}"
export COMPARISON_MODE="${COMPARISON_MODE:-whitespace}"
export MAX_NEW_TOKEN_HIT_RATE_THRESHOLD="${MAX_NEW_TOKEN_HIT_RATE_THRESHOLD:-0.5}"
export RUN_ROOT="${RUN_ROOT:-runs/linear50-iot-pruning}"
export EOS_RETUNE="${EOS_RETUNE:-1}"
export EOS_LOSS_WEIGHT="${EOS_LOSS_WEIGHT:-5.0}"
export EOS_RETUNE_EPOCHS="${EOS_RETUNE_EPOCHS:-1.0}"
export EOS_RETUNE_MODE="${EOS_RETUNE_MODE:-sft}"

exec bash run_5epoch_sft_contrastive_one_shot_pruning.sh "$@"
