#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

usage() {
  cat <<'EOF'
Usage:
  bash run_model_path_pruning_results.sh BASE_MODEL_PATH SFT_MODEL_PATH CONTRASTIVE_MODEL_PATH [extra python args]

Environment overrides:
  TRAINING_DATASET      default: data/scenic/SCENIC_full_training_dataset.json
  BENCHMARK_DATASET     default: data/benchmarks/iot_instruction_benchmark_200.json
  MAX_NEW_TOKEN_HIT_RATE_THRESHOLD default: 1.01
  EOS_RETUNE            default: 0; set to 1 to add masked EOS-weighted recovery rows.
  EOS_LOSS_WEIGHT       default: 5.0 when EOS_RETUNE=1.
  EOS_RETUNE_EPOCHS     default: 1.0 when EOS_RETUNE=1.
  RUN_ROOT              default: runs/model-path-pruning-results
  RESULTS_JSON          default: ${RUN_ROOT}/model_path_pruning_results.json
  METHODS               default: "wanda magnitude taylor nvidia"
  PRUNE_FAMILIES        default: "base_model sft contrastive"
  PYTHON                default: python3, then python
  CUDA_VISIBLE_DEVICES  default: 0,1,2,3,4,5,6,7
  NPROC_PER_NODE        default: 8

Examples:
  bash run_model_path_pruning_results.sh /path/base /path/sft/final /path/contrastive/final
  PRUNE_FAMILIES="sft contrastive" DRY_RUN=1 bash run_model_path_pruning_results.sh /path/base /path/sft /path/contrastive
EOF
}

run_model_path_pruning_results() {
  local base_model="${BASE_MODEL_PATH:-}"
  local sft_model="${SFT_MODEL_PATH:-}"
  local contrastive_model="${CONTRASTIVE_MODEL_PATH:-}"
  if [[ $# -ge 3 && "${1}" != -* && "${2}" != -* && "${3}" != -* ]]; then
    base_model="${1}"
    sft_model="${2}"
    contrastive_model="${3}"
    shift 3
  fi
  if [[ -z "${base_model}" || -z "${sft_model}" || -z "${contrastive_model}" ]]; then
    usage
    return 2
  fi

  local python_bin="${PYTHON:-}"
  if [[ -z "${python_bin}" ]]; then
    if command -v python3 >/dev/null 2>&1; then
      python_bin="python3"
    else
      python_bin="python"
    fi
  fi

  local args=(
    "run_model_path_pruning_results.py"
    "--base-model-path" "${base_model}"
    "--sft-model-path" "${sft_model}"
    "--contrastive-model-path" "${contrastive_model}"
    "--training-dataset" "${TRAINING_DATASET:-data/scenic/SCENIC_full_training_dataset.json}"
    "--benchmark-dataset" "${BENCHMARK_DATASET:-data/benchmarks/iot_instruction_benchmark_200.json}"
    "--run-root" "${RUN_ROOT:-runs/model-path-pruning-results}"
    "--prune-config" "${PRUNE_CONFIG:-configs/prune_50.yaml}"
    "--methods" "${METHODS:-wanda magnitude taylor nvidia}"
    "--prune-families" "${PRUNE_FAMILIES:-base_model sft contrastive}"
    "--cuda-visible-devices" "${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
    "--nproc-per-node" "${NPROC_PER_NODE:-8}"
    "--omp-num-threads" "${OMP_NUM_THREADS:-8}"
    "--eval-runs" "${EVAL_RUNS:-1}"
    "--top-k-exact-match" "${TOP_K_EXACT_MATCH:-5}"
    "--comparison-mode" "${COMPARISON_MODE:-whitespace}"
    "--dtype" "${DTYPE:-bf16}"
    "--max-length" "${MAX_LENGTH:-128}"
    "--max-new-tokens" "${MAX_NEW_TOKENS:-64}"
    "--max-new-token-hit-rate-threshold" "${MAX_NEW_TOKEN_HIT_RATE_THRESHOLD:-1.01}"
    "--eval-batch-size" "${EVAL_BATCH_SIZE:-16}"
    "--sparsity" "${SPARSITY:-0.5}"
    "--pruning-scope" "${PRUNING_SCOPE:-transformer_linears}"
    "--sparsity-denominator" "${SPARSITY_DENOMINATOR:-prunable}"
    "--granularity" "${GRANULARITY:-global}"
    "--include-lm-head" "${INCLUDE_LM_HEAD:-false}"
    "--calibration-batches" "${CALIBRATION_BATCHES:-128}"
    "--prune-batch-size" "${PRUNE_BATCH_SIZE:-2}"
    "--prune-num-workers" "${PRUNE_NUM_WORKERS:-0}"
    "--sparsity-tolerance" "${SPARSITY_TOLERANCE:-0.001}"
  )

  if [[ "${EOS_RETUNE:-0}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
    args+=(
      "--eos-retune"
      "--eos-loss-weight" "${EOS_LOSS_WEIGHT:-5.0}"
      "--eos-retune-epochs" "${EOS_RETUNE_EPOCHS:-1.0}"
      "--eos-retune-mode" "${EOS_RETUNE_MODE:-sft}"
    )
    if [[ -n "${EOS_RETUNE_MAX_STEPS:-}" ]]; then
      args+=("--eos-retune-max-steps" "${EOS_RETUNE_MAX_STEPS}")
    fi
  fi

  if [[ -n "${CALIBRATION_DATASET:-}" ]]; then
    args+=("--calibration-dataset" "${CALIBRATION_DATASET}")
  fi
  if [[ -n "${GENERATED_CONFIG_DIR:-}" ]]; then
    args+=("--generated-config-dir" "${GENERATED_CONFIG_DIR}")
  fi
  if [[ -n "${RESULTS_JSON:-}" ]]; then
    args+=("--results-json" "${RESULTS_JSON}")
  fi
  if [[ "${DRY_RUN:-0}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
    args+=("--dry-run")
  fi
  if [[ "${KEEP_GOING:-1}" =~ ^(0|false|FALSE|no|NO|off|OFF)$ ]]; then
    args+=("--stop-on-error")
  else
    args+=("--continue-on-error")
  fi

  exec "${python_bin}" "${args[@]}" "$@"
}

run_model_path_pruning_results "$@"
