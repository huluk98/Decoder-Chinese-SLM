#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

usage() {
  cat <<'EOF'
Usage:
  bash run_linear_sparsity_revision_from_base.sh BASE_MODEL_PATH

Run shape:
  1. Train regular SFT for 5 epochs from BASE_MODEL_PATH, then train
     contrastive SFT for 5 epochs from the regular SFT checkpoint.
  2. Original four methods: magnitude, WANDA, Taylor, NVIDIA 2:4.
     These run one-shot only at 30% and 50% against the trained SFT and
     contrastive SFT checkpoints; no masked retune rows are added.
     Exact NVIDIA 2:4 remains a fixed 50% structured condition and is reported
     with its achieved sparsity instead of pretending it can be true 30%.
     The BASE_MODEL_PATH itself is only a precursor and is not included as a
     dense baseline row by default.
  3. Added linear-sparsity experiment: dense 0%, one-shot 30/50%,
     progressive 30/50% with one recovery epoch per pruning stage and one
     final recovery epoch.

Environment overrides:
  PYTHON                         default: python3, then python
  ORIGINAL_RUN_ROOT              default: runs/revision-original-four-one-shot
  NATIVE_SPARSITY_LEVELS         default: "0.3 0.5"
  LINEAR_SPARSITY_OUTPUT_DIR     default: results/scenic_linear_sparsity_0_30_50_from_base
  BENCHMARK_PATH                 default: data/benchmarks/iot_instruction_benchmark_200.json
  RECOVERY_TRAIN_PATH            default: data/scenic/SCENIC_full_training_dataset.json
  CUDA_VISIBLE_DEVICES           default: 0,1,2,3,4,5,6,7
  NPROC_PER_NODE                 default: 8
  MODEL_FAMILY                   default: decoder_only
  DTYPE                          default: bf16

Example:
  PYTHON=/path/to/env/bin/python bash run_linear_sparsity_revision_from_base.sh /path/to/base_model
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -ne 1 ]]; then
  usage
  exit 2
fi

BASE_MODEL_PATH="$1"

python_bin="${PYTHON:-}"
if [[ -z "${python_bin}" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    python_bin="python3"
  else
    python_bin="python"
  fi
fi

export PYTHON="${python_bin}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

export RUN_ROOT="${ORIGINAL_RUN_ROOT:-runs/revision-original-four-one-shot}"
export METHODS="${METHODS:-magnitude wanda taylor 2of4}"
export SPARSITY_LEVELS="${SPARSITY_LEVELS:-${NATIVE_SPARSITY_LEVELS:-0.3 0.5}}"
export RUN_ORIGINAL_DECODER_EVAL="${RUN_ORIGINAL_DECODER_EVAL:-0}"
export EOS_RETUNE=0
export BENCHMARK_FILE="${BENCHMARK_FILE:-${BENCHMARK_PATH:-data/benchmarks/iot_instruction_benchmark_200.json}}"
export TOP_K_EXACT_MATCH="${TOP_K_EXACT_MATCH:-5}"
export COMPARISON_MODE="${COMPARISON_MODE:-whitespace}"
export MAX_NEW_TOKEN_HIT_RATE_THRESHOLD="${MAX_NEW_TOKEN_HIT_RATE_THRESHOLD:-0.5}"

echo "== Original four methods: one-shot only at ${SPARSITY_LEVELS} =="
bash run_5epoch_sft_contrastive_one_shot_pruning.sh one-shot "${BASE_MODEL_PATH}"

echo
echo "== Added linear-sparsity experiment: one recovery epoch =="
"${python_bin}" scripts/run_sparsity_experiments.py \
  --experiment_name "${EXPERIMENT_NAME:-scenic_linear_sparsity_0_30_50_from_base}" \
  --model_family "${MODEL_FAMILY:-decoder_only}" \
  --model_checkpoint "${BASE_MODEL_PATH}" \
  --benchmark_path "${BENCHMARK_PATH:-data/benchmarks/iot_instruction_benchmark_200.json}" \
  --sparsity_levels 0 0.3 0.5 \
  --pruning_modes dense oneshot progressive \
  --prune_scope linear_weights \
  --prune_method magnitude \
  --recovery_train_path "${RECOVERY_TRAIN_PATH:-data/scenic/SCENIC_full_training_dataset.json}" \
  --recovery_epochs_per_stage 1 \
  --final_recovery_epochs 1 \
  --num_beams "${NUM_BEAMS:-5}" \
  --num_return_sequences "${NUM_RETURN_SEQUENCES:-5}" \
  --max_new_tokens "${MAX_NEW_TOKENS:-64}" \
  --normalization_mode "${NORMALIZATION_MODE:-command}" \
  --seed "${SEED:-42}" \
  --dtype "${DTYPE:-bf16}" \
  --output_dir "${LINEAR_SPARSITY_OUTPUT_DIR:-results/scenic_linear_sparsity_0_30_50_from_base}"
