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
  3. Added linear-sparsity experiment: progressive 30/50% from each trained
     checkpoint, with one recovery epoch per pruning stage and two final
     recovery epochs by default.
  4. Write one final revision summary JSON that points to native one-shot and
     progressive artifacts.

Environment overrides:
  PYTHON                         default: python3, then python
  ORIGINAL_RUN_ROOT              default: runs/revision-original-four-one-shot
  NATIVE_SPARSITY_LEVELS         default: "0.3 0.5"
  LINEAR_SPARSITY_OUTPUT_DIR     default: results/scenic_linear_sparsity_0_30_50_from_base
  REVISION_RESULTS_JSON          default: results/scenic_revision_sparsity_summary.json
  BENCHMARK_PATH                 default: data/benchmarks/iot_instruction_benchmark_200.json
  RECOVERY_TRAIN_PATH            default: data/scenic/SCENIC_full_training_dataset.json
  RECOVERY_EPOCHS_PER_STAGE      default: 1
  FINAL_RECOVERY_EPOCHS          default: 2
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

REGULAR_SFT_FINAL="${RUN_ROOT}/training/base_sft_5ep/final"
CONTRASTIVE_SFT_FINAL="${RUN_ROOT}/training/contrastive_sft_5ep/final"
LINEAR_SPARSITY_BASE_DIR="${LINEAR_SPARSITY_OUTPUT_DIR:-results/scenic_linear_sparsity_0_30_50_from_base}"

run_progressive_linear_sparsity() {
  local label="$1"
  local checkpoint="$2"
  local output_dir="${LINEAR_SPARSITY_BASE_DIR}/${label}"

  echo
  echo "== Added linear-sparsity experiment for ${label}: progressive 30/50% =="
  "${python_bin}" scripts/run_sparsity_experiments.py \
    --experiment_name "${EXPERIMENT_NAME:-scenic_linear_sparsity_0_30_50_from_base}_${label}" \
    --model_family "${MODEL_FAMILY:-decoder_only}" \
    --model_checkpoint "${checkpoint}" \
    --benchmark_path "${BENCHMARK_PATH:-data/benchmarks/iot_instruction_benchmark_200.json}" \
    --sparsity_levels 0 0.3 0.5 \
    --pruning_modes dense progressive \
    --prune_scope linear_weights \
    --prune_method magnitude \
    --recovery_train_path "${RECOVERY_TRAIN_PATH:-data/scenic/SCENIC_full_training_dataset.json}" \
    --recovery_epochs_per_stage "${RECOVERY_EPOCHS_PER_STAGE:-1}" \
    --final_recovery_epochs "${FINAL_RECOVERY_EPOCHS:-2}" \
    --num_beams "${NUM_BEAMS:-5}" \
    --num_return_sequences "${NUM_RETURN_SEQUENCES:-5}" \
    --max_new_tokens "${MAX_NEW_TOKENS:-64}" \
    --normalization_mode "${NORMALIZATION_MODE:-command}" \
    --seed "${SEED:-42}" \
    --dtype "${DTYPE:-bf16}" \
    --output_dir "${output_dir}"
}

run_progressive_linear_sparsity "regular_sft" "${REGULAR_SFT_FINAL}"
run_progressive_linear_sparsity "contrastive_sft" "${CONTRASTIVE_SFT_FINAL}"

echo
echo "== Writing final revision sparsity summary JSON =="
"${python_bin}" scripts/write_revision_sparsity_summary.py \
  --native-results-json "${RUN_ROOT}/journal_results.json" \
  --regular-progressive-summary "${LINEAR_SPARSITY_BASE_DIR}/regular_sft/summary_metrics.csv" \
  --contrastive-progressive-summary "${LINEAR_SPARSITY_BASE_DIR}/contrastive_sft/summary_metrics.csv" \
  --output-json "${REVISION_RESULTS_JSON:-results/scenic_revision_sparsity_summary.json}"
