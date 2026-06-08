#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

usage() {
  cat <<'EOF'
Usage:
  bash run_progressive_magnitude_revision_only.sh
  bash run_progressive_magnitude_revision_only.sh REGULAR_SFT_CHECKPOINT CONTRASTIVE_SFT_CHECKPOINT

Runs only the added progressive magnitude pruning jobs:
  - regular SFT: 30% and 50%
  - contrastive SFT: 30% and 50%

With no checkpoint arguments, the script uses:
  runs/revision-original-four-one-shot/training/base_sft_5ep/final
  runs/revision-original-four-one-shot/training/contrastive_sft_5ep/final

Environment overrides:
  PYTHON                         default: python3, then python
  TORCHRUN                       default: torchrun
  RUN_ROOT                       default: runs/revision-original-four-one-shot
  LINEAR_SPARSITY_OUTPUT_DIR     default: results/scenic_linear_sparsity_progressive_only
  BENCHMARK_PATH                 default: data/benchmarks/iot_instruction_benchmark_200.json
  RECOVERY_TRAIN_PATH            default: data/scenic/SCENIC_full_training_dataset.json
  RECOVERY_EPOCHS_PER_STAGE      default: 1
  FINAL_RECOVERY_EPOCHS          default: 1
  EOS_LOSS_WEIGHT                default: 5.0
  CUDA_VISIBLE_DEVICES           default: 0,1,2,3,4,5,6,7
  NPROC_PER_NODE                 default: 8
  EXPECTED_GPU_COUNT             default: 8
  WRITE_REVISION_SUMMARY         default: 1 when native results JSON exists
  NATIVE_RESULTS_JSON            default: ${RUN_ROOT}/journal_results.json
  REVISION_RESULTS_JSON          default: results/scenic_progressive_only_revision_sparsity_summary.json

Example:
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  NPROC_PER_NODE=8 \
  DTYPE=fp16 \
  bash run_progressive_magnitude_revision_only.sh
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

torchrun_bin="${TORCHRUN:-torchrun}"
if [[ "${torchrun_bin}" != */* ]]; then
  if ! torchrun_bin="$(command -v "${torchrun_bin}")"; then
    echo "Could not find TORCHRUN executable '${TORCHRUN:-torchrun}' on PATH." >&2
    echo "Activate the chatlm-decoder environment or set TORCHRUN=/path/to/env/bin/torchrun." >&2
    exit 2
  fi
fi

export PYTHON="${python_bin}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export EXPECTED_GPU_COUNT="${EXPECTED_GPU_COUNT:-8}"
export SYMPY_GROUND_TYPES="${SYMPY_GROUND_TYPES:-python}"
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
export TORCH_COMPILE_DISABLE="${TORCH_COMPILE_DISABLE:-1}"
export ACCELERATE_DYNAMO_BACKEND="${ACCELERATE_DYNAMO_BACKEND:-no}"
export EOS_LOSS_WEIGHT="${EOS_LOSS_WEIGHT:-5.0}"

RUN_ROOT="${RUN_ROOT:-runs/revision-original-four-one-shot}"
if [[ $# -eq 2 ]]; then
  REGULAR_SFT_FINAL="$1"
  CONTRASTIVE_SFT_FINAL="$2"
else
  REGULAR_SFT_FINAL="${REGULAR_SFT_FINAL:-${RUN_ROOT}/training/base_sft_5ep/final}"
  CONTRASTIVE_SFT_FINAL="${CONTRASTIVE_SFT_FINAL:-${RUN_ROOT}/training/contrastive_sft_5ep/final}"
fi

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
if [[ "${EXPECTED_GPU_COUNT}" -gt 0 ]]; then
  if [[ "${VISIBLE_GPU_COUNT}" -ne "${EXPECTED_GPU_COUNT}" ]]; then
    echo "Expected ${EXPECTED_GPU_COUNT} visible GPUs, got ${VISIBLE_GPU_COUNT}: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" >&2
    exit 2
  fi
  if [[ "${NPROC_PER_NODE}" -ne "${EXPECTED_GPU_COUNT}" ]]; then
    echo "Expected NPROC_PER_NODE=${EXPECTED_GPU_COUNT}, got ${NPROC_PER_NODE}." >&2
    exit 2
  fi
fi

for checkpoint in "${REGULAR_SFT_FINAL}" "${CONTRASTIVE_SFT_FINAL}"; do
  if [[ ! -d "${checkpoint}" ]]; then
    echo "Checkpoint directory not found: ${checkpoint}" >&2
    exit 2
  fi
done

echo "== Progressive magnitude only =="
echo "regular checkpoint:     ${REGULAR_SFT_FINAL}"
echo "contrastive checkpoint: ${CONTRASTIVE_SFT_FINAL}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "visible_gpu_count=${VISIBLE_GPU_COUNT}"
echo "NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "EOS_LOSS_WEIGHT=${EOS_LOSS_WEIGHT}"
echo

LINEAR_SPARSITY_BASE_DIR="${LINEAR_SPARSITY_OUTPUT_DIR:-results/scenic_linear_sparsity_progressive_only}"

run_progressive_linear_sparsity() {
  local label="$1"
  local checkpoint="$2"
  local target_sparsity="$3"
  local output_dir="${LINEAR_SPARSITY_BASE_DIR}/${label}"

  echo
  echo "== Progressive magnitude ${label}: target ${target_sparsity} on ${NPROC_PER_NODE} GPU(s) =="
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
    "${torchrun_bin}" --standalone --nproc_per_node "${NPROC_PER_NODE}" scripts/run_sparsity_experiments.py \
      --experiment_name "${EXPERIMENT_NAME:-scenic_progressive_magnitude_only}_${label}_${target_sparsity}" \
      --model_family "${MODEL_FAMILY:-decoder_only}" \
      --model_checkpoint "${checkpoint}" \
      --benchmark_path "${BENCHMARK_PATH:-data/benchmarks/iot_instruction_benchmark_200.json}" \
      --extra_eval_path "training_dataset=${RECOVERY_TRAIN_PATH:-data/scenic/SCENIC_full_training_dataset.json}" \
      --sparsity_levels "${target_sparsity}" \
      --pruning_modes progressive \
      --prune_scope linear_weights \
      --prune_method magnitude \
      --recovery_train_path "${RECOVERY_TRAIN_PATH:-data/scenic/SCENIC_full_training_dataset.json}" \
      --recovery_epochs_per_stage "${RECOVERY_EPOCHS_PER_STAGE:-1}" \
      --final_recovery_epochs "${FINAL_RECOVERY_EPOCHS:-1}" \
      --eos_loss_weight "${EOS_LOSS_WEIGHT}" \
      --num_beams "${NUM_BEAMS:-5}" \
      --num_return_sequences "${NUM_RETURN_SEQUENCES:-5}" \
      --max_new_tokens "${MAX_NEW_TOKENS:-64}" \
      --normalization_mode "${NORMALIZATION_MODE:-command}" \
      --seed "${SEED:-42}" \
      --dtype "${DTYPE:-fp16}" \
      --expected_world_size "${EXPECTED_GPU_COUNT}" \
      --expected_visible_gpu_count "${EXPECTED_GPU_COUNT}" \
      --output_dir "${output_dir}/sparsity_${target_sparsity//./p}"
}

run_progressive_linear_sparsity "regular_sft" "${REGULAR_SFT_FINAL}" "0.3"
run_progressive_linear_sparsity "regular_sft" "${REGULAR_SFT_FINAL}" "0.5"
run_progressive_linear_sparsity "contrastive_sft" "${CONTRASTIVE_SFT_FINAL}" "0.3"
run_progressive_linear_sparsity "contrastive_sft" "${CONTRASTIVE_SFT_FINAL}" "0.5"

NATIVE_RESULTS_JSON="${NATIVE_RESULTS_JSON:-${RUN_ROOT}/journal_results.json}"
if [[ "${WRITE_REVISION_SUMMARY:-1}" == "1" && -f "${NATIVE_RESULTS_JSON}" ]]; then
  echo
  echo "== Writing combined summary JSON with refreshed progressive rows =="
  "${python_bin}" scripts/write_revision_sparsity_summary.py \
    --native-results-json "${NATIVE_RESULTS_JSON}" \
    --regular-progressive-summary "${LINEAR_SPARSITY_BASE_DIR}/regular_sft/sparsity_0p3/summary_metrics.csv" \
    --regular-progressive-summary "${LINEAR_SPARSITY_BASE_DIR}/regular_sft/sparsity_0p5/summary_metrics.csv" \
    --contrastive-progressive-summary "${LINEAR_SPARSITY_BASE_DIR}/contrastive_sft/sparsity_0p3/summary_metrics.csv" \
    --contrastive-progressive-summary "${LINEAR_SPARSITY_BASE_DIR}/contrastive_sft/sparsity_0p5/summary_metrics.csv" \
    --output-json "${REVISION_RESULTS_JSON:-results/scenic_progressive_only_revision_sparsity_summary.json}"
else
  echo
  echo "Skipping combined JSON because native results JSON was not found: ${NATIVE_RESULTS_JSON}"
  echo "Progressive CSVs are under: ${LINEAR_SPARSITY_BASE_DIR}"
fi

