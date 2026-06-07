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
  2. Original four methods: magnitude, WANDA, gradient/Taylor, NVIDIA 2:4.
     These run one-shot only at 30% and 50% against the trained SFT and
     contrastive SFT checkpoints; no masked retune rows are added.
     Exact NVIDIA 2:4 remains a fixed 50% structured condition and is reported
     with its achieved sparsity instead of pretending it can be true 30%.
     The BASE_MODEL_PATH itself is only a precursor and is not included as a
     dense baseline row by default.
  3. Added linear-sparsity experiment: progressive magnitude 30/50%
     from each trained checkpoint, with one recovery epoch per pruning stage
     and one final recovery epoch by default.
  4. Write one final revision summary JSON with a fused 20-row matrix:
     18 pruning outcomes plus 2 dense baselines, each carrying training-data
     EM@1/EM@5 and benchmark EM@1/EM@5 plus easy/medium/hard benchmark metrics.

Environment overrides:
  PYTHON                         default: python3, then python
  ORIGINAL_RUN_ROOT              default: runs/revision-original-four-one-shot
  NATIVE_SPARSITY_LEVELS         default: "0.3 0.5"
  LINEAR_SPARSITY_OUTPUT_DIR     default: results/scenic_linear_sparsity_0_30_50_from_base
  REVISION_RESULTS_JSON          default: results/scenic_revision_sparsity_summary.json
  BENCHMARK_PATH                 default: data/benchmarks/iot_instruction_benchmark_200.json
  RECOVERY_TRAIN_PATH            default: data/scenic/SCENIC_full_training_dataset.json
  RECOVERY_EPOCHS_PER_STAGE      default: 1
  FINAL_RECOVERY_EPOCHS          default: 1
  SPARSITY_GPU_IDS               default: 0,1,2,3,4,5,6,7
                                  Progressive jobs are launched in parallel
                                  and assigned to these GPU ids round-robin.
  CUDA_VISIBLE_DEVICES           default: 0,1,2,3,4,5,6,7
  NPROC_PER_NODE                 default: 8
  MODEL_FAMILY                   default: decoder_only
  DTYPE                          default: fp16

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
export SPARSITY_GPU_IDS="${SPARSITY_GPU_IDS:-0,1,2,3,4,5,6,7}"
export SYMPY_GROUND_TYPES="${SYMPY_GROUND_TYPES:-python}"
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
export TORCH_COMPILE_DISABLE="${TORCH_COMPILE_DISABLE:-1}"
export ACCELERATE_DYNAMO_BACKEND="${ACCELERATE_DYNAMO_BACKEND:-no}"

export RUN_ROOT="${ORIGINAL_RUN_ROOT:-runs/revision-original-four-one-shot}"
export METHODS="${METHODS:-magnitude wanda gradient 2of4}"
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

IFS=',' read -r -a SPARSITY_GPU_ID_ARRAY <<< "${SPARSITY_GPU_IDS}"
if [[ ${#SPARSITY_GPU_ID_ARRAY[@]} -lt 1 || -z "${SPARSITY_GPU_ID_ARRAY[0]}" ]]; then
  echo "SPARSITY_GPU_IDS must list at least one GPU id, for example: 0,1,2,3,4,5,6,7" >&2
  exit 2
fi

gpu_for_progressive_job() {
  local index="$1"
  local gpu_count="${#SPARSITY_GPU_ID_ARRAY[@]}"
  local gpu_index=$((index % gpu_count))
  echo "${SPARSITY_GPU_ID_ARRAY[$gpu_index]}"
}

run_progressive_linear_sparsity() {
  local label="$1"
  local checkpoint="$2"
  local target_sparsity="$3"
  local job_index="$4"
  local output_dir="${LINEAR_SPARSITY_BASE_DIR}/${label}"
  local gpu_id
  gpu_id="$(gpu_for_progressive_job "${job_index}")"

  echo
  echo "== Added linear-sparsity experiment for ${label}: progressive magnitude ${target_sparsity} on GPU ${gpu_id} =="
  (
    CUDA_VISIBLE_DEVICES="${gpu_id}" \
    "${python_bin}" scripts/run_sparsity_experiments.py \
      --experiment_name "${EXPERIMENT_NAME:-scenic_linear_sparsity_0_30_50_from_base}_${label}_${target_sparsity}" \
      --model_family "${MODEL_FAMILY:-decoder_only}" \
      --model_checkpoint "${checkpoint}" \
      --benchmark_path "${BENCHMARK_PATH:-data/benchmarks/iot_instruction_benchmark_200.json}" \
      --extra_eval_path "training_dataset=${RECOVERY_TRAIN_PATH:-data/scenic/SCENIC_full_training_dataset.json}" \
      --sparsity_levels 0 "${target_sparsity}" \
      --pruning_modes dense progressive \
      --prune_scope linear_weights \
      --prune_method magnitude \
      --recovery_train_path "${RECOVERY_TRAIN_PATH:-data/scenic/SCENIC_full_training_dataset.json}" \
      --recovery_epochs_per_stage "${RECOVERY_EPOCHS_PER_STAGE:-1}" \
      --final_recovery_epochs "${FINAL_RECOVERY_EPOCHS:-1}" \
      --num_beams "${NUM_BEAMS:-5}" \
      --num_return_sequences "${NUM_RETURN_SEQUENCES:-5}" \
      --max_new_tokens "${MAX_NEW_TOKENS:-64}" \
      --normalization_mode "${NORMALIZATION_MODE:-command}" \
      --seed "${SEED:-42}" \
      --dtype "${DTYPE:-fp16}" \
      --output_dir "${output_dir}/sparsity_${target_sparsity//./p}"
  ) &
}

progressive_pids=()
run_progressive_linear_sparsity "regular_sft" "${REGULAR_SFT_FINAL}" "0.3" 0
progressive_pids+=("$!")
run_progressive_linear_sparsity "regular_sft" "${REGULAR_SFT_FINAL}" "0.5" 1
progressive_pids+=("$!")
run_progressive_linear_sparsity "contrastive_sft" "${CONTRASTIVE_SFT_FINAL}" "0.3" 2
progressive_pids+=("$!")
run_progressive_linear_sparsity "contrastive_sft" "${CONTRASTIVE_SFT_FINAL}" "0.5" 3
progressive_pids+=("$!")

for pid in "${progressive_pids[@]}"; do
  wait "${pid}"
done

echo
echo "== Writing final revision sparsity summary JSON =="
"${python_bin}" scripts/write_revision_sparsity_summary.py \
  --native-results-json "${RUN_ROOT}/journal_results.json" \
  --regular-progressive-summary "${LINEAR_SPARSITY_BASE_DIR}/regular_sft/sparsity_0p3/summary_metrics.csv" \
  --regular-progressive-summary "${LINEAR_SPARSITY_BASE_DIR}/regular_sft/sparsity_0p5/summary_metrics.csv" \
  --contrastive-progressive-summary "${LINEAR_SPARSITY_BASE_DIR}/contrastive_sft/sparsity_0p3/summary_metrics.csv" \
  --contrastive-progressive-summary "${LINEAR_SPARSITY_BASE_DIR}/contrastive_sft/sparsity_0p5/summary_metrics.csv" \
  --output-json "${REVISION_RESULTS_JSON:-results/scenic_revision_sparsity_summary.json}"
