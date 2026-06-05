#!/usr/bin/env bash
set -euo pipefail

ORIGINAL_CWD="$(pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

usage() {
  cat <<'EOF'
Run the NVIDIA-only ONNX/TensorRT edge correctness pass:
  1. train regular SFT for 5 epochs from the base decoder checkpoint;
  2. make a NVIDIA 2:4 pruned checkpoint from the trained SFT checkpoint;
  3. export separate no-cache ONNX graphs for dense SFT and 2:4 pruned SFT models;
  4. build FP16 dense SFT, FP16 2:4, INT8 dense SFT, and INT8 2:4 TensorRT engines;
  5. evaluate EM@1 and EM@5 on training data and benchmark data.

Usage:
  bash run_nvidia_onnx_edge_eval.sh /path/to/base_decoder_model

Common overrides:
  PYTHON=/path/to/python
  RUN_ROOT=runs/nvidia-onnx-edge-eval
  SFT_EPOCHS=5
  SFT_OUTPUT_DIR=runs/nvidia-onnx-edge-eval/training/sft_5epoch
  TRAIN_NPROC_PER_NODE=8
  TRAINING_DATASET=data/scenic/SCENIC_full_training_dataset.json
  BENCHMARK_DATASET=data/benchmarks/iot_instruction_benchmark_200.json
  OVERWRITE=1
  MAX_SEQ_LEN=128
  MAX_NEW_TOKENS=64
  EXACT_MATCH_TOP_K=5
  RUN_INT8=1
  TRAINING_MAX_SAMPLES=200   # 0 means full training dataset; full can be very slow
  BENCHMARK_MAX_SAMPLES=0    # 0 means full benchmark dataset, normally 200 rows
  MAX_SAMPLES=20             # optional quick smoke override for both eval datasets
  CALIB_SAMPLES=128
  EDGE_ATTN_IMPLEMENTATION=eager
  PROMPT_FORMAT=raw          # raw, legacy, or chat-template
  COMPARISON_MODE=whitespace # whitespace, normalized, or command
  TRUST_REMOTE_CODE=1
  SKIP_TRAIN=1             # reuse SFT_OUTPUT_DIR/final instead of training again
  SKIP_PRUNE=1
  SKIP_EXPORT=1
  SKIP_BUILD=1
  SKIP_EVAL=1
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -lt 1 ]]; then
  usage
  exit 2
fi

resolve_input_path() {
  local value="$1"
  if [[ "${value}" == /* ]]; then
    printf '%s\n' "${value}"
  elif [[ -e "${ORIGINAL_CWD}/${value}" ]]; then
    printf '%s\n' "${ORIGINAL_CWD}/${value}"
  else
    printf '%s\n' "${value}"
  fi
}

BASE_DECODER_MODEL="$(resolve_input_path "$1")"
shift 1

PYTHON_BIN="${PYTHON:-python3}"
RUN_ROOT="${RUN_ROOT:-runs/nvidia-onnx-edge-eval}"
SFT_EPOCHS="${SFT_EPOCHS:-5}"
SFT_OUTPUT_DIR="${SFT_OUTPUT_DIR:-${RUN_ROOT}/training/sft_5epoch}"
SFT_CHECKPOINT="${SFT_CHECKPOINT:-${SFT_OUTPUT_DIR}/final}"
PRUNED_MODEL_DIR="${PRUNED_MODEL_DIR:-${RUN_ROOT}/models/nvidia-2of4-pruned}"
TRAINING_DATASET="${TRAINING_DATASET:-data/scenic/SCENIC_full_training_dataset.json}"
BENCHMARK_DATASET="${BENCHMARK_DATASET:-data/benchmarks/iot_instruction_benchmark_200.json}"
CALIB_DATASET="${CALIB_DATASET:-${TRAINING_DATASET}}"
MIN_SEQ_LEN="${MIN_SEQ_LEN:-1}"
OPT_SEQ_LEN="${OPT_SEQ_LEN:-64}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-128}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
EXACT_MATCH_TOP_K="${EXACT_MATCH_TOP_K:-5}"
MAX_NEW_TOKEN_HIT_RATE_THRESHOLD="${MAX_NEW_TOKEN_HIT_RATE_THRESHOLD:-0.5}"
CALIB_SAMPLES="${CALIB_SAMPLES:-128}"
CALIB_SEQ_LEN="${CALIB_SEQ_LEN:-${OPT_SEQ_LEN}}"
WORKSPACE_GB="${WORKSPACE_GB:-4}"
PROMPT_FORMAT="${PROMPT_FORMAT:-raw}"
SYSTEM_PROMPT="${SYSTEM_PROMPT:-}"
COMPARISON_MODE="${COMPARISON_MODE:-whitespace}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
TRAINING_MAX_SAMPLES="${TRAINING_MAX_SAMPLES:-200}"
BENCHMARK_MAX_SAMPLES="${BENCHMARK_MAX_SAMPLES:-0}"
EXPORT_DTYPE="${EXPORT_DTYPE:-fp16}"
OPSET="${OPSET:-18}"
SPARSITY="${SPARSITY:-0.5}"
SPARSE_WEIGHTS_FOR_2OF4="${SPARSE_WEIGHTS_FOR_2OF4:-1}"
RUN_INT8="${RUN_INT8:-1}"
EDGE_ATTN_IMPLEMENTATION="${EDGE_ATTN_IMPLEMENTATION:-eager}"
TRAIN_NPROC_PER_NODE="${TRAIN_NPROC_PER_NODE:-8}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

export CUDA_VISIBLE_DEVICES
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS

OVERWRITE_ARGS=()
if [[ "${OVERWRITE:-1}" == "1" ]]; then
  OVERWRITE_ARGS=(--overwrite)
fi

TRUST_ARGS=()
if [[ "${TRUST_REMOTE_CODE:-0}" == "1" ]]; then
  TRUST_ARGS=(--trust-remote-code)
fi

SYSTEM_PROMPT_ARGS=()
if [[ -n "${SYSTEM_PROMPT}" ]]; then
  SYSTEM_PROMPT_ARGS=(--system-prompt "${SYSTEM_PROMPT}")
fi

if [[ -n "${MAX_SAMPLES}" && "${MAX_SAMPLES}" != "0" ]]; then
  TRAINING_MAX_SAMPLES="${MAX_SAMPLES}"
  BENCHMARK_MAX_SAMPLES="${MAX_SAMPLES}"
fi

mkdir -p "${RUN_ROOT}/generated_configs"

SFT_CONFIG_PATH="${RUN_ROOT}/generated_configs/sft_5epoch.yaml"
cat > "${SFT_CONFIG_PATH}" <<EOF
model_name_or_path: ${BASE_DECODER_MODEL}
train_file: ${TRAINING_DATASET}
eval_file: ${TRAINING_DATASET}
benchmark_file: ${BENCHMARK_DATASET}
output_dir: ${SFT_OUTPUT_DIR}

max_seq_length: ${MAX_SEQ_LEN}
max_new_tokens: ${MAX_NEW_TOKENS}
benchmark_runs: 1
top_k_exact_match: ${EXACT_MATCH_TOP_K}

num_train_epochs: ${SFT_EPOCHS}
learning_rate: 2.0e-5
warmup_ratio: 0.03
weight_decay: 0.01
lr_scheduler_type: cosine
max_grad_norm: 1.0

bf16: true
fp16: false
tf32: true
attn_implementation: ${EDGE_ATTN_IMPLEMENTATION}
sdp_flash: false
sdp_mem_efficient: false
sdp_math: true

per_device_train_batch_size: 16
per_device_eval_batch_size: 16
gradient_accumulation_steps: 1

gradient_checkpointing: false
flash_attention: false
torch_compile: false

logging_steps: 10
save_strategy: epoch
eval_strategy: none
save_final_only: true
save_total_limit: 1

dataloader_num_workers: 8
dataloader_pin_memory: true
persistent_workers: true
group_by_length: true
remove_unused_columns: false

generation:
  max_new_tokens: ${MAX_NEW_TOKENS}
  do_sample: false
  num_beams: 1
EOF

PRUNE_CONFIG_PATH="${RUN_ROOT}/generated_configs/prune_nvidia_2of4.yaml"
cat > "${PRUNE_CONFIG_PATH}" <<EOF
run:
  seed: 42
model:
  block_size: ${MAX_SEQ_LEN}
train:
  batch_size: 2
prune:
  base_model: ${SFT_CHECKPOINT}
  output_dir: ${PRUNED_MODEL_DIR}
  method: 2of4
  sparsity: ${SPARSITY}
  scope: transformer_linears
  sparsity_denominator: prunable
  granularity: global
  include_lm_head: false
  attn_implementation: ${EDGE_ATTN_IMPLEMENTATION}
  calibration_data_path: ${CALIB_DATASET}
  calibration_batches: ${CALIB_SAMPLES}
  max_length: ${MAX_SEQ_LEN}
  batch_size: 2
  num_workers: 0
  recovery_steps: 0
  overwrite: true
EOF

train_sft() {
  echo
  echo "[train] 5-epoch SFT from base decoder"
  echo "  base:      ${BASE_DECODER_MODEL}"
  echo "  config:    ${SFT_CONFIG_PATH}"
  echo "  output:    ${SFT_OUTPUT_DIR}"
  echo "  epochs:    ${SFT_EPOCHS}"
  if [[ "${SKIP_TRAIN:-0}" == "1" ]]; then
    echo "[skip] train; expecting existing checkpoint at ${SFT_CHECKPOINT}"
  else
    torchrun --standalone --nproc_per_node="${TRAIN_NPROC_PER_NODE}" scripts/train.py \
      --config "${SFT_CONFIG_PATH}"
  fi
  if [[ ! -d "${SFT_CHECKPOINT}" || ! -f "${SFT_CHECKPOINT}/config.json" ]]; then
    echo "SFT checkpoint not found after training: ${SFT_CHECKPOINT}"
    echo "Expected ${SFT_OUTPUT_DIR}/final. Check training logs and SFT_OUTPUT_DIR."
    exit 1
  fi
}

export_model() {
  local label="$1"
  local model_path="$2"
  local onnx_dir="${RUN_ROOT}/onnx/${label}"
  echo
  echo "[export] ${label}: ${model_path}"
  "${PYTHON_BIN}" scripts/export_decoder_onnx.py \
    --model-path "${model_path}" \
    --onnx-dir "${onnx_dir}" \
    --dtype "${EXPORT_DTYPE}" \
    --opset "${OPSET}" \
    --seq-len "${OPT_SEQ_LEN}" \
    --batch-size "${BATCH_SIZE}" \
    --attn-implementation "${EDGE_ATTN_IMPLEMENTATION}" \
    --no-export-cache \
    "${TRUST_ARGS[@]}" \
    "${OVERWRITE_ARGS[@]}"
}

build_engine() {
  local label="$1"
  local model_path="$2"
  local precision="$3"
  local sparse_weights="$4"
  local onnx_path="${RUN_ROOT}/onnx/${label}/model_decoder_nocache.onnx"
  local trt_dir="${RUN_ROOT}/trt/${label}"
  local sparse_args=()
  if [[ "${sparse_weights}" == "1" && "${SPARSE_WEIGHTS_FOR_2OF4}" == "1" ]]; then
    sparse_args=(--sparse-weights)
  fi
  echo
  echo "[build] ${precision} ${label}"
  if [[ "${precision}" == "int8" ]]; then
    "${PYTHON_BIN}" scripts/build_trt_engines.py \
      --onnx "${onnx_path}" \
      --output-dir "${trt_dir}" \
      --precision int8 \
      --model-path "${model_path}" \
      --calib_json "${CALIB_DATASET}" \
      --calib-samples "${CALIB_SAMPLES}" \
      --calib-seq-len "${CALIB_SEQ_LEN}" \
      --prompt-format "${PROMPT_FORMAT}" \
      "${SYSTEM_PROMPT_ARGS[@]}" \
      --min_seq_len "${MIN_SEQ_LEN}" \
      --opt_seq_len "${OPT_SEQ_LEN}" \
      --max_seq_len "${MAX_SEQ_LEN}" \
      --batch_size "${BATCH_SIZE}" \
      --workspace-gb "${WORKSPACE_GB}" \
      "${sparse_args[@]}" \
      "${TRUST_ARGS[@]}" \
      "${OVERWRITE_ARGS[@]}"
  else
    "${PYTHON_BIN}" scripts/build_trt_engines.py \
      --onnx "${onnx_path}" \
      --output-dir "${trt_dir}" \
      --precision fp16 \
      --model-path "${model_path}" \
      --min_seq_len "${MIN_SEQ_LEN}" \
      --opt_seq_len "${OPT_SEQ_LEN}" \
      --max_seq_len "${MAX_SEQ_LEN}" \
      --batch_size "${BATCH_SIZE}" \
      --workspace-gb "${WORKSPACE_GB}" \
      "${sparse_args[@]}" \
      "${TRUST_ARGS[@]}" \
      "${OVERWRITE_ARGS[@]}"
  fi
}

eval_engine() {
  local label="$1"
  local model_path="$2"
  local precision="$3"
  local dataset_name="$4"
  local dataset_path="$5"
  local eval_limit="$6"
  shift 6
  local engine_path="${RUN_ROOT}/trt/${label}/model_${precision}.engine"
  local onnx_path="${RUN_ROOT}/onnx/${label}/model_decoder_nocache.onnx"
  local output_dir="${RUN_ROOT}/eval/${label}/${precision}/${dataset_name}"
  local limit_args=()
  local limit_label="full"
  if [[ -n "${eval_limit}" && "${eval_limit}" != "0" ]]; then
    limit_args=(--limit "${eval_limit}")
    limit_label="${eval_limit}"
  fi
  echo
  echo "[eval] ${precision} ${label} on ${dataset_name} (limit=${limit_label})"
  "${PYTHON_BIN}" scripts/eval_trt_prompt_response.py \
    --engine "${engine_path}" \
    --model-path "${model_path}" \
    --dataset "${dataset_path}" \
    --output-dir "${output_dir}" \
    --precision "${precision}" \
    --variant "${precision}_${label}" \
    --runtime "TensorRT" \
    --onnx-path "${onnx_path}" \
    --batch-size "${BATCH_SIZE}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --max-seq-len "${MAX_SEQ_LEN}" \
    --exact-match-top-k "${EXACT_MATCH_TOP_K}" \
    --max-new-token-hit-rate-threshold "${MAX_NEW_TOKEN_HIT_RATE_THRESHOLD}" \
    --prompt-format "${PROMPT_FORMAT}" \
    "${SYSTEM_PROMPT_ARGS[@]}" \
    --comparison-mode "${COMPARISON_MODE}" \
    "${limit_args[@]}" \
    "${TRUST_ARGS[@]}" \
    "${OVERWRITE_ARGS[@]}" \
    "$@"
}

echo "NVIDIA ONNX edge eval"
echo "  base decoder:     ${BASE_DECODER_MODEL}"
echo "  dense SFT model:  ${SFT_CHECKPOINT}"
echo "  2:4 model:        ${PRUNED_MODEL_DIR}"
echo "  training data:    ${TRAINING_DATASET}"
echo "  benchmark data:   ${BENCHMARK_DATASET}"
echo "  run root:         ${RUN_ROOT}"
echo "  sparse TRT flag:  ${SPARSE_WEIGHTS_FOR_2OF4}"
echo "  run INT8:         ${RUN_INT8}"
echo "  attention impl:   ${EDGE_ATTN_IMPLEMENTATION}"
echo "  SFT epochs:       ${SFT_EPOCHS}"
echo "  train GPUs:       ${TRAIN_NPROC_PER_NODE}"
echo "  training limit:   ${TRAINING_MAX_SAMPLES} (0 means full; full can be slow with no-cache EM@K)"
echo "  benchmark limit:  ${BENCHMARK_MAX_SAMPLES} (0 means full)"
echo

train_sft

if [[ "${SKIP_PRUNE:-0}" != "1" ]]; then
  echo "[prune] NVIDIA 2:4 checkpoint"
  "${PYTHON_BIN}" scripts/prune.py \
    --config "${PRUNE_CONFIG_PATH}" \
    --method 2of4 \
    --checkpoint "${SFT_CHECKPOINT}" \
    --output-dir "${PRUNED_MODEL_DIR}"
else
  echo "[skip] prune"
fi

if [[ "${SKIP_EXPORT:-0}" != "1" ]]; then
  export_model "dense" "${SFT_CHECKPOINT}"
  export_model "nvidia-2of4" "${PRUNED_MODEL_DIR}"
else
  echo "[skip] export"
fi

if [[ "${SKIP_BUILD:-0}" != "1" ]]; then
  build_engine "dense" "${SFT_CHECKPOINT}" "fp16" "0"
  build_engine "nvidia-2of4" "${PRUNED_MODEL_DIR}" "fp16" "1"
  if [[ "${RUN_INT8}" == "1" ]]; then
    build_engine "dense" "${SFT_CHECKPOINT}" "int8" "0"
    build_engine "nvidia-2of4" "${PRUNED_MODEL_DIR}" "int8" "1"
  fi
else
  echo "[skip] build"
fi

if [[ "${SKIP_EVAL:-0}" != "1" ]]; then
  eval_engine "dense" "${SFT_CHECKPOINT}" "fp16" "training_data" "${TRAINING_DATASET}" "${TRAINING_MAX_SAMPLES}" "$@"
  eval_engine "nvidia-2of4" "${PRUNED_MODEL_DIR}" "fp16" "training_data" "${TRAINING_DATASET}" "${TRAINING_MAX_SAMPLES}" "$@"
  eval_engine "dense" "${SFT_CHECKPOINT}" "fp16" "benchmark" "${BENCHMARK_DATASET}" "${BENCHMARK_MAX_SAMPLES}" "$@"
  eval_engine "nvidia-2of4" "${PRUNED_MODEL_DIR}" "fp16" "benchmark" "${BENCHMARK_DATASET}" "${BENCHMARK_MAX_SAMPLES}" "$@"
  if [[ "${RUN_INT8}" == "1" ]]; then
    eval_engine "dense" "${SFT_CHECKPOINT}" "int8" "training_data" "${TRAINING_DATASET}" "${TRAINING_MAX_SAMPLES}" "$@"
    eval_engine "nvidia-2of4" "${PRUNED_MODEL_DIR}" "int8" "training_data" "${TRAINING_DATASET}" "${TRAINING_MAX_SAMPLES}" "$@"
    eval_engine "dense" "${SFT_CHECKPOINT}" "int8" "benchmark" "${BENCHMARK_DATASET}" "${BENCHMARK_MAX_SAMPLES}" "$@"
    eval_engine "nvidia-2of4" "${PRUNED_MODEL_DIR}" "int8" "benchmark" "${BENCHMARK_DATASET}" "${BENCHMARK_MAX_SAMPLES}" "$@"
  fi
else
  echo "[skip] eval"
fi

"${PYTHON_BIN}" scripts/collect_edge_eval_summaries.py \
  --run-root "${RUN_ROOT}" \
  --output-csv "${RUN_ROOT}/nvidia_onnx_edge_em_summary.csv" \
  --output-json "${RUN_ROOT}/nvidia_onnx_edge_em_summary.json"

echo
echo "Finished. Summary:"
echo "  ${RUN_ROOT}/nvidia_onnx_edge_em_summary.csv"
echo "  ${RUN_ROOT}/nvidia_onnx_edge_em_summary.json"
