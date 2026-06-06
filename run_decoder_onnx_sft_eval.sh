#!/usr/bin/env bash
set -euo pipefail

ORIGINAL_CWD="$(pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

usage() {
  cat <<'EOF'
Run the decoder-only base SFT ONNX edge format:
  1. train regular base SFT for 5 epochs from a base decoder checkpoint;
  2. export the trained SFT checkpoint to no-cache FP32 ONNX;
  3. benchmark FP32, FP16, and INT8 ONNX Runtime precision/provider rows;
  4. build FP16 and INT8 TensorRT engines from the ONNX graph;
  5. evaluate latency, P95 latency, memory, EM@1, and EM@5;
  6. write CSV/JSON summaries plus LaTeX ONNX result tables.

Usage:
  bash run_decoder_onnx_sft_eval.sh /path/to/base_decoder_model

Common overrides:
  PYTHON=/path/to/python
  RUN_ROOT=runs/decoder-onnx-sft-eval
  SFT_EPOCHS=5
  SFT_OUTPUT_DIR=runs/decoder-onnx-sft-eval/training/base_sft_5epoch
  TRAIN_NPROC_PER_NODE=8
  TRAINING_DATASET=data/scenic/SCENIC_full_training_dataset.json
  BENCHMARK_DATASET=data/benchmarks/iot_instruction_benchmark_200.json
  OVERWRITE=1
  MAX_SEQ_LEN=128
  OPT_SEQ_LEN=64
  MAX_NEW_TOKENS=64
  EXACT_MATCH_TOP_K=5
  RUN_INT8=1
  SHARDED_EVAL=1
  EVAL_NPROC_PER_NODE=8
  EVAL_CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  BENCHMARK_MAX_SAMPLES=0    # 0 means full benchmark dataset
  MAX_SAMPLES=20             # optional quick smoke override
  CALIB_SAMPLES=128
  RUN_ONNX_PRECISION_BENCHMARK=1
  ONNX_BENCHMARK_PROVIDERS="CPUExecutionProvider CUDAExecutionProvider"
  ONNX_BENCHMARK_WARMUP=30
  ONNX_BENCHMARK_RUNS=200
  DEVICE_NAME="Jetson Orin Nano / Raspberry Pi 5 / Snapdragon / etc."
  POWER_LOG=/path/to/power.csv # optional; energy stays N/A if omitted
  EDGE_ATTN_IMPLEMENTATION=eager
  PROMPT_FORMAT=raw          # raw, legacy, or chat-template
  COMPARISON_MODE=whitespace # whitespace, normalized, or command
  TRUST_REMOTE_CODE=1
  SKIP_TRAIN=1               # reuse SFT_OUTPUT_DIR/final
  SKIP_EXPORT=1
  SKIP_ONNX_PRECISION_BENCHMARK=1
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
RUN_ROOT="${RUN_ROOT:-runs/decoder-onnx-sft-eval}"
SFT_EPOCHS="${SFT_EPOCHS:-5}"
SFT_OUTPUT_DIR="${SFT_OUTPUT_DIR:-${RUN_ROOT}/training/base_sft_5epoch}"
SFT_CHECKPOINT="${SFT_CHECKPOINT:-${SFT_OUTPUT_DIR}/final}"
TRAINING_DATASET="$(resolve_input_path "${TRAINING_DATASET:-data/scenic/SCENIC_full_training_dataset.json}")"
BENCHMARK_DATASET="$(resolve_input_path "${BENCHMARK_DATASET:-data/benchmarks/iot_instruction_benchmark_200.json}")"
CALIB_DATASET="${CALIB_DATASET:-${TRAINING_DATASET}}"
CALIB_DATASET="$(resolve_input_path "${CALIB_DATASET}")"
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
BENCHMARK_MAX_SAMPLES="${BENCHMARK_MAX_SAMPLES:-0}"
EXPORT_DTYPE="${EXPORT_DTYPE:-fp32}"
OPSET="${OPSET:-18}"
RUN_INT8="${RUN_INT8:-1}"
RUN_ONNX_PRECISION_BENCHMARK="${RUN_ONNX_PRECISION_BENCHMARK:-1}"
ONNX_BENCHMARK_OUTPUT_DIR="${ONNX_BENCHMARK_OUTPUT_DIR:-${RUN_ROOT}/onnx-precision-benchmark/base-sft}"
ONNX_BENCHMARK_PROVIDERS="${ONNX_BENCHMARK_PROVIDERS:-CPUExecutionProvider CUDAExecutionProvider}"
ONNX_BENCHMARK_WARMUP="${ONNX_BENCHMARK_WARMUP:-30}"
ONNX_BENCHMARK_RUNS="${ONNX_BENCHMARK_RUNS:-200}"
ONNX_BENCHMARK_TABLE_FORMATS="${ONNX_BENCHMARK_TABLE_FORMATS:-csv markdown latex}"
ONNX_QUANTIZATION_MODE="${ONNX_QUANTIZATION_MODE:-static}"
ONNX_QUANT_FORMAT="${ONNX_QUANT_FORMAT:-qdq}"
ONNX_DRIFT_SAMPLES="${ONNX_DRIFT_SAMPLES:-16}"
ONNX_NUM_THREADS="${ONNX_NUM_THREADS:-}"
POWER_LOG="${POWER_LOG:-}"
POWER_COLUMN="${POWER_COLUMN:-power_w}"
TIMESTAMP_COLUMN="${TIMESTAMP_COLUMN:-timestamp_s}"
DEVICE_NAME="${DEVICE_NAME:-}"
EDGE_ATTN_IMPLEMENTATION="${EDGE_ATTN_IMPLEMENTATION:-eager}"
TRAIN_NPROC_PER_NODE="${TRAIN_NPROC_PER_NODE:-8}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
SHARDED_EVAL="${SHARDED_EVAL:-1}"
EVAL_NPROC_PER_NODE="${EVAL_NPROC_PER_NODE:-${TRAIN_NPROC_PER_NODE}}"
EVAL_CUDA_VISIBLE_DEVICES="${EVAL_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES}}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

export CUDA_VISIBLE_DEVICES
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS

if [[ -n "${MAX_SAMPLES:-}" && "${MAX_SAMPLES}" != "0" ]]; then
  BENCHMARK_MAX_SAMPLES="${MAX_SAMPLES}"
fi

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

mkdir -p "${RUN_ROOT}/generated_configs"

SFT_CONFIG_PATH="${RUN_ROOT}/generated_configs/base_sft_5epoch.yaml"
cat > "${SFT_CONFIG_PATH}" <<EOF
model_name_or_path: "${BASE_DECODER_MODEL}"
train_file: "${TRAINING_DATASET}"
eval_file: "${TRAINING_DATASET}"
benchmark_file: "${BENCHMARK_DATASET}"
output_dir: "${SFT_OUTPUT_DIR}"

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

train_sft() {
  echo
  echo "[train] base decoder SFT (${SFT_EPOCHS} epochs)"
  echo "  base:   ${BASE_DECODER_MODEL}"
  echo "  config: ${SFT_CONFIG_PATH}"
  echo "  output: ${SFT_OUTPUT_DIR}"
  if [[ "${SKIP_TRAIN:-0}" == "1" ]]; then
    echo "[skip] train; expecting existing checkpoint at ${SFT_CHECKPOINT}"
  else
    torchrun --standalone --nproc_per_node="${TRAIN_NPROC_PER_NODE}" scripts/train.py \
      --config "${SFT_CONFIG_PATH}"
  fi
  if [[ ! -d "${SFT_CHECKPOINT}" || ! -f "${SFT_CHECKPOINT}/config.json" ]]; then
    echo "SFT checkpoint not found: ${SFT_CHECKPOINT}"
    echo "Expected ${SFT_OUTPUT_DIR}/final. Check training logs and SFT_OUTPUT_DIR."
    exit 1
  fi
}

export_model() {
  local onnx_dir="${RUN_ROOT}/onnx/base-sft"
  echo
  echo "[export] base-sft ONNX: ${SFT_CHECKPOINT}"
  "${PYTHON_BIN}" scripts/export_decoder_onnx.py \
    --model-path "${SFT_CHECKPOINT}" \
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

run_onnx_precision_benchmark() {
  local onnx_path="${RUN_ROOT}/onnx/base-sft/model_decoder_nocache.onnx"
  if [[ ! -f "${onnx_path}" ]]; then
    echo "ONNX model not found for precision benchmark: ${onnx_path}"
    echo "Run without SKIP_EXPORT=1 or set RUN_ONNX_PRECISION_BENCHMARK=0."
    exit 1
  fi

  local provider_args=()
  local table_format_args=()
  local power_args=()
  local device_args=()
  local thread_args=()
  local iobinding_args=()
  local profile_args=()

  read -r -a provider_args <<< "${ONNX_BENCHMARK_PROVIDERS}"
  read -r -a table_format_args <<< "${ONNX_BENCHMARK_TABLE_FORMATS}"

  if [[ -n "${POWER_LOG}" ]]; then
    power_args=(--power-log "${POWER_LOG}" --power-column "${POWER_COLUMN}" --timestamp-column "${TIMESTAMP_COLUMN}")
  fi
  if [[ -n "${DEVICE_NAME}" ]]; then
    device_args=(--device-name "${DEVICE_NAME}")
  fi
  if [[ -n "${ONNX_NUM_THREADS}" ]]; then
    thread_args=(--num-threads "${ONNX_NUM_THREADS}")
  fi
  if [[ "${ONNX_DISABLE_IOBINDING:-0}" == "1" ]]; then
    iobinding_args=(--disable-iobinding)
  fi
  if [[ "${ONNX_PROFILE_ORT:-0}" == "1" ]]; then
    profile_args=(--profile-ort)
  fi

  echo
  echo "[benchmark] ONNX Runtime precision/provider comparison"
  echo "  fp32 onnx:  ${onnx_path}"
  echo "  output dir: ${ONNX_BENCHMARK_OUTPUT_DIR}"
  echo "  providers:  ${ONNX_BENCHMARK_PROVIDERS}"
  "${PYTHON_BIN}" tools/benchmark_onnx_precision.py \
    --fp32-onnx "${onnx_path}" \
    --output-dir "${ONNX_BENCHMARK_OUTPUT_DIR}" \
    --providers "${provider_args[@]}" \
    --batch-size "${BATCH_SIZE}" \
    --warmup "${ONNX_BENCHMARK_WARMUP}" \
    --runs "${ONNX_BENCHMARK_RUNS}" \
    --calibration-samples "${CALIB_SAMPLES}" \
    --quantization-mode "${ONNX_QUANTIZATION_MODE}" \
    --quant-format "${ONNX_QUANT_FORMAT}" \
    --table-formats "${table_format_args[@]}" \
    --dataset "${BENCHMARK_DATASET}" \
    --model-path "${SFT_CHECKPOINT}" \
    --prompt-format "${PROMPT_FORMAT}" \
    "${SYSTEM_PROMPT_ARGS[@]}" \
    --max-seq-len "${MAX_SEQ_LEN}" \
    --drift-samples "${ONNX_DRIFT_SAMPLES}" \
    "${power_args[@]}" \
    "${device_args[@]}" \
    "${thread_args[@]}" \
    "${iobinding_args[@]}" \
    "${profile_args[@]}" \
    "${TRUST_ARGS[@]}"
}

build_engine() {
  local precision="$1"
  local onnx_path="${RUN_ROOT}/onnx/base-sft/model_decoder_nocache.onnx"
  local trt_dir="${RUN_ROOT}/trt/base-sft"
  echo
  echo "[build] ONNX -> TensorRT ${precision}"
  if [[ "${precision}" == "int8" ]]; then
    "${PYTHON_BIN}" scripts/build_trt_engines.py \
      --onnx "${onnx_path}" \
      --output-dir "${trt_dir}" \
      --precision int8 \
      --model-path "${SFT_CHECKPOINT}" \
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
      "${TRUST_ARGS[@]}" \
      "${OVERWRITE_ARGS[@]}"
  else
    "${PYTHON_BIN}" scripts/build_trt_engines.py \
      --onnx "${onnx_path}" \
      --output-dir "${trt_dir}" \
      --precision fp16 \
      --model-path "${SFT_CHECKPOINT}" \
      --min_seq_len "${MIN_SEQ_LEN}" \
      --opt_seq_len "${OPT_SEQ_LEN}" \
      --max_seq_len "${MAX_SEQ_LEN}" \
      --batch_size "${BATCH_SIZE}" \
      --workspace-gb "${WORKSPACE_GB}" \
      "${TRUST_ARGS[@]}" \
      "${OVERWRITE_ARGS[@]}"
  fi
}

eval_engine() {
  local precision="$1"
  local onnx_path="${RUN_ROOT}/onnx/base-sft/model_decoder_nocache.onnx"
  local engine_path="${RUN_ROOT}/trt/base-sft/model_${precision}.engine"
  local output_dir="${RUN_ROOT}/eval/base-sft/${precision}/benchmark"
  local limit_args=()
  local limit_label="full"
  if [[ -n "${BENCHMARK_MAX_SAMPLES}" && "${BENCHMARK_MAX_SAMPLES}" != "0" ]]; then
    limit_args=(--limit "${BENCHMARK_MAX_SAMPLES}")
    limit_label="${BENCHMARK_MAX_SAMPLES}"
  fi
  echo
  echo "[eval] ONNX/TensorRT ${precision} on benchmark (limit=${limit_label})"
  "${PYTHON_BIN}" scripts/eval_trt_prompt_response.py \
    --engine "${engine_path}" \
    --model-path "${SFT_CHECKPOINT}" \
    --dataset "${BENCHMARK_DATASET}" \
    --output-dir "${output_dir}" \
    --precision "${precision}" \
    --variant "base_sft_${precision}" \
    --runtime "ONNX/TensorRT" \
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

eval_engine_sharded() {
  local precision="$1"
  shift 1
  local onnx_path="${RUN_ROOT}/onnx/base-sft/model_decoder_nocache.onnx"
  local engine_path="${RUN_ROOT}/trt/base-sft/model_${precision}.engine"
  local output_dir="${RUN_ROOT}/eval/base-sft/${precision}/benchmark"
  local shard_root="${RUN_ROOT}/eval-shards/base-sft/${precision}/benchmark"
  local limit_args=()
  local limit_label="full"
  if [[ -n "${BENCHMARK_MAX_SAMPLES}" && "${BENCHMARK_MAX_SAMPLES}" != "0" ]]; then
    limit_args=(--limit "${BENCHMARK_MAX_SAMPLES}")
    limit_label="${BENCHMARK_MAX_SAMPLES}"
  fi

  IFS=',' read -r -a eval_gpus <<< "${EVAL_CUDA_VISIBLE_DEVICES}"
  if (( EVAL_NPROC_PER_NODE < 1 )); then
    echo "EVAL_NPROC_PER_NODE must be at least 1."
    exit 1
  fi
  if (( EVAL_NPROC_PER_NODE > ${#eval_gpus[@]} )); then
    echo "EVAL_NPROC_PER_NODE=${EVAL_NPROC_PER_NODE} exceeds EVAL_CUDA_VISIBLE_DEVICES count (${#eval_gpus[@]}): ${EVAL_CUDA_VISIBLE_DEVICES}"
    exit 1
  fi

  if [[ "${OVERWRITE:-1}" == "1" ]]; then
    rm -rf "${shard_root}" "${output_dir}"
  fi
  mkdir -p "${shard_root}"

  echo
  echo "[eval] sharded ONNX/TensorRT ${precision} on benchmark (shards=${EVAL_NPROC_PER_NODE}, limit=${limit_label})"
  local pids=()
  local shard_index
  for (( shard_index = 0; shard_index < EVAL_NPROC_PER_NODE; shard_index++ )); do
    local gpu="${eval_gpus[$shard_index]}"
    local shard_output_dir="${shard_root}/shard_${shard_index}"
    echo "  shard ${shard_index}/${EVAL_NPROC_PER_NODE} -> CUDA_VISIBLE_DEVICES=${gpu}"
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" scripts/eval_trt_prompt_response.py \
      --engine "${engine_path}" \
      --model-path "${SFT_CHECKPOINT}" \
      --dataset "${BENCHMARK_DATASET}" \
      --output-dir "${shard_output_dir}" \
      --precision "${precision}" \
      --variant "base_sft_${precision}" \
      --runtime "ONNX/TensorRT" \
      --onnx-path "${onnx_path}" \
      --batch-size "${BATCH_SIZE}" \
      --max-new-tokens "${MAX_NEW_TOKENS}" \
      --max-seq-len "${MAX_SEQ_LEN}" \
      --exact-match-top-k "${EXACT_MATCH_TOP_K}" \
      --max-new-token-hit-rate-threshold "${MAX_NEW_TOKEN_HIT_RATE_THRESHOLD}" \
      --prompt-format "${PROMPT_FORMAT}" \
      "${SYSTEM_PROMPT_ARGS[@]}" \
      --comparison-mode "${COMPARISON_MODE}" \
      --num-shards "${EVAL_NPROC_PER_NODE}" \
      --shard-index "${shard_index}" \
      "${limit_args[@]}" \
      "${TRUST_ARGS[@]}" \
      "${OVERWRITE_ARGS[@]}" \
      "$@" &
    pids+=("$!")
  done

  local failed=0
  local pid
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      failed=1
    fi
  done
  if [[ "${failed}" != "0" ]]; then
    echo "One or more ${precision} eval shards failed."
    exit 1
  fi

  "${PYTHON_BIN}" scripts/merge_trt_sharded_eval.py \
    --shard-root "${shard_root}" \
    --output-dir "${output_dir}" \
    "${OVERWRITE_ARGS[@]}"
}

echo "Decoder base SFT ONNX eval"
echo "  base decoder:    ${BASE_DECODER_MODEL}"
echo "  SFT checkpoint:  ${SFT_CHECKPOINT}"
echo "  training data:   ${TRAINING_DATASET}"
echo "  benchmark data:  ${BENCHMARK_DATASET}"
echo "  run root:        ${RUN_ROOT}"
echo "  run INT8:        ${RUN_INT8}"
echo "  ONNX precision: ${RUN_ONNX_PRECISION_BENCHMARK}"
echo "  ORT providers:  ${ONNX_BENCHMARK_PROVIDERS}"
echo "  attention impl:  ${EDGE_ATTN_IMPLEMENTATION}"
echo "  SFT epochs:      ${SFT_EPOCHS}"
echo "  export dtype:    ${EXPORT_DTYPE}"
echo "  train GPUs:      ${TRAIN_NPROC_PER_NODE}"
echo "  sharded eval:    ${SHARDED_EVAL}"
echo "  eval GPUs:       ${EVAL_NPROC_PER_NODE} (${EVAL_CUDA_VISIBLE_DEVICES})"
echo "  benchmark limit: ${BENCHMARK_MAX_SAMPLES} (0 means full)"
echo

train_sft

if [[ "${SKIP_EXPORT:-0}" != "1" ]]; then
  export_model
else
  echo "[skip] export"
fi

if [[ "${RUN_ONNX_PRECISION_BENCHMARK}" == "1" && "${SKIP_ONNX_PRECISION_BENCHMARK:-0}" != "1" ]]; then
  run_onnx_precision_benchmark
else
  echo "[skip] ONNX precision benchmark"
fi

if [[ "${SKIP_BUILD:-0}" != "1" ]]; then
  build_engine "fp16"
  if [[ "${RUN_INT8}" == "1" ]]; then
    build_engine "int8"
  fi
else
  echo "[skip] build"
fi

if [[ "${SKIP_EVAL:-0}" != "1" ]]; then
  if [[ "${SHARDED_EVAL}" == "1" ]]; then
    eval_engine_sharded "fp16" "$@"
  else
    eval_engine "fp16" "$@"
  fi
  if [[ "${RUN_INT8}" == "1" ]]; then
    if [[ "${SHARDED_EVAL}" == "1" ]]; then
      eval_engine_sharded "int8" "$@"
    else
      eval_engine "int8" "$@"
    fi
  fi
else
  echo "[skip] eval"
fi

"${PYTHON_BIN}" scripts/collect_edge_eval_summaries.py \
  --run-root "${RUN_ROOT}" \
  --output-csv "${RUN_ROOT}/decoder_onnx_edge_summary.csv" \
  --output-json "${RUN_ROOT}/decoder_onnx_edge_summary.json"

"${PYTHON_BIN}" scripts/format_decoder_onnx_table.py \
  --run-root "${RUN_ROOT}" \
  --dataset benchmark \
  --architecture "Base SFT" \
  --output-tex "${RUN_ROOT}/decoder_onnx_table.tex" \
  --output-accuracy-tex "${RUN_ROOT}/decoder_onnx_accuracy_levels.tex"

echo
echo "Finished. Summary:"
echo "  ${RUN_ROOT}/decoder_onnx_edge_summary.csv"
echo "  ${RUN_ROOT}/decoder_onnx_edge_summary.json"
echo "  ${RUN_ROOT}/decoder_onnx_table.tex"
echo "  ${RUN_ROOT}/decoder_onnx_accuracy_levels.tex"
if [[ "${RUN_ONNX_PRECISION_BENCHMARK}" == "1" && "${SKIP_ONNX_PRECISION_BENCHMARK:-0}" != "1" ]]; then
  echo "  ${ONNX_BENCHMARK_OUTPUT_DIR}/onnx_precision_benchmark.md"
  echo "  ${ONNX_BENCHMARK_OUTPUT_DIR}/onnx_precision_benchmark_summary.txt"
fi
