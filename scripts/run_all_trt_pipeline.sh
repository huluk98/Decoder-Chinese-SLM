#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: bash scripts/run_all_trt_pipeline.sh ./model_path ./data/eval.json [extra benchmark args...]"
  echo "Environment overrides: PYTHON=python3 OVERWRITE=1 OPT_SEQ_LEN=64 MAX_SEQ_LEN=128 MAX_NEW_TOKENS=64"
  exit 2
fi

MODEL_PATH="$1"
DATASET_PATH="$2"
shift 2

PYTHON_BIN="${PYTHON:-python3}"
MIN_SEQ_LEN="${MIN_SEQ_LEN:-1}"
OPT_SEQ_LEN="${OPT_SEQ_LEN:-64}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-128}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
CALIB_SAMPLES="${CALIB_SAMPLES:-128}"
WORKSPACE_GB="${WORKSPACE_GB:-4}"
PROMPT_FORMAT="${PROMPT_FORMAT:-raw}"
OVERWRITE_ARGS=()
if [[ "${OVERWRITE:-0}" == "1" ]]; then
  OVERWRITE_ARGS=(--overwrite)
fi

ONNX_PATH="outputs/onnx/model_decoder_nocache.onnx"

echo "[1/5] Exporting decoder ONNX"
"${PYTHON_BIN}" scripts/export_decoder_onnx.py \
  --model-path "${MODEL_PATH}" \
  --seq-len "${OPT_SEQ_LEN}" \
  --batch-size "${BATCH_SIZE}" \
  "${OVERWRITE_ARGS[@]}"

echo "[2/5] Building FP16 TensorRT engine"
"${PYTHON_BIN}" scripts/build_trt_engines.py \
  --onnx "${ONNX_PATH}" \
  --precision fp16 \
  --model-path "${MODEL_PATH}" \
  --min_seq_len "${MIN_SEQ_LEN}" \
  --opt_seq_len "${OPT_SEQ_LEN}" \
  --max_seq_len "${MAX_SEQ_LEN}" \
  --batch_size "${BATCH_SIZE}" \
  --workspace-gb "${WORKSPACE_GB}" \
  "${OVERWRITE_ARGS[@]}"

echo "[3/5] Building INT8 TensorRT engine with calibration prompts"
"${PYTHON_BIN}" scripts/build_trt_engines.py \
  --onnx "${ONNX_PATH}" \
  --precision int8 \
  --model-path "${MODEL_PATH}" \
  --calib_json "${DATASET_PATH}" \
  --calib-samples "${CALIB_SAMPLES}" \
  --prompt-format "${PROMPT_FORMAT}" \
  --min_seq_len "${MIN_SEQ_LEN}" \
  --opt_seq_len "${OPT_SEQ_LEN}" \
  --max_seq_len "${MAX_SEQ_LEN}" \
  --batch_size "${BATCH_SIZE}" \
  --workspace-gb "${WORKSPACE_GB}" \
  "${OVERWRITE_ARGS[@]}"

echo "[4/5] Attempting INT4 ModelOpt/TensorRT-LLM weight-only path"
if "${PYTHON_BIN}" scripts/build_trt_engines.py \
  --onnx "${ONNX_PATH}" \
  --precision int4 \
  --model-path "${MODEL_PATH}" \
  --calib_json "${DATASET_PATH}" \
  --calib-samples "${CALIB_SAMPLES}" \
  --prompt-format "${PROMPT_FORMAT}" \
  --min_seq_len "${MIN_SEQ_LEN}" \
  --opt_seq_len "${OPT_SEQ_LEN}" \
  --max_seq_len "${MAX_SEQ_LEN}" \
  --batch_size "${BATCH_SIZE}" \
  --workspace-gb "${WORKSPACE_GB}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  "${OVERWRITE_ARGS[@]}"; then
  echo "INT4 build succeeded."
else
  echo "INT4 build did not produce an engine in this environment; continuing with successful engines."
fi

echo "[5/5] Benchmarking available TensorRT engines"
for PRECISION in fp16 int8 int4; do
  ENGINE_PATH="outputs/trt/model_${PRECISION}.engine"
  if [[ -f "${ENGINE_PATH}" ]]; then
    echo "Benchmarking ${PRECISION}: ${ENGINE_PATH}"
    "${PYTHON_BIN}" scripts/benchmark_trt_decoder.py \
      --engine "${ENGINE_PATH}" \
      --model-path "${MODEL_PATH}" \
      --dataset "${DATASET_PATH}" \
      --precision "${PRECISION}" \
      --max-new-tokens "${MAX_NEW_TOKENS}" \
      --max-seq-len "${MAX_SEQ_LEN}" \
      --prompt-format "${PROMPT_FORMAT}" \
      "${OVERWRITE_ARGS[@]}" \
      "$@"
  else
    echo "Skipping ${PRECISION} benchmark because ${ENGINE_PATH} does not exist."
  fi
done

echo "TensorRT edge pipeline finished. Artifacts are under outputs/."

