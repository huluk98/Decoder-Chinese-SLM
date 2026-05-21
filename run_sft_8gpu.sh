#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=WARN
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8

CONFIG_PATH="${CONFIG_PATH:-configs/sft_0p2b_8gpu.yaml}"
SFT_BENCHMARK_RUNS="${SFT_BENCHMARK_RUNS:-}"

torchrun --standalone --nproc_per_node=8 scripts/train.py \
  --config "${CONFIG_PATH}"

readarray -t SFT_EVAL_VALUES < <(python - "${CONFIG_PATH}" <<'PY'
from pathlib import Path
import sys
import yaml

config_path = Path(sys.argv[1])
with config_path.open("r", encoding="utf-8") as handle:
    config = yaml.safe_load(handle) or {}

def resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (Path.cwd() / path)


output_dir = resolve_path(str(config.get("output_dir") or config.get("run", {}).get("output_dir") or "outputs/sft_0p2b_8gpu"))
train_file = str(config.get("train_file") or config.get("sft", {}).get("data_path") or "")
eval_file = str(config.get("eval_file") or config.get("sft", {}).get("eval_path") or "")
train_path = resolve_path(train_file) if train_file else None
eval_path = resolve_path(eval_file) if eval_file else None
if (eval_path is None or not eval_path.exists()) and train_path is not None and train_path.exists():
    eval_path = train_path

latest_path = output_dir / "latest"
model_path = latest_path if latest_path.exists() else None
if model_path is None:
    checkpoints = sorted(
        [path for path in output_dir.glob("step-*") if path.is_dir()],
        key=lambda path: path.name,
    )
    if checkpoints:
        model_path = checkpoints[-1]

max_new_tokens = int(config.get("max_new_tokens") or config.get("generation", {}).get("max_new_tokens") or 64)
eval_batch_size = int(config.get("per_device_eval_batch_size") or config.get("train", {}).get("eval_batch_size") or 8)
bf16 = bool(config.get("bf16", True))
fp16 = bool(config.get("fp16", False))
dtype = "bf16" if bf16 else ("fp16" if fp16 else "fp32")
benchmark_runs = int(config.get("benchmark_runs") or config.get("eval", {}).get("benchmark_runs") or 5)

print(output_dir)
print(eval_path or "")
print(model_path or "")
print(max_new_tokens)
print(eval_batch_size)
print(dtype)
print(benchmark_runs)
PY
)

OUTPUT_DIR="${SFT_EVAL_VALUES[0]}"
EVAL_FILE="${SFT_EVAL_VALUES[1]}"
MODEL_PATH="${SFT_EVAL_VALUES[2]}"
MAX_NEW_TOKENS="${SFT_EVAL_VALUES[3]}"
EVAL_BATCH_SIZE="${SFT_EVAL_VALUES[4]}"
EVAL_DTYPE="${SFT_EVAL_VALUES[5]}"
CONFIG_BENCHMARK_RUNS="${SFT_EVAL_VALUES[6]}"
SFT_BENCHMARK_RUNS="${SFT_BENCHMARK_RUNS:-${CONFIG_BENCHMARK_RUNS}}"

if [[ -n "${EVAL_FILE}" && -f "${EVAL_FILE}" ]]; then
  if [[ -z "${MODEL_PATH}" || ! -d "${MODEL_PATH}" ]]; then
    echo "SFT training finished, but final eval cannot find a saved checkpoint."
    echo "Expected ${OUTPUT_DIR}/latest or a ${OUTPUT_DIR}/step-* directory."
    echo "Check your output_dir in ${CONFIG_PATH}: ${OUTPUT_DIR}"
    exit 1
  fi
  echo "SFT training finished. Running final 8-GPU exact-match benchmark (${SFT_BENCHMARK_RUNS} runs) on: ${EVAL_FILE}"
  echo "Using model checkpoint: ${MODEL_PATH}"
  torchrun --standalone --nproc_per_node=8 scripts/eval_prompt_response.py \
    --model-path "${MODEL_PATH}" \
    --dataset-file "${EVAL_FILE}" \
    --output-dir "${OUTPUT_DIR}/eval/final_prompt_response_benchmark" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --temperature 0 \
    --num-beams 1 \
    --batch-size "${EVAL_BATCH_SIZE}" \
    --dtype "${EVAL_DTYPE}" \
    --benchmark-runs "${SFT_BENCHMARK_RUNS}"
else
  echo "SFT training finished. Skipping final eval because no eval_file/train_file exists in ${CONFIG_PATH}."
fi
