#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=WARN
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8

CONFIG_PATH="${CONFIG_PATH:-configs/sft_qwen25_0p5b_instruct.yaml}"
QWEN_BENCHMARK_RUNS="${QWEN_BENCHMARK_RUNS:-}"

torchrun --standalone --nproc_per_node=8 scripts/sft_qwen25_instruct.py \
  --config "${CONFIG_PATH}"

readarray -t QWEN_EVAL_VALUES < <(python - "${CONFIG_PATH}" <<'PY'
from pathlib import Path
import sys
import yaml

config_path = Path(sys.argv[1])
with config_path.open("r", encoding="utf-8") as handle:
    config = yaml.safe_load(handle) or {}


def resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (Path.cwd() / path)


output_dir = resolve_path(str(config.get("output_dir") or "outputs/qwen25_0p5b_instruct_sft"))
train_file = str(config.get("train_file") or "")
eval_file = str(config.get("eval_file") or "")
train_path = resolve_path(train_file) if train_file else None
eval_path = resolve_path(eval_file) if eval_file else None

final_path = output_dir / "final"
final_manifest = output_dir / "final_checkpoint.json"
has_model = any(final_path.glob("*.safetensors")) or any(final_path.glob("pytorch_model*.bin"))
has_tokenizer = (final_path / "tokenizer_config.json").exists() and any(
    (final_path / name).exists() for name in ("tokenizer.json", "tokenizer.model", "vocab.json")
)
model_path = (
    final_path
    if final_path.is_dir() and final_manifest.exists() and (final_path / "config.json").exists() and has_model and has_tokenizer
    else None
)

max_new_tokens = int(config.get("max_new_tokens") or config.get("generation", {}).get("max_new_tokens") or 64)
eval_batch_size = int(config.get("per_device_eval_batch_size") or 16)
bf16 = bool(config.get("bf16", True))
fp16 = bool(config.get("fp16", False))
dtype = "bf16" if bf16 else ("fp16" if fp16 else "fp32")
benchmark_runs = int(config.get("benchmark_runs") or 5)
system_prompt = str(config.get("system_prompt") or "")
seed = int(config.get("seed") or 42)

print(output_dir)
print(train_path or "")
print(eval_path or "")
print(model_path or "")
print(max_new_tokens)
print(eval_batch_size)
print(dtype)
print(benchmark_runs)
print(system_prompt)
print(seed)
PY
)

OUTPUT_DIR="${QWEN_EVAL_VALUES[0]}"
TRAIN_FILE="${QWEN_EVAL_VALUES[1]}"
EVAL_FILE="${QWEN_EVAL_VALUES[2]}"
MODEL_PATH="${QWEN_EVAL_VALUES[3]}"
MAX_NEW_TOKENS="${QWEN_EVAL_VALUES[4]}"
EVAL_BATCH_SIZE="${QWEN_EVAL_VALUES[5]}"
EVAL_DTYPE="${QWEN_EVAL_VALUES[6]}"
CONFIG_BENCHMARK_RUNS="${QWEN_EVAL_VALUES[7]}"
SYSTEM_PROMPT="${QWEN_EVAL_VALUES[8]}"
QWEN_SEED="${QWEN_EVAL_VALUES[9]}"
QWEN_BENCHMARK_RUNS="${QWEN_BENCHMARK_RUNS:-${CONFIG_BENCHMARK_RUNS}}"

if [[ -z "${MODEL_PATH}" || ! -d "${MODEL_PATH}" ]]; then
  echo "Qwen SFT training finished, but final eval cannot find the checkpoint from this run."
  echo "Expected exactly ${OUTPUT_DIR}/final with model tensors, tokenizer files, config.json, and ${OUTPUT_DIR}/final_checkpoint.json."
  exit 1
fi

if [[ -z "${EVAL_FILE}" || ! -f "${EVAL_FILE}" ]]; then
  echo "Qwen SFT training finished, but eval_file is missing or does not exist in ${CONFIG_PATH}: ${EVAL_FILE}"
  echo "Set eval_file in ${CONFIG_PATH} before running the full train+eval launcher."
  exit 1
fi

TRAIN_FILE_ARG=()
if [[ -n "${TRAIN_FILE}" && -f "${TRAIN_FILE}" && "${TRAIN_FILE}" != "${EVAL_FILE}" ]]; then
  TRAIN_FILE_ARG=(--train-file "${TRAIN_FILE}")
fi

echo "Qwen2.5-Instruct SFT finished. Running final 8-GPU Qwen chat-template benchmark (${QWEN_BENCHMARK_RUNS} runs)."
echo "Using exact final SFT checkpoint from this completed run: ${MODEL_PATH}"

torchrun --standalone --nproc_per_node=8 scripts/eval_qwen25_instruct.py \
  --model-path "${MODEL_PATH}" \
  --dataset-file "${EVAL_FILE}" \
  "${TRAIN_FILE_ARG[@]}" \
  --output-dir "${OUTPUT_DIR}/eval/final_qwen25_instruct_benchmark" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --benchmark-runs "${QWEN_BENCHMARK_RUNS}" \
  --batch-size "${EVAL_BATCH_SIZE}" \
  --dtype "${EVAL_DTYPE}" \
  --seed "${QWEN_SEED}" \
  --system-prompt "${SYSTEM_PROMPT}"
