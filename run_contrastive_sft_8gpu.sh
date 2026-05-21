#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

CONFIG_PATH="${CONFIG_PATH:-configs/contrastive_sft_8gpu.yaml}"
CONTRASTIVE_BENCHMARK_RUNS="${CONTRASTIVE_BENCHMARK_RUNS:-}"

torchrun --standalone --nproc_per_node=8 scripts/train.py \
  --config "${CONFIG_PATH}" \
  --mode contrastive

readarray -t CONTRASTIVE_EVAL_VALUES < <(python - "${CONFIG_PATH}" <<'PY'
from pathlib import Path
import sys
import yaml

config_path = Path(sys.argv[1])
with config_path.open("r", encoding="utf-8") as handle:
    config = yaml.safe_load(handle) or {}

def resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (Path.cwd() / path)

sft_config = config.get("sft", {}) or {}
train_config = config.get("train", {}) or {}
generation_config = config.get("generation", {}) or {}

output_dir = resolve_path(str(config.get("output_dir") or config.get("run", {}).get("output_dir") or "runs/contrastive-sft-0p2b-8gpu"))
train_file = str(config.get("train_file") or sft_config.get("data_path") or "")
eval_file = str(config.get("eval_file") or sft_config.get("eval_path") or "")
train_path = resolve_path(train_file) if train_file else None
eval_path = resolve_path(eval_file) if eval_file else None
if (eval_path is None or not eval_path.exists()) and train_path is not None and train_path.exists():
    eval_path = train_path
audit_train_path = train_path if train_path is not None and eval_path is not None and train_path != eval_path else None

final_path = output_dir / "final"
final_manifest = output_dir / "final_checkpoint.json"
has_model = any(final_path.glob("*.safetensors")) or any(final_path.glob("pytorch_model*.bin"))
has_tokenizer = (final_path / "tokenizer_config.json").exists() and any((final_path / name).exists() for name in ("tokenizer.json", "tokenizer.model", "vocab.json"))
model_path = final_path if final_path.is_dir() and final_manifest.exists() and (final_path / "config.json").exists() and has_model and has_tokenizer else None

max_new_tokens = int(config.get("max_new_tokens") or generation_config.get("max_new_tokens") or 64)
eval_batch_size = int(
    config.get("per_device_eval_batch_size")
    or train_config.get("eval_batch_size")
    or max(1, int(train_config.get("batch_size", 8)))
)
precision = str(train_config.get("precision", "bf16")).lower()
bf16 = bool(config.get("bf16", precision == "bf16"))
fp16 = bool(config.get("fp16", precision == "fp16"))
dtype = "bf16" if bf16 else ("fp16" if fp16 else "fp32")
benchmark_runs = int(config.get("benchmark_runs") or sft_config.get("benchmark_runs") or config.get("eval", {}).get("benchmark_runs") or 5)
seed = int(config.get("run", {}).get("seed") or config.get("seed") or 42)
data_seed = int(config.get("data", {}).get("seed") or seed)

print(output_dir)
print(audit_train_path or "")
print(eval_path or "")
print(model_path or "")
print(max_new_tokens)
print(eval_batch_size)
print(dtype)
print(benchmark_runs)
print(seed)
print(data_seed)
PY
)

OUTPUT_DIR="${CONTRASTIVE_EVAL_VALUES[0]}"
TRAIN_FILE="${CONTRASTIVE_EVAL_VALUES[1]}"
EVAL_FILE="${CONTRASTIVE_EVAL_VALUES[2]}"
MODEL_PATH="${CONTRASTIVE_EVAL_VALUES[3]}"
MAX_NEW_TOKENS="${CONTRASTIVE_EVAL_VALUES[4]}"
EVAL_BATCH_SIZE="${CONTRASTIVE_EVAL_VALUES[5]}"
EVAL_DTYPE="${CONTRASTIVE_EVAL_VALUES[6]}"
CONFIG_BENCHMARK_RUNS="${CONTRASTIVE_EVAL_VALUES[7]}"
CONTRASTIVE_SEED="${CONTRASTIVE_EVAL_VALUES[8]}"
CONTRASTIVE_DATA_SEED="${CONTRASTIVE_EVAL_VALUES[9]}"
CONTRASTIVE_BENCHMARK_RUNS="${CONTRASTIVE_BENCHMARK_RUNS:-${CONFIG_BENCHMARK_RUNS}}"

if [[ -n "${EVAL_FILE}" && -f "${EVAL_FILE}" ]]; then
  if [[ -z "${MODEL_PATH}" || ! -d "${MODEL_PATH}" ]]; then
    echo "Contrastive SFT finished, but final eval cannot find a saved checkpoint."
    echo "Expected exactly ${OUTPUT_DIR}/final with model tensors, tokenizer files, config.json, and ${OUTPUT_DIR}/final_checkpoint.json."
    echo "Check your output_dir in ${CONFIG_PATH}: ${OUTPUT_DIR}"
    exit 1
  fi
  echo "Contrastive SFT finished. Running final 8-GPU exact-match benchmark (${CONTRASTIVE_BENCHMARK_RUNS} runs) on: ${EVAL_FILE}"
  echo "Using exact final contrastive SFT checkpoint from this completed run: ${MODEL_PATH}"
  torchrun --standalone --nproc_per_node=8 scripts/eval_prompt_response.py \
    --model-path "${MODEL_PATH}" \
    --dataset-file "${EVAL_FILE}" \
    --train-file "${TRAIN_FILE}" \
    --output-dir "${OUTPUT_DIR}/eval/final_prompt_response_benchmark" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --temperature 0 \
    --num-beams 1 \
    --seed "${CONTRASTIVE_SEED}" \
    --data-seed "${CONTRASTIVE_DATA_SEED}" \
    --batch-size "${EVAL_BATCH_SIZE}" \
    --dtype "${EVAL_DTYPE}" \
    --benchmark-runs "${CONTRASTIVE_BENCHMARK_RUNS}"
else
  echo "Contrastive SFT finished. Skipping final eval because no eval_path/data_path exists in ${CONFIG_PATH}."
fi
