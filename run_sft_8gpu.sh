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
SFT_TOP_K_EXACT_MATCH="${SFT_TOP_K_EXACT_MATCH:-}"

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
eval_config = config.get("eval", {}) or {}
benchmark_file = str(
    config.get("benchmark_file")
    or config.get("benchmark_eval_file")
    or eval_config.get("benchmark_file")
    or eval_config.get("benchmark_path")
    or config.get("sft", {}).get("benchmark_path")
    or ""
)
train_path = resolve_path(train_file) if train_file else None
eval_path = resolve_path(eval_file) if eval_file else None
benchmark_path = resolve_path(benchmark_file) if benchmark_file else None
if (eval_path is None or not eval_path.exists()) and train_path is not None and train_path.exists():
    eval_path = train_path
audit_train_path = train_path if train_path is not None and eval_path is not None and train_path != eval_path else None

final_path = output_dir / "final"
final_manifest = output_dir / "final_checkpoint.json"
has_model = any(final_path.glob("*.safetensors")) or any(final_path.glob("pytorch_model*.bin"))
has_tokenizer = (final_path / "tokenizer_config.json").exists() and any((final_path / name).exists() for name in ("tokenizer.json", "tokenizer.model", "vocab.json"))
model_path = final_path if final_path.is_dir() and final_manifest.exists() and (final_path / "config.json").exists() and has_model and has_tokenizer else None

max_new_tokens = int(config.get("max_new_tokens") or config.get("generation", {}).get("max_new_tokens") or 64)
eval_batch_size = int(config.get("per_device_eval_batch_size") or config.get("train", {}).get("eval_batch_size") or 8)
bf16 = bool(config.get("bf16", True))
fp16 = bool(config.get("fp16", False))
dtype = "bf16" if bf16 else ("fp16" if fp16 else "fp32")
benchmark_runs = int(config.get("benchmark_runs") or config.get("eval", {}).get("benchmark_runs") or 5)
top_k_exact_match = int(config.get("top_k_exact_match") or eval_config.get("top_k_exact_match") or 5)
seed = int(config.get("run", {}).get("seed") or config.get("seed") or 42)
data_seed = int(config.get("data", {}).get("seed") or seed)

print(output_dir)
print(train_path or "")
print(audit_train_path or "")
print(eval_path or "")
print(model_path or "")
print(benchmark_path or "")
print(max_new_tokens)
print(eval_batch_size)
print(dtype)
print(benchmark_runs)
print(top_k_exact_match)
print(seed)
print(data_seed)
PY
)

OUTPUT_DIR="${SFT_EVAL_VALUES[0]}"
TRAIN_FILE="${SFT_EVAL_VALUES[1]}"
EVAL_AUDIT_TRAIN_FILE="${SFT_EVAL_VALUES[2]}"
EVAL_FILE="${SFT_EVAL_VALUES[3]}"
MODEL_PATH="${SFT_EVAL_VALUES[4]}"
BENCHMARK_FILE="${SFT_EVAL_VALUES[5]}"
MAX_NEW_TOKENS="${SFT_EVAL_VALUES[6]}"
EVAL_BATCH_SIZE="${SFT_EVAL_VALUES[7]}"
EVAL_DTYPE="${SFT_EVAL_VALUES[8]}"
CONFIG_BENCHMARK_RUNS="${SFT_EVAL_VALUES[9]}"
CONFIG_TOP_K_EXACT_MATCH="${SFT_EVAL_VALUES[10]}"
SFT_SEED="${SFT_EVAL_VALUES[11]}"
SFT_DATA_SEED="${SFT_EVAL_VALUES[12]}"
SFT_BENCHMARK_RUNS="${SFT_BENCHMARK_RUNS:-${CONFIG_BENCHMARK_RUNS}}"
SFT_TOP_K_EXACT_MATCH="${SFT_TOP_K_EXACT_MATCH:-${CONFIG_TOP_K_EXACT_MATCH}}"

if { [[ -n "${EVAL_FILE}" && -f "${EVAL_FILE}" ]]; } || { [[ -n "${BENCHMARK_FILE}" && -f "${BENCHMARK_FILE}" ]]; }; then
  if [[ -z "${MODEL_PATH}" || ! -d "${MODEL_PATH}" ]]; then
    echo "SFT training finished, but final eval cannot find a saved checkpoint."
    echo "Expected exactly ${OUTPUT_DIR}/final with model tensors, tokenizer files, config.json, and ${OUTPUT_DIR}/final_checkpoint.json."
    echo "Check your output_dir in ${CONFIG_PATH}: ${OUTPUT_DIR}"
    exit 1
  fi
  run_prompt_response_eval() {
    local eval_label="$1"
    local dataset_file="$2"
    local audit_train_file="$3"
    local output_name="$4"
    local fail_on_leakage="$5"

    echo "SFT training finished. Running final 8-GPU ${eval_label} eval (${SFT_BENCHMARK_RUNS} runs, exact-match@${SFT_TOP_K_EXACT_MATCH}) on: ${dataset_file}"
    echo "Using exact final SFT checkpoint from this completed run: ${MODEL_PATH}"
    local eval_args=(
      --model-path "${MODEL_PATH}"
      --dataset-file "${dataset_file}"
      --output-dir "${OUTPUT_DIR}/eval/${output_name}"
      --max-new-tokens "${MAX_NEW_TOKENS}"
      --temperature 0
      --num-beams 1
      --exact-match-top-k "${SFT_TOP_K_EXACT_MATCH}"
      --seed "${SFT_SEED}"
      --data-seed "${SFT_DATA_SEED}"
      --batch-size "${EVAL_BATCH_SIZE}"
      --dtype "${EVAL_DTYPE}"
      --benchmark-runs "${SFT_BENCHMARK_RUNS}"
    )
    if [[ -n "${audit_train_file}" && -f "${audit_train_file}" ]]; then
      eval_args+=(--train-file "${audit_train_file}")
    fi
    if [[ "${fail_on_leakage}" != "true" ]]; then
      eval_args+=(--no-fail-on-leakage)
    fi
    torchrun --standalone --nproc_per_node=8 scripts/eval_prompt_response.py "${eval_args[@]}"
  }

  if [[ -n "${EVAL_FILE}" && -f "${EVAL_FILE}" ]]; then
    run_prompt_response_eval "SFT dataset" "${EVAL_FILE}" "${EVAL_AUDIT_TRAIN_FILE}" "final_sft_dataset" "true"
  fi
  if [[ -n "${BENCHMARK_FILE}" && -f "${BENCHMARK_FILE}" ]]; then
    run_prompt_response_eval "benchmark" "${BENCHMARK_FILE}" "${TRAIN_FILE}" "final_benchmark" "false"
  fi
else
  echo "SFT training finished. Skipping final eval because no eval_file/train_file/benchmark_file exists in ${CONFIG_PATH}."
fi
