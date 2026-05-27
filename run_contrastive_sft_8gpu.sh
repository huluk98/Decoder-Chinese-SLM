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
CONTRASTIVE_TOP_K_EXACT_MATCH="${CONTRASTIVE_TOP_K_EXACT_MATCH:-}"

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
eval_config = config.get("eval", {}) or {}

output_dir = resolve_path(str(config.get("output_dir") or config.get("run", {}).get("output_dir") or "runs/contrastive-sft-0p2b-8gpu"))
train_file = str(config.get("train_file") or sft_config.get("data_path") or "")
eval_file = str(
    config.get("anchor_eval_file")
    or eval_config.get("anchor_file")
    or eval_config.get("anchor_path")
    or sft_config.get("anchor_eval_path")
    or config.get("eval_file")
    or sft_config.get("eval_path")
    or ""
)
benchmark_file = str(
    config.get("benchmark_file")
    or config.get("benchmark_eval_file")
    or eval_config.get("benchmark_file")
    or eval_config.get("benchmark_path")
    or sft_config.get("benchmark_path")
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
top_k_exact_match = int(config.get("top_k_exact_match") or eval_config.get("top_k_exact_match") or sft_config.get("top_k_exact_match") or 5)
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

OUTPUT_DIR="${CONTRASTIVE_EVAL_VALUES[0]}"
TRAIN_FILE="${CONTRASTIVE_EVAL_VALUES[1]}"
ANCHOR_AUDIT_TRAIN_FILE="${CONTRASTIVE_EVAL_VALUES[2]}"
ANCHOR_EVAL_FILE="${CONTRASTIVE_EVAL_VALUES[3]}"
MODEL_PATH="${CONTRASTIVE_EVAL_VALUES[4]}"
BENCHMARK_FILE="${CONTRASTIVE_EVAL_VALUES[5]}"
MAX_NEW_TOKENS="${CONTRASTIVE_EVAL_VALUES[6]}"
EVAL_BATCH_SIZE="${CONTRASTIVE_EVAL_VALUES[7]}"
EVAL_DTYPE="${CONTRASTIVE_EVAL_VALUES[8]}"
CONFIG_BENCHMARK_RUNS="${CONTRASTIVE_EVAL_VALUES[9]}"
CONFIG_TOP_K_EXACT_MATCH="${CONTRASTIVE_EVAL_VALUES[10]}"
CONTRASTIVE_SEED="${CONTRASTIVE_EVAL_VALUES[11]}"
CONTRASTIVE_DATA_SEED="${CONTRASTIVE_EVAL_VALUES[12]}"
CONTRASTIVE_BENCHMARK_RUNS="${CONTRASTIVE_BENCHMARK_RUNS:-${CONFIG_BENCHMARK_RUNS}}"
CONTRASTIVE_TOP_K_EXACT_MATCH="${CONTRASTIVE_TOP_K_EXACT_MATCH:-${CONFIG_TOP_K_EXACT_MATCH}}"

if { [[ -n "${ANCHOR_EVAL_FILE}" && -f "${ANCHOR_EVAL_FILE}" ]]; } || { [[ -n "${BENCHMARK_FILE}" && -f "${BENCHMARK_FILE}" ]]; }; then
  if [[ -z "${MODEL_PATH}" || ! -d "${MODEL_PATH}" ]]; then
    echo "Contrastive SFT finished, but final eval cannot find a saved checkpoint."
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

    echo "Contrastive SFT finished. Running final 8-GPU ${eval_label} eval (${CONTRASTIVE_BENCHMARK_RUNS} runs, exact-match@${CONTRASTIVE_TOP_K_EXACT_MATCH}) on: ${dataset_file}"
    echo "Using exact final contrastive SFT checkpoint from this completed run: ${MODEL_PATH}"
    local eval_args=(
      --model-path "${MODEL_PATH}"
      --dataset-file "${dataset_file}"
      --output-dir "${OUTPUT_DIR}/eval/${output_name}"
      --max-new-tokens "${MAX_NEW_TOKENS}"
      --temperature 0
      --num-beams 1
      --exact-match-top-k "${CONTRASTIVE_TOP_K_EXACT_MATCH}"
      --seed "${CONTRASTIVE_SEED}"
      --data-seed "${CONTRASTIVE_DATA_SEED}"
      --batch-size "${EVAL_BATCH_SIZE}"
      --dtype "${EVAL_DTYPE}"
      --benchmark-runs "${CONTRASTIVE_BENCHMARK_RUNS}"
    )
    if [[ -n "${audit_train_file}" && -f "${audit_train_file}" ]]; then
      eval_args+=(--train-file "${audit_train_file}")
    fi
    if [[ "${fail_on_leakage}" != "true" ]]; then
      eval_args+=(--no-fail-on-leakage)
    fi
    torchrun --standalone --nproc_per_node=8 scripts/eval_prompt_response.py "${eval_args[@]}"
  }

  if [[ -n "${ANCHOR_EVAL_FILE}" && -f "${ANCHOR_EVAL_FILE}" ]]; then
    run_prompt_response_eval "anchor-only" "${ANCHOR_EVAL_FILE}" "${ANCHOR_AUDIT_TRAIN_FILE}" "final_anchor" "false"
  fi
  if [[ -n "${BENCHMARK_FILE}" && -f "${BENCHMARK_FILE}" ]]; then
    run_prompt_response_eval "benchmark" "${BENCHMARK_FILE}" "${TRAIN_FILE}" "final_benchmark" "false"
  fi
else
  echo "Contrastive SFT finished. Skipping final eval because no anchor eval/data_path/benchmark_file exists in ${CONFIG_PATH}."
fi
