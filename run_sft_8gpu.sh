#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=WARN
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8

CONFIG_PATH="${CONFIG_PATH:-configs/sft_0p2b_8gpu.yaml}"

torchrun --standalone --nproc_per_node=8 scripts/train.py \
  --config "${CONFIG_PATH}"

readarray -t SFT_EVAL_VALUES < <(python - "${CONFIG_PATH}" <<'PY'
from pathlib import Path
import sys
import yaml

config_path = Path(sys.argv[1])
with config_path.open("r", encoding="utf-8") as handle:
    config = yaml.safe_load(handle) or {}

output_dir = str(config.get("output_dir") or config.get("run", {}).get("output_dir") or "outputs/sft_0p2b_8gpu")
train_file = str(config.get("train_file") or config.get("sft", {}).get("data_path") or "")
eval_file = str(config.get("eval_file") or config.get("sft", {}).get("eval_path") or "")
if (not eval_file or not Path(eval_file).expanduser().exists()) and train_file and Path(train_file).expanduser().exists():
    eval_file = train_file

max_new_tokens = int(config.get("max_new_tokens") or config.get("generation", {}).get("max_new_tokens") or 64)
eval_batch_size = int(config.get("per_device_eval_batch_size") or config.get("train", {}).get("eval_batch_size") or 8)
bf16 = bool(config.get("bf16", True))
fp16 = bool(config.get("fp16", False))
dtype = "bf16" if bf16 else ("fp16" if fp16 else "fp32")

print(output_dir)
print(eval_file)
print(max_new_tokens)
print(eval_batch_size)
print(dtype)
PY
)

OUTPUT_DIR="${SFT_EVAL_VALUES[0]}"
EVAL_FILE="${SFT_EVAL_VALUES[1]}"
MAX_NEW_TOKENS="${SFT_EVAL_VALUES[2]}"
EVAL_BATCH_SIZE="${SFT_EVAL_VALUES[3]}"
EVAL_DTYPE="${SFT_EVAL_VALUES[4]}"

if [[ -n "${EVAL_FILE}" && -f "${EVAL_FILE}" ]]; then
  echo "SFT training finished. Running final 8-GPU exact-match eval on: ${EVAL_FILE}"
  torchrun --standalone --nproc_per_node=8 scripts/eval_prompt_response.py \
    --model-path "${OUTPUT_DIR}/latest" \
    --dataset-file "${EVAL_FILE}" \
    --output-dir "${OUTPUT_DIR}/eval/final_prompt_response" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --temperature 0 \
    --num-beams 1 \
    --batch-size "${EVAL_BATCH_SIZE}" \
    --dtype "${EVAL_DTYPE}"
else
  echo "SFT training finished. Skipping final eval because no eval_file/train_file exists in ${CONFIG_PATH}."
fi
