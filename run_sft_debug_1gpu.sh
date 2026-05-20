#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=7
export TOKENIZERS_PARALLELISM=false

python scripts/train.py \
  --config configs/sft_0p2b_8gpu.yaml \
  --debug_overfit_samples 32 \
  --max_steps 100 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 1
