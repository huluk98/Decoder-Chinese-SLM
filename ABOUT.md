# About Decoder-Chinese-SLM

Decoder-Chinese-SLM is a compact decoder-only Chinese language model project focused on training and evaluating a 0.2B-parameter small language model for edge and domain-specific use.

The project is inspired by the public size target and Chinese data direction of `charent/ChatLM-mini-Chinese`, while using a modern causal decoder architecture instead of the original T5-style encoder-decoder format. The goal is to make a small Chinese model that is practical to train, fine-tune, evaluate, and deploy when latency, privacy, memory, and cost matter.

## Project Focus

- 0.2B-parameter decoder-only Chinese SLM.
- Llama-style causal LM architecture with RoPE, RMSNorm, SwiGLU, grouped-query attention, and BF16 training.
- Public Chinese pretraining data pipeline based on downloadable Hugging Face sources.
- 8x NVIDIA H20 training recipes with DDP, DeepSpeed, packed-token pretraining, and bounded checkpoint saving.
- Supervised fine-tuning for structured smart-home command generation.
- Prompt/response SFT where loss is computed only on response tokens.
- Exact-match and command-normalized evaluation for short generated responses.
- C-Eval evaluation support for Chinese benchmark comparison.
- Loss plotting and training metrics for model-card style reporting.
- Optional post-training pruning workflows including magnitude, Wanda, Taylor-saliency pruning, and NVIDIA 2:4-style sparsity experiments.

## Why This Size

The 0.2B parameter target is intentional. It is small enough to support fast iteration, affordable multi-GPU training, and edge-oriented deployment, while still large enough to study tokenizer choices, data quality, SFT behavior, pruning, and Chinese domain adaptation.

This model is not meant to compete with frontier-scale general chat models. It is designed as a practical research and engineering base for Chinese smart-home control, private domain assistants, command understanding, small retrieval-augmented systems, and other settings where a focused small model may be better than a larger general model.

## Training Path

The repository supports a full local workflow:

1. Download public Chinese dataset sources.
2. Normalize and merge data into a stable JSONL corpus.
3. Train or load a 29,298-token tokenizer.
4. Pack token IDs for high-throughput pretraining.
5. Train the 0.2B decoder-only model on 8x H20 GPUs.
6. Fine-tune on local prompt/response data.
7. Evaluate with C-Eval or exact-match command generation.
8. Plot loss curves and summarize training cost.
9. Optionally prune and recover the model for smaller deployment targets.

## Smart-Home SFT

The smart-home SFT path is built for short normalized responses. Prompts and responses are joined for decoder-only training, prompt tokens and padding tokens are masked with `-100`, and the model only learns from response tokens.

The default SFT generation cap is `max_new_tokens=64`, because smart-home outputs should be short. The evaluator compares generated responses with targets using command-aware normalization so harmless formatting differences, such as Chinese versus ASCII punctuation, do not hide real model quality.

## Recommended Hardware

The main training recipes target a single node with 8 NVIDIA H20 GPUs. The project also includes a 7-GPU launch path for machines where one physical GPU is occupied.

For full training, use all available H20 GPUs. For large exact-match generation evaluation, the evaluator can shard the local JSON/JSONL dataset across all 8 GPUs and write one combined accuracy report.

## Repository

Main repository:

```text
https://github.com/huluk98/Decoder-Chinese-SLM
```

Suggested citation information is available in `CITATION.cff`.
