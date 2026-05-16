# Decoder-Only Chinese Mini LM

This is a clean starter codebase for training a decoder-only autoregressive Chinese language model at roughly the same parameter budget as [`charent/ChatLM-mini-Chinese`](https://huggingface.co/charent/ChatLM-mini-Chinese), with an 8x NVIDIA H20 launch recipe for the same 0.2B target.

The upstream ChatLM-mini-Chinese model is a T5-style text-to-text model, not decoder-only. Its model card reports a 0.2B parameter model, a 29,298 token vocabulary, and public dataset sources. This project keeps the size target and Chinese data recipe, but uses a modern Llama-family causal LM.

## Why 0.2B?

The goal of this project is not to compete with frontier-scale general chat models. It is to train a compact Chinese decoder model that can be adapted for edge and domain-specific deployment.

A 0.2B parameter model is intentionally small enough to be practical when latency, memory, privacy, and deployment cost matter. It can be trained and iterated on with modest multi-GPU resources, fine-tuned for narrow Chinese domains, and served closer to users or internal systems without requiring large inference infrastructure. This makes it a useful base for domain assistants, local retrieval-augmented generation, command understanding, classification-style generation, private enterprise workflows, and other Chinese-language tasks where a specialized small model can be preferable to a much larger general one.

The size also keeps the experiment legible: training runs finish faster, ablations are cheaper, tokenizer and data choices are easier to study, and the model can still use modern decoder architecture choices such as RoPE, RMSNorm, SwiGLU, grouped-query attention, and BF16 training.

## What Is Included

- Config-driven model/training setup in `configs/`.
- Hugging Face tokenizer training with Chinese-friendly BPE special tokens.
- Dataset preprocessing that downloads all configured sources, normalizes schemas, and writes one merged JSONL before tokenizer training.
- Streaming dataset pipeline for public Hugging Face datasets and local JSONL files.
- Llama-style decoder definition backed by `transformers.LlamaForCausalLM`.
- RoPE positions, RMSNorm, SwiGLU, grouped-query attention, bias-free projections, and SDPA attention.
- Plain PyTorch training loop with checkpointing, BF16 mixed precision, gradient accumulation, and `torchrun` distributed data parallel support.
- Tiny local smoke config and sample data so the skeleton can run before downloading full corpora.

## Upstream Data Notes

ChatLM-mini-Chinese says its pretraining data comes from public single-turn dialogue sources, cleaned and formatted into parquet. The listed sources include webtext2019zh, baike_qa2019, Chinese medical dialogue data, Zhihu-KOL, BELLE instruction data, and Chinese Wikipedia dumps.

This repo now configures public Hugging Face mirrors/equivalents for those sources:

- [`YeungNLP/firefly-pretrain-dataset`](https://huggingface.co/datasets/YeungNLP/firefly-pretrain-dataset), `webText2019zh.jsonl`, community QA.
- [`ZhouLV/Chinese-Train-Datasets`](https://huggingface.co/datasets/ZhouLV/Chinese-Train-Datasets/tree/main/baike2018qa), `baike2018qa/baike_qa_train.json`, encyclopedia QA.
- [`ticoAg/Chinese-medical-dialogue`](https://huggingface.co/datasets/ticoAg/Chinese-medical-dialogue), Chinese medical QA.
- [`wangrui6/Zhihu-KOL`](https://huggingface.co/datasets/wangrui6/Zhihu-KOL), 1.01M rows, Zhihu QA.
- [`BelleGroup/train_1M_CN`](https://huggingface.co/datasets/BelleGroup/train_1M_CN), [`BelleGroup/train_2M_CN`](https://huggingface.co/datasets/BelleGroup/train_2M_CN), and [`BelleGroup/train_3.5M_CN`](https://huggingface.co/datasets/BelleGroup/train_3.5M_CN), BELLE instruction/chat data.
- [`YeungNLP/firefly-pretrain-dataset`](https://huggingface.co/datasets/YeungNLP/firefly-pretrain-dataset), `wiki_zh.jsonl`, Chinese Wikipedia-like text.

The exact cleaned upstream parquet blend is not exposed as one canonical artifact, so this project reconstructs the public recipe from downloadable sources and normalizes everything into one JSONL. The preprocessing config has `min_rows: 9000000`; if the merged dataset falls below that floor, the script raises an error instead of silently training on a small corpus.

## Conda Setup

Python 3.11 plus CUDA 12.4 PyTorch is defined in `environment.yml`.

```bash
conda env create -f environment.yml
conda activate chatlm-decoder
```

## Smoke Run

The smoke config first normalizes `data/sample_zh_dialog.jsonl` into `data/processed/smoke_normalized.jsonl`, then trains a tiny tokenizer and tiny model.

```bash
python scripts/train.py --config configs/smoke.yaml
```

Generate from the latest smoke checkpoint:

```bash
python scripts/generate.py \
  --checkpoint runs/smoke/latest \
  --prompt "如何开始学习机器学习？"
```

## Train A 0.2B-ish Model

Train a tokenizer first. This command automatically downloads every configured dataset source, normalizes the records into one JSONL file, then trains the tokenizer from that merged file. The target vocab size is 29,298 to mirror ChatLM-mini-Chinese.

```bash
python scripts/train_tokenizer.py --config configs/model_0p2b.yaml
```

To run only the download/normalization step:

```bash
python scripts/prepare_data.py --config configs/model_0p2b.yaml
```

The merged dataset is written to `data/processed/chatlm_public_sources_0p2b.jsonl`, with counts in `data/processed/chatlm_public_sources_0p2b.manifest.json`. The raw Hugging Face cache goes under `data/raw/huggingface`.

If a Hugging Face source fails with an `HTTPSConnectionPool` or read-timeout error, it is usually a transient network issue rather than a bad config. The loader has retry/backoff settings in the `data:` section, and `wangrui6/Zhihu-KOL` has extra retries because it is a common long download. Re-run with `--force-prepare` after the connection stabilizes:

```bash
HF_HUB_ENABLE_HF_TRANSFER=1 python scripts/train_tokenizer.py \
  --config configs/h20_8gpu_llama_0p2b.yaml \
  --force-prepare
```

On networks where Hugging Face is slow or blocked, try a mirror endpoint:

```bash
HF_ENDPOINT=https://hf-mirror.com HF_HUB_ENABLE_HF_TRANSFER=1 python scripts/train_tokenizer.py \
  --config configs/h20_8gpu_llama_0p2b.yaml \
  --force-prepare
```

You can also raise `data.hf_download_timeout`, `data.hf_etag_timeout`, or per-source `retries` in the YAML if one dataset is especially flaky.

Then launch training on one GPU:

```bash
python scripts/train.py --config configs/model_0p2b.yaml
```

Or launch across 8 GPUs:

```bash
torchrun --standalone --nproc_per_node=8 scripts/train.py \
  --config configs/model_0p2b.yaml
```

The default 0.2B-ish config uses:

- 24 decoder blocks
- hidden size 768
- 12 query heads and 4 key/value heads
- MLP size 2048
- sequence length 2048
- vocab size 29,298
- untied input/output embeddings

This lands near the 0.2B parameter class while using a Llama-like decoder stack.

## 8x H20 Training Recipe

NVIDIA's MIG documentation lists H20 as a Hopper/GH100 GPU with compute capability 9.0 and 96 GB memory. The provided H20 config keeps the model at the same 0.2B target, but uses BF16 training, SDPA attention, DDP over 8 GPUs, and a larger effective batch.

Train the tokenizer once:

```bash
HF_HUB_ENABLE_HF_TRANSFER=1 python scripts/train_tokenizer.py \
  --config configs/h20_8gpu_llama_0p2b.yaml
```

That command also prepares `data/processed/chatlm_public_sources_0p2b.jsonl` if it does not already exist. Use `--force-prepare` to rebuild it, or `--skip-prepare` to train the tokenizer directly from the raw configured sources.

Then launch the 0.2B Llama-style run:

```bash
HF_HUB_ENABLE_HF_TRANSFER=1 torchrun --standalone --nproc_per_node=8 scripts/train.py \
  --config configs/h20_8gpu_llama_0p2b.yaml
```

The H20 config is about the 0.2B class:

- 24 decoder blocks
- hidden size 768
- 12 query heads and 4 key/value heads
- MLP size 2048
- sequence length 2048
- per-GPU microbatch 8, gradient accumulation 8

For a first full-data run, watch memory with `nvidia-smi` and tune `train.batch_size`, `train.grad_accum_steps`, and `model.block_size`. Keep the tokenizer step single-process unless you intentionally add a shared tokenizer artifact first.

## Add More Public Sources

The preprocessing script can normalize each extra local source if it is JSONL with either:

```json
{"prompt": "...", "response": "..."}
```

or:

```json
{"text": "<|user|>\n...\n<|assistant|>\n...<|eos|>"}
```

Then add it to the config:

```yaml
data:
  sources:
    - type: local_jsonl
      path: data/my_cleaned_source.jsonl
      format: prompt_response
      prompt_fields: [prompt]
      response_fields: [response]
```

For public Hugging Face datasets, add another `type: hf` entry with the dataset `path`, `split`, and `format`. The preprocessing step will download it through `datasets.load_dataset(...)` and append normalized rows to the same merged JSONL.

## Checkpoints

Checkpoints are written under `run.output_dir` as `step-000000/` directories, plus a copied `latest/` directory for convenient generation and resume experiments.

## References

- ChatLM-mini-Chinese model card and config: [`charent/ChatLM-mini-Chinese`](https://huggingface.co/charent/ChatLM-mini-Chinese), [`config.json`](https://huggingface.co/charent/ChatLM-mini-Chinese/blob/main/config.json).
- Llama architecture fields are mapped to Hugging Face [`LlamaConfig`](https://huggingface.co/docs/transformers/model_doc/llama).
- H20 memory and compute capability reference: NVIDIA [`Supported GPUs`](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/supported-gpus.html).
- Multi-GPU launch uses PyTorch [`DistributedDataParallel`](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html); for larger-than-memory variants, PyTorch [`FSDP`](https://docs.pytorch.org/docs/stable/fsdp.html) is the natural next step.
