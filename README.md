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
- Dataset staging that downloads all configured Hugging Face sources locally before normalization.
- Dataset preprocessing that normalizes cached/local sources and writes one merged JSONL before tokenizer training.
- Streaming dataset pipeline for public Hugging Face datasets and local JSONL files.
- Llama-style decoder definition backed by `transformers.LlamaForCausalLM`.
- RoPE positions, RMSNorm, SwiGLU, grouped-query attention, bias-free projections, and SDPA attention.
- Plain PyTorch training loop with checkpointing, BF16 mixed precision, gradient accumulation, and `torchrun` distributed data parallel support.
- Rank-0 metrics logging plus loss-curve generation for model-card comparisons.
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

The config pins actual remote file names for the sources that need them: `data/train_0001_of_0001.json` for `ticoAg/Chinese-medical-dialogue`, the five `data/train-00000-of-00005-*.parquet` through `data/train-00004-of-00005-*.parquet` shards for `wangrui6/Zhihu-KOL`, `Belle_open_source_1M.json`, `train_2M_CN.json`, and `train_3.5M_CN.json` for the BELLE datasets, plus the Firefly and Baike files listed above. This avoids relying on Hugging Face auto-discovery when downloading local snapshots.

The exact cleaned upstream parquet blend is not exposed as one canonical artifact, so this project reconstructs the public recipe from downloadable sources and normalizes everything into one JSONL. The full-data configs keep `min_rows: 9000000` as a target-size warning. They also set `download_first: true` and `continue_on_source_error: true`, so the pipeline first downloads dataset snapshots into the local Hugging Face cache, then normalizes from local/cache-backed files. A flaky source is logged in the manifest and the preprocessor moves on to the next source instead of discarding hours of completed work.

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

Train a tokenizer first. This command automatically downloads every configured dataset source into the local cache, normalizes the cached/local records into one JSONL file, then trains the tokenizer from that merged file. The target vocab size is 29,298 to mirror ChatLM-mini-Chinese.

```bash
python scripts/train_tokenizer.py --config configs/model_0p2b.yaml
```

To run only the download stage:

```bash
python scripts/download_data.py --config configs/model_0p2b.yaml
```

To run the download and normalization stages:

```bash
python scripts/prepare_data.py --config configs/model_0p2b.yaml
```

The download manifest is written to `data/raw/chatlm_public_sources_0p2b.download_manifest.json`. The merged dataset is written to `data/processed/chatlm_public_sources_0p2b.jsonl`, with counts in `data/processed/chatlm_public_sources_0p2b.manifest.json`. The raw Hugging Face cache goes under `data/raw/huggingface`.

If a Hugging Face source fails with an `HTTPSConnectionPool` or read-timeout error, it is usually a transient network issue rather than a bad config. The downloader has retry/backoff settings in the `data:` section, and `wangrui6/Zhihu-KOL` has extra retries because it is a common long download. After retries are exhausted, the full-data configs skip that source, write the error in the download manifest, and continue with the next dataset.

If the download stage is interrupted, rerun the same command. Hugging Face cache snapshots resume/reuse files under `data/raw/huggingface`, so you do not need to start from zero. Use `--force-download` only when you intentionally want fresh remote copies.

If the normalization stage is interrupted, you may see `data/processed/chatlm_public_sources_0p2b.jsonl.tmp`. That file is only the in-progress write target. It is not used by training or tokenizer scripts, and it can be deleted before a clean rebuild:

```bash
rm -f data/processed/chatlm_public_sources_0p2b.jsonl.tmp
```

Re-run normalization from the downloaded cache after the connection stabilizes:

```bash
HF_HUB_ENABLE_HF_TRANSFER=1 python scripts/train_tokenizer.py \
  --config configs/h20_8gpu_llama_0p2b.yaml \
  --force-prepare
```

`--force-prepare` rebuilds the normalized JSONL. It does not force a full re-download. Add `--force-download` only when you want to refresh the cached dataset snapshots too.

On networks where Hugging Face is slow or blocked, try a mirror endpoint:

```bash
HF_ENDPOINT=https://hf-mirror.com HF_HUB_ENABLE_HF_TRANSFER=1 python scripts/download_data.py \
  --config configs/h20_8gpu_llama_0p2b.yaml \
  --force-download
```

You can also raise `data.hf_download_timeout`, `data.hf_etag_timeout`, or per-source `retries` in the YAML if one dataset is especially flaky.

If you already have a completed `data/processed/chatlm_public_sources_0p2b.jsonl`, the tokenizer and training scripts use that final JSONL file and do not need the `.tmp` file. Check the row count with:

```bash
wc -l data/processed/chatlm_public_sources_0p2b.jsonl
```

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

That command also downloads source snapshots and prepares `data/processed/chatlm_public_sources_0p2b.jsonl` if it does not already exist. Use `--force-prepare` to rebuild normalized data from the cache, `--force-download` to refresh remote snapshots, or `--skip-prepare` to train the tokenizer directly from the raw configured sources.

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

## 7x H20 When GPU 1 Is Busy

Use this one-line launch command when physical GPU 1 is occupied and training should use only GPUs 0, 2, 3, 4, 5, 6, and 7:

```bash
CUDA_VISIBLE_DEVICES=0,2,3,4,5,6,7 HF_HUB_ENABLE_HF_TRANSFER=1 NCCL_DEBUG=WARN TORCH_NCCL_ASYNC_ERROR_HANDLING=1 torchrun --standalone --nproc_per_node=7 scripts/train.py --config configs/h20_7gpu_llama_0p2b_fast.yaml
```

The 7-GPU fast config keeps the same 0.2B Llama-style model shape, but uses `train.batch_size: 16` and `train.grad_accum_steps: 5`, giving `7 * 16 * 5 = 560` samples per optimizer update.

The H20 configs also set `train.tf32: true` and `train.float32_matmul_precision: high`, which enables TensorFloat-32 Tensor Cores for any FP32 matrix multiplication paths while keeping the main training precision at BF16.

If the 7-GPU ETA looks higher than the old 8-GPU ETA, compare token throughput rather than only wall-clock ETA. The 8-GPU config runs `8 * 8 * 8 = 512` sequences per optimizer step, while the 7-GPU fast config runs `7 * 16 * 5 = 560`; that is 9.4% more tokens per step on 12.5% fewer GPUs, so a fixed `max_steps: 100000` run naturally has a longer ETA. The progress bar reports `tok_s` and `step_s` so you can check real throughput.

This training script supports two multi-GPU backends:

- `configs/h20_7gpu_llama_0p2b_fast.yaml` uses plain PyTorch DDP.
- `configs/h20_7gpu_llama_0p2b_deepspeed.yaml` uses DeepSpeed with BF16, FusedAdam when available, and ZeRO-1 optimizer partitioning.

The conda environment installs DeepSpeed. If you are managing packages manually on the H20 machine, install the optional extra with `pip install -e ".[deepspeed]"`.

Use this one-line DeepSpeed launch when physical GPU 1 is occupied:

```bash
CUDA_VISIBLE_DEVICES=0,2,3,4,5,6,7 HF_HUB_ENABLE_HF_TRANSFER=1 NCCL_DEBUG=WARN TORCH_NCCL_ASYNC_ERROR_HANDLING=1 deepspeed --num_gpus=7 scripts/train.py --config configs/h20_7gpu_llama_0p2b_deepspeed.yaml
```

DeepSpeed is not required for this 0.2B model to fit in 96 GB+ H20 memory, so ZeRO-1 is the first recommended DeepSpeed mode. ZeRO-2 or ZeRO-3 can save more optimizer/parameter memory, but they add extra communication and are usually slower for a model this small unless memory pressure is the real bottleneck. For throughput, compare `tok_s` and `step_s` between the DDP and DeepSpeed commands, and then tune `train.batch_size`, `train.grad_accum_steps`, `train.num_workers`, `train.pin_memory`, and whether `model.gradient_checkpointing` is worth the recompute overhead on your GPUs.

You do not need exactly 24 layers to stay near 0.2B parameters. The current model is roughly 196M parameters with 24 layers, `hidden_size: 768`, and `intermediate_size: 2048`. Other Llama-style shapes in the same class include about 197M parameters at 20 layers with hidden size 832 and MLP 2240, about 206M at 18 layers with hidden size 896 and MLP 2368, or about 195M at 12 layers with hidden size 1024 and MLP 2752. Fewer wider layers can improve hardware utilization, but they change the model shape and you should treat that as a new run, not a resume of the 24-layer checkpoints.

## Loss Curves For Comparison

The trainer writes rank-0 metrics to:

```text
runs/<run-name>/metrics/training_metrics.csv
```

Use the plotting script to generate ChatLM-mini-Chinese-style loss images for a README or model card:

```bash
python scripts/plot_loss.py \
  --metrics runs/h20-7gpu-llama-0p2b-deepspeed/metrics/training_metrics.csv \
  --stage pretrain \
  --title "Decoder-Chinese-SLM 0.2B Pretraining Loss"
```

This writes `pretrain_loss.png`, `loss.png`, and `pretrain_loss_lr.png` beside the metrics CSV. Use `--output-dir img` if you want the PNGs in a model-card image folder. The CSV stores `step`, averaged causal-LM `loss`, `lr`, `tokens_per_second`, `seconds_per_step`, `world_size`, and effective batch details so the decoder-only run can be compared against the same-size ChatLM-mini-Chinese reference transparently.

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

For public Hugging Face datasets, add another `type: hf` entry with the dataset `path`, `split`, and `format`. The download step stores a local dataset snapshot first, and the preprocessing step appends normalized rows from that local/cache-backed source to the merged JSONL.

For a different local dataset blend, make a new config or change `preprocess.output_path` so you do not overwrite the current merged file:

```yaml
preprocess:
  output_path: data/processed/my_domain_mix.jsonl
  manifest_path: data/processed/my_domain_mix.manifest.json

data:
  sources:
    - type: local_jsonl
      path: /absolute/path/to/my_domain_data.jsonl
      format: prompt_response
      prompt_fields: [prompt]
      response_fields: [response]
```

Then rebuild that blend:

```bash
python scripts/prepare_data.py --config configs/my_domain_mix.yaml --force
```

To pull the latest code onto another desktop, clone once:

```bash
git clone https://github.com/huluk98/Decoder-Chinese-SLM.git
```

Inside an existing clone, update it with:

```bash
git pull origin main
```

## Checkpoints

Checkpoints are written under `run.output_dir` as `step-000000/` directories, plus a copied `latest/` directory for convenient generation and resume experiments.

## References

- ChatLM-mini-Chinese model card and config: [`charent/ChatLM-mini-Chinese`](https://huggingface.co/charent/ChatLM-mini-Chinese), [`config.json`](https://huggingface.co/charent/ChatLM-mini-Chinese/blob/main/config.json).
- Llama architecture fields are mapped to Hugging Face [`LlamaConfig`](https://huggingface.co/docs/transformers/model_doc/llama).
- H20 memory and compute capability reference: NVIDIA [`Supported GPUs`](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/supported-gpus.html).
- Multi-GPU launch uses PyTorch [`DistributedDataParallel`](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html); for larger-than-memory variants, PyTorch [`FSDP`](https://docs.pytorch.org/docs/stable/fsdp.html) is the natural next step.
