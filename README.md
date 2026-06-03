# Decoder-Only Chinese Mini LM

This is a clean starter codebase for training a decoder-only autoregressive Chinese language model at roughly the same parameter budget as [`charent/ChatLM-mini-Chinese`](https://huggingface.co/charent/ChatLM-mini-Chinese), with an 8x NVIDIA H20 launch recipe for the same 0.2B target.

The upstream ChatLM-mini-Chinese model is a T5-style text-to-text model, not decoder-only. Its model card reports a 0.2B parameter model, a 29,298 token vocabulary, and public dataset sources. This project keeps the size target and Chinese data recipe, but uses a modern Llama-family causal LM.

For a concise project overview, see [ABOUT.md](ABOUT.md).

## Why 0.2B?

The goal of this project is not to compete with frontier-scale general chat models. It is to train a compact Chinese decoder model that can be adapted for edge and domain-specific deployment.

A 0.2B parameter model is intentionally small enough to be practical when latency, memory, privacy, and deployment cost matter. It can be trained and iterated on with modest multi-GPU resources, fine-tuned for narrow Chinese domains, and served closer to users or internal systems without requiring large inference infrastructure. This makes it a useful base for domain assistants, local retrieval-augmented generation, command understanding, classification-style generation, private enterprise workflows, and other Chinese-language tasks where a specialized small model can be preferable to a much larger general one.

The size also keeps the experiment legible: training runs finish faster, ablations are cheaper, tokenizer and data choices are easier to study, and the model can still use modern decoder architecture choices such as RoPE, RMSNorm, SwiGLU, grouped-query attention, and BF16 training.

## What Is Included

- Config-driven model/training setup in `configs/`.
- Hugging Face tokenizer training with Chinese-friendly BPE special tokens.
- Dataset staging that downloads all configured Hugging Face sources locally before normalization.
- Dataset preprocessing that normalizes cached/local sources and writes one merged JSONL before tokenizer training.
- Optional packed-token pretraining path that stores token IDs once and streams fixed-length blocks from disk.
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

## Training Method Focus

The goal is to follow the useful parts of ChatLM-mini-Chinese's training method while keeping this project decoder-only. ChatLM-mini-Chinese uses cleaned public Chinese single-turn dialogue data, tokenizer-first preparation, streaming/shuffled loading, training logs, Accelerate, and arbitrary stop/resume support. This repo mirrors those ideas for causal LM pretraining:

- Normalize public Chinese prompt/response, QA, BELLE, Zhihu, medical, web, and wiki-like sources into one cleaned corpus.
- Train or load the 29,298-token tokenizer before pretraining so model vocab and token IDs stay fixed.
- Prefer the packed-token path for H20 runs so each GPU reads fixed 2048-token causal-LM blocks instead of repeatedly tokenizing JSONL rows.
- Log true mean causal-LM loss, learning rate, throughput, world size, effective global batch, and tokens per step.
- Save bounded Hugging Face safetensors checkpoints and support resuming from `latest`.
- Catch graceful stop signals and save at the next optimizer-step boundary.

Unlike the upstream T5-style model, this project does not use text-to-text encoder-decoder pretraining or masked prediction. Each normalized dialogue/text record is concatenated into decoder-only causal-LM text, then the model learns next-token prediction over 2048-token packed blocks.

## Conda Setup

Python 3.11 plus CUDA 12.4 PyTorch is defined in `environment.yml`.

```bash
conda env create -f environment.yml
conda activate chatlm-decoder
```

For an existing environment:

```bash
conda env update -f environment.yml --prune
conda activate chatlm-decoder
pip install -e ".[deepspeed]"
```

## Start-To-Finish Workflow

For a fresh 8x H20 training run:

```bash
git clone https://github.com/huluk98/Decoder-Chinese-SLM.git
cd Decoder-Chinese-SLM
conda env create -f environment.yml
conda activate chatlm-decoder
pip install -e ".[deepspeed]"
HF_HUB_ENABLE_HF_TRANSFER=1 python scripts/download_data.py --config configs/h20_8gpu_llama_0p2b_deepspeed.yaml
python scripts/prepare_data.py --config configs/h20_8gpu_llama_0p2b_deepspeed.yaml
python scripts/train_tokenizer.py --config configs/h20_8gpu_llama_0p2b_deepspeed.yaml
python scripts/pack_tokens.py --config configs/h20_8gpu_llama_0p2b_deepspeed.yaml
./scripts/launch_h20_8gpu.sh
```

For the 7-GPU run that skips physical GPU 1:

```bash
./scripts/launch_h20_7gpu_no_gpu1.sh
```

Resume after stopping or crashing:

```bash
./scripts/launch_h20_8gpu.sh --resume runs/h20-8gpu-llama-0p2b-deepspeed/latest
```

Monitor loss, throughput, GPU-hours, and estimated tokens:

```bash
tail -f runs/h20-8gpu-llama-0p2b-deepspeed/metrics/training_metrics.csv
python scripts/summarize_training_run.py runs/h20-8gpu-llama-0p2b-deepspeed
```

After training:

```bash
python scripts/plot_loss.py --metrics runs/h20-8gpu-llama-0p2b-deepspeed/metrics/training_metrics.csv
python scripts/eval_ceval.py --checkpoint runs/h20-8gpu-llama-0p2b-deepspeed/latest --subjects all --split val --n-shot 5
```

Optional post-pretraining alignment and pruning:

```bash
python scripts/sft.py --config configs/sft.yaml --checkpoint runs/h20-8gpu-llama-0p2b-deepspeed/latest
python scripts/sft.py --config configs/contrastive_sft.yaml --mode contrastive --checkpoint runs/sft-0p2b/latest
CHECKPOINT=runs/contrastive-sft-0p2b/latest ./scripts/run_pruning_suite.sh
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

## Run Locally With Custom Paths

For local inference, point `--checkpoint` at any Hugging Face-style checkpoint directory that contains `config.json`, tokenizer files, and `model.safetensors`.

Single prompt:

```bash
python scripts/generate.py \
  --checkpoint /absolute/path/to/your/checkpoint \
  --prompt "请介绍一下边缘端中文小模型的优势。" \
  --device cuda:0 \
  --dtype bf16 \
  --max-new-tokens 256
```

Run a local dataset file and write generations to JSONL:

```bash
python scripts/generate.py \
  --checkpoint /absolute/path/to/your/checkpoint \
  --dataset-file /absolute/path/to/prompts.jsonl \
  --text-field prompt \
  --output-file runs/local_generations.jsonl \
  --device cuda:0 \
  --dtype bf16
```

`--dataset-file` accepts `.jsonl`, `.json`, `.csv`, or `.txt`. For JSONL/JSON/CSV, the script automatically looks for `prompt`, `instruction`, `question`, `input`, `text`, or `query`; pass `--text-field your_column_name` if your file uses a different column. For base pretraining checkpoints, add `--no-chat-format` if you want the raw prompt without `<|user|>` and `<|assistant|>` wrappers.

Evaluate a local instruction/response SFT file by generated exact-match accuracy:

```bash
python scripts/eval_prompt_response.py \
  --model-path /absolute/path/to/sft/checkpoint \
  --dataset-file /absolute/path/to/eval.json \
  --max-new-tokens 64 \
  --temperature 0 \
  --batch-size 8 \
  --device cuda:0 \
  --dtype bf16
```

The eval file can be `.json`, `.jsonl`, or `.csv`. Rows may use `instruction` + `response`, `prompt` + `response`, `question` + `answer`, or a `messages`/`conversations` transcript. By default the script generates every row and reports whitespace-insensitive exact-match accuracy with `--comparison-mode whitespace`. Use `--comparison-mode normalized` for light formatting cleanup, `--comparison-mode command` only when intentionally measuring semantic smart-home command equivalence, or `--no-exact-match` when you want loss/perplexity without generation.

Add `--exact-match-top-k 5` to also run deterministic beam search and report whether the target appears in the top five generated candidates. The summaries include `exact_match_at_5_accuracy` and `top5_exact_match_accuracy` aliases when K is 5.

For large eval files on your 8x H20 machine, shard generation across all GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 TOKENIZERS_PARALLELISM=false \
torchrun --standalone --nproc_per_node=8 scripts/eval_prompt_response.py \
  --model-path /absolute/path/to/sft/checkpoint \
  --dataset-file /absolute/path/to/eval.json \
  --output-dir runs/eval/my_smart_home_eval \
  --max-new-tokens 64 \
  --temperature 0 \
  --num-beams 1 \
  --batch-size 8 \
  --dtype bf16 \
  --benchmark-runs 5
```

With `--benchmark-runs 5`, rank 0 writes `run_01` through `run_05` prediction folders plus `prompt_response_eval_benchmark_summary.json`, including mean ± sample std for loss, perplexity, exact-match accuracy, exact-match@K, generated length, and eval wall time.

For audit-safe comparisons, eval now writes a unique timestamped child folder under `--output-dir` plus `run_config.json`, `split_audit.json`, `metrics.json`, `prediction_debug.csv`, and `prompt_response_eval_predictions.jsonl`. Compare two runs with:

```bash
python scripts/compare_predictions.py \
  /path/to/old/prompt_response_eval_predictions.jsonl \
  /path/to/new/prompt_response_eval_predictions.jsonl \
  --output-json runs/eval/compare_predictions.json
```

For local SFT/fine-tuning with your chosen model and dataset paths:

```bash
python scripts/sft.py \
  --config configs/sft.yaml \
  --checkpoint /absolute/path/to/base/checkpoint \
  --data-path /absolute/path/to/sft_train.jsonl \
  --output-dir runs/my-local-sft \
  --epochs 3
```

## 8-GPU Smart-Home SFT

For structured smart-home command generation, use the dedicated short-output SFT config:

1. Edit `configs/sft_0p2b_8gpu.yaml`.
2. Set `model_name_or_path` to the pretrained 0.2B checkpoint directory.
3. Set `train_file` and `eval_file` to JSON/JSONL files containing prompt/response rows. The checked-in 8-GPU config now points both at `data/scenic/SCENIC_full_training_dataset.json` and uses `data/benchmarks/iot_instruction_benchmark_200.json` as `benchmark_file`.

The SFT trainer formats each row as decoder-only `prompt + response`, masks all prompt and padding labels with `-100`, and computes loss only on response tokens. The 8-GPU launch script trains through the configured epochs first, then runs a five-pass exact-match generation benchmark by default. It does not stop at every epoch for full-dataset validation.

Debug overfit on one GPU first:

```bash
./run_sft_debug_1gpu.sh
```

Full 8x H20 SFT run followed by final 8-GPU exact-match eval:

```bash
./run_sft_8gpu.sh
```

For this repo's updated smart-home run, set only the base checkpoint in `configs/sft_0p2b_8gpu.yaml`; the SFT data, SFT eval data, benchmark file, five benchmark repeats, and exact-match@5 are already wired:

```yaml
model_name_or_path: /absolute/path/to/pretrained-or-pruned-checkpoint
train_file: data/scenic/SCENIC_full_training_dataset.json
eval_file: data/scenic/SCENIC_full_training_dataset.json
benchmark_file: data/benchmarks/iot_instruction_benchmark_200.json
benchmark_runs: 5
top_k_exact_match: 5
```

Then run:

```bash
CONFIG_PATH=configs/sft_0p2b_8gpu.yaml ./run_sft_8gpu.sh
```

The final reports land under:

```text
outputs/sft_0p2b_8gpu/eval/final_sft_dataset/
outputs/sft_0p2b_8gpu/eval/final_benchmark/
```

Read `prompt_response_eval_summary.json` or `metrics.json` in each folder for `exact_match_accuracy`, `exact_match_at_5_accuracy`, and `top5_exact_match_accuracy`.

Equivalent one-line launch:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 TOKENIZERS_PARALLELISM=false NCCL_DEBUG=WARN PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8 torchrun --standalone --nproc_per_node=8 scripts/train.py --config configs/sft_0p2b_8gpu.yaml
```

The one-line `torchrun` command trains only. `./run_sft_8gpu.sh` trains, then launches `scripts/eval_prompt_response.py` on all 8 GPUs using exactly `output_dir/final`, the checkpoint written by the run that just completed. It does not fall back to `latest` or `step-*`, so stale SFT checkpoints cannot be evaluated by accident. The launcher evaluates the SFT dataset and `benchmark_file`, reporting exact match and exact-match@5 by default. Set `benchmark_runs` or `top_k_exact_match` in `configs/sft_0p2b_8gpu.yaml`, or override once with `SFT_BENCHMARK_RUNS=3 SFT_TOP_K_EXACT_MATCH=5 ./run_sft_8gpu.sh`.

Default SFT settings are `num_train_epochs=3`, `max_seq_length=128`, `max_new_tokens=64`, BF16, TF32, per-device batch size 16, gradient accumulation 1, cosine LR, `eval_strategy=none`, and `save_final_only=true`. Startup logging prints world size, local rank, GPU name, effective batch size, trainable parameter count, sequence length, generation cap, and a decoded tokenized sample showing the supervised response region.

## Qwen2.5-0.5B-Instruct SFT

Qwen2.5-Instruct must use its official chat template. Do not run it through the legacy decoder SFT path that inserts `<|user|>`, `<|assistant|>`, `<|system|>`, and `<|eos|>` tokens.

1. Edit `configs/sft_qwen25_0p5b_instruct.yaml`.
2. Set `train_file` and `eval_file` to your local JSON/JSONL/CSV files.
3. Leave `model_name_or_path: Qwen/Qwen2.5-0.5B-Instruct`, or point it at a Qwen2.5-Instruct-compatible local checkpoint.

Full 8x H20 run with final five-pass eval:

```bash
./run_qwen25_instruct_sft_8gpu.sh
```

Train-only one-liner:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 TOKENIZERS_PARALLELISM=false NCCL_DEBUG=WARN PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8 torchrun --standalone --nproc_per_node=8 scripts/sft_qwen25_instruct.py --config configs/sft_qwen25_0p5b_instruct.yaml
```

The Qwen data path builds prompts with `tokenizer.apply_chat_template(..., add_generation_prompt=True)`, appends `response + tokenizer.eos_token`, and masks all prompt and padding labels with `-100`. The evaluator uses the same Qwen chat template, decodes only continuation tokens, caps generation at `max_new_tokens=64`, and writes `qwen25_instruct_eval_summary.json`, `qwen25_instruct_predictions.jsonl`, `qwen25_instruct_prediction_debug.csv`, and `failed_examples_20.json`.

Before trusting a full run, use the required overfit diagnostic:

```bash
python scripts/sft_qwen25_instruct.py \
  --config configs/sft_qwen25_0p5b_instruct.yaml \
  --debug-overfit-samples 20 \
  --epochs 20 \
  --output-dir outputs/debug_qwen25_instruct_overfit

python scripts/eval_qwen25_instruct.py \
  --model-path outputs/debug_qwen25_instruct_overfit/final \
  --dataset-file /absolute/path/to/the/same/train.json \
  --output-dir outputs/debug_qwen25_instruct_overfit/eval \
  --limit 20 \
  --max-new-tokens 64 \
  --dtype bf16
```

The debug exact match should approach 100%. If it does not, treat any previous Qwen2.5-Instruct result, especially a 0% result from the legacy formatter, as invalid until the chat-template pipeline is fixed.

For model comparison, keep the paths separate:

- Qwen2.5-0.5B base using the legacy decoder SFT path: `scripts/sft.py` and `scripts/eval_prompt_response.py`.
- Qwen2.5-0.5B-Instruct using the Qwen chat-template path: `scripts/sft_qwen25_instruct.py` and `scripts/eval_qwen25_instruct.py`.
- LLaMA-style 0.2B using the legacy decoder SFT path: `scripts/sft.py` and `scripts/eval_prompt_response.py`.

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

Your H20 machine reports about 143711 MiB per GPU, so the H20 configs now spend memory to improve throughput instead of saving memory too aggressively. They keep the same 0.2B Llama-style model shape, but use BF16, TF32 for any FP32 matmul paths, SDPA flash kernels when PyTorch can dispatch them, larger per-GPU microbatches, fewer accumulation steps, persistent DataLoader workers, and gradient checkpointing off by default.

Train the tokenizer once:

```bash
HF_HUB_ENABLE_HF_TRANSFER=1 python scripts/train_tokenizer.py \
  --config configs/h20_8gpu_llama_0p2b_deepspeed.yaml
```

That command also downloads source snapshots and prepares `data/processed/chatlm_public_sources_0p2b.jsonl` if it does not already exist. Use `--force-prepare` to rebuild normalized data from the cache, `--force-download` to refresh remote snapshots, or `--skip-prepare` to train the tokenizer directly from the raw configured sources.

For the fastest H20 pretraining path, pack the normalized text into token IDs once before the GPU run:

```bash
python scripts/pack_tokens.py --config configs/h20_8gpu_llama_0p2b_deepspeed.yaml
```

This writes `data/processed/chatlm_public_sources_0p2b.tokens.uint16.bin` plus a manifest. The H20 configs automatically train from that packed file when it exists, avoiding repeated tokenizer work in every DataLoader worker. The H20 launch scripts also run this packing preflight before starting GPU training; it is idempotent and returns immediately when the packed file already exists. Set `PACK_TOKENS=0` only for a deliberate debug run where you want to skip this check.

Then launch the 8-GPU run:

```bash
./scripts/launch_h20_8gpu.sh
```

That script defaults to Accelerate as the process launcher and DeepSpeed ZeRO-1 as the training backend. It verifies that exactly 8 CUDA devices are visible and that the Accelerate config launches 8 processes, then uses physical GPUs 0-7:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 HF_HUB_ENABLE_HF_TRANSFER=1 NCCL_DEBUG=WARN TORCH_NCCL_ASYNC_ERROR_HANDLING=1 accelerate launch --config_file configs/accelerate_h20_8gpu.yaml scripts/train.py --config configs/h20_8gpu_llama_0p2b_deepspeed.yaml
```

The 8-GPU speed config stays in the 0.2B class:

- 24 decoder blocks
- hidden size 768
- 12 query heads and 4 key/value heads
- MLP size 2048
- sequence length 2048
- per-GPU microbatch 32, gradient accumulation 2
- effective batch `8 * 32 * 2 = 512` sequences per optimizer update

The launcher exits with a clear error if the 8-GPU run is started with fewer than 8 visible devices or an Accelerate config that launches fewer than 8 workers.

If you want plain DDP instead of DeepSpeed, use:

```bash
LAUNCHER=torchrun CONFIG=configs/h20_8gpu_llama_0p2b.yaml ./scripts/launch_h20_8gpu.sh
```

## 7x H20 When GPU 1 Is Busy

Use this script when physical GPU 1 is occupied and training should use only GPUs 0, 2, 3, 4, 5, 6, and 7:

```bash
./scripts/launch_h20_7gpu_no_gpu1.sh
```

It expands to:

```bash
CUDA_VISIBLE_DEVICES=0,2,3,4,5,6,7 HF_HUB_ENABLE_HF_TRANSFER=1 NCCL_DEBUG=WARN TORCH_NCCL_ASYNC_ERROR_HANDLING=1 accelerate launch --config_file configs/accelerate_h20_7gpu.yaml scripts/train.py --config configs/h20_7gpu_llama_0p2b_deepspeed.yaml
```

The 7-GPU speed config keeps the same 0.2B model shape, but uses `train.batch_size: 24` and `train.grad_accum_steps: 3`, giving `7 * 24 * 3 = 504` sequences per optimizer update. That is close to the 8-GPU effective batch of 512 while giving each H20 a much larger microbatch than the earlier `16 * 5` setup.

This training script supports two multi-GPU backends:

- `configs/h20_7gpu_llama_0p2b_fast.yaml` uses plain PyTorch DDP.
- `configs/h20_7gpu_llama_0p2b_deepspeed.yaml` and `configs/h20_8gpu_llama_0p2b_deepspeed.yaml` use DeepSpeed with BF16, FusedAdam when available, and ZeRO-1 optimizer partitioning.

It also supports Accelerate as the process launcher. The checked-in Accelerate configs launch 7 or 8 local BF16 processes; `CUDA_VISIBLE_DEVICES` controls the physical GPU list, so GPU 1 stays unused in the 7-GPU script.

```bash
CUDA_VISIBLE_DEVICES=0,2,3,4,5,6,7 HF_HUB_ENABLE_HF_TRANSFER=1 NCCL_DEBUG=WARN TORCH_NCCL_ASYNC_ERROR_HANDLING=1 accelerate launch --config_file configs/accelerate_h20_7gpu.yaml scripts/train.py --config configs/h20_7gpu_llama_0p2b_fast.yaml
```

For plain DDP on the 7-GPU set:

```bash
LAUNCHER=torchrun CONFIG=configs/h20_7gpu_llama_0p2b_fast.yaml ./scripts/launch_h20_7gpu_no_gpu1.sh
```

In this repo, Accelerate is used as the launcher while `scripts/train.py` keeps ownership of DDP, DeepSpeed initialization, loss averaging, metrics logging, and checkpoint saving. That keeps the DDP, DeepSpeed, and Accelerate launch commands comparable.

The conda environment installs DeepSpeed. If you are managing packages manually on the H20 machine, install the optional extra with `pip install -e ".[deepspeed]"`.

DeepSpeed is not required for this 0.2B model to fit in 143711 MiB H20 memory, so ZeRO-1 is the first recommended DeepSpeed mode. ZeRO-2 or ZeRO-3 can save more optimizer/parameter memory, but they add extra communication and are usually slower for a model this small unless memory pressure is the real bottleneck.

For throughput, compare `tok_s` and `step_s`, not only ETA. The trainer now prints the visible GPUs, world size, effective global batch, effective tokens per optimizer step, tokenizer/model vocab sanity checks, SDPA/TF32 settings, DataLoader settings, and whether gradient checkpointing is enabled.

Speed knobs to test in order:

- Increase `train.batch_size` until memory is comfortably high but not close to OOM, then lower `train.grad_accum_steps` to keep the effective batch near 512.
- Keep `model.gradient_checkpointing: false` on 143711 MiB H20s unless memory forces it back on.
- Run `scripts/pack_tokens.py` before training so the GPUs read packed token IDs instead of waiting on repeated JSONL parsing and tokenizer calls.
- Compare `LAUNCHER=accelerate` plus DeepSpeed ZeRO-1 against `LAUNCHER=torchrun` plus DDP. For this size, DDP can be faster if optimizer memory is not a problem.
- If GPU utilization dips between steps, raise `train.num_workers` from 4 to 6 or 8 and keep `persistent_workers: true`.
- If `tok_s` is good but the total ETA still looks high, check whether `max_steps` now represents more total tokens. The 8-GPU config is `512 * 2048` tokens per optimizer step; the 7-GPU config is `504 * 2048`.

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

## GPU-Hour Accounting

New training runs write cumulative cost columns into `runs/<run-name>/metrics/training_metrics.csv`:

- `wall_hours`
- `gpu_hours`
- `estimated_total_tokens`
- `estimated_billion_tokens`
- `gpu_hours_per_billion_tokens`

The progress bar also shows `gpuh`. For old logs that only have `time_seconds`, `world_size`, and throughput columns, use the summarizer:

```bash
python scripts/summarize_training_run.py runs/h20-8gpu-llama-0p2b-deepspeed
```

If an older CSV does not include `world_size`, pass the GPU count:

```bash
python scripts/summarize_training_run.py runs/old-run/metrics/training_metrics.csv --gpus 8
```

The estimate detects appended resume segments by watching `time_seconds` reset. For a single uninterrupted run, GPU-hours are simply `last time_seconds / 3600 * world_size`.

## C-Eval Evaluation

C-Eval is a Chinese multiple-choice benchmark with 52 subjects and `dev`, `val`, and `test` splits. This repo evaluates your decoder-only checkpoint by scoring the conditional log probability of answer choices `A`, `B`, `C`, and `D`, which is more stable for a base/pretraining checkpoint than asking it to generate free-form answers.

Latest checked-in decoder-only result:

| Checkpoint | Split | Shots | Prompt Format | Correct / Total | Accuracy |
| --- | --- | ---: | --- | ---: | ---: |
| `runs/h20-8gpu-llama-0p2b-deepspeed/latest` | `val` | 5 | chat format | 320 / 1346 | 23.77% |

Category split, matching the reporting format used by ChatLM-mini-Chinese:

| category | correct | question_count | accuracy |
| --- | ---: | ---: | ---: |
| Humanities | 67 | 257 | 26.07% |
| Other | 89 | 384 | 23.18% |
| STEM | 90 | 430 | 20.93% |
| Social Science | 74 | 275 | 26.91% |

Raw result files are stored in [`eval_results/ceval/latest`](eval_results/ceval/latest).

Run a quick two-subject smoke check:

```bash
python scripts/eval_ceval.py \
  --checkpoint runs/h20-8gpu-llama-0p2b-deepspeed/latest \
  --subjects computer_network,operating_system \
  --split val \
  --n-shot 5 \
  --limit 20
```

Run the full validation set:

```bash
python scripts/eval_ceval.py \
  --model-path runs/h20-8gpu-llama-0p2b-deepspeed/latest \
  --subjects all \
  --split val \
  --n-shot 5
```

The script writes `ceval_summary.json`, `ceval_category_summary.csv`, and `ceval_predictions.csv` under `runs/.../latest/eval/ceval_<split>_<n-shot>shot/` by default. Use `--split test` after you are ready for a final reported score, and use `--no-chat-format` if you want the plain prompt without `<|user|>` and `<|assistant|>` wrappers.

## Prompt/Response Evaluation

Custom prompt/response datasets are evaluated separately from C-Eval. This evaluator expects only `prompt` and `response` fields, normalizes them into the same `<|user|>` / `<|assistant|>` format used by SFT, and reports teacher-forced response loss and perplexity.

```bash
python scripts/eval_prompt_response.py \
  --model-path /absolute/path/to/model/checkpoint \
  --dataset-file /absolute/path/to/eval.json \
  --output-dir runs/my-prompt-response-eval \
  --batch-size 8 \
  --dtype bf16
```

The local eval file can be `.json`, `.jsonl`, or `.csv`, but each row should be simple:

```json
[
  {
    "prompt": "请解释边缘端中文小模型的优势。",
    "response": "边缘端中文小模型通常具有低延迟、低成本和更好的本地隐私保护。"
  }
]
```

To also inspect generations for the first few rows:

```bash
python scripts/eval_prompt_response.py \
  --model-path /absolute/path/to/model/checkpoint \
  --dataset-file /absolute/path/to/eval.json \
  --generate-samples 10 \
  --max-new-tokens 64
```

This writes `prompt_response_eval_summary.json`, `metrics.json`, `split_audit.json`, `prediction_debug.csv`, and `prompt_response_eval_predictions.jsonl`. By default `--comparison-mode whitespace` is the strict whitespace-insensitive exact-match metric. Use `--comparison-mode command` only when you intentionally want conservative smart-home semantic equivalence. Add `--benchmark-runs 5` to repeat the full eval five times and write a mean ± std benchmark summary.

## SFT And Contrastive SFT

After pretraining, run standard supervised fine-tuning on instruction/answer rows:

```bash
python scripts/sft.py \
  --config configs/sft.yaml \
  --checkpoint runs/h20-8gpu-llama-0p2b-deepspeed/latest \
  --epochs 3
```

SFT files can be `.jsonl` or normal `.json`. Rows can use `prompt`/`response`, `anchor`/`response`, `prompt`/`responses`, `instruction`/`response`, `question`/`answer`, or similar fields:

```json
{"prompt": "什么是边缘端中文小模型？", "response": "..."}
```

For a normal JSON file, use either a list:

```json
[
  {"prompt": "什么是边缘端中文小模型？", "response": "..."},
  {"prompt": "它适合哪些场景？", "response": "..."}
]
```

or a wrapper object:

```json
{
  "data": [
    {"prompt": "什么是边缘端中文小模型？", "response": "..."}
  ]
}
```

The default SFT mode is regular causal-LM supervised fine-tuning. Contrastive positive/negative SFT is only enabled when you explicitly pass `--mode contrastive`.

Use `--epochs N` to train for `N` passes over your SFT dataset. If `--epochs` is omitted, the script uses `train.max_steps` from the config. Internally, epochs are converted into optimizer steps using the dataloader length and `train.grad_accum_steps`.

SFT data is normalized automatically before tokenization by `normalize_sft_record` in `src/chatlm_decoder/sft_data.py`. This is important: the dataset should be shaped to the model's role-token format instead of asking the model to infer inconsistent dataset schemas. The normalizer:

- strips duplicated `<|user|>`, `<|assistant|>`, `<|system|>`, and `<|eos|>` tokens from raw fields;
- cleans line endings and repeated blank lines;
- combines `instruction` plus `input`/`context` into one user prompt;
- converts `messages` or `conversations` rows into the same chat format;
- always writes prompts as `<|user|>\n...\n<|assistant|>\n`, with optional `<|system|>` before the user turn;
- always appends one final `<|eos|>` to the response.

For ChatML/ShareGPT-style rows, the last assistant message is trained as the answer and earlier turns are used as the masked prompt:

```json
{
  "messages": [
    {"role": "system", "content": "你是一个中文助手。"},
    {"role": "user", "content": "什么是小模型？"},
    {"role": "assistant", "content": "小模型是参数量较小、部署成本较低的模型。"}
  ]
}
```

For contrastive SFT, each row also includes a positive semantic example and a negative example:

```json
{"anchor": "...", "response": "...", "positive": "...", "negative": "..."}
```

The 8-GPU contrastive launcher evaluates only prompt/response anchor rows after training; positive and negative fields are used for the contrastive objective, not as generated-response targets. It also runs the configured smart-home benchmark, using the same exact-match and exact-match@5 evaluator.

Run it with:

```bash
python scripts/sft.py \
  --config configs/contrastive_sft.yaml \
  --mode contrastive \
  --checkpoint runs/sft-0p2b/latest
```

For the 8x H20 workflow, edit only `configs/contrastive_sft_8gpu.yaml`:

```yaml
sft:
  base_model: /absolute/path/to/base-or-sft-checkpoint
  data_path: data/scenic/SCENIC_full_anchor_positive_negative.json
  anchor_eval_path: data/scenic/SCENIC_full_training_dataset.json
  benchmark_path: data/benchmarks/iot_instruction_benchmark_200.json
  benchmark_runs: 5
  top_k_exact_match: 5
  max_length: 128
  alignment_weight: 0.1
  margin: 0.5
```

Then run:

```bash
CONFIG_PATH=configs/contrastive_sft_8gpu.yaml ./run_contrastive_sft_8gpu.sh
```

The script uses all 8 visible GPUs, runs contrastive SFT, then evaluates only the anchor prompt/response rows plus the smart-home benchmark. Positive and negative fields are used only for contrastive training. The anchor eval can intentionally be the prompt/response projection of the contrastive training file, so overlap is logged in `split_audit.json` but does not fail the run. The final reports land under:

```text
runs/contrastive-sft-0p2b-8gpu/eval/final_anchor/
runs/contrastive-sft-0p2b-8gpu/eval/final_benchmark/
```

Override repeats or top-K once with:

```bash
CONTRASTIVE_BENCHMARK_RUNS=3 CONTRASTIVE_TOP_K_EXACT_MATCH=5 ./run_contrastive_sft_8gpu.sh
```

The contrastive objective follows the compatibility-aware triplet SFT algorithm:

```text
loss =
  GenLoss(anchor, response)
  + GenLoss(positive, response)
  + lambda * relu(margin + distance(anchor, positive) - distance(anchor, negative))
```

The implementation prompt-formats anchor, positive, and negative before representation scoring, uses mean-pooled last hidden states and cosine distance, and does not train on `negative_response`. `configs/contrastive_sft.yaml` controls `alignment_weight` and `margin`.

For hyperparameter testing, start small. A good first screen is 6 runs:

```text
alignment_weight: 0.03, 0.1, 0.3
margin:           0.3, 0.5
```

If the best two are close, expand to a 3x3 grid with `margin: 0.3, 0.5, 0.7` and the same three alignment weights. Avoid going much above `alignment_weight: 0.3` at first; if the contrastive term dominates, exact command generation can get worse even when representation alignment improves.

## 50% Pruning

Pruning is a post-training checkpoint transform. It writes a new checkpoint with zeroed weights and a `pruning_report.json`; it does not mutate your original model. The default configs now use `prune.scope: transformer_linears`, `sparsity_denominator: whole_model`, `granularity: layer`, and `include_lm_head: false`. That keeps embeddings, output heads, tied output embeddings, norms, biases, and non-Linear tensors protected while still asking the pruning code to resolve enough Linear sparsity for `achieved_whole_model_sparsity` to land at the 50% target on dense checkpoints.

Run one pruning method:

```bash
python scripts/prune.py \
  --config configs/prune_50.yaml \
  --method magnitude \
  --checkpoint runs/contrastive-sft-0p2b/latest \
  --output-dir runs/pruned-magnitude-50
```

Available methods:

- `magnitude`: layerwise unstructured pruning by `abs(parameter)` over protected transformer Linear weights; with the default whole-model denominator, the resolved Linear sparsity is raised above 50% when needed to achieve real 50% model sparsity.
- `2of4`: exact 2:4 masks on eligible Linear weights. Because exact 2:4 is fixed at 50% within those groups, it is reported as 50% prunable-Linear sparsity when protected parameters remain unchanged.
- `wanda`: activation-aware rowwise scoring for protected Linear weights.
- `gradient`: layerwise gradient-score pruning using `abs(parameter * grad)` on calibration batches.

Run all four:

```bash
CHECKPOINT=runs/contrastive-sft-0p2b/latest ./scripts/run_pruning_suite.sh
```

For the dense-vs-pruned check you asked for most often, use the SFT prompt/response evaluator directly. This avoids the Qwen-Instruct or IoT benchmark paths: it evaluates the dense SFT checkpoint on your data file, prunes each method, reloads the saved pruned checkpoint, evaluates that checkpoint with the same SFT evaluator, then prints accuracy and pruning stats.

You can edit the `SCRIPT_MODEL_PATH` and `SCRIPT_DATA_FILE` block at the top of `scripts/run_sft_pruning_eval.sh`, then run:

```bash
bash scripts/run_sft_pruning_eval.sh
```

Or keep the script unchanged and pass paths at launch time:

```bash
MODEL_PATH=/absolute/path/to/sft_or_hf_checkpoint \
DATA_FILE=/absolute/path/to/eval_prompt_response.json \
bash scripts/run_sft_pruning_eval.sh
```

Useful overrides:

```bash
METHODS="magnitude 2of4 wanda gradient" \
CALIBRATION_FILE=/absolute/path/to/calibration_or_sft.jsonl \
OUTPUT_DIR=runs/sft-pruning-eval \
NPROC=8 \
bash scripts/run_sft_pruning_eval.sh /absolute/path/to/model /absolute/path/to/eval.json
```

The summary is written to `OUTPUT_DIR/sft_pruning_eval_summary.csv` and `OUTPUT_DIR/sft_pruning_eval_summary.json`. The `checkpoint_evaluated` column is the exact dense or saved pruned checkpoint passed to `scripts/eval_prompt_response.py`; pruned checkpoints are saved under `OUTPUT_DIR/one_shot/{magnitude,nvidia-2of4,wanda,gradient}/`.

For one-off dense/pruned exact-match eval on one local model, use the standalone scripts in `single_pruning/`:

```bash
python single_pruning/magnitude_prune_8gpu_exact.py /path/to/local_model /path/to/eval.json
python single_pruning/gradient_prune_8gpu_exact.py /path/to/local_model /path/to/eval.json
python single_pruning/wanda_prune_8gpu_exact.py /path/to/local_model /path/to/eval.json
python single_pruning/prune_2of4_8gpu_exact.py /path/to/local_model /path/to/eval.json
```

They auto-launch with `torchrun` on up to 8 GPUs, evaluate the dense model, apply one pruning method, evaluate the pruned model, and save the pruned checkpoint plus summaries next to the model path.

You can also edit `MODEL_PATH` and `EVAL_DATASET_PATH` at the top of any `single_pruning/*.py` file and run it with no path arguments. These scripts use one full model copy per GPU and shard evaluation across up to 8 visible GPUs; increase `BATCH_SIZE` near the top of the script if GPU utilization is low and memory allows it. The pruned model is saved immediately after pruning, then the pruned benchmark reloads that saved `pruned_model/` checkpoint from disk. The terminal prints dense accuracy, pruned accuracy, real whole-model sparsity, selected-linear sparsity, and output paths at the end.

Wanda and gradient pruning require `prune.calibration_data_path`; the default config points at `data/sft/contrastive_train.jsonl`. Keep `prune.recovery_steps: 0` for one-shot pruning. Use the benchmark suite's separate retune phase for post-pruning SFT; that path reapplies masks after every optimizer step and reloads the final checkpoint to confirm pruned weights stayed zero.

Important: the `2of4` method creates the correct 2:4 zero pattern in linear weights. Real NVIDIA sparse Tensor Core speedups still require an inference/training stack that actually dispatches 2:4 kernels, such as a compatible TensorRT-LLM, cuSPARSELt, or other semi-structured sparse runtime path.

### Qwen2.5-Instruct Pruning

Qwen2.5-0.5B-Instruct has a separate pruning path because Wanda/gradient calibration and prune+retune must use the same Qwen chat template as Qwen SFT. Do not use `scripts/run_pruning_benchmark.py` for Qwen-Instruct comparisons.

Edit `configs/qwen25_instruct_pruning_benchmark.yaml`:

```yaml
benchmark:
  base_checkpoint: outputs/qwen25_0p5b_instruct_sft/final
  eval_file: /absolute/path/to/eval.json

prune:
  calibration_data_path: /absolute/path/to/calibration_or_sft.json

retune:
  data_path: /absolute/path/to/retune_sft.json
  epochs: 3
  max_steps: null
```

Then run:

```bash
./run_qwen25_instruct_pruning_benchmark_8gpu.sh
```

Or use the single YAML-driven sequential wrapper. It auto-detects Qwen2.5-Instruct configs by name/content:

```bash
CONFIG_PATH=configs/qwen25_instruct_pruning_benchmark.yaml ./scripts/run_pruning_benchmark_8way.sh
```

Base `Qwen/Qwen2.5-0.5B` and custom decoder-only models should use the generic pruning benchmark, not the Qwen2.5-Instruct benchmark. If you want to force that path, run with `MODE=generic`.

This first evaluates the dense Qwen SFT checkpoint and writes `dense_baseline_eval.json`. It then runs all four pruning methods with `scripts/prune_qwen25_instruct.py`, checks the saved pruning report for the configured sparsity denominator and zero mask violations, evaluates the reloaded one-shot pruned checkpoint once with `scripts/eval_qwen25_instruct.py`, then retunes for 3 epochs with `scripts/sft_qwen25_instruct.py --pruning-mask ...` so zeroed weights stay zero after every optimizer step. Outputs are grouped under `benchmark.output_dir` with Qwen-specific summaries:

- `one_shot/<method>/`
- `retuned/<method>/final/`
- `benchmarks/one_shot/<method>/`
- `benchmarks/retuned/<method>/`
- `qwen25_instruct_pruning_benchmark_summary.csv`
- `qwen25_instruct_pruning_benchmark_summary.json`
- `benchmark_summary_one_shot.csv`
- `benchmark_summary_retuned.csv`

### Pruning Benchmark Suite

For a full pruning comparison, use the YAML-driven benchmark suite. It evaluates the dense SFT baseline first, then runs all four pruning methods in two separate phases:

1. one-shot prune with no retuning, then exact-match generation benchmark;
2. prune, SFT-retune on your dataset while reapplying the pruning mask after every optimizer step, then exact-match generation benchmark.

Use this generic suite for your custom decoder-only model and for base `Qwen/Qwen2.5-0.5B`. The Qwen2.5-Instruct suite is only for checkpoints/configs that explicitly use the Instruct chat-template path.

For base Qwen2.5-0.5B, start from `configs/qwen25_0p5b_pruning_benchmark.yaml`:

```bash
MODE=generic CONFIG_PATH=configs/qwen25_0p5b_pruning_benchmark.yaml ./scripts/run_pruning_benchmark_8way.sh
```

That config uses the generic decoder-only pruning code and writes one-shot pruned checkpoints under `runs/qwen25-0p5b-pruning-benchmark/one_shot/{magnitude,nvidia-2of4,wanda,gradient}/`.

For the current pruning benchmark, use your own prompt/response eval file through `benchmark.eval_file` or the one-run `EVAL_FILE` override. The IoT benchmark launcher is separate and should only be used for the later final IoT command benchmark.

To run the exact current setup for both trained 0.2B SFT families in one command:

```bash
conda activate chatlm-decoder
./run_sft_and_contrastive_pruning_benchmarks.sh
```

That launches `configs/pruning_benchmark_regular_sft.yaml` and then `configs/pruning_benchmark_contrastive_sft.yaml`. Each config prunes the checkpoint with `magnitude`, `wanda`, `gradient`, and NVIDIA `2of4`; protects embeddings, norms, biases, and `lm_head` while targeting 50% whole-model sparsity for methods that can satisfy it; evaluates the dense baseline, one-shot pruned checkpoints, and masked-retuned checkpoints; and reports both `exact_match_accuracy` and `exact_match_at_top_k_accuracy` with `top_k_exact_match: 5`. The exact `2of4` row should be read as the fixed 50% prunable-Linear hardware-pattern condition, not as an exact 50% whole-model condition.

The regular SFT config evaluates both `sft_dataset` and `benchmark`. The contrastive SFT config evaluates both `anchor_dataset` and `benchmark`. The expensive prune/retune step runs once per method per model, and the saved checkpoint is then evaluated on both named eval files. The benchmark split has difficulty labels (`easy`, `medium`, `hard`; currently 70/65/65 examples), and the summaries include overall accuracy plus per-hardness columns such as `difficulty_easy_exact_match_accuracy`, `difficulty_medium_exact_match_accuracy`, `difficulty_hard_exact_match_accuracy`, and matching exact-match@5 columns.

For the full journal run that trains regular SFT for 5 epochs, trains contrastive SFT for 5 epochs, evaluates the original decoder, dense regular SFT, and dense contrastive SFT on the original SFT dataset and benchmark, then runs one-shot `wanda`, `gradient`, `magnitude`, and NVIDIA `2of4` on both SFT checkpoints, use:

```bash
PYTHON=/path/to/training/env/bin/python \
bash run_5epoch_sft_contrastive_one_shot_pruning.sh /path/to/base_model
```

If you only want to generate the two 5-epoch checkpoints from a base model path, use the train-only wrapper:

```bash
PYTHON=/path/to/training/env/bin/python \
bash run_5epoch_sft_contrastive_from_base.sh /path/to/base_model
```

The SCENIC full-training and contrastive data are versioned under `data/scenic/`, so this command only needs the untuned base model path. The wrapper uses that one path for original decoder dense eval and regular SFT training, then starts contrastive SFT from the freshly written regular SFT final checkpoint. By default it evaluates both the SFT dataset (`data/scenic/SCENIC_full_training_dataset.json`) and benchmark (`data/benchmarks/iot_instruction_benchmark_200.json`), and writes to `runs/5epoch-sft-contrastive-prunable50-comparable/`.

The main full-journal artifact is one consolidated JSON file at `runs/5epoch-sft-contrastive-prunable50-comparable/journal_results.json`; it contains 22 expected EM@1/EM@5 rows by default: original decoder dense accuracy on `training_dataset` and `benchmark`; dense regular SFT and dense contrastive SFT on both eval splits; and `wanda`, `gradient`, `magnitude`, and `2of4` one-shot pruning rows for both SFT checkpoints on both eval splits. The pruning target is ED-comparable 50% targeted/prunable decoder Transformer Linear weights with global masks, not 50% whole-model sparsity. Change `BASE_MODEL`, dataset paths, or `RESULTS_JSON` only when you intentionally want to override the default run shape.

To test whether the aggressive-pruning collapse is mainly an EOS/termination-control failure, run the EOS-reinforced variant. It keeps the same 50% pruning masks fixed, then adds masked SFT recovery rows with supervised EOS labels upweighted in the loss:

```bash
PYTHON=/path/to/training/env/bin/python \
bash run_5epoch_sft_contrastive_eos_reinforced_pruning.sh /path/to/base_model
```

The same retune path can also be requested explicitly as a shell command:

```bash
PYTHON=/path/to/training/env/bin/python \
bash run_5epoch_sft_contrastive_one_shot_pruning.sh retune /path/to/base_model
```

The EOS experiment writes both `one_shot` and `retuned` rows in the same `journal_results.json`. Treat EOS as a major cause only if the `retuned` rows reduce `max_token_hit_rate` and recover EM/loss under the same sparsity. If `max_token_hit_rate` improves but `mean_response_loss` and EM stay collapsed, EOS was a symptom of broader token-distribution damage.

If you already have the three checkpoint paths and only want dense EM@1/EM@5 plus 50% one-shot pruning results in one JSON, use the path-driven wrapper:

```bash
PYTHON=/path/to/training/env/bin/python \
bash run_model_path_pruning_results.sh \
  /path/to/base_model \
  /path/to/sft/final \
  /path/to/contrastive/final
```

By default it evaluates all three dense checkpoints on `data/scenic/SCENIC_full_training_dataset.json` and `data/benchmarks/iot_instruction_benchmark_200.json`, then runs `wanda`, `magnitude`, `gradient`, and NVIDIA `2of4` one-shot 50% pruning for `base_model`, `sft`, and `contrastive`. It writes `runs/model-path-pruning-results/model_path_pruning_results.json`. Set `TRAINING_DATASET`, `BENCHMARK_DATASET`, `RESULTS_JSON`, `METHODS`, or `PRUNE_FAMILIES="sft contrastive"` to override the default shape.

For already-trained checkpoints, set `EOS_RETUNE=1` on the path-driven wrapper to add the same fixed-mask EOS-reinforced recovery rows:

```bash
EOS_RETUNE=1 EOS_LOSS_WEIGHT=5.0 EOS_RETUNE_EPOCHS=1.0 \
PYTHON=/path/to/training/env/bin/python \
bash run_model_path_pruning_results.sh /path/base /path/sft/final /path/contrastive/final
```

Quick prune commands:

```bash
# Run regular SFT + contrastive SFT pruning/eval in one go.
conda activate chatlm-decoder
./run_sft_and_contrastive_pruning_benchmarks.sh

# Only regular SFT.
SKIP_CONTRASTIVE=1 ./run_sft_and_contrastive_pruning_benchmarks.sh

# Only contrastive SFT.
SKIP_REGULAR=1 ./run_sft_and_contrastive_pruning_benchmarks.sh

# Preview without launching prune/train/eval commands.
DRY_RUN=1 ./run_sft_and_contrastive_pruning_benchmarks.sh
```

Wanda-only rerun commands:

```bash
# Run only Wanda for regular SFT + contrastive SFT.
./run_wanda_pruning_benchmarks.sh

# Only regular SFT Wanda.
SKIP_CONTRASTIVE=1 ./run_wanda_pruning_benchmarks.sh

# Only contrastive SFT Wanda.
SKIP_REGULAR=1 ./run_wanda_pruning_benchmarks.sh

# Preview Wanda-only commands.
DRY_RUN=1 ./run_wanda_pruning_benchmarks.sh
```

For one model/config instead of both:

```bash
MODE=generic CONFIG_PATH=configs/pruning_benchmark_regular_sft.yaml ./scripts/run_pruning_benchmark_8way.sh
MODE=generic CONFIG_PATH=configs/pruning_benchmark_contrastive_sft.yaml ./scripts/run_pruning_benchmark_8way.sh

# Or run one method through the generic launcher without editing YAML:
PRUNING_METHODS=wanda MODE=generic CONFIG_PATH=configs/pruning_benchmark_contrastive_sft.yaml ./scripts/run_pruning_benchmark_8way.sh
```

For a single method without the full benchmark suite:

```bash
python scripts/prune.py \
  --config configs/prune_50.yaml \
  --method magnitude \
  --checkpoint outputs/sft_0p2b_8gpu/final \
  --output-dir runs/pruned-regular-sft-magnitude-50
```

Edit `configs/pruning_benchmark.yaml`:

```yaml
benchmark:
  base_checkpoint: /absolute/path/to/model/latest
  eval_file: /absolute/path/to/eval.json
  # Or evaluate several files after the same prune/retune pass:
  eval_files:
    dataset: /absolute/path/to/model_specific_eval.json
    benchmark: data/benchmarks/iot_instruction_benchmark_200.json
  top_k_exact_match: 5
  output_dir: runs/pruning-benchmark-0p2b

prune:
  calibration_data_path: /absolute/path/to/calibration_or_sft.jsonl

retune:
  data_path: /absolute/path/to/retune_sft.jsonl
  epochs: 3
  max_steps: null
```

Then run:

```bash
./run_pruning_benchmark_suite.sh
```

Or override only the eval file for one run:

```bash
EVAL_FILE=/absolute/path/to/your_eval.json ./run_pruning_benchmark_suite.sh
EVAL_FILE=/absolute/path/to/your_eval.json CONFIG_PATH=configs/qwen25_0p5b_pruning_benchmark.yaml MODE=generic ./scripts/run_pruning_benchmark_8way.sh
```

Or use the single YAML-driven sequential wrapper:

```bash
CONFIG_PATH=configs/pruning_benchmark.yaml ./scripts/run_pruning_benchmark_8way.sh
CONFIG_PATH=configs/pruning_benchmark.yaml python scripts/run_pruning_benchmark.py --eval-file /absolute/path/to/your_eval.json
```

Preview the commands without running them:

```bash
DRY_RUN=1 CONFIG_PATH=configs/pruning_benchmark.yaml ./scripts/run_pruning_benchmark_8way.sh
```

It runs in this order: `magnitude` one-shot prune and eval, optional 3-epoch masked retune and eval; then `2of4`; then `wanda`; then `gradient`. By default `KEEP_GOING=1`, so a failed method is recorded in the summary and the next method still runs. Set `KEEP_GOING=0` if you want fail-fast behavior. The final CSV has the 8 accuracy rows from four methods times one-shot/retuned phases when all phases succeed.

Or point at another config file or a directory containing `pruning_benchmark.yaml`:

```bash
CONFIG_PATH=configs/pruning_benchmark.yaml ./run_pruning_benchmark_suite.sh
python scripts/run_pruning_benchmark.py --config configs/pruning_benchmark.yaml
```

Outputs are grouped under `benchmark.output_dir`:

```text
one_shot/<method>/               # 4 one-shot pruned checkpoints
retuned/<method>/                # 4 SFT-retuned pruned checkpoints
benchmarks/one_shot/<method>/    # one-shot eval outputs, or <method>/<eval_name>/ with eval_files
benchmarks/retuned/<method>/     # retuned eval outputs, or <method>/<eval_name>/ with eval_files
pruning_benchmark_summary.csv
pruning_benchmark_summary.json
```

That gives 8 model outputs total and 8 benchmark output folders total for a single eval file, plus the dense baseline eval folder. With `benchmark.eval_files`, the same 8 model outputs are reused across each named eval file and the CSV includes `eval_name` and `eval_file`. Each one-shot checkpoint writes `pruning_report.json`, `module_filter_report.json`, `mask_validation.json`, `checkpoint_reload_validation.json`, `sparsity_by_module.csv`, `layerwise_zero_fraction.csv`, and `layerwise_weight_norms_before_after.csv`; gradient, Wanda, and 2:4 runs also write their method-specific diagnostics. Each eval writes `generation_samples.json`, `exact_match_failure_cases.json`, and `top_k_exact_match_failure_cases.json`. When rows contain `difficulty` or `hardness`, the eval summary also writes `by_difficulty` and flat CSV-ready fields for easy/medium/hard top-1 and top-5 accuracy.

Use `benchmark_summary_one_shot.csv` for the one-shot pruning comparison. Retuned rows are written separately to `benchmark_summary_retuned.csv` and are post-pruning SFT results, not one-shot pruning results. The summary reports both `achieved_prunable_sparsity` and `achieved_whole_model_sparsity`; with the default protected whole-model target, `achieved_whole_model_sparsity` should land at 50% for magnitude, Wanda, and gradient on dense checkpoints, while exact `2of4` remains the fixed 50% prunable-Linear condition.

The runner checks each report before evaluation and fails the phase if the configured sparsity denominator is not at 50% within `benchmark.sparsity_tolerance` (`0.001` by default for protected whole-model pruning), any masked weight is nonzero, the evaluated checkpoint is not the pruned checkpoint, 2:4 structure is invalid, gradient/Wanda calibration statistics are missing or degenerate, generated predictions are all empty/all identical/mostly prompt copies, or too many generations hit `max_new_tokens` without EOS when `benchmark.max_new_token_hit_rate_threshold` is at or below the observed rate. The SCENIC journal defaults set that threshold to `1.01`, so high length-cap rates remain visible as `reached_max_new_tokens_rate` diagnostics instead of stopping the whole report. Failed rows include `error_type` so generation/eval issues can be separated from pruning failures. By default, each eval runs one benchmark pass; set `benchmark.benchmark_runs` higher if you want repeated mean/std measurements.

### Final IoT Benchmark Eval

Evaluate any dense, pruned, or retuned checkpoint on the final 200-example IoT benchmark:

```bash
# Edit only configs/iot_benchmark_eval.yaml:model_path for the normal path.
bash run_iot_benchmark_eval.sh
```

`model_path` can be either an absolute local checkpoint path or a Hugging Face model id. The default config uses `prompt_format: auto`, which uses the Qwen chat template only for `*Instruct*` checkpoints and otherwise uses the repo's decoder-only `<|user|>` / `<|assistant|>` format. You can still override any value for a single run:

```bash
bash run_iot_benchmark_eval.sh --model-path Qwen/Qwen2.5-0.5B --prompt-format raw
```

The script writes `iot_benchmark_summary.json`, `iot_benchmark_predictions.jsonl`, `iot_benchmark_predictions.csv`, `generation_samples.json`, and `exact_match_failure_cases.json`.

Inspect any one-shot or retuned pruning report:

```bash
python scripts/inspect_pruning_report.py runs/pruning-benchmark-0p2b/one_shot/magnitude
python scripts/inspect_pruning_report.py runs/pruning-benchmark-0p2b/retuned/magnitude/final
```

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

SFT checkpoints are written once under `run.output_dir/final/`. Before a new SFT starts, old `step-*`, `latest`, and `final` SFT checkpoint folders in that output directory are removed so the final eval cannot accidentally load a stale model. The final folder contains the model safetensors, tokenizer files, config, `trainer_state.pt`, and `checkpoint_manifest.json`; `run.output_dir/final_checkpoint.json` records the exact checkpoint path used for eval.

Pretraining checkpoints are still step-based under `run.output_dir` because long pretraining runs need resume and crash recovery.

The trainer also handles stoppage:

- Scheduled saves happen every `train.save_every` optimizer steps and at the final step.
- Pressing `Ctrl+C` or sending `SIGTERM` requests a graceful stop; the trainer finishes the current optimizer step, saves `stop-step-000000/`, updates `latest`, prunes old checkpoints, and exits cleanly.
- If an exception or CUDA OOM happens after at least one optimizer step, the trainer attempts a best-effort `crash-step-000000/` checkpoint before re-raising the error.

Resume with:

```bash
./scripts/launch_h20_8gpu.sh --resume runs/h20-8gpu-llama-0p2b-deepspeed/latest
```

or, if GPU 1 must stay unused:

```bash
./scripts/launch_h20_7gpu_no_gpu1.sh --resume runs/h20-7gpu-llama-0p2b-deepspeed/latest
```

Training is step-based rather than epoch-based. The 8-H20 config uses `max_steps: 100000`, effective batch `8 * 32 * 2 = 512` sequences/update, and block size `2048`, so the planned run is about `100000 * 512 * 2048 = 104,857,600,000` training tokens. When a packed-token manifest exists, startup logs also print the approximate number of corpus passes.

## Push Model Tensors To GitHub

GitHub rejects normal Git blobs above 100 MB, so model tensors should be pushed with Git LFS. This repo tracks common tensor extensions in `.gitattributes`, but you still need Git LFS installed on the machine that pushes the checkpoint.

One-time setup on the training machine:

```bash
git lfs install
git pull origin main
```

Keep `runs/` ignored for active training. After training, copy the final checkpoint into a versioned artifact directory:

```bash
mkdir -p model-artifacts/decoder-chinese-slm-0p2b
cp -R runs/h20-8gpu-llama-0p2b-deepspeed/latest/* model-artifacts/decoder-chinese-slm-0p2b/
```

Then commit and push the tensor files through LFS:

```bash
git status
git add .gitattributes model-artifacts/decoder-chinese-slm-0p2b
git commit -m "Add 0.2B checkpoint tensors"
git push origin main
```

For large public model distribution, Hugging Face Hub is usually a better home for checkpoints than GitHub LFS because it is built for model files, safetensors, model cards, and downloads.

## References

This project is an independent decoder-only implementation and experiment scaffold. If you use this repository, also cite and respect the licenses/terms of the upstream repositories, datasets, benchmarks, and software that your run depends on.

### Reference Model And Benchmark

- `charent/ChatLM-mini-Chinese`: Hugging Face model repository used as the same-size reference target for the 0.2B parameter budget, 29,298-token vocabulary, Chinese public-data recipe, loss-curve style, and C-Eval reporting comparison. Repository: [`https://huggingface.co/charent/ChatLM-mini-Chinese`](https://huggingface.co/charent/ChatLM-mini-Chinese). Config reference: [`config.json`](https://huggingface.co/charent/ChatLM-mini-Chinese/blob/main/config.json).
- `ceval/ceval-exam`: Hugging Face dataset repository used for C-Eval validation and test loading. Repository: [`https://huggingface.co/datasets/ceval/ceval-exam`](https://huggingface.co/datasets/ceval/ceval-exam).
- `hkust-nlp/ceval`: Official C-Eval benchmark repository used for benchmark context and subject/category mapping. Repository: [`https://github.com/hkust-nlp/ceval`](https://github.com/hkust-nlp/ceval). Subject mapping reference: [`subject_mapping.json`](https://raw.githubusercontent.com/hkust-nlp/ceval/main/subject_mapping.json).

### Public Data Repositories

- `YeungNLP/firefly-pretrain-dataset`: Hugging Face dataset repository used for public Chinese web/wiki-style pretraining sources, including `webText2019zh.jsonl` and `wiki_zh.jsonl`. Repository: [`https://huggingface.co/datasets/YeungNLP/firefly-pretrain-dataset`](https://huggingface.co/datasets/YeungNLP/firefly-pretrain-dataset).
- `ZhouLV/Chinese-Train-Datasets`: Hugging Face dataset repository used for the Baike QA source. Repository: [`https://huggingface.co/datasets/ZhouLV/Chinese-Train-Datasets`](https://huggingface.co/datasets/ZhouLV/Chinese-Train-Datasets). Baike path used in config: [`baike2018qa`](https://huggingface.co/datasets/ZhouLV/Chinese-Train-Datasets/tree/main/baike2018qa).
- `ticoAg/Chinese-medical-dialogue`: Hugging Face dataset repository used for Chinese medical dialogue/QA data. Repository: [`https://huggingface.co/datasets/ticoAg/Chinese-medical-dialogue`](https://huggingface.co/datasets/ticoAg/Chinese-medical-dialogue).
- `wangrui6/Zhihu-KOL`: Hugging Face dataset repository used for Zhihu KOL question-answer style data. Repository: [`https://huggingface.co/datasets/wangrui6/Zhihu-KOL`](https://huggingface.co/datasets/wangrui6/Zhihu-KOL).
- `BelleGroup/train_1M_CN`: Hugging Face dataset repository used for BELLE Chinese instruction/chat data. Repository: [`https://huggingface.co/datasets/BelleGroup/train_1M_CN`](https://huggingface.co/datasets/BelleGroup/train_1M_CN).
- `BelleGroup/train_2M_CN`: Hugging Face dataset repository used for BELLE Chinese instruction/chat data. Repository: [`https://huggingface.co/datasets/BelleGroup/train_2M_CN`](https://huggingface.co/datasets/BelleGroup/train_2M_CN).
- `BelleGroup/train_3.5M_CN`: Hugging Face dataset repository used for BELLE Chinese instruction/chat data. Repository: [`https://huggingface.co/datasets/BelleGroup/train_3.5M_CN`](https://huggingface.co/datasets/BelleGroup/train_3.5M_CN).

### Software And Systems

- Hugging Face Transformers: model definitions, `LlamaConfig`, tokenizer/model loading, generation, and checkpoint serialization. Repository: [`https://github.com/huggingface/transformers`](https://github.com/huggingface/transformers). Llama docs: [`https://huggingface.co/docs/transformers/model_doc/llama`](https://huggingface.co/docs/transformers/model_doc/llama).
- Hugging Face Datasets: streaming/loading public datasets and C-Eval splits. Repository: [`https://github.com/huggingface/datasets`](https://github.com/huggingface/datasets).
- Hugging Face Tokenizers: BPE tokenizer training/runtime components. Repository: [`https://github.com/huggingface/tokenizers`](https://github.com/huggingface/tokenizers).
- Hugging Face Accelerate: optional launcher configs for H20 multi-GPU runs. Repository: [`https://github.com/huggingface/accelerate`](https://github.com/huggingface/accelerate).
- PyTorch: tensor runtime, BF16/TF32 training, SDPA attention kernels, and distributed data parallel training. Repository: [`https://github.com/pytorch/pytorch`](https://github.com/pytorch/pytorch). DDP docs: [`DistributedDataParallel`](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html). FSDP docs: [`FullyShardedDataParallel`](https://docs.pytorch.org/docs/stable/fsdp.html).
- DeepSpeed: optional ZeRO-1 optimizer/runtime path for H20 pretraining. Repository: [`https://github.com/deepspeedai/DeepSpeed`](https://github.com/deepspeedai/DeepSpeed).
- Safetensors: safe model tensor serialization for checkpoints. Repository: [`https://github.com/huggingface/safetensors`](https://github.com/huggingface/safetensors).
- NVIDIA H20/system references: H20 memory and compute capability should be checked against the current NVIDIA documentation for the exact machine being used. MIG supported GPU reference: [`https://docs.nvidia.com/datacenter/tesla/mig-user-guide/supported-gpus.html`](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/supported-gpus.html).

## Cite This Repository

GitHub can read [`CITATION.cff`](CITATION.cff) and show a "Cite this repository" button. A plain BibTeX entry is also provided below:

```bibtex
@software{decoder_chinese_slm_2026,
  author = {huluk98},
  title = {Decoder-Chinese-SLM: A Decoder-Only Chinese Small Language Model Training Codebase},
  year = {2026},
  version = {0.1.0},
  url = {https://github.com/huluk98/Decoder-Chinese-SLM}
}
```
