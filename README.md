# Decoder-Only Chinese SLM Local Pipeline

This repository is currently set up for one local H20 workflow:

1. Start from your local decoder checkpoint.
2. Fine-tune regular SFT for 5 epochs.
3. Fine-tune contrastive SFT for 5 epochs from the regular SFT output.
4. Run the original one-shot pruning methods.
5. Run progressive magnitude pruning.
6. Write one final JSON summary for the local model.

Everything below assumes the model is already on the machine as a local checkpoint directory.

## How The Model Was Made

The local checkpoint is a decoder-only Chinese SLM built to stay near the same 0.2B parameter budget and 29,298-token vocabulary target as `charent/ChatLM-mini-Chinese`, but with a modern causal decoder architecture instead of a T5-style encoder-decoder model.

The base model recipe uses:

- Llama-family causal LM modeling through `transformers.LlamaForCausalLM`.
- RoPE position embeddings, RMSNorm, SwiGLU MLPs, grouped-query attention, bias-free projections, and SDPA attention.
- A Chinese-friendly BPE tokenizer with the project special tokens.
- Public Chinese prompt/response, QA, instruction, web, wiki-like, and domain dialogue sources normalized into one decoder-only text corpus.
- H20 multi-GPU training with checkpointing, metrics logging, and safetensors checkpoint output.

The pruning pipeline below starts from the resulting local checkpoint. It does not download or create the base model during the pruning run.

## Public Data Provenance

The base model data recipe follows the public-data spirit of `charent/ChatLM-mini-Chinese` while converting the task to decoder-only causal-LM training. The configured public sources include:

- `YeungNLP/firefly-pretrain-dataset`, including `webText2019zh.jsonl`, for Chinese web/community text.
- `ZhouLV/Chinese-Train-Datasets`, `baike2018qa/baike_qa_train.json`, for encyclopedia QA.
- `ticoAg/Chinese-medical-dialogue` for Chinese medical dialogue and QA.
- `wangrui6/Zhihu-KOL` for Zhihu-style QA.
- `BelleGroup/train_1M_CN`, `BelleGroup/train_2M_CN`, and `BelleGroup/train_3.5M_CN` for Chinese instruction/chat data.
- `YeungNLP/firefly-pretrain-dataset`, including `wiki_zh.jsonl`, for Chinese Wikipedia-like text.

The data preparation path downloads/cache-stages those sources, normalizes local/cache-backed records into one JSONL corpus, trains or loads the fixed tokenizer, and then trains the decoder-only model with next-token prediction.

## Local Checkpoint

Set the model path once:

```bash
export MODEL="/PATH/TO/MY/DECODER_SLM_CHECKPOINT"
```

The folder should contain at least:

```text
config.json
model.safetensors
tokenizer.json
```

Run a quick sanity check:

```bash
python - <<'PY'
import os, pathlib

p = pathlib.Path(os.environ["MODEL"]).expanduser()
print("path exists:", p.exists())
print("config:", (p / "config.json").exists())
print("model:", (p / "model.safetensors").exists())
print("tokenizer:", (p / "tokenizer.json").exists())
PY
```

## Run Pipeline

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
DTYPE=fp16 \
bash run_linear_sparsity_revision_from_base.sh "$MODEL"
```

The launcher uses:

- `torchrun` with `NPROC_PER_NODE=8` for regular SFT, contrastive SFT, original one-shot evaluation, and progressive magnitude pruning.
- `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7` to expose all 8 H20 GPUs to every distributed stage.
- The launcher fails fast unless it sees `EXPECTED_GPU_COUNT=8`, 8 visible GPU ids, and `NPROC_PER_NODE=8`. Set `ALLOW_H20_WORLD_SIZE_MISMATCH=1` only for deliberate debug runs.
- Every prompt/response eval launch passes `--expected-world-size 8` and `--expected-visible-gpu-count 8`; if an eval starts with only 4 ranks or 4 visible GPUs, it aborts before generating metrics.
- Progressive magnitude jobs run sequentially at 30% and 50%; each progressive job uses all visible GPUs instead of splitting one job per GPU.
- The active terminal environment's `python`/`python3`; bare executable names are resolved through `PATH`.
- FP16 autocast training with GradScaler, while keeping trainable weights in full precision.
- EOS-reinforced SFT and progressive recovery. The pipeline defaults `EOS_LOSS_WEIGHT=5.0`, so supervised `<|eos|>` labels get extra loss weight during stopping recovery.
- SDPA attention by default, not FlashAttention 2.
- `SYMPY_GROUND_TYPES=python` and disabled Dynamo/compile paths to avoid the H20 `gmp: overflow in mpz type` abort.
- Evaluator progress bars show rank 0's shard only. Check `world_size=8` and `all_rank_shards=[...]` in the log to confirm all 8 ranks are active.
- Progressive sparsity logs print `Linear sparsity runtime: world_size=8 ... rank_devices=[...]` before each progressive job.
- `MAX_NEW_TOKEN_HIT_RATE_THRESHOLD` defaults to `1.01` for pruning runs, so damaged pruned checkpoints still produce metric rows with max-token-hit rates instead of aborting the final matrix.

## Expected Outputs

The final combined JSON is written to:

```text
results/scenic_revision_sparsity_summary.json
```

Counting dense baselines, the expected final JSON has 20 result rows:

- 2 dense baselines: regular SFT and contrastive SFT.
- 14 original one-shot pruning rows:
  - regular SFT: 7 rows.
  - contrastive SFT: 7 rows.
  - 30%: magnitude, wanda, gradient.
  - 50%: magnitude, wanda, gradient, nvidia24.
- 4 progressive magnitude pruning rows:
  - regular SFT at 30% and 50%.
  - contrastive SFT at 30% and 50%.

Progressive recovery uses 1 recovery epoch after every pruning stage and 1 final recovery epoch after all stages.

The summary is intended to include training-data EM@1/EM@5, benchmark EM@1/EM@5, and benchmark easy/medium/hard breakdowns when those eval files are available.

## Decoder-Only C-Eval Metrics

C-Eval is still part of the decoder-only model reporting. It is separate from the pruning summary JSON and is run directly on whichever local checkpoint you want to report.

Run C-Eval on the local base checkpoint:

```bash
python scripts/eval_ceval.py \
  --checkpoint "$MODEL" \
  --split val \
  --n-shot 5 \
  --dtype fp16 \
  --device cuda \
  --output-dir results/ceval_local_decoder
```

Run C-Eval on the regular SFT checkpoint:

```bash
python scripts/eval_ceval.py \
  --checkpoint runs/revision-original-four-one-shot/training/base_sft_5ep/final \
  --split val \
  --n-shot 5 \
  --dtype fp16 \
  --device cuda \
  --output-dir results/ceval_regular_sft
```

Run C-Eval on the contrastive SFT checkpoint:

```bash
python scripts/eval_ceval.py \
  --checkpoint runs/revision-original-four-one-shot/training/contrastive_sft_5ep/final \
  --split val \
  --n-shot 5 \
  --dtype fp16 \
  --device cuda \
  --output-dir results/ceval_contrastive_sft
```

The evaluator scores the conditional log probability of answer choices `A`, `B`, `C`, and `D`, which is more stable for decoder-only checkpoints than free-form generation. Each C-Eval run writes:

```text
ceval_summary.json
ceval_predictions.csv
ceval_category_summary.csv
```

The category summary reports Humanities, STEM, Social Science, and Other accuracy.

## Important Defaults

The pipeline-facing configs are:

```text
configs/sft_0p2b_8gpu.yaml
configs/contrastive_sft_8gpu.yaml
configs/pruning_benchmark_regular_sft.yaml
configs/pruning_benchmark_contrastive_sft.yaml
```

The active precision defaults are:

```yaml
bf16: false
fp16: true
load_in_training_dtype: false
attn_implementation: sdpa
flash_attention: false
sdp_flash: true
torch_compile: false
```

Do not set `load_in_training_dtype: true` for this FP16 run unless you intentionally want pure FP16 trainable weights and have disabled scaler-based gradient clipping. Pure FP16 weights can trigger:

```text
ValueError: Attempting to unscale FP16 gradients.
```

## Tokenizer Repair

If the checkpoint fails with a tokenizer-class error, inspect the tokenizer metadata:

```bash
export MODEL="/path/to/the/folder/that/contains/tokenizer.json"

python - <<'PY'
import json, os, pathlib

p = pathlib.Path(os.environ["MODEL"]).expanduser()
for name in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "config.json", "generation_config.json", "model.safetensors"]:
    f = p / name
    print(name, "exists:", f.exists())

for name in ["tokenizer_config.json", "config.json"]:
    f = p / name
    if f.exists():
        data = json.loads(f.read_text())
        print(name, "tokenizer_class:", data.get("tokenizer_class"))
        print(name, "model_type:", data.get("model_type"))
PY
```

Regenerate the fast tokenizer wrapper files from `tokenizer.json`:

```bash
export MODEL="/path/to/the/folder/that/contains/tokenizer.json"

python - <<'PY'
import os, pathlib, shutil
from transformers import AutoTokenizer, PreTrainedTokenizerFast

p = pathlib.Path(os.environ["MODEL"]).expanduser()
tok_json = p / "tokenizer.json"
if not tok_json.exists():
    raise SystemExit(f"Missing tokenizer.json in {p}")

for name in ["tokenizer_config.json", "special_tokens_map.json"]:
    f = p / name
    if f.exists():
        shutil.copy2(f, p / f"{name}.bak")

tok = PreTrainedTokenizerFast(
    tokenizer_file=str(tok_json),
    pad_token="<|pad|>",
    unk_token="<|unk|>",
    bos_token="<|bos|>",
    eos_token="<|eos|>",
    additional_special_tokens=["<|user|>", "<|assistant|>", "<|system|>"],
    model_max_length=2048,
)
tok.save_pretrained(str(p))

loaded = AutoTokenizer.from_pretrained(str(p), trust_remote_code=False, use_fast=True)
print("loaded:", loaded.__class__.__name__)
print("vocab:", len(loaded))
print("eos:", loaded.eos_token, loaded.eos_token_id)
print("pad:", loaded.pad_token, loaded.pad_token_id)
PY
```

## Troubleshooting

If you see FlashAttention 2 errors, pull latest `main` and confirm `configs/sft_0p2b_8gpu.yaml` has:

```yaml
attn_implementation: sdpa
flash_attention: false
```

If you see `Attempting to unscale FP16 gradients`, confirm the local configs have:

```yaml
fp16: true
load_in_training_dtype: false
```

If you see `gmp: overflow in mpz type`, run with:

```bash
SYMPY_GROUND_TYPES=python \
TORCHDYNAMO_DISABLE=1 \
TORCH_COMPILE_DISABLE=1 \
ACCELERATE_DYNAMO_BACKEND=no \
bash run_linear_sparsity_revision_from_base.sh "$MODEL"
```

The main launcher sets those defaults automatically.

## How To Cite

GitHub can read [`CITATION.cff`](CITATION.cff) and show a "Cite this repository" button. Use this BibTeX entry for this codebase:

```bibtex
@software{decoder_chinese_slm_2026,
  author = {huluk98},
  title = {Decoder-Chinese-SLM: A Decoder-Only Chinese Small Language Model Training Codebase},
  year = {2026},
  version = {0.1.0},
  url = {https://github.com/huluk98/Decoder-Chinese-SLM}
}
```

When describing the model design target, also cite the same-size public reference model:

- `charent/ChatLM-mini-Chinese`: `https://huggingface.co/charent/ChatLM-mini-Chinese`.

## References And Citations

Reference model and benchmark:

- `charent/ChatLM-mini-Chinese`: same-size public reference target for the 0.2B parameter budget, 29,298-token vocabulary, Chinese public-data recipe, loss-curve style, and C-Eval reporting comparison. Repository: `https://huggingface.co/charent/ChatLM-mini-Chinese`; config: `https://huggingface.co/charent/ChatLM-mini-Chinese/blob/main/config.json`.
- `ceval/ceval-exam`: C-Eval validation/test loading: `https://huggingface.co/datasets/ceval/ceval-exam`.
- `hkust-nlp/ceval`: official C-Eval benchmark context and subject/category mapping: `https://github.com/hkust-nlp/ceval`.

Public data repositories:

- `YeungNLP/firefly-pretrain-dataset`: `https://huggingface.co/datasets/YeungNLP/firefly-pretrain-dataset`.
- `ZhouLV/Chinese-Train-Datasets`: `https://huggingface.co/datasets/ZhouLV/Chinese-Train-Datasets`.
- `ticoAg/Chinese-medical-dialogue`: `https://huggingface.co/datasets/ticoAg/Chinese-medical-dialogue`.
- `wangrui6/Zhihu-KOL`: `https://huggingface.co/datasets/wangrui6/Zhihu-KOL`.
- `BelleGroup/train_1M_CN`: `https://huggingface.co/datasets/BelleGroup/train_1M_CN`.
- `BelleGroup/train_2M_CN`: `https://huggingface.co/datasets/BelleGroup/train_2M_CN`.
- `BelleGroup/train_3.5M_CN`: `https://huggingface.co/datasets/BelleGroup/train_3.5M_CN`.

Software:

- Hugging Face Transformers for model definitions, tokenizer/model loading, generation, and checkpoint serialization: `https://github.com/huggingface/transformers`.
- Hugging Face Datasets for public data and C-Eval loading: `https://github.com/huggingface/datasets`.
- Hugging Face Tokenizers for BPE tokenizer training/runtime components: `https://github.com/huggingface/tokenizers`.
- Safetensors for model tensor serialization: `https://github.com/huggingface/safetensors`.
