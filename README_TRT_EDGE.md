# TensorRT Edge Deployment Pipeline

This repo includes a TensorRT-oriented deployment path for decoder-only Chinese SLM checkpoints on NVIDIA Jetson Orin NX or another NVIDIA GPU. The first stable path is a no-cache decoder engine. It is simple and reliable, but slower than KV-cache decoding because each generated token reruns the whole context window.

## Artifacts

All generated files stay under `outputs/`:

- `outputs/onnx/model_decoder_nocache.onnx`
- `outputs/onnx/model_decoder_cache.onnx` when cached export works for the model
- `outputs/trt/model_fp16.engine`
- `outputs/trt/model_int8.engine`
- `outputs/trt/model_int8_calib.cache`
- `outputs/trt/model_int4.engine` only when the local ModelOpt/TensorRT-LLM stack supports it
- `outputs/benchmarks/{precision}_benchmark.json`
- `outputs/benchmarks/{precision}_benchmark.csv`

## Install Notes

Jetson TensorRT normally comes from JetPack, not plain pip:

```bash
sudo apt install python3-libnvinfer python3-libnvinfer-dev
python -m pip install -r requirements_edge.txt
```

If `onnxruntime` or `nvidia-modelopt` is unavailable on aarch64, the scripts still fail clearly for the specific feature that needs it.

## One-Command Pipeline

```bash
bash scripts/run_all_trt_pipeline.sh ./model_path ./data/eval.json
```

Useful overrides:

```bash
OVERWRITE=1 OPT_SEQ_LEN=64 MAX_SEQ_LEN=128 MAX_NEW_TOKENS=64 \
bash scripts/run_all_trt_pipeline.sh ./model_path ./data/eval.json
```

For Qwen-style instruct tokenizers, pass a chat template prompt format:

```bash
PROMPT_FORMAT=chat-template bash scripts/run_all_trt_pipeline.sh ./model_path ./data/eval.json
```

## FP16 Baseline

FP16 is the baseline TensorRT engine:

```bash
python scripts/export_decoder_onnx.py --model-path ./model_path --overwrite
python scripts/build_trt_engines.py \
  --onnx outputs/onnx/model_decoder_nocache.onnx \
  --precision fp16 \
  --model-path ./model_path \
  --min_seq_len 1 --opt_seq_len 64 --max_seq_len 128 --batch_size 1 \
  --overwrite
```

The ONNX exporter uses int32 dummy `input_ids` and `attention_mask` so TensorRT bindings are int32 where the model supports it. The builder uses TensorRT 10 memory-pool APIs when available.

## INT8 Calibration

INT8 uses TensorRT calibration with representative prompts from JSON, JSONL, CSV, or TXT data:

```bash
python scripts/build_trt_engines.py \
  --onnx outputs/onnx/model_decoder_nocache.onnx \
  --precision int8 \
  --model-path ./model_path \
  --calib_json ./data/eval.json \
  --calib-samples 128 \
  --min_seq_len 1 --opt_seq_len 64 --max_seq_len 128 --batch_size 1 \
  --overwrite
```

The calibrator looks for fields such as `prompt`, `instruction`, `input`, `question`, `text`, `response`, and `output`. It writes `outputs/trt/model_int8_calib.cache` and reuses it on later builds.

## INT4 Caveats

INT4 is not a TensorRT builder flag. The script treats INT4 as a best-effort NVIDIA ModelOpt AWQ or weight-only path:

```bash
python scripts/build_trt_engines.py \
  --precision int4 \
  --model-path ./model_path \
  --calib_json ./data/eval.json
```

If ModelOpt, TensorRT-LLM export, or `trtllm-build` is missing or incompatible, the script exits nonzero and prints:

```text
INT4 TensorRT build is not supported in this environment. Use ModelOpt/TensorRT-LLM weight-only INT4 path or upgrade TensorRT/ModelOpt.
```

That is intentional. A fake `--int4` flag on a normal TensorRT engine would not produce a valid weight-only INT4 decoder.

## Shape Choices

The default edge profile is bounded and batch-1:

- batch size: min/opt/max = `1`
- prompt length: min `1`, opt `64`, max `128`
- no-cache generation: each step feeds the current context window
- cached decoding export: decode input is intended to be one token per step, but model-specific ONNX cache export can be unstable

If engine building is killed on Jetson due to memory pressure, reduce `--max_seq_len`, `--opt_seq_len`, and `--workspace-gb`.

## No-Cache vs KV Cache

`scripts/export_decoder_onnx.py` always exports `model_decoder_nocache.onnx`. It also attempts `model_decoder_cache.onnx` by flattening `past_key_values`, but not every Hugging Face decoder exports cleanly to ONNX with cache tensors.

The benchmark script currently targets the stable no-cache TensorRT engine. This gives correct end-to-end greedy generation and is a good first deployment baseline, but latency per token grows with context length. A production cached path should use a prefill engine plus a one-token decode engine, or a TensorRT-LLM runtime that manages KV cache directly.

## Benchmarking

```bash
python scripts/benchmark_trt_decoder.py \
  --engine outputs/trt/model_fp16.engine \
  --model-path ./model_path \
  --dataset ./data/eval.json \
  --precision fp16 \
  --max-new-tokens 64 \
  --max-seq-len 128 \
  --overwrite
```

The summary table reports:

- `engine_size_mb`: serialized engine size on disk
- `avg_latency_ms`: average total generation time per sample
- `tokens_per_sec`: generated tokens divided by measured generation latency
- `exact_match`: whitespace-normalized generated text versus reference fields
- `peak_memory_mb`: CUDA memory high-water estimate from `cudaMemGetInfo`

For short smart-home commands, exact match matters more than long-form language quality. Always compare FP16, INT8, and any supported INT4 path against the same dataset.

## Numerical Validation

Use the comparison script after export and build:

```bash
python scripts/compare_pytorch_onnx_trt.py \
  --model-path ./model_path \
  --prompt "打开客厅灯" \
  --onnx outputs/onnx/model_decoder_nocache.onnx \
  --engine outputs/trt/model_fp16.engine
```

It reports logits shape, max absolute difference versus PyTorch, argmax token agreement, and greedy decoded text for PyTorch, ONNX Runtime when installed, and TensorRT.

