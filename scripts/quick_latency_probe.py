#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from chatlm_decoder.tokenizer import prepare_decoder_tokenizer, strip_unused_decoder_model_kwargs


DEFAULT_PROMPT = "Explain artificial intelligence in one short sentence."


def select_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def dtype_for(name: str, device: torch.device) -> torch.dtype | str:
    name = str(name).lower()
    if name == "auto":
        return "auto"
    if name == "bf16":
        return torch.bfloat16 if device.type == "cuda" else torch.float32
    if name == "fp16":
        return torch.float16 if device.type == "cuda" else torch.float32
    if name == "fp32":
        return torch.float32
    raise ValueError(f"Unknown dtype: {name}")


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    index = (len(ordered) - 1) * float(q)
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def format_prompt(prompt: str, use_chat_format: bool) -> str:
    if not use_chat_format:
        return prompt
    return f"<|user|>\n{prompt}\n<|assistant|>\n"


def load_prompts(args: argparse.Namespace) -> list[str]:
    if args.prompt_file:
        path = Path(args.prompt_file).expanduser()
        prompts = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not prompts:
            raise ValueError(f"Prompt file has no non-empty lines: {path}")
    else:
        prompts = [args.prompt or DEFAULT_PROMPT]
    formatted = [format_prompt(prompt, use_chat_format=not bool(args.no_chat_format)) for prompt in prompts]
    batch_size = max(1, int(args.batch_size))
    while len(formatted) < batch_size:
        formatted.extend(formatted)
    return formatted[:batch_size]


def count_generated_tokens(ids: torch.Tensor, pad_token_id: int | None) -> int:
    values = ids.detach().cpu().tolist()
    if pad_token_id is not None:
        while values and int(values[-1]) == int(pad_token_id):
            values.pop()
    return len(values)


@torch.inference_mode()
def timed_generate(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    device: torch.device,
    max_new_tokens: int,
    num_beams: int,
) -> dict[str, Any]:
    token_start = time.perf_counter()
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        add_special_tokens=False,
    ).to(device)
    strip_unused_decoder_model_kwargs(inputs)
    tokenization_ms = (time.perf_counter() - token_start) * 1000.0

    prompt_width = int(inputs["input_ids"].shape[-1])
    synchronize(device)
    generate_start = time.perf_counter()
    output_ids = model.generate(
        **inputs,
        max_new_tokens=int(max_new_tokens),
        do_sample=False,
        num_beams=max(1, int(num_beams)),
        num_return_sequences=1,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    synchronize(device)
    generate_ms = (time.perf_counter() - generate_start) * 1000.0

    continuation_ids = output_ids[:, prompt_width:]
    generated_counts = [
        count_generated_tokens(row, getattr(tokenizer, "pad_token_id", None)) for row in continuation_ids
    ]
    generated_tokens = int(sum(generated_counts))
    return {
        "tokenization_ms": tokenization_ms,
        "generate_ms": generate_ms,
        "overall_ms": tokenization_ms + generate_ms,
        "prompt_tokens_per_sample": prompt_width,
        "generated_tokens": generated_tokens,
        "generated_tokens_per_sample": generated_counts,
        "tokens_per_second": (generated_tokens / (generate_ms / 1000.0)) if generate_ms > 0 else 0.0,
        "sample_output": tokenizer.decode(continuation_ids[0], skip_special_tokens=True).strip(),
    }


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "avg": float(statistics.mean(values)) if values else 0.0,
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "min": float(min(values)) if values else 0.0,
        "max": float(max(values)) if values else 0.0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quick local HF decoder latency probe.")
    parser.add_argument("checkpoint", help="Local HF decoder checkpoint directory.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--prompt-file", default=None, help="Optional text file with one prompt per line.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=16, help="Small quick-test decode length; default is 16, not 64.")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmup-runs", type=int, default=2)
    parser.add_argument("--num-beams", type=int, default=1, help="Use 1 for quick greedy latency.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="fp16", choices=("auto", "bf16", "fp16", "fp32"))
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--no-chat-format", action="store_true")
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    checkpoint = str(Path(args.checkpoint).expanduser())
    prompts = load_prompts(args)

    tokenizer = prepare_decoder_tokenizer(
        AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=bool(args.trust_remote_code))
    )
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        torch_dtype=dtype_for(args.dtype, device),
        trust_remote_code=bool(args.trust_remote_code),
    ).to(device)
    model.eval()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for _ in range(max(0, int(args.warmup_runs))):
        timed_generate(
            model,
            tokenizer,
            prompts,
            device,
            max_new_tokens=int(args.max_new_tokens),
            num_beams=int(args.num_beams),
        )

    ttft_runs = [
        timed_generate(model, tokenizer, prompts, device, max_new_tokens=1, num_beams=int(args.num_beams))
        for _ in range(max(1, int(args.runs)))
    ]
    decode_runs = [
        timed_generate(
            model,
            tokenizer,
            prompts,
            device,
            max_new_tokens=int(args.max_new_tokens),
            num_beams=int(args.num_beams),
        )
        for _ in range(max(1, int(args.runs)))
    ]

    generate_latencies = [float(row["generate_ms"]) for row in decode_runs]
    overall_latencies = [float(row["overall_ms"]) for row in decode_runs]
    ttft_latencies = [float(row["generate_ms"]) for row in ttft_runs]
    throughputs = [float(row["tokens_per_second"]) for row in decode_runs]
    generated_tokens = [int(row["generated_tokens"]) for row in decode_runs]

    cuda_info: dict[str, Any] = {}
    if device.type == "cuda":
        cuda_info = {
            "gpu_name": torch.cuda.get_device_name(device),
            "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
            "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 1024**3,
        }

    summary = {
        "checkpoint": checkpoint,
        "device": str(device),
        "dtype": str(args.dtype),
        "batch_size": int(args.batch_size),
        "max_new_tokens": int(args.max_new_tokens),
        "num_beams": int(args.num_beams),
        "runs": int(args.runs),
        "warmup_runs": int(args.warmup_runs),
        "prompt_tokens_per_sample": int(decode_runs[0]["prompt_tokens_per_sample"]),
        "generated_tokens_avg": float(statistics.mean(generated_tokens)) if generated_tokens else 0.0,
        "ttft_ms": summarize(ttft_latencies),
        "decode_generate_ms": summarize(generate_latencies),
        "overall_with_tokenization_ms": summarize(overall_latencies),
        "generated_tokens_per_second": summarize(throughputs),
        "sample_output": str(decode_runs[0]["sample_output"]),
        **cuda_info,
    }

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.output_json:
        output_path = Path(args.output_json).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
