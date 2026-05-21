#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import logging
import statistics
import sys
import time
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from trt_edge_common import (
    TensorRTEngineRunner,
    apply_prompt_format,
    ensure_output_path,
    expand_path,
    format_table,
    import_required,
    infer_precision_from_engine,
    normalize_exact,
    prompt_and_reference,
    read_records,
    setup_logging,
)


def load_tokenizer(model_path: str, trust_remote_code: bool) -> Any:
    transformers = import_required("transformers", "benchmark tokenizer loading")
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token_id", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def output_logits(outputs: dict[str, Any]) -> Any:
    if "logits" in outputs:
        return outputs["logits"]
    for name, value in outputs.items():
        if "logits" in name.lower():
            return value
    return next(iter(outputs.values()))


def encode_prompt(tokenizer: Any, text: str, max_seq_len: int, add_special_tokens: bool) -> list[int]:
    encoded = tokenizer(
        text,
        return_tensors="np",
        truncation=True,
        max_length=int(max_seq_len),
        add_special_tokens=bool(add_special_tokens),
    )
    input_ids = encoded["input_ids"][0].astype("int64").tolist()
    if not input_ids:
        fallback = getattr(tokenizer, "bos_token_id", None) or getattr(tokenizer, "eos_token_id", None)
        if fallback is None:
            raise ValueError("Tokenizer produced empty input_ids and has no bos/eos token fallback.")
        input_ids = [int(fallback)]
    return [int(token_id) for token_id in input_ids]


def greedy_generate_nocache(
    runner: TensorRTEngineRunner,
    tokenizer: Any,
    prompt_text: str,
    max_new_tokens: int,
    max_seq_len: int,
    add_special_tokens: bool,
) -> dict[str, Any]:
    np = import_required("numpy", "TensorRT benchmark arrays")
    context_ids = encode_prompt(tokenizer, prompt_text, max_seq_len=max_seq_len, add_special_tokens=add_special_tokens)
    generated: list[int] = []
    token_latencies_ms: list[float] = []
    first_token_latency_ms: float | None = None
    truncated_context = False
    eos_token_id = getattr(tokenizer, "eos_token_id", None)

    for step in range(int(max_new_tokens)):
        window_ids = context_ids[-int(max_seq_len) :]
        truncated_context = truncated_context or len(context_ids) > int(max_seq_len)
        input_ids = np.asarray([window_ids], dtype=np.int32)
        attention_mask = np.ones_like(input_ids, dtype=np.int32)
        start = time.perf_counter()
        outputs = runner.infer({"input_ids": input_ids, "attention_mask": attention_mask})
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        logits = output_logits(outputs)
        next_token = int(np.argmax(logits[0, -1, :]))
        if step == 0:
            first_token_latency_ms = elapsed_ms
        token_latencies_ms.append(elapsed_ms)
        generated.append(next_token)
        context_ids.append(next_token)
        if eos_token_id is not None and next_token == int(eos_token_id):
            break

    decoded = tokenizer.decode(generated, skip_special_tokens=True).strip()
    total_latency_ms = float(sum(token_latencies_ms))
    return {
        "generated_ids": generated,
        "generated_text": decoded,
        "first_token_latency_ms": first_token_latency_ms,
        "per_token_latencies_ms": token_latencies_ms,
        "avg_per_token_latency_ms": float(statistics.mean(token_latencies_ms)) if token_latencies_ms else 0.0,
        "total_generation_latency_ms": total_latency_ms,
        "tokens_per_sec": (len(generated) / (total_latency_ms / 1000.0)) if total_latency_ms > 0 else 0.0,
        "generated_tokens": len(generated),
        "prompt_tokens": len(context_ids) - len(generated),
        "truncated_context": truncated_context,
    }


def write_outputs(json_path: Path, csv_path: Path, summary: dict[str, Any], results: list[dict[str, Any]], overwrite: bool) -> None:
    ensure_output_path(json_path, overwrite=overwrite, kind="benchmark JSON")
    ensure_output_path(csv_path, overwrite=overwrite, kind="benchmark CSV")
    payload = {"summary": summary, "results": results}
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    fieldnames = [
        "index",
        "prompt",
        "reference",
        "generated_text",
        "exact_match",
        "prompt_tokens",
        "generated_tokens",
        "first_token_latency_ms",
        "avg_per_token_latency_ms",
        "total_generation_latency_ms",
        "tokens_per_sec",
        "truncated_context",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def summarize_results(precision: str, engine_path: Path, results: list[dict[str, Any]], peak_memory_mb: float | None) -> dict[str, Any]:
    latency_values = [float(row["total_generation_latency_ms"]) for row in results]
    tps_values = [float(row["tokens_per_sec"]) for row in results]
    comparable = [row for row in results if row.get("reference") is not None]
    matches = [row for row in comparable if row.get("exact_match") is True]
    return {
        "precision": precision,
        "engine_path": str(engine_path),
        "engine_size_mb": engine_path.stat().st_size / 1024**2,
        "num_samples": len(results),
        "num_reference_samples": len(comparable),
        "avg_latency_ms": float(statistics.mean(latency_values)) if latency_values else 0.0,
        "p50_latency_ms": float(statistics.median(latency_values)) if latency_values else 0.0,
        "tokens_per_sec": float(statistics.mean(tps_values)) if tps_values else 0.0,
        "exact_match": (len(matches) / len(comparable)) if comparable else None,
        "peak_memory_mb": peak_memory_mb,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark a TensorRT decoder engine on real JSON prompts.")
    parser.add_argument("--engine", required=True, help="Path to TensorRT engine.")
    parser.add_argument("--model-path", required=True, help="HF tokenizer/model directory for tokenization.")
    parser.add_argument("--dataset", required=True, help="JSON/JSONL/CSV/TXT prompt dataset.")
    parser.add_argument("--precision", default=None, help="Precision label. Defaults to inferred engine filename.")
    parser.add_argument("--output-dir", default="outputs/benchmarks")
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--warmup-samples", type=int, default=2)
    parser.add_argument("--prompt-field", default=None)
    parser.add_argument("--prompt-format", choices=("raw", "legacy", "chat-template"), default="raw")
    parser.add_argument("--system-prompt", default=None)
    parser.add_argument("--add-special-tokens", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)
    engine_path = expand_path(args.engine)
    if not engine_path.exists():
        raise FileNotFoundError(f"TensorRT engine does not exist: {engine_path}")
    precision = args.precision or infer_precision_from_engine(engine_path)

    tokenizer = load_tokenizer(args.model_path, trust_remote_code=bool(args.trust_remote_code))
    records = read_records(args.dataset, limit=int(args.max_samples))
    if not records:
        raise ValueError(f"Dataset produced zero records: {args.dataset}")
    runner = TensorRTEngineRunner(engine_path, verbose=bool(args.verbose))
    cache_inputs = [name for name in runner.input_names if "past_key_values" in name or name.startswith("past_")]
    if cache_inputs:
        raise RuntimeError(
            "This generic benchmark currently supports the stable no-cache TensorRT engine. "
            f"Cached engine inputs were detected: {cache_inputs}"
        )
    logging.info("TensorRT inputs: %s outputs: %s", runner.input_names, runner.output_names)

    if args.warmup_samples > 0:
        logging.info("Running %d warmup sample(s).", args.warmup_samples)
        for record in records[: int(args.warmup_samples)]:
            prompt, _ = prompt_and_reference(record, prompt_field=args.prompt_field)
            prompt = apply_prompt_format(tokenizer, prompt, args.prompt_format, args.system_prompt)
            greedy_generate_nocache(
                runner,
                tokenizer,
                prompt,
                max_new_tokens=min(4, int(args.max_new_tokens)),
                max_seq_len=int(args.max_seq_len),
                add_special_tokens=bool(args.add_special_tokens),
            )

    free_start, total_memory = runner.cuda.mem_info()
    min_free = free_start
    results: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        prompt, reference = prompt_and_reference(record, prompt_field=args.prompt_field)
        formatted_prompt = apply_prompt_format(tokenizer, prompt, args.prompt_format, args.system_prompt)
        generated = greedy_generate_nocache(
            runner,
            tokenizer,
            formatted_prompt,
            max_new_tokens=int(args.max_new_tokens),
            max_seq_len=int(args.max_seq_len),
            add_special_tokens=bool(args.add_special_tokens),
        )
        free_now, _ = runner.cuda.mem_info()
        min_free = min(min_free, free_now)
        normalized_prediction = normalize_exact(generated["generated_text"], tokenizer)
        normalized_reference = normalize_exact(reference, tokenizer) if reference is not None else None
        exact_match = normalized_prediction == normalized_reference if normalized_reference is not None else None
        row = {
            "index": index,
            "prompt": prompt,
            "formatted_prompt": formatted_prompt,
            "reference": reference,
            "generated_text": generated["generated_text"],
            "normalized_prediction": normalized_prediction,
            "normalized_reference": normalized_reference,
            "exact_match": exact_match,
            **generated,
        }
        results.append(row)
        logging.info(
            "[%d/%d] tokens=%d latency=%.2fms tps=%.2f exact=%s",
            index + 1,
            len(records),
            row["generated_tokens"],
            row["total_generation_latency_ms"],
            row["tokens_per_sec"],
            exact_match,
        )

    peak_memory_mb = (total_memory - min_free) / 1024**2 if total_memory else None
    summary = summarize_results(precision, engine_path, results, peak_memory_mb)
    output_dir = Path(args.output_dir).expanduser()
    json_path = output_dir / f"{precision}_benchmark.json"
    csv_path = output_dir / f"{precision}_benchmark.csv"
    write_outputs(json_path, csv_path, summary, results, overwrite=bool(args.overwrite))

    table = format_table(
        ["precision", "engine_size_mb", "avg_latency_ms", "tokens_per_sec", "exact_match", "peak_memory_mb"],
        [
            [
                summary["precision"],
                f"{summary['engine_size_mb']:.2f}",
                f"{summary['avg_latency_ms']:.2f}",
                f"{summary['tokens_per_sec']:.2f}",
                "n/a" if summary["exact_match"] is None else f"{summary['exact_match']:.4f}",
                "n/a" if summary["peak_memory_mb"] is None else f"{summary['peak_memory_mb']:.2f}",
            ]
        ],
    )
    print(table)
    print(f"Wrote benchmark JSON: {json_path}")
    print(f"Wrote benchmark CSV: {csv_path}")
    runner.close()


if __name__ == "__main__":
    main()

