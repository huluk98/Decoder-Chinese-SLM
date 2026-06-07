#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import statistics
import sys
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[0]
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from chatlm_decoder.command_eval import canonicalize_command_response
from chatlm_decoder.tokenizer import prepare_decoder_tokenizer
from trt_edge_common import (
    TensorRTEngineRunner,
    apply_prompt_format,
    ensure_output_path,
    expand_path,
    import_required,
    prompt_and_reference,
    read_records,
    setup_logging,
)


DIFFICULTY_FIELDS = ("difficulty", "hardness", "difficulty_level", "hardness_level", "level")
ZERO_WIDTH_PATTERN = __import__("re").compile(r"[\u200b-\u200f\u202a-\u202e\u2060\ufeff]")
TRAILING_STOP_PATTERN = __import__("re").compile(r"[\s。．.；;，,]+$")
QUOTE_TRANSLATION = str.maketrans(
    {
        "“": '"',
        "”": '"',
        "„": '"',
        "‘": "'",
        "’": "'",
        "‚": "'",
        "（": "(",
        "）": ")",
        "，": ",",
        "：": ":",
        "；": ";",
        "［": "[",
        "］": "]",
        "｛": "{",
        "｝": "}",
        "＝": "=",
    }
)
CHAT_MARKERS = (
    "<|user|>",
    "<|assistant|>",
    "<|system|>",
    "<|eos|>",
    "assistant:",
    "Assistant:",
    "ASSISTANT:",
    "回答:",
    "回答：",
    "答案:",
    "答案：",
    "输出:",
    "输出：",
)


def percentile(values: list[float], percent: float) -> float | None:
    finite = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not finite:
        return None
    if len(finite) == 1:
        return finite[0]
    rank = (len(finite) - 1) * float(percent)
    low = int(math.floor(rank))
    high = int(math.ceil(rank))
    if low == high:
        return finite[low]
    weight = rank - low
    return finite[low] * (1.0 - weight) + finite[high] * weight


def file_size_mb(path: str | Path | None) -> float | None:
    if not path:
        return None
    candidate = Path(path).expanduser()
    if not candidate.is_file():
        return None
    return candidate.stat().st_size / 1024**2


def artifact_size_mb(path: str | Path | None) -> float | None:
    if not path:
        return None
    candidate = Path(path).expanduser()
    if candidate.is_file():
        return candidate.stat().st_size / 1024**2
    if not candidate.is_dir():
        return None
    total = 0
    for root, _dirs, files in os.walk(candidate):
        for name in files:
            file_path = Path(root) / name
            try:
                total += file_path.stat().st_size
            except OSError:
                continue
    return total / 1024**2


@dataclass(frozen=True)
class Beam:
    tokens: tuple[int, ...]
    score: float
    done: bool
    reached_max_new_tokens: bool


def load_tokenizer(model_path: str, trust_remote_code: bool) -> Any:
    transformers = import_required("transformers", "TensorRT prompt-response eval tokenizer loading")
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    return prepare_decoder_tokenizer(tokenizer)


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


def stop_token_ids(tokenizer: Any) -> set[int]:
    stops: set[int] = set()
    for attr in ("eos_token_id", "pad_token_id"):
        value = getattr(tokenizer, attr, None)
        if value is not None:
            stops.add(int(value))
    return stops


def apply_repetition_penalty(logits: Any, token_ids: list[int], penalty: float) -> Any:
    if float(penalty) == 1.0:
        return logits
    adjusted = logits.copy()
    for token_id in set(int(token_id) for token_id in token_ids):
        if 0 <= token_id < adjusted.shape[-1]:
            adjusted[token_id] = adjusted[token_id] * penalty if adjusted[token_id] < 0 else adjusted[token_id] / penalty
    return adjusted


def top_log_probs(logits: Any, count: int) -> list[tuple[int, float]]:
    np = import_required("numpy", "TensorRT beam-search math")
    scores = logits.astype(np.float32)
    scores = scores - np.max(scores)
    log_probs = scores - np.log(np.exp(scores).sum())
    count = max(1, min(int(count), int(log_probs.shape[-1])))
    if count == int(log_probs.shape[-1]):
        top_ids = np.arange(int(log_probs.shape[-1]))
    else:
        top_ids = np.argpartition(log_probs, -count)[-count:]
    ranked = sorted(((int(token_id), float(log_probs[token_id])) for token_id in top_ids), key=lambda item: item[1], reverse=True)
    return ranked[:count]


def infer_next_logits(runner: TensorRTEngineRunner, context_ids: list[int], max_seq_len: int) -> Any:
    np = import_required("numpy", "TensorRT prompt-response eval arrays")
    window_ids = context_ids[-int(max_seq_len) :]
    input_ids = np.asarray([window_ids], dtype=np.int32)
    attention_mask = np.ones_like(input_ids, dtype=np.int32)
    outputs = runner.infer({"input_ids": input_ids, "attention_mask": attention_mask})
    logits = output_logits(outputs)
    return logits[0, -1, :]


def decode_completion(tokenizer: Any, token_ids: tuple[int, ...]) -> str:
    stops = stop_token_ids(tokenizer)
    trimmed: list[int] = []
    for token_id in token_ids:
        if int(token_id) in stops:
            break
        trimmed.append(int(token_id))
    return tokenizer.decode(trimmed, skip_special_tokens=True).strip()


def beam_generate_nocache(
    runner: TensorRTEngineRunner,
    tokenizer: Any,
    prompt_text: str,
    max_new_tokens: int,
    max_seq_len: int,
    beam_width: int,
    add_special_tokens: bool,
    repetition_penalty: float,
) -> dict[str, Any]:
    prompt_ids = encode_prompt(tokenizer, prompt_text, max_seq_len=max_seq_len, add_special_tokens=add_special_tokens)
    beams = [Beam(tokens=tuple(), score=0.0, done=False, reached_max_new_tokens=False)]
    stops = stop_token_ids(tokenizer)
    token_latencies_ms: list[float] = []
    first_token_latency_ms: float | None = None
    truncated_context = False

    for step in range(int(max_new_tokens)):
        candidates: list[Beam] = []
        all_done = True
        for beam in beams:
            if beam.done:
                candidates.append(beam)
                continue
            all_done = False
            context_ids = [*prompt_ids, *beam.tokens]
            truncated_context = truncated_context or len(context_ids) > int(max_seq_len)
            start = time.perf_counter()
            logits = infer_next_logits(runner, context_ids, max_seq_len=max_seq_len)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            if step == 0 and first_token_latency_ms is None:
                first_token_latency_ms = elapsed_ms
            token_latencies_ms.append(elapsed_ms)
            logits = apply_repetition_penalty(logits, context_ids, float(repetition_penalty))
            for token_id, log_prob in top_log_probs(logits, count=max(beam_width * 2, beam_width)):
                done = int(token_id) in stops
                new_tokens = beam.tokens if done else (*beam.tokens, int(token_id))
                reached_max = (not done) and len(new_tokens) >= int(max_new_tokens)
                candidates.append(
                    Beam(
                        tokens=new_tokens,
                        score=float(beam.score + log_prob),
                        done=done or reached_max,
                        reached_max_new_tokens=reached_max,
                    )
                )
        if all_done:
            break
        beams = sorted(candidates, key=lambda item: item.score, reverse=True)[: int(beam_width)]
        if all(beam.done for beam in beams):
            break

    beams = sorted(beams, key=lambda item: item.score, reverse=True)[: int(beam_width)]
    candidates = [
        {
            "rank": rank,
            "generated_text": decode_completion(tokenizer, beam.tokens),
            "generated_tokens": len(beam.tokens),
            "score": beam.score,
            "reached_max_new_tokens": beam.reached_max_new_tokens,
        }
        for rank, beam in enumerate(beams, start=1)
    ]
    best = candidates[0] if candidates else {
        "rank": 1,
        "generated_text": "",
        "generated_tokens": 0,
        "score": 0.0,
        "reached_max_new_tokens": False,
    }
    total_latency_ms = float(sum(token_latencies_ms))
    return {
        "generated_text": best["generated_text"],
        "generated_tokens": best["generated_tokens"],
        "reached_max_new_tokens": best["reached_max_new_tokens"],
        "top_k_candidates": candidates,
        "first_token_latency_ms": first_token_latency_ms,
        "avg_per_token_latency_ms": float(statistics.mean(token_latencies_ms)) if token_latencies_ms else 0.0,
        "total_generation_latency_ms": total_latency_ms,
        "tokens_per_sec": (sum(candidate["generated_tokens"] for candidate in candidates[:1]) / (total_latency_ms / 1000.0))
        if total_latency_ms > 0
        else 0.0,
        "truncated_context": truncated_context,
    }


def strip_wrapping_quotes(text: str) -> str:
    text = text.strip()
    changed = True
    while changed and len(text) >= 2:
        changed = False
        for left, right in (("`", "`"), ('"', '"'), ("'", "'")):
            if text.startswith(left) and text.endswith(right):
                text = text[1:-1].strip()
                changed = True
    return text


def normalize_whitespace_exact(text: str, tokenizer: Any | None = None) -> str:
    text = unicodedata.normalize("NFKC", str(text))
    text = ZERO_WIDTH_PATTERN.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    for attr in ("bos_token", "eos_token", "pad_token", "unk_token"):
        token = getattr(tokenizer, attr, None) if tokenizer is not None else None
        if token:
            text = text.replace(str(token), "")
    for marker in CHAT_MARKERS:
        if text.startswith(marker):
            text = text[len(marker) :].strip()
    for marker in ("<|user|>", "<|system|>", "\nUser:", "\n用户:"):
        if marker in text:
            text = text.split(marker, 1)[0].strip()
    text = strip_wrapping_quotes(text)
    return " ".join(text.split()).strip()


def comparison_text(text: str, tokenizer: Any | None, comparison_mode: str) -> str:
    normalized = normalize_whitespace_exact(text, tokenizer=tokenizer)
    if comparison_mode == "whitespace":
        return normalized
    normalized = normalized.translate(QUOTE_TRANSLATION)
    normalized = TRAILING_STOP_PATTERN.sub("", normalized)
    normalized = " ".join(normalized.split()).strip()
    if comparison_mode == "command":
        return canonicalize_command_response(normalized)
    return normalized


def first_present(record: dict[str, Any], fields: tuple[str, ...]) -> str:
    for field in fields:
        value = record.get(field)
        if value not in (None, ""):
            return str(value)
    return "unknown"


def grouped_metrics(results: list[dict[str, Any]], field: str) -> dict[str, Any]:
    groups = sorted({str(row.get(field) or "unknown") for row in results})
    output: dict[str, Any] = {}
    for group in groups:
        rows = [row for row in results if str(row.get(field) or "unknown") == group]
        exact = sum(1 for row in rows if bool(row.get("exact_match")))
        top_k = sum(1 for row in rows if bool(row.get("exact_match_at_top_k")))
        prefix = f"{field}_{group}"
        output[f"{prefix}_total_examples"] = len(rows)
        output[f"{prefix}_exact_match_correct"] = exact
        output[f"{prefix}_exact_match_accuracy"] = exact / float(len(rows) or 1)
        output[f"{prefix}_exact_match_at_top_k_correct"] = top_k
        output[f"{prefix}_exact_match_at_top_k_accuracy"] = top_k / float(len(rows) or 1)
    return output


def write_outputs(output_dir: Path, summary: dict[str, Any], results: list[dict[str, Any]], overwrite: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "prompt_response_eval_summary.json"
    metrics_path = output_dir / "metrics.json"
    predictions_path = output_dir / "prompt_response_eval_predictions.jsonl"
    debug_csv_path = output_dir / "prediction_debug.csv"
    for path, kind in (
        (summary_path, "summary JSON"),
        (metrics_path, "metrics JSON"),
        (predictions_path, "prediction JSONL"),
        (debug_csv_path, "prediction debug CSV"),
    ):
        ensure_output_path(path, overwrite=overwrite, kind=kind)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    metrics_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with predictions_path.open("w", encoding="utf-8") as handle:
        for row in results:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    fieldnames = [
        "index",
        "prompt",
        "reference",
        "generated_text",
        "exact_match",
        "exact_match_at_top_k",
        "top_k_match_rank",
        "generated_tokens",
        "reached_max_new_tokens",
        "difficulty",
        "total_generation_latency_ms",
        "tokens_per_sec",
    ]
    with debug_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    (output_dir / "generation_samples.json").write_text(
        json.dumps(results[:20], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "exact_match_failure_cases.json").write_text(
        json.dumps([row for row in results if not bool(row.get("exact_match"))][:50], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "top_k_exact_match_failure_cases.json").write_text(
        json.dumps([row for row in results if not bool(row.get("exact_match_at_top_k"))][:50], ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a no-cache TensorRT decoder engine with EM@1 and EM@K.")
    parser.add_argument("--engine", required=True)
    parser.add_argument("--model-path", required=True, help="HF tokenizer/model path for tokenization.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--precision", default="trt")
    parser.add_argument("--variant", default=None)
    parser.add_argument("--runtime", default="TensorRT")
    parser.add_argument("--onnx-path", default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=1, help="Total number of dataset shards for parallel eval.")
    parser.add_argument("--shard-index", type=int, default=0, help="Zero-based shard index for this eval process.")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--exact-match-top-k", type=int, default=5)
    parser.add_argument("--comparison-mode", choices=("whitespace", "normalized", "command"), default="whitespace")
    parser.add_argument("--prompt-field", default=None)
    parser.add_argument("--prompt-format", choices=("raw", "legacy", "chat-template"), default="raw")
    parser.add_argument("--system-prompt", default=None)
    parser.add_argument("--add-special-tokens", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--warmup-samples", type=int, default=0)
    parser.add_argument("--max-new-token-hit-rate-threshold", type=float, default=0.5)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(bool(args.verbose))
    if int(args.exact_match_top_k) < 1:
        raise ValueError("--exact-match-top-k must be at least 1.")
    if int(args.num_shards) < 1:
        raise ValueError("--num-shards must be at least 1.")
    if not 0 <= int(args.shard_index) < int(args.num_shards):
        raise ValueError("--shard-index must satisfy 0 <= shard-index < num-shards.")
    max_records = args.limit if args.limit is not None else args.max_samples
    engine_path = expand_path(args.engine)
    if not engine_path.exists():
        raise FileNotFoundError(f"TensorRT engine does not exist: {engine_path}")
    tokenizer = load_tokenizer(args.model_path, trust_remote_code=bool(args.trust_remote_code))
    records = read_records(args.dataset, limit=max_records)
    if int(args.num_shards) > 1:
        records = [
            {**record, "_global_index": original_index}
            for original_index, record in enumerate(records)
            if original_index % int(args.num_shards) == int(args.shard_index)
        ]
    if not records:
        if int(args.num_shards) > 1:
            empty_summary: dict[str, Any] = {
                "runtime": str(args.runtime),
                "engine_path": str(engine_path),
                "onnx_path": str(args.onnx_path or ""),
                "model_path": str(args.model_path),
                "dataset_file": str(args.dataset),
                "precision": str(args.precision),
                "variant": str(args.variant or args.precision),
                "batch_size": int(args.batch_size),
                "total_examples": 0,
                "shard_index": int(args.shard_index),
                "num_shards": int(args.num_shards),
                "exact_match_accuracy": 0.0,
                "exact_match_correct": 0,
                "exact_match_at_top_k_accuracy": 0.0,
                "exact_match_at_top_k_correct": 0,
                "top_k_exact_match": int(args.exact_match_top_k),
                "comparison_mode": args.comparison_mode,
                "input_length": int(args.max_seq_len),
                "max_new_tokens": int(args.max_new_tokens),
                "max_seq_len": int(args.max_seq_len),
                "reached_max_new_tokens": 0,
                "reached_max_new_tokens_rate": 0.0,
                "avg_latency_ms": 0.0,
                "mean_latency_ms": 0.0,
                "p95_latency_ms": None,
                "queries_per_second": 0.0,
                "tokens_per_sec": 0.0,
                "peak_memory_mb": None,
                "engine_size_mb": file_size_mb(engine_path),
                "onnx_size_mb": file_size_mb(args.onnx_path),
                "model_artifact_size_mb": artifact_size_mb(args.model_path),
                "empty_shard": True,
            }
            if int(args.exact_match_top_k) == 5:
                empty_summary["exact_match_at_5_accuracy"] = 0.0
                empty_summary["exact_match_at_5_correct"] = 0
                empty_summary["top5_exact_match_accuracy"] = 0.0
                empty_summary["top5_exact_match_correct"] = 0
            write_outputs(Path(args.output_dir).expanduser(), empty_summary, [], overwrite=bool(args.overwrite))
            print(f"{empty_summary['variant']} shard {int(args.shard_index)}: empty")
            return
        raise ValueError(f"Dataset produced zero records: {args.dataset}")
    runner = TensorRTEngineRunner(engine_path, verbose=bool(args.verbose))
    cache_inputs = [name for name in runner.input_names if "past_key_values" in name or name.startswith("past_")]
    if cache_inputs:
        raise RuntimeError(f"This evaluator supports no-cache engines only. Cached inputs detected: {cache_inputs}")

    free_start, total_memory = runner.cuda.mem_info()
    min_free = free_start
    try:
        for record in records[: int(args.warmup_samples)]:
            prompt, _reference = prompt_and_reference(record, prompt_field=args.prompt_field)
            formatted = apply_prompt_format(tokenizer, prompt, args.prompt_format, args.system_prompt)
            beam_generate_nocache(
                runner,
                tokenizer,
                formatted,
                max_new_tokens=min(4, int(args.max_new_tokens)),
                max_seq_len=int(args.max_seq_len),
                beam_width=1,
                add_special_tokens=bool(args.add_special_tokens),
                repetition_penalty=float(args.repetition_penalty),
            )

        results: list[dict[str, Any]] = []
        for index, record in enumerate(records):
            global_index = int(record.get("_global_index", index))
            prompt, reference = prompt_and_reference(record, prompt_field=args.prompt_field)
            formatted = apply_prompt_format(tokenizer, prompt, args.prompt_format, args.system_prompt)
            generation = beam_generate_nocache(
                runner,
                tokenizer,
                formatted,
                max_new_tokens=int(args.max_new_tokens),
                max_seq_len=int(args.max_seq_len),
                beam_width=int(args.exact_match_top_k),
                add_special_tokens=bool(args.add_special_tokens),
                repetition_penalty=float(args.repetition_penalty),
            )
            normalized_reference = comparison_text(reference or "", tokenizer, args.comparison_mode) if reference is not None else None
            normalized_prediction = comparison_text(generation["generated_text"], tokenizer, args.comparison_mode)
            exact_match = normalized_reference is not None and normalized_prediction == normalized_reference
            top_k_match_rank = None
            candidates = []
            for candidate in generation["top_k_candidates"]:
                candidate_text = str(candidate["generated_text"])
                candidate_normalized = comparison_text(candidate_text, tokenizer, args.comparison_mode)
                candidate_exact = normalized_reference is not None and candidate_normalized == normalized_reference
                if candidate_exact and top_k_match_rank is None:
                    top_k_match_rank = int(candidate["rank"])
                candidates.append({**candidate, "normalized_text": candidate_normalized, "exact_match": candidate_exact})
            row = {
                "index": global_index,
                "shard_index": int(args.shard_index),
                "num_shards": int(args.num_shards),
                "prompt": prompt,
                "formatted_prompt": formatted,
                "reference": reference,
                "generated_text": generation["generated_text"],
                "normalized_prediction": normalized_prediction,
                "normalized_reference": normalized_reference,
                "exact_match": exact_match,
                "exact_match_at_top_k": top_k_match_rank is not None,
                "top_k_match_rank": top_k_match_rank,
                "top_k": int(args.exact_match_top_k),
                "top_k_candidates": candidates,
                "difficulty": first_present(record, DIFFICULTY_FIELDS),
                **{key: value for key, value in generation.items() if key != "top_k_candidates"},
            }
            results.append(row)
            free_now, _ = runner.cuda.mem_info()
            min_free = min(min_free, free_now)
            logging.info(
                "[%d/%d] EM1=%s EM@%d=%s tokens=%s latency=%.2fms",
                index + 1,
                len(records),
                exact_match,
                int(args.exact_match_top_k),
                row["exact_match_at_top_k"],
                row["generated_tokens"],
                row["total_generation_latency_ms"],
            )
    finally:
        runner.close()

    exact_correct = sum(1 for row in results if bool(row.get("exact_match")))
    top_k_correct = sum(1 for row in results if bool(row.get("exact_match_at_top_k")))
    reached_max = sum(1 for row in results if bool(row.get("reached_max_new_tokens")))
    latency_values = [float(row["total_generation_latency_ms"]) for row in results]
    tps_values = [float(row["tokens_per_sec"]) for row in results]
    total_latency_seconds = sum(latency_values) / 1000.0
    peak_memory_mb = (total_memory - min_free) / 1024**2 if total_memory else None
    summary: dict[str, Any] = {
        "runtime": str(args.runtime),
        "engine_path": str(engine_path),
        "onnx_path": str(args.onnx_path or ""),
        "model_path": str(args.model_path),
        "dataset_file": str(args.dataset),
        "precision": str(args.precision),
        "variant": str(args.variant or args.precision),
        "batch_size": int(args.batch_size),
        "total_examples": len(results),
        "shard_index": int(args.shard_index),
        "num_shards": int(args.num_shards),
        "exact_match_accuracy": exact_correct / float(len(results) or 1),
        "exact_match_correct": exact_correct,
        "exact_match_at_top_k_accuracy": top_k_correct / float(len(results) or 1),
        "exact_match_at_top_k_correct": top_k_correct,
        "top_k_exact_match": int(args.exact_match_top_k),
        "comparison_mode": args.comparison_mode,
        "input_length": int(args.max_seq_len),
        "max_new_tokens": int(args.max_new_tokens),
        "max_seq_len": int(args.max_seq_len),
        "reached_max_new_tokens": reached_max,
        "reached_max_new_tokens_rate": reached_max / float(len(results) or 1),
        "avg_latency_ms": float(statistics.mean(latency_values)) if latency_values else 0.0,
        "mean_latency_ms": float(statistics.mean(latency_values)) if latency_values else 0.0,
        "p95_latency_ms": percentile(latency_values, 0.95),
        "queries_per_second": (len(results) / total_latency_seconds) if total_latency_seconds > 0 else 0.0,
        "tokens_per_sec": float(statistics.mean(tps_values)) if tps_values else 0.0,
        "peak_memory_mb": peak_memory_mb,
        "engine_size_mb": file_size_mb(engine_path),
        "onnx_size_mb": file_size_mb(args.onnx_path),
        "model_artifact_size_mb": artifact_size_mb(args.model_path),
    }
    if int(args.exact_match_top_k) == 5:
        summary["exact_match_at_5_accuracy"] = summary["exact_match_at_top_k_accuracy"]
        summary["exact_match_at_5_correct"] = summary["exact_match_at_top_k_correct"]
        summary["top5_exact_match_accuracy"] = summary["exact_match_at_top_k_accuracy"]
        summary["top5_exact_match_correct"] = summary["exact_match_at_top_k_correct"]
    summary.update(grouped_metrics(results, "difficulty"))

    write_outputs(Path(args.output_dir).expanduser(), summary, results, overwrite=bool(args.overwrite))
    print(
        f"{summary['variant']} {Path(args.dataset).name}: "
        f"EM1={summary['exact_match_accuracy']:.4f} "
        f"EM@{int(args.exact_match_top_k)}={summary['exact_match_at_top_k_accuracy']:.4f}"
    )
    if float(args.max_new_token_hit_rate_threshold) <= float(summary["reached_max_new_tokens_rate"]):
        raise RuntimeError(
            "Generated predictions are frequently hitting max_new_tokens without EOS: "
            f"rate={summary['reached_max_new_tokens_rate']:.4f}, "
            f"threshold={float(args.max_new_token_hit_rate_threshold):.4f}"
        )


if __name__ == "__main__":
    main()
