#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
from pathlib import Path
from typing import Any


DIFFICULTY_FIELDS = ("difficulty",)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number} must be a JSON object.")
            rows.append(payload)
    return rows


def ensure_output_path(path: Path, overwrite: bool, kind: str) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing {kind}: {path}. Pass --overwrite.")
    path.parent.mkdir(parents=True, exist_ok=True)


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


def first_present(rows: list[dict[str, Any]], key: str, default: Any = "") -> Any:
    for row in rows:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return default


def max_numeric(rows: list[dict[str, Any]], key: str) -> float | None:
    values = []
    for row in rows:
        value = row.get(key)
        if value in (None, ""):
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            values.append(number)
    return max(values) if values else None


def sum_numeric(rows: list[dict[str, Any]], key: str) -> float | None:
    values = []
    for row in rows:
        value = row.get(key)
        if value in (None, ""):
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            values.append(number)
    return sum(values) if values else None


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
        "shard_index",
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


def load_shards(shard_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summary_paths = sorted(shard_root.rglob("prompt_response_eval_summary.json"))
    if not summary_paths:
        raise FileNotFoundError(f"No shard summaries found under {shard_root}")
    summaries: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    for summary_path in summary_paths:
        shard_dir = summary_path.parent
        predictions_path = shard_dir / "prompt_response_eval_predictions.jsonl"
        if not predictions_path.exists():
            raise FileNotFoundError(f"Missing shard predictions: {predictions_path}")
        summary = read_json(summary_path)
        summaries.append(summary)
        results.extend(read_jsonl(predictions_path))
    results.sort(key=lambda row: int(row.get("index", 0)))
    return summaries, results


def aggregate_summary(summaries: list[dict[str, Any]], results: list[dict[str, Any]], output_dir: Path) -> dict[str, Any]:
    exact_correct = sum(1 for row in results if bool(row.get("exact_match")))
    top_k_correct = sum(1 for row in results if bool(row.get("exact_match_at_top_k")))
    reached_max = sum(1 for row in results if bool(row.get("reached_max_new_tokens")))
    latency_values = [
        float(row["total_generation_latency_ms"])
        for row in results
        if row.get("total_generation_latency_ms") not in (None, "")
    ]
    tps_values = [float(row["tokens_per_sec"]) for row in results if row.get("tokens_per_sec") not in (None, "")]
    total_latency_seconds = sum(latency_values) / 1000.0
    top_k = int(first_present(summaries, "top_k_exact_match", first_present(results, "top_k", 5)) or 5)
    engine_path = str(first_present(summaries, "engine_path"))
    onnx_path = str(first_present(summaries, "onnx_path"))
    model_path = str(first_present(summaries, "model_path"))
    peak_memory_mb = max_numeric(summaries, "peak_memory_mb")
    aggregate_memory_mb = sum_numeric(summaries, "peak_memory_mb")

    summary: dict[str, Any] = {
        "runtime": str(first_present(summaries, "runtime", "ONNX/TensorRT")),
        "engine_path": engine_path,
        "onnx_path": onnx_path,
        "model_path": model_path,
        "dataset_file": str(first_present(summaries, "dataset_file")),
        "precision": str(first_present(summaries, "precision")),
        "variant": str(first_present(summaries, "variant")),
        "batch_size": int(first_present(summaries, "batch_size", 1) or 1),
        "total_examples": len(results),
        "exact_match_accuracy": exact_correct / float(len(results) or 1),
        "exact_match_correct": exact_correct,
        "exact_match_at_top_k_accuracy": top_k_correct / float(len(results) or 1),
        "exact_match_at_top_k_correct": top_k_correct,
        "top_k_exact_match": top_k,
        "comparison_mode": str(first_present(summaries, "comparison_mode")),
        "input_length": int(first_present(summaries, "input_length", first_present(summaries, "max_seq_len", 0)) or 0),
        "max_new_tokens": int(first_present(summaries, "max_new_tokens", 0) or 0),
        "max_seq_len": int(first_present(summaries, "max_seq_len", 0) or 0),
        "reached_max_new_tokens": reached_max,
        "reached_max_new_tokens_rate": reached_max / float(len(results) or 1),
        "avg_latency_ms": float(statistics.mean(latency_values)) if latency_values else 0.0,
        "mean_latency_ms": float(statistics.mean(latency_values)) if latency_values else 0.0,
        "p95_latency_ms": percentile(latency_values, 0.95),
        "queries_per_second": (len(results) / total_latency_seconds) if total_latency_seconds > 0 else 0.0,
        "tokens_per_sec": float(statistics.mean(tps_values)) if tps_values else 0.0,
        "peak_memory_mb": peak_memory_mb,
        "aggregate_peak_memory_mb": aggregate_memory_mb,
        "engine_size_mb": file_size_mb(engine_path),
        "onnx_size_mb": file_size_mb(onnx_path),
        "model_artifact_size_mb": artifact_size_mb(model_path),
        "sharded_eval": True,
        "num_shards": len(summaries),
        "shard_summary_count": len(summaries),
        "output_dir": str(output_dir),
    }
    if top_k == 5:
        summary["exact_match_at_5_accuracy"] = summary["exact_match_at_top_k_accuracy"]
        summary["exact_match_at_5_correct"] = summary["exact_match_at_top_k_correct"]
        summary["top5_exact_match_accuracy"] = summary["exact_match_at_top_k_accuracy"]
        summary["top5_exact_match_correct"] = summary["exact_match_at_top_k_correct"]
    for field in DIFFICULTY_FIELDS:
        summary.update(grouped_metrics(results, field))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge sharded TensorRT prompt-response eval outputs.")
    parser.add_argument("--shard-root", required=True, help="Directory containing per-shard eval output directories.")
    parser.add_argument("--output-dir", required=True, help="Directory for merged prompt_response_eval_summary.json.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    shard_root = Path(args.shard_root).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    summaries, results = load_shards(shard_root)
    summary = aggregate_summary(summaries, results, output_dir)
    write_outputs(output_dir, summary, results, overwrite=bool(args.overwrite))
    print(
        f"Merged {len(summaries)} shards / {len(results)} examples: "
        f"EM1={summary['exact_match_accuracy']:.4f} "
        f"EM@{int(summary['top_k_exact_match'])}={summary['exact_match_at_top_k_accuracy']:.4f}"
    )


if __name__ == "__main__":
    main()
