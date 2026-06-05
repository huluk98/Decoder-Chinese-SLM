#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


FIELDS = [
    "variant",
    "runtime",
    "precision",
    "dataset_name",
    "dataset_file",
    "batch_size",
    "input_length",
    "max_seq_len",
    "engine_path",
    "onnx_path",
    "engine_size_mb",
    "onnx_size_mb",
    "model_artifact_size_mb",
    "exact_match_accuracy",
    "exact_match_correct",
    "exact_match_at_5_accuracy",
    "exact_match_at_5_correct",
    "exact_match_at_top_k_accuracy",
    "exact_match_at_top_k_correct",
    "total_examples",
    "mean_latency_ms",
    "avg_latency_ms",
    "p95_latency_ms",
    "queries_per_second",
    "reached_max_new_tokens_rate",
    "tokens_per_sec",
    "peak_memory_mb",
]


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def dataset_name_from_path(summary_path: Path) -> str:
    # Expected layout: <run_root>/eval/<model>/<precision>/<dataset>/prompt_response_eval_summary.json
    parts = summary_path.parts
    if len(parts) >= 2:
        return parts[-2]
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect edge prompt-response eval summaries into CSV/JSON.")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    run_root = Path(args.run_root).expanduser()
    summary_paths = sorted((run_root / "eval").rglob("prompt_response_eval_summary.json"))
    rows: list[dict[str, Any]] = []
    for path in summary_paths:
        summary = read_json(path)
        row = {field: summary.get(field, "") for field in FIELDS}
        row["dataset_name"] = dataset_name_from_path(path)
        rows.append(row)

    output_csv = Path(args.output_csv).expanduser() if args.output_csv else run_root / "nvidia_onnx_edge_em_summary.csv"
    output_json = Path(args.output_json).expanduser() if args.output_json else run_root / "nvidia_onnx_edge_em_summary.json"
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDS})
    output_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} summary rows: {output_csv}")
    print(f"Wrote summary JSON: {output_json}")


if __name__ == "__main__":
    main()
