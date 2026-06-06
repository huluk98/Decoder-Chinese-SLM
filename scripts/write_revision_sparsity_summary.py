#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def numeric(value: Any) -> Any:
    if value in (None, ""):
        return value
    try:
        if "." not in str(value) and "e" not in str(value).lower():
            return int(value)
        return float(value)
    except (TypeError, ValueError):
        return value


def normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    return {key: numeric(value) for key, value in row.items()}


def progressive_rows(path: Path, label: str) -> list[dict[str, Any]]:
    rows = []
    for row in read_csv_rows(path):
        normalized = normalize_row(row)
        normalized["trained_checkpoint_family"] = label
        if normalized.get("pruning_mode") == "progressive":
            rows.append(normalized)
    return rows


def native_result_counts(native_payload: dict[str, Any]) -> dict[str, Any]:
    rows = native_payload.get("results", []) or []
    eval_rows_by_family: dict[str, int] = {}
    eval_rows_by_family_and_sparsity: dict[str, int] = {}
    unique_outputs_by_family: dict[str, set[tuple[str, str]]] = {}
    unique_outputs_by_family_and_sparsity: dict[str, set[str]] = {}
    for row in rows:
        if row.get("phase") != "one_shot":
            continue
        family = str(row.get("model_family", ""))
        sparsity = str(row.get("target_sparsity", ""))
        method = str(row.get("method", ""))
        eval_rows_by_family[family] = eval_rows_by_family.get(family, 0) + 1
        key = f"{family}@{sparsity}"
        eval_rows_by_family_and_sparsity[key] = eval_rows_by_family_and_sparsity.get(key, 0) + 1
        unique_outputs_by_family.setdefault(family, set()).add((method, sparsity))
        unique_outputs_by_family_and_sparsity.setdefault(key, set()).add(method)
    return {
        "native_unique_outputs_by_family": {family: len(outputs) for family, outputs in unique_outputs_by_family.items()},
        "native_unique_outputs_by_family_and_sparsity": {
            family_sparsity: len(methods) for family_sparsity, methods in unique_outputs_by_family_and_sparsity.items()
        },
        "one_shot_eval_rows_by_family": eval_rows_by_family,
        "one_shot_eval_rows_by_family_and_sparsity": eval_rows_by_family_and_sparsity,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine native and progressive SCENIC sparsity revision summaries.")
    parser.add_argument("--native-results-json", required=True)
    parser.add_argument("--regular-progressive-summary", required=True)
    parser.add_argument("--contrastive-progressive-summary", required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    native_path = Path(args.native_results_json).expanduser()
    regular_path = Path(args.regular_progressive_summary).expanduser()
    contrastive_path = Path(args.contrastive_progressive_summary).expanduser()
    output_path = Path(args.output_json).expanduser()

    native_payload = read_json(native_path)
    regular_progressive = progressive_rows(regular_path, "regular_sft")
    contrastive_progressive = progressive_rows(contrastive_path, "contrastive_sft")
    progressive = regular_progressive + contrastive_progressive

    payload = {
        "schema_version": 1,
        "native_results_json": str(native_path),
        "progressive_summary_csvs": {
            "regular_sft": str(regular_path),
            "contrastive_sft": str(contrastive_path),
        },
        "matrix_counts": {
            **native_result_counts(native_payload),
            "progressive_rows_by_family": {
                "regular_sft": len(regular_progressive),
                "contrastive_sft": len(contrastive_progressive),
            },
        },
        "native_results": native_payload.get("results", []),
        "progressive_results": progressive,
        "notes": [
            "Native one-shot rows evaluate trained regular SFT and contrastive SFT checkpoints on training_dataset and benchmark eval files.",
            "Progressive rows are benchmark-only rows from scripts/run_sparsity_experiments.py and include easy/medium/hard EM@1/EM@5 fields.",
        ],
    }
    write_json(output_path, payload)
    print(f"Wrote revision sparsity summary: {output_path}", flush=True)


if __name__ == "__main__":
    main()
