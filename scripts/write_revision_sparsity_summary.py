#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


NATIVE_FAMILY_LABELS = {
    "base_sft": "regular_sft",
    "regular_sft": "regular_sft",
    "contrastive_sft": "contrastive_sft",
}


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
        normalized.setdefault("eval_name", "benchmark")
        if normalized.get("pruning_mode") == "progressive":
            rows.append(normalized)
    return rows


def family_label(value: Any) -> str:
    text = str(value or "")
    return NATIVE_FAMILY_LABELS.get(text, text)


def sparsity_key(value: Any) -> str:
    if value in (None, ""):
        return ""
    try:
        return f"{float(value):.6g}"
    except (TypeError, ValueError):
        return str(value)


def preferred_metric(row: dict[str, Any], *names: str) -> Any:
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            return value
    return ""


def empty_matrix_row(
    *,
    row_type: str,
    family: str,
    source: str,
    phase: str,
    method: str,
    target_sparsity: Any,
) -> dict[str, Any]:
    target = numeric(target_sparsity)
    return {
        "row_type": row_type,
        "trained_checkpoint_family": family,
        "source": source,
        "phase": phase,
        "method": method,
        "target_sparsity": target,
        "target_sparsity_label": f"{float(target):.0%}" if isinstance(target, (int, float)) else "",
        "training_data_em1": "",
        "training_data_em5": "",
        "training_data_status": "",
        "training_data_eval_path": "",
        "benchmark_em1": "",
        "benchmark_em5": "",
        "benchmark_status": "",
        "benchmark_eval_path": "",
        "benchmark_easy_count": "",
        "benchmark_easy_em1": "",
        "benchmark_easy_em5": "",
        "benchmark_medium_count": "",
        "benchmark_medium_em1": "",
        "benchmark_medium_em5": "",
        "benchmark_hard_count": "",
        "benchmark_hard_em1": "",
        "benchmark_hard_em5": "",
        "checkpoint_path": "",
        "mask_path": "",
        "achieved_whole_model_sparsity": "",
        "achieved_prunable_sparsity": "",
    }


def assign_native_eval_metrics(matrix_row: dict[str, Any], eval_name: str, row: dict[str, Any]) -> None:
    if eval_name == "training_dataset":
        matrix_row["training_data_em1"] = preferred_metric(row, "em1")
        matrix_row["training_data_em5"] = preferred_metric(row, "em5")
        matrix_row["training_data_status"] = preferred_metric(row, "status")
        matrix_row["training_data_eval_path"] = preferred_metric(row, "eval_file")
    elif eval_name == "benchmark":
        matrix_row["benchmark_em1"] = preferred_metric(row, "em1")
        matrix_row["benchmark_em5"] = preferred_metric(row, "em5")
        matrix_row["benchmark_status"] = preferred_metric(row, "status")
        matrix_row["benchmark_eval_path"] = preferred_metric(row, "eval_file")
        for level in ("easy", "medium", "hard"):
            matrix_row[f"benchmark_{level}_count"] = preferred_metric(row, f"count_{level}")
            matrix_row[f"benchmark_{level}_em1"] = preferred_metric(row, f"em1_{level}")
            matrix_row[f"benchmark_{level}_em5"] = preferred_metric(row, f"em5_{level}")
    matrix_row["checkpoint_path"] = preferred_metric(row, "checkpoint_evaluated", "checkpoint_path") or matrix_row["checkpoint_path"]
    matrix_row["achieved_whole_model_sparsity"] = (
        preferred_metric(row, "achieved_whole_model_sparsity") or matrix_row["achieved_whole_model_sparsity"]
    )
    matrix_row["achieved_prunable_sparsity"] = (
        preferred_metric(row, "achieved_prunable_sparsity") or matrix_row["achieved_prunable_sparsity"]
    )


def assign_progressive_eval_metrics(matrix_row: dict[str, Any], eval_name: str, row: dict[str, Any]) -> None:
    if eval_name == "training_dataset":
        matrix_row["training_data_em1"] = preferred_metric(row, "em1_overall")
        matrix_row["training_data_em5"] = preferred_metric(row, "em5_overall")
        matrix_row["training_data_status"] = "ok"
        matrix_row["training_data_eval_path"] = preferred_metric(row, "eval_path")
    elif eval_name == "benchmark":
        matrix_row["benchmark_em1"] = preferred_metric(row, "em1_overall")
        matrix_row["benchmark_em5"] = preferred_metric(row, "em5_overall")
        matrix_row["benchmark_status"] = "ok"
        matrix_row["benchmark_eval_path"] = preferred_metric(row, "eval_path")
        for level in ("easy", "medium", "hard"):
            matrix_row[f"benchmark_{level}_count"] = preferred_metric(row, f"count_{level}")
            matrix_row[f"benchmark_{level}_em1"] = preferred_metric(row, f"em1_{level}")
            matrix_row[f"benchmark_{level}_em5"] = preferred_metric(row, f"em5_{level}")
    matrix_row["checkpoint_path"] = preferred_metric(row, "checkpoint_path") or matrix_row["checkpoint_path"]
    matrix_row["mask_path"] = preferred_metric(row, "mask_path") or matrix_row["mask_path"]
    matrix_row["achieved_whole_model_sparsity"] = (
        preferred_metric(row, "whole_model_sparsity_actual") or matrix_row["achieved_whole_model_sparsity"]
    )
    matrix_row["achieved_prunable_sparsity"] = (
        preferred_metric(row, "targeted_linear_sparsity_actual") or matrix_row["achieved_prunable_sparsity"]
    )


def final_matrix_rows(native_payload: dict[str, Any], progressive: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    for row in native_payload.get("results", []) or []:
        family = family_label(row.get("model_family"))
        if family not in {"regular_sft", "contrastive_sft"}:
            continue
        phase = str(row.get("phase", ""))
        if phase not in {"dense_baseline", "one_shot"}:
            continue
        method = "base_model" if phase == "dense_baseline" else str(row.get("method", ""))
        target_sparsity = 0.0 if phase == "dense_baseline" else row.get("target_sparsity", "")
        source = "dense_baseline" if phase == "dense_baseline" else "native_one_shot"
        row_type = "dense_baseline" if phase == "dense_baseline" else "pruning"
        key = (family, source, phase, method, sparsity_key(target_sparsity))
        if key not in grouped:
            grouped[key] = empty_matrix_row(
                row_type=row_type,
                family=family,
                source=source,
                phase=phase,
                method=method,
                target_sparsity=target_sparsity,
            )
        assign_native_eval_metrics(grouped[key], str(row.get("eval_name", "")), row)

    for row in progressive:
        family = family_label(row.get("trained_checkpoint_family"))
        if family not in {"regular_sft", "contrastive_sft"}:
            continue
        method = str(row.get("pruning_method", "gradient") or "gradient")
        target_sparsity = row.get("target_sparsity", "")
        key = (family, "progressive_gradient", "progressive", method, sparsity_key(target_sparsity))
        if key not in grouped:
            grouped[key] = empty_matrix_row(
                row_type="pruning",
                family=family,
                source="progressive_gradient",
                phase="progressive",
                method=method,
                target_sparsity=target_sparsity,
            )
        assign_progressive_eval_metrics(grouped[key], str(row.get("eval_name", "benchmark")), row)

    return sorted(
        grouped.values(),
        key=lambda row: (
            row["trained_checkpoint_family"],
            0 if row["row_type"] == "dense_baseline" else 1,
            {"native_one_shot": 0, "progressive_gradient": 1, "dense_baseline": -1}.get(str(row["source"]), 9),
            float(row["target_sparsity"]) if isinstance(row["target_sparsity"], (int, float)) else 0.0,
            str(row["method"]),
        ),
    )


def final_matrix_counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    pruning_rows = [row for row in rows if row.get("row_type") == "pruning"]
    dense_rows = [row for row in rows if row.get("row_type") == "dense_baseline"]
    by_source: dict[str, int] = {}
    by_family_source: dict[str, int] = {}
    for row in rows:
        source = str(row.get("source", ""))
        family = str(row.get("trained_checkpoint_family", ""))
        by_source[source] = by_source.get(source, 0) + 1
        key = f"{family}:{source}"
        by_family_source[key] = by_family_source.get(key, 0) + 1
    expected = {
        "regular_sft:native_one_shot": 7,
        "contrastive_sft:native_one_shot": 7,
        "regular_sft:progressive_gradient": 2,
        "contrastive_sft:progressive_gradient": 2,
        "regular_sft:dense_baseline": 1,
        "contrastive_sft:dense_baseline": 1,
    }
    missing_or_mismatched = {
        key: {"expected": expected_count, "actual": by_family_source.get(key, 0)}
        for key, expected_count in expected.items()
        if by_family_source.get(key, 0) != expected_count
    }
    return {
        "final_pruning_result_rows": len(pruning_rows),
        "final_dense_baseline_rows": len(dense_rows),
        "final_total_rows_with_dense_baselines": len(rows),
        "final_rows_by_source": by_source,
        "final_rows_by_family_and_source": by_family_source,
        "expected_final_counts": expected,
        "missing_or_mismatched_final_counts": missing_or_mismatched,
        "final_matrix_complete": not missing_or_mismatched
        and len(pruning_rows) == 18
        and len(dense_rows) == 2
        and len(rows) == 20,
    }


def native_result_counts(native_payload: dict[str, Any]) -> dict[str, Any]:
    rows = native_payload.get("results", []) or []
    eval_rows_by_family: dict[str, int] = {}
    eval_rows_by_family_and_sparsity: dict[str, int] = {}
    unique_outputs_by_family: dict[str, set[tuple[str, str]]] = {}
    unique_outputs_by_family_and_sparsity: dict[str, set[str]] = {}
    for row in rows:
        if row.get("phase") != "one_shot":
            continue
        family = family_label(row.get("model_family"))
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


def progressive_result_counts(progressive: list[dict[str, Any]]) -> dict[str, Any]:
    eval_rows_by_family: dict[str, int] = {}
    unique_outputs_by_family: dict[str, set[tuple[str, str]]] = {}
    for row in progressive:
        family = family_label(row.get("trained_checkpoint_family"))
        method = str(row.get("pruning_method", ""))
        sparsity = sparsity_key(row.get("target_sparsity", ""))
        eval_rows_by_family[family] = eval_rows_by_family.get(family, 0) + 1
        unique_outputs_by_family.setdefault(family, set()).add((method, sparsity))
    return {
        "progressive_unique_outputs_by_family": {
            family: len(outputs) for family, outputs in unique_outputs_by_family.items()
        },
        "progressive_eval_rows_by_family": eval_rows_by_family,
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
    final_rows = final_matrix_rows(native_payload, progressive)

    payload = {
        "schema_version": 1,
        "native_results_json": str(native_path),
        "progressive_summary_csvs": {
            "regular_sft": str(regular_path),
            "contrastive_sft": str(contrastive_path),
        },
        "matrix_counts": {
            **native_result_counts(native_payload),
            **progressive_result_counts(progressive),
            **final_matrix_counts(final_rows),
        },
        "final_matrix_rows": final_rows,
        "native_results": native_payload.get("results", []),
        "progressive_results": progressive,
        "notes": [
            "Native one-shot rows evaluate trained regular SFT and contrastive SFT checkpoints on training_dataset and benchmark eval files.",
            "Progressive rows evaluate benchmark plus training_dataset when launched through run_linear_sparsity_revision_from_base.sh.",
            "final_matrix_rows fuses eval splits so the expected design is 18 pruning rows plus 2 dense baselines.",
        ],
    }
    write_json(output_path, payload)
    print(f"Wrote revision sparsity summary: {output_path}", flush=True)


if __name__ == "__main__":
    main()
