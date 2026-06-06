from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts import write_revision_sparsity_summary as summary


def native_eval_row(
    family: str,
    phase: str,
    method: str,
    sparsity: float,
    eval_name: str,
) -> dict:
    benchmark_metrics = (
        {
            "count_easy": 10,
            "em1_easy": 0.1,
            "em5_easy": 0.2,
            "count_medium": 20,
            "em1_medium": 0.3,
            "em5_medium": 0.4,
            "count_hard": 30,
            "em1_hard": 0.5,
            "em5_hard": 0.6,
        }
        if eval_name == "benchmark"
        else {}
    )
    return {
        "model_family": family,
        "eval_name": eval_name,
        "eval_file": f"{eval_name}.json",
        "phase": phase,
        "method": method,
        "target_sparsity": sparsity,
        "status": "ok",
        "em1": 0.7 if eval_name == "training_dataset" else 0.8,
        "em5": 0.75 if eval_name == "training_dataset" else 0.85,
        "checkpoint_evaluated": f"checkpoints/{family}/{phase}/{method}/{sparsity}",
        "achieved_whole_model_sparsity": sparsity,
        "achieved_prunable_sparsity": sparsity,
        **benchmark_metrics,
    }


def write_progressive_csv(path: Path, family: str, sparsities: tuple[float, ...] = (0.3, 0.5)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "eval_name",
        "eval_path",
        "pruning_mode",
        "pruning_method",
        "target_sparsity",
        "em1_overall",
        "em5_overall",
        "em1_easy",
        "em5_easy",
        "count_easy",
        "em1_medium",
        "em5_medium",
        "count_medium",
        "em1_hard",
        "em5_hard",
        "count_hard",
        "checkpoint_path",
        "mask_path",
        "whole_model_sparsity_actual",
        "targeted_linear_sparsity_actual",
    ]
    rows = []
    for sparsity in sparsities:
        for eval_name in ("benchmark", "training_dataset"):
            rows.append(
                {
                    "eval_name": eval_name,
                    "eval_path": f"{eval_name}.json",
                    "pruning_mode": "progressive",
                    "pruning_method": "magnitude",
                    "target_sparsity": sparsity,
                    "em1_overall": 0.61 if eval_name == "training_dataset" else 0.51,
                    "em5_overall": 0.62 if eval_name == "training_dataset" else 0.52,
                    "em1_easy": 0.1,
                    "em5_easy": 0.2,
                    "count_easy": 10,
                    "em1_medium": 0.3,
                    "em5_medium": 0.4,
                    "count_medium": 20,
                    "em1_hard": 0.5,
                    "em5_hard": 0.6,
                    "count_hard": 30,
                    "checkpoint_path": f"checkpoints/{family}/progressive/{sparsity}",
                    "mask_path": f"masks/{family}/progressive/{sparsity}.pt",
                    "whole_model_sparsity_actual": sparsity,
                    "targeted_linear_sparsity_actual": sparsity,
                }
            )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_final_revision_matrix_has_eighteen_pruning_rows_plus_two_dense_rows(tmp_path: Path) -> None:
    native_rows = []
    native_methods_by_sparsity = {
        0.3: ("magnitude", "wanda", "taylor"),
        0.5: ("magnitude", "wanda", "taylor", "2of4"),
    }
    for family in ("base_sft", "contrastive_sft"):
        for eval_name in ("training_dataset", "benchmark"):
            native_rows.append(native_eval_row(family, "dense_baseline", "base_model", 0.0, eval_name))
        for sparsity, methods in native_methods_by_sparsity.items():
            for method in methods:
                for eval_name in ("training_dataset", "benchmark"):
                    native_rows.append(native_eval_row(family, "one_shot", method, sparsity, eval_name))

    regular_csv = tmp_path / "regular_sft" / "summary_metrics.csv"
    contrastive_csv = tmp_path / "contrastive_sft" / "summary_metrics.csv"
    write_progressive_csv(regular_csv, "regular_sft")
    write_progressive_csv(contrastive_csv, "contrastive_sft")

    native_payload = {"results": native_rows}
    progressive_rows = summary.progressive_rows(regular_csv, "regular_sft") + summary.progressive_rows(
        contrastive_csv,
        "contrastive_sft",
    )
    matrix_rows = summary.final_matrix_rows(native_payload, progressive_rows)
    counts = summary.final_matrix_counts(matrix_rows)

    assert counts["final_pruning_result_rows"] == 18
    assert counts["final_dense_baseline_rows"] == 2
    assert counts["final_total_rows_with_dense_baselines"] == 20
    assert counts["final_matrix_complete"] is True
    assert counts["final_rows_by_family_and_source"] == {
        "contrastive_sft:dense_baseline": 1,
        "contrastive_sft:native_one_shot": 7,
        "contrastive_sft:progressive_magnitude": 2,
        "regular_sft:dense_baseline": 1,
        "regular_sft:native_one_shot": 7,
        "regular_sft:progressive_magnitude": 2,
    }
    assert {
        row["method"]
        for row in matrix_rows
        if row["source"] == "native_one_shot" and row["target_sparsity"] == 0.5
    } == {"magnitude", "wanda", "gradient", "nvidia24"}

    regular_progressive = next(
        row
        for row in matrix_rows
        if row["trained_checkpoint_family"] == "regular_sft"
        and row["source"] == "progressive_magnitude"
        and row["target_sparsity"] == 0.3
    )
    assert regular_progressive["training_data_em1"] == 0.61
    assert regular_progressive["training_data_em5"] == 0.62
    assert regular_progressive["benchmark_em1"] == 0.51
    assert regular_progressive["benchmark_em5"] == 0.52
    assert regular_progressive["benchmark_hard_em1"] == 0.5
    assert regular_progressive["benchmark_hard_em5"] == 0.6


def test_write_revision_summary_emits_final_matrix(tmp_path: Path, monkeypatch) -> None:
    native_path = tmp_path / "native.json"
    regular_30_csv = tmp_path / "regular_sft" / "sparsity_0p3" / "summary_metrics.csv"
    regular_50_csv = tmp_path / "regular_sft" / "sparsity_0p5" / "summary_metrics.csv"
    contrastive_30_csv = tmp_path / "contrastive_sft" / "sparsity_0p3" / "summary_metrics.csv"
    contrastive_50_csv = tmp_path / "contrastive_sft" / "sparsity_0p5" / "summary_metrics.csv"
    output_path = tmp_path / "revision_summary.json"

    native_rows = []
    for family in ("base_sft", "contrastive_sft"):
        for eval_name in ("training_dataset", "benchmark"):
            native_rows.append(native_eval_row(family, "dense_baseline", "base_model", 0.0, eval_name))
        for sparsity, methods in {0.3: ("magnitude", "wanda", "taylor"), 0.5: ("magnitude", "wanda", "taylor", "2of4")}.items():
            for method in methods:
                for eval_name in ("training_dataset", "benchmark"):
                    native_rows.append(native_eval_row(family, "one_shot", method, sparsity, eval_name))
    native_path.write_text(json.dumps({"results": native_rows}), encoding="utf-8")
    write_progressive_csv(regular_30_csv, "regular_sft", (0.3,))
    write_progressive_csv(regular_50_csv, "regular_sft", (0.5,))
    write_progressive_csv(contrastive_30_csv, "contrastive_sft", (0.3,))
    write_progressive_csv(contrastive_50_csv, "contrastive_sft", (0.5,))

    monkeypatch.setattr(
        "sys.argv",
        [
            "write_revision_sparsity_summary.py",
            "--native-results-json",
            str(native_path),
            "--regular-progressive-summary",
            str(regular_30_csv),
            "--regular-progressive-summary",
            str(regular_50_csv),
            "--contrastive-progressive-summary",
            str(contrastive_30_csv),
            "--contrastive-progressive-summary",
            str(contrastive_50_csv),
            "--output-json",
            str(output_path),
        ],
    )
    summary.main()

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert len(payload["final_matrix_rows"]) == 20
    assert payload["matrix_counts"]["final_pruning_result_rows"] == 18
    assert payload["matrix_counts"]["final_matrix_complete"] is True
    assert payload["execution_plan"]["sft_training"]["distributed"] is True
    assert payload["execution_plan"]["progressive_magnitude"]["distributed"] is False
    assert payload["execution_plan"]["progressive_magnitude"]["parallel_jobs"] is True
