from __future__ import annotations

import csv
import json
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

sys.modules.setdefault(
    "yaml",
    types.SimpleNamespace(
        safe_load=lambda handle: {},
        safe_dump=lambda payload, handle, **kwargs: handle.write(json.dumps(payload)),
    ),
)

from scripts import run_pruning_benchmark as benchmark  # noqa: E402


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def make_eval_result(parent: Path, checkpoint: Path, summary: dict | None = None) -> Path:
    child = parent / "619_Luke_clean_plus_tv_natural_language__magnitude_checkpoint_seed42_20260522T062446Z"
    write_json(
        child / "run_config.json",
        {
            "model_path": str(checkpoint),
            "checkpoint_path_used_for_evaluation": str(checkpoint),
            "dataset_file": "data/benchmark.json",
            "max_length": 128,
            "generation": {"max_new_tokens": 64, "num_beams": 1, "temperature": None},
            "tokenizer": {"pad_token_id": 0, "eos_token_id": 3},
            "eval_args": {"comparison_mode": "whitespace"},
        },
    )
    write_json(
        child / "prompt_response_eval_summary.json",
        summary
        or {
            "exact_match_accuracy_mean": 0.0,
            "correct_examples": 0,
            "total_examples": 4772,
            "mean_response_loss_mean": 5.737810,
            "response_perplexity_mean": 310.383830,
            "avg_generated_tokens_mean": 64.0,
        },
    )
    write_json(parent / "prompt_response_eval_summary.json", {"mirrored": True})
    (parent / "latest_eval_dir.txt").write_text(str(child), encoding="utf-8")
    return child


def make_pruning_report(path: Path) -> None:
    stats = {
        "target_prunable_sparsity": 0.5,
        "achieved_prunable_sparsity": 0.5,
        "achieved_whole_model_sparsity": 0.38512769683412146,
        "active_prunable_parameters": 75497472,
        "pruned_prunable_parameters": 75497472,
        "total_prunable_parameters": 150994944,
        "prunable_parameter_count": 150994944,
        "protected_parameter_count": 45039360,
        "total_parameter_count": 196034304,
        "nonzero_parameters": 120536064,
        "zero_parameters": 75498240,
        "model_zero_fraction": 0.38512769683412146,
    }
    write_json(path, {"checkpoint_reload_validation": stats, **stats})


def test_eval_parent_resolves_timestamp_child(tmp_path: Path) -> None:
    checkpoint = tmp_path / "one_shot" / "magnitude"
    parent = tmp_path / "benchmarks" / "one_shot" / "magnitude"
    child = make_eval_result(parent, checkpoint)

    assert benchmark.resolve_eval_result_dir(parent) == child
    assert benchmark.read_eval_run_config(parent)["checkpoint_path_used_for_evaluation"] == str(checkpoint)
    assert benchmark.read_eval_summary(parent)["exact_match_accuracy_mean"] == 0.0


def test_metric_falls_back_to_per_run_summaries() -> None:
    summary = {
        "per_run_summaries": [
            {"exact_match_accuracy": 0.0},
            {"exact_match_accuracy": 0.039396},
        ]
    }

    assert benchmark.metric(summary, "exact_match_accuracy") == pytest.approx(0.019698)


def test_summary_row_uses_real_pruning_stats(tmp_path: Path) -> None:
    checkpoint = tmp_path / "one_shot" / "magnitude"
    eval_parent = tmp_path / "benchmarks" / "one_shot" / "magnitude"
    make_eval_result(eval_parent, checkpoint)
    report_path = checkpoint / "pruning_report.json"
    make_pruning_report(report_path)

    row = benchmark.summary_row(
        "magnitude",
        "one_shot",
        checkpoint,
        eval_parent,
        "ok",
        pruning_report_path=report_path,
    )

    assert row["active_model_parameters"] == 120536064
    assert row["active_prunable_parameters"] == 75497472
    assert row["real_sparsity"] == pytest.approx(0.38512769683412146)
    assert row["exact_match_accuracy"] == 0.0
    assert row["checkpoint_evaluated"] == str(checkpoint)
    assert row["checkpoint_evaluated"] != str(tmp_path)


def test_write_summary_has_eight_pruning_rows_with_missing_retuned(tmp_path: Path) -> None:
    rows: list[dict] = []
    for method in benchmark.METHODS:
        slug = benchmark.method_slug(method)
        checkpoint = tmp_path / "one_shot" / slug
        eval_parent = tmp_path / "benchmarks" / "one_shot" / slug
        make_eval_result(eval_parent, checkpoint)
        report_path = checkpoint / "pruning_report.json"
        make_pruning_report(report_path)
        rows.append(
            benchmark.summary_row(
                method,
                "one_shot",
                checkpoint,
                eval_parent,
                "ok",
                pruning_report_path=report_path,
            )
        )

    benchmark.ensure_expected_pruning_rows(
        rows=rows,
        methods=list(benchmark.METHODS),
        output_dir=tmp_path,
        retune_enabled=True,
        dense_exact_match=0.923722,
        cmc_comparable=True,
        comparability_issues=[],
    )
    benchmark.write_summary(tmp_path, rows)

    with (tmp_path / "pruning_benchmark_summary.csv").open("r", encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))

    assert len(csv_rows) == 8
    assert {(row["method"], row["phase"]) for row in csv_rows} == {
        (method, phase) for method in benchmark.METHODS for phase in ("one_shot", "retuned")
    }
    assert all(row["exact_match_accuracy"] != "" for row in csv_rows if row["phase"] == "one_shot")
    assert all(row["status"] == "missing" for row in csv_rows if row["phase"] == "retuned")


def test_whole_model_target_resolves_to_higher_prunable_sparsity() -> None:
    torch = pytest.importorskip("torch")
    from chatlm_decoder.pruning import (
        apply_masks,
        global_magnitude_masks,
        resolve_prunable_sparsity_for_target,
        sparsity_accounting,
    )

    class TinyDecoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed_tokens = torch.nn.Embedding(2, 4)
            self.block = torch.nn.Module()
            self.block.self_attn = torch.nn.Module()
            self.block.self_attn.q_proj = torch.nn.Linear(4, 4, bias=False)
            self.block.self_attn.v_proj = torch.nn.Linear(4, 4, bias=False)
            self.ln_f = torch.nn.LayerNorm(4)
            self.lm_head = torch.nn.Linear(4, 2, bias=False)

    model = TinyDecoder()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(1.0)

    resolution = resolve_prunable_sparsity_for_target(
        model,
        target_sparsity=0.5,
        denominator="whole_model",
    )
    assert resolution["target_sparsity_denominator"] == "whole_model"
    assert resolution["target_prunable_sparsity"] == pytest.approx(28 / 32)

    masks = global_magnitude_masks(model, sparsity=resolution["target_prunable_sparsity"])
    apply_masks(model, masks)
    accounting = sparsity_accounting(model, masks, target=resolution["target_prunable_sparsity"])

    assert accounting["achieved_prunable_sparsity"] == pytest.approx(28 / 32)
    assert accounting["achieved_whole_model_sparsity"] == pytest.approx(0.5)
