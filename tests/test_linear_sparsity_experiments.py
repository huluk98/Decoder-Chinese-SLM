from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from chatlm_decoder.pruning import apply_masks, layerwise_magnitude_masks, mask_sparsity, named_prunable_linears  # noqa: E402
from chatlm_decoder.sparsity_experiments import (  # noqa: E402
    add_retention_metrics,
    attach_external_difficulty,
    exact_match_flags,
    load_benchmark_samples,
    prediction_result_rows,
    summarize_prediction_rows,
    write_csv_rows,
)


class LinearScopeToy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(8, 4)
        self.block = torch.nn.Module()
        self.block.linear = torch.nn.Linear(10, 10, bias=True)
        self.norm = torch.nn.LayerNorm(4)
        self.lm_head = torch.nn.Linear(4, 8, bias=False)
        self.classifier = torch.nn.Linear(4, 2, bias=False)
        self.final_response_projection = torch.nn.Linear(4, 4, bias=False)


def fill_ranked_weights(model: torch.nn.Module) -> None:
    with torch.no_grad():
        for index, parameter in enumerate(model.parameters(), start=1):
            parameter.copy_(torch.arange(1, parameter.numel() + 1, dtype=parameter.dtype).reshape_as(parameter) * index)


def test_linear_scope_excludes_heads_embeddings_norms_and_biases() -> None:
    model = LinearScopeToy()

    names = [name for name, _module in named_prunable_linears(model)]

    assert names == ["block.linear"]
    assert all("embed" not in name for name in names)
    assert all("norm" not in name for name in names)
    assert all("lm_head" not in name for name in names)
    assert all("classifier" not in name for name in names)
    assert all("response_projection" not in name for name in names)


def test_magnitude_pruning_hits_30_and_50_percent_targets() -> None:
    model = LinearScopeToy()
    fill_ranked_weights(model)

    masks_30 = layerwise_magnitude_masks(model, sparsity=0.30)
    masks_50 = layerwise_magnitude_masks(model, sparsity=0.50)

    assert mask_sparsity(masks_30) == pytest.approx(0.30)
    assert mask_sparsity(masks_50) == pytest.approx(0.50)


def test_mask_enforcement_keeps_pruned_weights_zero_after_optimizer_step() -> None:
    model = LinearScopeToy()
    fill_ranked_weights(model)
    masks = layerwise_magnitude_masks(model, sparsity=0.50)
    apply_masks(model, masks)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    loss = model.block.linear.weight.sum()
    loss.backward()
    optimizer.step()
    apply_masks(model, masks)

    mask = masks["block.linear"].bool()
    pruned_values = model.block.linear.weight.detach().masked_select(~mask)
    assert torch.count_nonzero(pruned_values).item() == 0


def test_em1_and_em5_on_synthetic_candidates() -> None:
    em1, em5 = exact_match_flags("turn on light", ["wrong", "dim light", "turn on light"], normalization_mode="normalized")

    assert not em1
    assert em5


def test_difficulty_join_by_id_and_input(tmp_path: Path) -> None:
    samples = [
        {"sample_id": "a1", "input": "Turn on the light", "target": "ok", "difficulty": "", "raw_record": {}},
        {"sample_id": "2", "input": "It is too dark", "target": "ok", "difficulty": "", "raw_record": {}},
    ]
    by_id = tmp_path / "by_id.csv"
    by_id.write_text("id,difficulty\na1,easy\n2,medium\n", encoding="utf-8")
    joined = attach_external_difficulty(samples, by_id)
    assert [row["difficulty"] for row in joined] == ["easy", "medium"]

    by_input = tmp_path / "by_input.csv"
    by_input.write_text("input,difficulty\nTurn on the light,easy\nIt is too dark,hard\n", encoding="utf-8")
    joined = attach_external_difficulty(samples, by_input)
    assert [row["difficulty"] for row in joined] == ["easy", "hard"]


def test_benchmark_loader_requires_external_difficulty_when_missing(tmp_path: Path) -> None:
    benchmark = tmp_path / "benchmark.json"
    benchmark.write_text(json.dumps([{"id": "1", "input": "x", "target": "y"}]), encoding="utf-8")

    with pytest.raises(ValueError, match="benchmark_difficulty_path"):
        load_benchmark_samples(benchmark)


def test_summary_csv_columns_include_difficulty_counts(tmp_path: Path) -> None:
    samples = [
        {"sample_id": "1", "input": "x", "target": "a", "difficulty": "easy"},
        {"sample_id": "2", "input": "y", "target": "b", "difficulty": "medium"},
        {"sample_id": "3", "input": "z", "target": "c", "difficulty": "hard"},
    ]
    prediction_rows = prediction_result_rows(
        samples,
        [["a"], ["wrong", "b"], ["wrong"]],
        normalization_mode="normalized",
        model_family="decoder_only",
        pruning_mode="dense",
        pruning_method="magnitude",
        target_sparsity=0.0,
        targeted_linear_sparsity_actual=0.0,
        whole_model_sparsity_actual=0.0,
        seed=42,
    )
    summary = summarize_prediction_rows(prediction_rows, bootstrap_resamples=10, seed=42)
    row = add_retention_metrics(summary, summary)
    output = tmp_path / "summary.csv"
    write_csv_rows(output, [row])

    with output.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        csv_rows = list(reader)

    assert csv_rows[0]["count_easy"] == "1"
    assert csv_rows[0]["count_medium"] == "1"
    assert csv_rows[0]["count_hard"] == "1"
    assert csv_rows[0]["em1_retention_overall"] == "1.0"
