from __future__ import annotations

import sys
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from chatlm_decoder.pruning import (  # noqa: E402
    exact_rowwise_score_masks,
    layerwise_gradient_score_masks,
    layerwise_magnitude_masks,
    mask_sparsity,
    named_prunable_linears,
    two_of_four_masks,
    validate_two_of_four_masks,
)


class MixedLinearDecoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block = torch.nn.Module()
        self.block.good = torch.nn.Linear(4, 2, bias=False)
        self.block.other = torch.nn.Linear(6, 2, bias=False)
        self.lm_head = torch.nn.Linear(4, 2, bias=False)


def fill_weights(model: torch.nn.Module) -> None:
    with torch.no_grad():
        for index, parameter in enumerate(model.parameters(), start=1):
            parameter.copy_(torch.arange(1, parameter.numel() + 1, dtype=parameter.dtype).reshape_as(parameter) * index)


def test_default_prunable_linears_include_decoder_linears_but_skip_lm_head() -> None:
    model = MixedLinearDecoder()

    names = [name for name, _module in named_prunable_linears(model)]

    assert names == ["block.good", "block.other"]
    assert [name for name, _module in named_prunable_linears(model, include_lm_head=True)] == [
        "block.good",
        "block.other",
        "lm_head",
    ]


def test_magnitude_prunes_50_percent_per_linear_layer() -> None:
    model = MixedLinearDecoder()
    fill_weights(model)

    masks = layerwise_magnitude_masks(model, sparsity=0.5)

    assert set(masks) == {"block.good", "block.other"}
    assert all(mask_sparsity({name: mask}) == pytest.approx(0.5) for name, mask in masks.items())


def test_gradient_taylor_prunes_50_percent_per_linear_layer() -> None:
    model = MixedLinearDecoder()
    fill_weights(model)
    scores = {
        name: module.weight.detach().abs() * (index + 1)
        for index, (name, module) in enumerate(named_prunable_linears(model))
    }

    masks = layerwise_gradient_score_masks(model, scores, sparsity=0.5)

    assert all(mask_sparsity({name: mask}) == pytest.approx(0.5) for name, mask in masks.items())


def test_wanda_rowwise_masks_prune_50_percent_per_output_row() -> None:
    scores = {
        "block.good": torch.tensor(
            [
                [1.0, 4.0, 3.0, 2.0],
                [8.0, 5.0, 6.0, 7.0],
            ]
        )
    }

    masks = exact_rowwise_score_masks(scores, sparsity=0.5)
    row_kept = masks["block.good"].sum(dim=1)

    assert row_kept.tolist() == [2, 2]
    assert mask_sparsity(masks) == pytest.approx(0.5)


def test_two_of_four_prunes_eligible_layers_and_skips_non_divisible_layers() -> None:
    model = MixedLinearDecoder()
    fill_weights(model)

    masks = two_of_four_masks(model)
    validation = validate_two_of_four_masks(masks, model=model)

    assert set(masks) == {"block.good"}
    assert validation["valid"]
    assert validation["achieved_2of4_sparsity"] == pytest.approx(0.5)
    assert validation["skipped_modules"] == [
        {
            "module": "block.other",
            "shape": [2, 6],
            "reason": "in_features_not_divisible_by_4",
        }
    ]
