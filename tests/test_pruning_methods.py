from __future__ import annotations

import sys
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from chatlm_decoder.pruning import (  # noqa: E402
    apply_masks,
    assert_protected_parameters_unchanged,
    exact_rowwise_score_masks,
    layerwise_gradient_score_masks,
    layerwise_magnitude_masks,
    masked_weight_stats,
    mask_sparsity,
    named_prunable_linears,
    protected_parameter_snapshot,
    two_of_four_masks,
    validate_two_of_four_masks,
    wanda_masks,
)


class MixedLinearDecoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block = torch.nn.Module()
        self.block.good = torch.nn.Linear(4, 2, bias=False)
        self.block.other = torch.nn.Linear(6, 2, bias=False)
        self.lm_head = torch.nn.Linear(4, 2, bias=False)


class TinyLlamaLikeDecoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = torch.nn.Module()
        self.model.embed_tokens = torch.nn.Embedding(32, 16)
        self.model.layers = torch.nn.ModuleList([self._block() for _ in range(2)])
        self.model.norm = torch.nn.LayerNorm(16)
        self.lm_head = torch.nn.Linear(16, 32, bias=False)

    @staticmethod
    def _block() -> torch.nn.Module:
        block = torch.nn.Module()
        block.self_attn = torch.nn.Module()
        block.self_attn.q_proj = torch.nn.Linear(16, 16, bias=False)
        block.self_attn.k_proj = torch.nn.Linear(16, 8, bias=False)
        block.self_attn.v_proj = torch.nn.Linear(16, 8, bias=False)
        block.self_attn.o_proj = torch.nn.Linear(16, 16, bias=False)
        block.mlp = torch.nn.Module()
        block.mlp.gate_proj = torch.nn.Linear(16, 32, bias=False)
        block.mlp.up_proj = torch.nn.Linear(16, 32, bias=False)
        block.mlp.down_proj = torch.nn.Linear(32, 16, bias=False)
        block.input_layernorm = torch.nn.LayerNorm(16)
        block.post_attention_layernorm = torch.nn.LayerNorm(16)
        return block


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


def test_wanda_transformer_global_masks_prune_across_all_linear_weights() -> None:
    model = MixedLinearDecoder()
    with torch.no_grad():
        model.block.good.weight.copy_(torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]))
        model.block.other.weight.copy_(
            torch.tensor(
                [
                    [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
                    [106.0, 107.0, 108.0, 109.0, 110.0, 111.0],
                ]
            )
        )

    masks = wanda_masks(model, activation_scalers={}, sparsity=0.5, granularity="global")

    assert masks["block.good"].sum().item() == 0
    assert masks["block.other"].sum().item() == 10
    assert mask_sparsity(masks) == pytest.approx(0.5)


def test_wanda_transformer_layer_masks_prune_per_output_row() -> None:
    model = MixedLinearDecoder()
    with torch.no_grad():
        model.block.good.weight.copy_(torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]))
        model.block.other.weight.copy_(
            torch.tensor(
                [
                    [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
                    [106.0, 107.0, 108.0, 109.0, 110.0, 111.0],
                ]
            )
        )

    masks = wanda_masks(model, activation_scalers={}, sparsity=0.5, granularity="layer")

    assert masks["block.good"].sum(dim=1).tolist() == [2, 2]
    assert masks["block.other"].sum(dim=1).tolist() == [3, 3]
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


def test_llama_transformer_scope_prunes_only_decoder_linears() -> None:
    model = TinyLlamaLikeDecoder()

    prunable_names = [name for name, _module in named_prunable_linears(model)]

    assert len(prunable_names) == 14
    assert all("lm_head" not in name for name in prunable_names)
    assert all("embed" not in name for name in prunable_names)
    assert all("norm" not in name for name in prunable_names)
    assert {name.rsplit(".", 1)[-1] for name in prunable_names} == {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    }

    masks = layerwise_magnitude_masks(model, sparsity=0.5)
    protected_snapshot = protected_parameter_snapshot(model, masks)
    apply_masks(model, masks)

    assert_protected_parameters_unchanged(model, protected_snapshot)
    assert masked_weight_stats(model, masks)["masked_weight_violation_count"] == 0
    assert all(mask_sparsity({name: mask}) == pytest.approx(0.5) for name, mask in masks.items())
