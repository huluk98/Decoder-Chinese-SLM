from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

sys.modules.setdefault(
    "transformers",
    types.SimpleNamespace(AutoModelForCausalLM=object, AutoTokenizer=object),
)

from scripts import sft  # noqa: E402


class TinyCausalLM(torch.nn.Module):
    def __init__(self, logits: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("fixed_logits", logits)

    def forward(self, input_ids, attention_mask=None, labels=None, use_cache=False):  # noqa: ANN001
        logits = self.fixed_logits.expand(input_ids.shape[0], -1, -1)
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        return SimpleNamespace(logits=logits, loss=loss)


class TinyCachePositionCausalLM(TinyCausalLM):
    def __init__(self, logits: torch.Tensor) -> None:
        super().__init__(logits)
        self.cache_position_shape = None

    def forward(self, input_ids, attention_mask=None, labels=None, use_cache=False, cache_position=None):  # noqa: ANN001
        self.cache_position_shape = None if cache_position is None else tuple(cache_position.shape)
        return super().forward(input_ids, attention_mask=attention_mask, labels=labels, use_cache=use_cache)


def test_causal_lm_loss_passes_1d_cache_position_when_supported() -> None:
    logits = torch.zeros((1, 4, 4), dtype=torch.float32)
    labels = torch.tensor([[-100, 1, 2, 3]], dtype=torch.long)
    input_ids = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    model = TinyCachePositionCausalLM(logits)

    sft.causal_lm_loss(
        model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        eos_token_id=3,
        eos_loss_weight=1.0,
    )

    assert model.cache_position_shape == (input_ids.shape[-1],)


def test_eos_loss_weight_upweights_only_supervised_eos_positions() -> None:
    logits = torch.tensor(
        [
            [
                [3.0, 0.0, 0.0, 0.0],
                [0.0, 3.0, 0.0, 0.0],
                [0.0, 0.0, 3.0, 0.0],
                [0.0, 0.0, 0.0, 3.0],
            ]
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([[-100, 1, 3, -100]], dtype=torch.long)
    input_ids = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    model = TinyCausalLM(logits)

    weighted_loss, _ = sft.causal_lm_loss(
        model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        eos_token_id=3,
        eos_loss_weight=5.0,
    )

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    token_loss = torch.nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="none",
    )
    weights = torch.tensor([1.0, 5.0, 1.0])
    valid = shift_labels.view(-1).ne(-100)
    expected = (token_loss * weights)[valid].sum() / weights[valid].sum()

    assert weighted_loss.item() == pytest.approx(expected.item())


def test_eos_loss_weight_one_uses_model_loss_path() -> None:
    logits = torch.zeros((1, 3, 4), dtype=torch.float32)
    labels = torch.tensor([[-100, 1, 3]], dtype=torch.long)
    input_ids = torch.tensor([[0, 1, 2]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    model = TinyCausalLM(logits)

    loss, outputs = sft.causal_lm_loss(
        model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        eos_token_id=3,
        eos_loss_weight=1.0,
    )

    assert loss is outputs.loss
