from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_sparsity_experiments as sparsity  # noqa: E402


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


def test_progressive_recovery_eos_weight_upweights_supervised_eos() -> None:
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

    weighted_loss, _ = sparsity.causal_lm_recovery_loss(
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

