from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.chatlm_decoder.pruning import apply_masks, masked_weight_stats  # noqa: E402


def test_apply_masks_hard_zeros_masked_weights_even_when_nan() -> None:
    model = torch.nn.Sequential(torch.nn.Linear(4, 1, bias=False))
    with torch.no_grad():
        model[0].weight.copy_(torch.tensor([[float("nan"), 2.0, 3.0, 4.0]]))
    masks = {"0": torch.tensor([[False, True, False, True]])}

    apply_masks(model, masks)

    assert model[0].weight.detach()[0, 0].item() == 0.0
    assert model[0].weight.detach()[0, 2].item() == 0.0
    assert masked_weight_stats(model, masks)["masked_weight_violation_count"] == 0

