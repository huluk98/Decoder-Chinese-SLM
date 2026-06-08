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


class TinyLinearModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(10, 1, bias=False)
        with torch.no_grad():
            self.proj.weight.copy_(torch.tensor([[10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]]))


def test_progressive_magnitude_preserves_previous_prunes_without_overshooting() -> None:
    model = TinyLinearModel()
    previous = {"proj": torch.ones_like(model.proj.weight, dtype=torch.bool)}
    previous["proj"].flatten()[0] = False
    args = SimpleNamespace(
        prune_method="magnitude",
        prune_output_heads=False,
        global_pruning=False,
        regrowth=False,
    )

    masks = sparsity.make_progressive_stage_masks(
        model,
        tokenizer=None,
        stage_sparsity=0.3,
        previous=previous,
        recovery_samples=[],
        args=args,
        device=torch.device("cpu"),
    )

    flat = masks["proj"].flatten()
    assert flat[0].item() is False
    assert int((~flat).sum().item()) == 3
    assert sparsity.mask_sparsity(masks) == pytest.approx(0.3)

