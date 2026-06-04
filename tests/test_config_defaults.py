from __future__ import annotations

import sys
from pathlib import Path

import pytest


pytest.importorskip("yaml")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from chatlm_decoder.config import load_config  # noqa: E402


def test_prune_defaults_protect_decoder_only_non_linear_parameters(tmp_path: Path) -> None:
    config_path = tmp_path / "minimal.yaml"
    config_path.write_text("run:\n  seed: 123\n", encoding="utf-8")

    config = load_config(config_path)

    assert config["prune"]["scope"] == "transformer_linears"
    assert config["prune"]["sparsity_denominator"] == "prunable"
    assert config["prune"]["include_lm_head"] is False
