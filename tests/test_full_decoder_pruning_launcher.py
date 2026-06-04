from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "run_5epoch_sft_contrastive_one_shot_pruning.py"


def load_launcher_module():
    spec = importlib.util.spec_from_file_location("full_decoder_pruning_launcher", LAUNCHER)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_full_decoder_launcher_defaults_are_guarded_and_report_em5() -> None:
    launcher = load_launcher_module()
    config = launcher.CONFIG

    assert config["epochs"] == 5
    assert config["methods"] == ["magnitude", "wanda", "gradient", "2of4"]
    assert config["top_k_exact_match"] == 5
    assert config["max_new_tokens"] == 64
    assert config["max_new_token_hit_rate_threshold"] == 0.5
    assert config["sparsity_denominator"] == "whole_model"
    assert config["granularity"] == "layer"
    assert config["pruning_scope"] == "transformer_linears"
    assert config["include_lm_head"] is False


def test_full_decoder_shell_wrapper_help_describes_one_argument_run() -> None:
    result = subprocess.run(
        ["bash", "run_full_decoder_sft_contrastive_pruning.sh", "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "bash run_full_decoder_sft_contrastive_pruning.sh /path/to/original_decoder_checkpoint" in result.stdout
    assert "EM@1 and EM@5" in result.stdout
    assert "MAX_NEW_TOKEN_HIT_RATE_THRESHOLD=0.5" in result.stdout
