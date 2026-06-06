from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "run_5epoch_sft_contrastive_one_shot_pruning.py"

sys.modules.setdefault(
    "yaml",
    types.SimpleNamespace(
        safe_load=lambda handle: {},
        safe_dump=lambda payload, handle, **kwargs: handle.write(json.dumps(payload, default=str)),
    ),
)


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
    assert config["methods"] == ["magnitude", "wanda", "taylor", "2of4"]
    assert config["top_k_exact_match"] == 5
    assert config["max_new_tokens"] == 64
    assert config["max_new_token_hit_rate_threshold"] == 0.5
    assert config["sparsity_denominator"] == "prunable"
    assert config["granularity"] == "global"
    assert config["pruning_scope"] == "transformer_linears"
    assert config["include_lm_head"] is False
    assert config["sparsity_levels"] is None


def test_full_decoder_launcher_expands_native_sparsity_levels(tmp_path: Path) -> None:
    launcher = load_launcher_module()
    config = dict(launcher.CONFIG)
    config.update(
        {
            "base_model": "dummy/base",
            "run_root": str(tmp_path / "run"),
            "generated_config_dir": str(tmp_path / "generated"),
            "sparsity_levels": [0.3, 0.5],
        }
    )
    settings = launcher.resolved_settings(config)

    assert settings["sparsity_levels"] == [0.3, 0.5]
    assert launcher.pruning_output_dir_for_level(settings["regular_pruning_output_dir"], 0.3, settings).name == "sparsity_0p3"
    assert launcher.generated_config_name("base_sft_one_shot_pruning", 0.5, settings).endswith("sparsity_0p5.yaml")

    benchmark_config = launcher.benchmark_config(
        label="base_sft",
        source_config={},
        config_path="configs/sft_0p2b_8gpu.yaml",
        settings=settings,
        base_checkpoint=settings["regular_final"],
        output_dir=launcher.pruning_output_dir_for_level(settings["regular_pruning_output_dir"], 0.3, settings),
        inputs={"training_dataset": "train.json", "benchmark": "bench.json", "calibration": "cal.json"},
        sparsity=0.3,
    )

    assert benchmark_config["prune"]["sparsity"] == 0.3
    assert benchmark_config["benchmark"]["output_dir"].endswith("sparsity_0p3")


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
