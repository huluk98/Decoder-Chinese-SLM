from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_launcher(config_path: Path) -> str:
    env = os.environ.copy()
    env.update(
        {
            "PYTHON": shutil.which("true") or "/usr/bin/true",
            "MODE": "auto",
            "DRY_RUN": "1",
            "KEEP_GOING": "1",
        }
    )
    result = subprocess.run(
        ["bash", "scripts/run_pruning_benchmark_8way.sh", str(config_path)],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    return result.stdout


def launcher_env(mode: str = "auto") -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "PYTHON": shutil.which("true") or "/usr/bin/true",
            "MODE": mode,
            "DRY_RUN": "1",
            "KEEP_GOING": "1",
        }
    )
    return env


def test_base_qwen_config_uses_generic_decoder_runner(tmp_path: Path) -> None:
    config_path = tmp_path / "base_qwen_pruning_benchmark.yaml"
    config_path.write_text(
        """
benchmark:
  base_checkpoint: Qwen/Qwen2.5-0.5B
  prune_config: configs/prune_50.yaml
retune:
  config: configs/sft_0p2b_8gpu.yaml
""".strip(),
        encoding="utf-8",
    )

    output = run_launcher(config_path)

    assert "mode:   generic" in output
    assert "runner: scripts/run_pruning_benchmark.py" in output
    assert "run_qwen25_instruct_pruning_benchmark.py" not in output


def test_qwen_instruct_config_uses_instruct_runner(tmp_path: Path) -> None:
    config_path = tmp_path / "qwen25_instruct_pruning_benchmark.yaml"
    config_path.write_text(
        """
benchmark:
  base_checkpoint: Qwen/Qwen2.5-0.5B-Instruct
  prune_config: configs/prune_qwen25_50.yaml
retune:
  config: configs/sft_qwen25_0p5b_instruct.yaml
""".strip(),
        encoding="utf-8",
    )

    output = run_launcher(config_path)

    assert "mode:   qwen_instruct" in output
    assert "runner: scripts/run_qwen25_instruct_pruning_benchmark.py" in output


def test_mode_qwen_is_rejected_because_it_is_ambiguous(tmp_path: Path) -> None:
    config_path = tmp_path / "base_qwen_pruning_benchmark.yaml"
    config_path.write_text("benchmark:\n  base_checkpoint: Qwen/Qwen2.5-0.5B\n", encoding="utf-8")

    result = subprocess.run(
        ["bash", "scripts/run_pruning_benchmark_8way.sh", str(config_path)],
        cwd=ROOT,
        env=launcher_env(mode="qwen"),
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "MODE=qwen is ambiguous" in result.stderr
