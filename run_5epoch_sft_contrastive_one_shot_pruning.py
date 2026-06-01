#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover - environment guard.
    raise SystemExit(
        "PyYAML is required. Run this from the training environment, or set "
        "PYTHON=/path/to/env/bin/python when using the .sh wrapper."
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parent


# Everything you are likely to change for the journal run is here.
CONFIG: dict[str, Any] = {
    "python": None,  # None means use the Python executable running this script.
    "torchrun": None,  # None means use torchrun beside the Python executable, then PATH.
    "cuda_visible_devices": "0,1,2,3,4,5,6,7",
    "nproc_per_node": 8,
    "omp_num_threads": 8,
    "epochs": 5,
    "run_root": "runs/5epoch-sft-contrastive-one-shot",
    "sft_config": "configs/sft_0p2b_8gpu.yaml",
    "contrastive_config": "configs/contrastive_sft_8gpu.yaml",
    "prune_config": "configs/prune_50.yaml",
    "base_model": None,  # Optional override for regular/base SFT starting checkpoint.
    "contrastive_base_model": None,  # None means use the freshly trained regular SFT final checkpoint.
    "regular_output_dir": None,  # None -> <run_root>/training/base_sft_5ep
    "contrastive_output_dir": None,  # None -> <run_root>/training/contrastive_sft_5ep
    "regular_pruning_output_dir": None,  # None -> <run_root>/pruning/base_sft
    "contrastive_pruning_output_dir": None,  # None -> <run_root>/pruning/contrastive_sft
    "generated_config_dir": None,  # None -> <run_root>/generated_configs
    "summary_csv": None,  # None -> <run_root>/em1_em5_summary.csv
    "summary_json": None,  # None -> <run_root>/em1_em5_summary.json
    "methods": ["wanda", "gradient", "magnitude", "2of4"],
    "eval_runs": 1,
    "top_k_exact_match": 5,
    "comparison_mode": "whitespace",
    "sparsity": 0.5,
    "pruning_scope": "transformer_linears",
    "sparsity_denominator": "whole_model",
    "granularity": "layer",
    "include_lm_head": False,
    "calibration_batches": 128,
    "prune_batch_size": 2,
    "prune_num_workers": 0,
    "sparsity_tolerance": 1.0e-6,
    "keep_going": True,
    "train_regular_sft": True,
    "train_contrastive_sft": True,
    "run_pruning_benchmarks": True,
    "dry_run": False,
}


ENV_OVERRIDES = {
    "PYTHON": "python",
    "TORCHRUN": "torchrun",
    "CUDA_VISIBLE_DEVICES": "cuda_visible_devices",
    "NPROC_PER_NODE": "nproc_per_node",
    "OMP_NUM_THREADS": "omp_num_threads",
    "EPOCHS": "epochs",
    "RUN_ROOT": "run_root",
    "SFT_CONFIG": "sft_config",
    "CONTRASTIVE_CONFIG": "contrastive_config",
    "PRUNE_CONFIG": "prune_config",
    "BASE_MODEL": "base_model",
    "CONTRASTIVE_BASE_MODEL": "contrastive_base_model",
    "REGULAR_OUTPUT_DIR": "regular_output_dir",
    "CONTRASTIVE_OUTPUT_DIR": "contrastive_output_dir",
    "REGULAR_PRUNING_OUTPUT_DIR": "regular_pruning_output_dir",
    "CONTRASTIVE_PRUNING_OUTPUT_DIR": "contrastive_pruning_output_dir",
    "GENERATED_CONFIG_DIR": "generated_config_dir",
    "SUMMARY_CSV": "summary_csv",
    "SUMMARY_JSON": "summary_json",
    "METHODS": "methods",
    "EVAL_RUNS": "eval_runs",
    "TOP_K_EXACT_MATCH": "top_k_exact_match",
    "KEEP_GOING": "keep_going",
    "DRY_RUN": "dry_run",
}


def coerce_value(value: str, current: Any) -> Any:
    if isinstance(current, bool):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    if isinstance(current, int) and not isinstance(current, bool):
        return int(value)
    if isinstance(current, float):
        return float(value)
    if isinstance(current, list):
        return str(value).replace(",", " ").split()
    return value


def apply_env_overrides(config: dict[str, Any]) -> None:
    for env_name, key in ENV_OVERRIDES.items():
        if env_name not in os.environ:
            continue
        config[key] = coerce_value(os.environ[env_name], config.get(key))


def apply_set_overrides(config: dict[str, Any], overrides: list[str]) -> None:
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"--set must use key=value syntax, got {override!r}")
        key, raw_value = override.split("=", 1)
        key = key.strip()
        if key not in config:
            raise ValueError(f"Unknown config key {key!r}. Edit CONFIG or use one of: {', '.join(sorted(config))}")
        try:
            parsed = yaml.safe_load(raw_value)
        except Exception:
            parsed = raw_value
        if isinstance(config.get(key), list) and isinstance(parsed, str):
            parsed = parsed.replace(",", " ").split()
        config[key] = parsed


def repo_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else Path.cwd() / candidate


def read_yaml(path: str | Path) -> dict[str, Any]:
    with repo_path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def resolve_path(value: str | None, config_path: str | Path) -> str:
    if not value:
        return ""
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return str(path)
    config_dir = repo_path(config_path).resolve().parent
    for base in (config_dir, Path.cwd()):
        candidate = base / path
        if candidate.exists():
            return str(candidate)
    return str(Path.cwd() / path)


def first_value(*values: Any) -> str:
    for value in values:
        if value not in (None, ""):
            return str(value)
    return ""


def sft_eval_inputs(config: dict[str, Any], config_path: str | Path) -> dict[str, str]:
    sft = config.get("sft", {}) or {}
    eval_config = config.get("eval", {}) or {}
    train_file = first_value(config.get("train_file"), sft.get("data_path"))
    eval_file = first_value(config.get("eval_file"), sft.get("eval_path"), train_file)
    benchmark_file = first_value(
        config.get("benchmark_file"),
        config.get("benchmark_eval_file"),
        eval_config.get("benchmark_file"),
        eval_config.get("benchmark_path"),
        sft.get("benchmark_path"),
    )
    return {
        "training_dataset": resolve_path(eval_file, config_path),
        "calibration": resolve_path(train_file, config_path),
        "benchmark": resolve_path(benchmark_file, config_path),
    }


def contrastive_eval_inputs(config: dict[str, Any], config_path: str | Path) -> dict[str, str]:
    sft = config.get("sft", {}) or {}
    eval_config = config.get("eval", {}) or {}
    train_file = first_value(config.get("train_file"), sft.get("data_path"))
    anchor_eval = first_value(
        config.get("anchor_eval_file"),
        eval_config.get("anchor_file"),
        eval_config.get("anchor_path"),
        sft.get("anchor_eval_path"),
        config.get("eval_file"),
        sft.get("eval_path"),
        train_file,
    )
    benchmark_file = first_value(
        config.get("benchmark_file"),
        config.get("benchmark_eval_file"),
        eval_config.get("benchmark_file"),
        eval_config.get("benchmark_path"),
        sft.get("benchmark_path"),
    )
    return {
        "training_dataset": resolve_path(anchor_eval, config_path),
        "calibration": resolve_path(anchor_eval, config_path),
        "benchmark": resolve_path(benchmark_file, config_path),
    }


def int_from_config(config: dict[str, Any], *keys: str, default: int) -> int:
    for key in keys:
        cursor: Any = config
        for part in key.split("."):
            if not isinstance(cursor, dict) or part not in cursor:
                cursor = None
                break
            cursor = cursor[part]
        if cursor not in (None, ""):
            return int(cursor)
    return int(default)


def dtype_from_config(config: dict[str, Any]) -> str:
    train = config.get("train", {}) or {}
    precision = str(train.get("precision", "")).lower()
    if bool(config.get("bf16", False)) or precision == "bf16":
        return "bf16"
    if bool(config.get("fp16", False)) or precision == "fp16":
        return "fp16"
    return "fp32"


def resolved_settings(config: dict[str, Any]) -> dict[str, Any]:
    run_root = repo_path(config["run_root"])
    settings = dict(config)
    settings["run_root"] = run_root
    settings["regular_output_dir"] = repo_path(config["regular_output_dir"]) if config.get("regular_output_dir") else run_root / "training" / "base_sft_5ep"
    settings["contrastive_output_dir"] = repo_path(config["contrastive_output_dir"]) if config.get("contrastive_output_dir") else run_root / "training" / "contrastive_sft_5ep"
    settings["regular_pruning_output_dir"] = repo_path(config["regular_pruning_output_dir"]) if config.get("regular_pruning_output_dir") else run_root / "pruning" / "base_sft"
    settings["contrastive_pruning_output_dir"] = repo_path(config["contrastive_pruning_output_dir"]) if config.get("contrastive_pruning_output_dir") else run_root / "pruning" / "contrastive_sft"
    settings["generated_config_dir"] = repo_path(config["generated_config_dir"]) if config.get("generated_config_dir") else run_root / "generated_configs"
    settings["summary_csv"] = repo_path(config["summary_csv"]) if config.get("summary_csv") else run_root / "em1_em5_summary.csv"
    settings["summary_json"] = repo_path(config["summary_json"]) if config.get("summary_json") else run_root / "em1_em5_summary.json"
    settings["regular_final"] = settings["regular_output_dir"] / "final"
    settings["contrastive_final"] = settings["contrastive_output_dir"] / "final"
    settings["contrastive_base_model"] = config.get("contrastive_base_model") or str(settings["regular_final"])
    settings["python"] = str(repo_path(config["python"])) if config.get("python") else sys.executable
    torchrun = config.get("torchrun")
    if torchrun:
        settings["torchrun"] = str(repo_path(torchrun))
    else:
        beside_python = Path(settings["python"]).with_name("torchrun")
        settings["torchrun"] = str(beside_python) if beside_python.exists() else "torchrun"
    return settings


def command_env(settings: dict[str, Any]) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(settings["cuda_visible_devices"])
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["NCCL_DEBUG"] = str(env.get("NCCL_DEBUG") or "WARN")
    env["PYTORCH_CUDA_ALLOC_CONF"] = str(env.get("PYTORCH_CUDA_ALLOC_CONF") or "expandable_segments:True")
    env["OMP_NUM_THREADS"] = str(settings["omp_num_threads"])
    env["PYTHON"] = str(settings["python"])
    python_bin = str(Path(settings["python"]).parent)
    env["PATH"] = python_bin + os.pathsep + env.get("PATH", "")
    return env


def run_command(cmd: list[str], *, env: dict[str, str], dry_run: bool) -> None:
    print("\n+ " + shlex.join(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, cwd=Path.cwd(), env=env, check=True)


def train_regular(settings: dict[str, Any], env: dict[str, str]) -> None:
    cmd = [
        str(settings["torchrun"]),
        "--standalone",
        f"--nproc_per_node={int(settings['nproc_per_node'])}",
        "scripts/train.py",
        "--config",
        str(settings["sft_config"]),
        "--epochs",
        str(settings["epochs"]),
        "--output-dir",
        str(settings["regular_output_dir"]),
    ]
    if settings.get("base_model"):
        cmd.extend(["--checkpoint", str(settings["base_model"])])
    run_command(cmd, env=env, dry_run=bool(settings["dry_run"]))
    if not settings["dry_run"] and not settings["regular_final"].is_dir():
        raise FileNotFoundError(f"Regular SFT final checkpoint not found: {settings['regular_final']}")


def train_contrastive(settings: dict[str, Any], env: dict[str, str]) -> None:
    cmd = [
        str(settings["torchrun"]),
        "--standalone",
        f"--nproc_per_node={int(settings['nproc_per_node'])}",
        "scripts/train.py",
        "--config",
        str(settings["contrastive_config"]),
        "--mode",
        "contrastive",
        "--epochs",
        str(settings["epochs"]),
        "--checkpoint",
        str(settings["contrastive_base_model"]),
        "--output-dir",
        str(settings["contrastive_output_dir"]),
    ]
    run_command(cmd, env=env, dry_run=bool(settings["dry_run"]))
    if not settings["dry_run"] and not settings["contrastive_final"].is_dir():
        raise FileNotFoundError(f"Contrastive SFT final checkpoint not found: {settings['contrastive_final']}")


def benchmark_config(
    *,
    label: str,
    source_config: dict[str, Any],
    config_path: str | Path,
    settings: dict[str, Any],
    base_checkpoint: Path,
    output_dir: Path,
    inputs: dict[str, str],
) -> dict[str, Any]:
    max_length = int_from_config(source_config, "max_seq_length", "sft.max_length", default=128)
    max_new_tokens = int_from_config(source_config, "max_new_tokens", "generation.max_new_tokens", default=64)
    eval_batch_size = int_from_config(
        source_config,
        "per_device_eval_batch_size",
        "train.eval_batch_size",
        "train.batch_size",
        default=16,
    )
    eval_files = {"training_dataset": inputs["training_dataset"], "benchmark": inputs["benchmark"]}
    missing = [name for name, path in eval_files.items() if not path]
    if missing:
        raise ValueError(f"{label} missing eval file(s): {missing}")
    return {
        "benchmark": {
            "output_dir": str(output_dir),
            "base_checkpoint": str(base_checkpoint),
            "prune_config": str(settings["prune_config"]),
            "methods": list(settings["methods"]),
            "eval_files": eval_files,
            "run_dense_baseline": True,
            "min_dense_exact_match_accuracy": None,
            "benchmark_runs": int(settings["eval_runs"]),
            "top_k_exact_match": int(settings["top_k_exact_match"]),
            "comparison_mode": str(settings["comparison_mode"]),
            "max_new_tokens": max_new_tokens,
            "max_length": max_length,
            "eval_batch_size": eval_batch_size,
            "dtype": dtype_from_config(source_config),
            "nproc_per_node": int(settings["nproc_per_node"]),
            "expected_gpu_count": int(settings["nproc_per_node"]),
            "cuda_visible_devices": str(settings["cuda_visible_devices"]),
            "sparsity_tolerance": float(settings["sparsity_tolerance"]),
            "continue_on_error": bool(settings["keep_going"]),
        },
        "prune": {
            "sparsity": float(settings["sparsity"]),
            "scope": str(settings["pruning_scope"]),
            "sparsity_denominator": str(settings["sparsity_denominator"]),
            "granularity": str(settings["granularity"]),
            "include_lm_head": bool(settings["include_lm_head"]),
            "calibration_data_path": inputs["calibration"],
            "calibration_batches": int(settings["calibration_batches"]),
            "max_length": max_length,
            "batch_size": int(settings["prune_batch_size"]),
            "num_workers": int(settings["prune_num_workers"]),
        },
        "one_shot": {"enabled": True},
        "retune": {"enabled": False},
        "_source_config": str(config_path),
    }


def write_generated_benchmark_configs(settings: dict[str, Any]) -> tuple[Path, Path]:
    sft_config = read_yaml(settings["sft_config"])
    contrastive_config = read_yaml(settings["contrastive_config"])
    generated_dir = Path(settings["generated_config_dir"])
    generated_dir.mkdir(parents=True, exist_ok=True)
    regular_path = generated_dir / "base_sft_one_shot_pruning.yaml"
    contrastive_path = generated_dir / "contrastive_sft_one_shot_pruning.yaml"
    write_yaml(
        regular_path,
        benchmark_config(
            label="base_sft",
            source_config=sft_config,
            config_path=settings["sft_config"],
            settings=settings,
            base_checkpoint=settings["regular_final"],
            output_dir=settings["regular_pruning_output_dir"],
            inputs=sft_eval_inputs(sft_config, settings["sft_config"]),
        ),
    )
    write_yaml(
        contrastive_path,
        benchmark_config(
            label="contrastive_sft",
            source_config=contrastive_config,
            config_path=settings["contrastive_config"],
            settings=settings,
            base_checkpoint=settings["contrastive_final"],
            output_dir=settings["contrastive_pruning_output_dir"],
            inputs=contrastive_eval_inputs(contrastive_config, settings["contrastive_config"]),
        ),
    )
    return regular_path, contrastive_path


def run_pruning_benchmark(config_path: Path, settings: dict[str, Any], env: dict[str, str]) -> None:
    cmd = [
        str(settings["python"]),
        "scripts/run_pruning_benchmark.py",
        "--config",
        str(config_path),
        "--methods",
        " ".join(str(method) for method in settings["methods"]),
    ]
    cmd.append("--continue-on-error" if settings["keep_going"] else "--stop-on-error")
    if settings["dry_run"]:
        cmd.append("--dry-run")
    run_command(cmd, env=env, dry_run=False)


def metric(row: dict[str, Any], *names: str) -> Any:
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            return value
    return ""


def safe_float(value: Any) -> Any:
    if value in (None, ""):
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return value
    return number if math.isfinite(number) else ""


def read_pruning_summary(output_dir: Path) -> dict[str, Any]:
    summary_path = output_dir / "pruning_benchmark_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing pruning benchmark summary: {summary_path}")
    with summary_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def rows_for_family(family: str, output_dir: Path) -> list[dict[str, Any]]:
    payload = read_pruning_summary(output_dir)
    rows: list[dict[str, Any]] = []
    all_rows = list(payload.get("dense_baseline", []) or []) + list(payload.get("results", []) or [])
    for row in all_rows:
        phase = str(row.get("phase", ""))
        if phase == "retuned":
            continue
        method = "base_model" if phase == "dense_baseline" else str(row.get("method", ""))
        rows.append(
            {
                "model_family": family,
                "eval_name": row.get("eval_name", ""),
                "phase": phase,
                "method": method,
                "status": row.get("status", ""),
                "em1": safe_float(metric(row, "exact_match_accuracy", "exact_match_accuracy_mean")),
                "em5": safe_float(
                    metric(
                        row,
                        "exact_match_at_5_accuracy",
                        "top5_exact_match_accuracy",
                        "exact_match_at_top_k_accuracy",
                        "exact_match_at_top_k_accuracy_mean",
                    )
                ),
                "em1_correct": metric(row, "correct_examples", "exact_match_correct"),
                "em5_correct": metric(row, "exact_match_at_top_k_correct", "exact_match_at_5_correct"),
                "total_examples": metric(row, "total_examples"),
                "avg_generated_tokens": safe_float(metric(row, "avg_generated_tokens", "avg_generated_tokens_mean")),
                "max_token_hit_rate": safe_float(metric(row, "reached_max_new_tokens_rate")),
                "achieved_whole_model_sparsity": safe_float(metric(row, "achieved_whole_model_sparsity", "real_sparsity")),
                "achieved_prunable_sparsity": safe_float(metric(row, "achieved_prunable_sparsity")),
                "checkpoint_evaluated": row.get("checkpoint_evaluated", row.get("checkpoint_path", "")),
                "eval_output_dir": row.get("eval_output_dir", ""),
                "error": row.get("error", ""),
            }
        )
    return rows


def write_compact_summary(settings: dict[str, Any]) -> None:
    rows = rows_for_family("base_sft", settings["regular_pruning_output_dir"]) + rows_for_family(
        "contrastive_sft",
        settings["contrastive_pruning_output_dir"],
    )
    fieldnames = [
        "model_family",
        "eval_name",
        "phase",
        "method",
        "status",
        "em1",
        "em5",
        "em1_correct",
        "em5_correct",
        "total_examples",
        "avg_generated_tokens",
        "max_token_hit_rate",
        "achieved_whole_model_sparsity",
        "achieved_prunable_sparsity",
        "checkpoint_evaluated",
        "eval_output_dir",
        "error",
    ]
    summary_csv = Path(settings["summary_csv"])
    summary_json = Path(settings["summary_json"])
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with summary_json.open("w", encoding="utf-8") as handle:
        json.dump({"results": rows}, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(f"\nWrote EM@1/EM@5 CSV:  {summary_csv}")
    print(f"Wrote EM@1/EM@5 JSON: {summary_json}")


def print_plan(settings: dict[str, Any]) -> None:
    print("5-epoch SFT + contrastive SFT + one-shot pruning benchmark")
    print(f"  epochs:                {settings['epochs']}")
    print(f"  methods:               {' '.join(settings['methods'])}")
    print(f"  eval runs:             {settings['eval_runs']}")
    print(f"  exact-match top-k:     {settings['top_k_exact_match']}")
    print(f"  run root:              {settings['run_root']}")
    print(f"  CUDA_VISIBLE_DEVICES:  {settings['cuda_visible_devices']}")
    print(f"  nproc_per_node:        {settings['nproc_per_node']}")
    print(f"  Python:                {settings['python']}")
    print(f"  torchrun:              {settings['torchrun']}")
    print(f"  base SFT checkpoint:   {settings.get('base_model') or 'from SFT config'}")
    print(f"  contrastive base:      {settings['contrastive_base_model']}")
    print(f"  compact summary CSV:   {settings['summary_csv']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run 5-epoch base SFT, contrastive SFT, and one-shot pruning EM@1/EM@5 benchmark.")
    parser.add_argument("--dry-run", action="store_true", help="Print and generate configs without running training/pruning.")
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="Override CONFIG keys, e.g. --set run_root=runs/test --set epochs=5")
    parser.add_argument("--print-config", action="store_true", help="Print the resolved config and exit.")
    return parser.parse_args()


def main() -> None:
    os.chdir(PROJECT_ROOT)
    args = parse_args()
    config = dict(CONFIG)
    apply_env_overrides(config)
    apply_set_overrides(config, args.overrides)
    if args.dry_run:
        config["dry_run"] = True
    settings = resolved_settings(config)
    if args.print_config:
        print(json.dumps({key: str(value) if isinstance(value, Path) else value for key, value in settings.items()}, indent=2))
        return

    env = command_env(settings)
    print_plan(settings)
    Path(settings["run_root"]).mkdir(parents=True, exist_ok=True)
    Path(settings["generated_config_dir"]).mkdir(parents=True, exist_ok=True)

    if settings["train_regular_sft"]:
        train_regular(settings, env)
    if settings["train_contrastive_sft"]:
        train_contrastive(settings, env)

    regular_config, contrastive_config = write_generated_benchmark_configs(settings)
    print(f"\nGenerated benchmark configs:\n  {regular_config}\n  {contrastive_config}")

    if settings["run_pruning_benchmarks"]:
        run_pruning_benchmark(regular_config, settings, env)
        run_pruning_benchmark(contrastive_config, settings, env)

    if settings["dry_run"]:
        print("\nDry run complete.")
        return

    write_compact_summary(settings)
    print("\nDone.")


if __name__ == "__main__":
    main()
