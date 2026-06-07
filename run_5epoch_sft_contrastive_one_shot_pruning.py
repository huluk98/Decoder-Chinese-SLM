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
    "run_root": "runs/full-decoder-sft-contrastive-pruning",
    "sft_config": "configs/sft_0p2b_8gpu.yaml",
    "contrastive_config": "configs/contrastive_sft_8gpu.yaml",
    "prune_config": "configs/prune_50.yaml",
    "original_model": None,  # None -> base_model override, then the base model in sft_config.
    "base_model": None,  # Optional override for regular/base SFT starting checkpoint.
    "contrastive_base_model": None,  # None means use the freshly trained regular SFT final checkpoint.
    "sft_train_file": None,  # Optional server-local override for regular SFT data.
    "sft_eval_file": None,  # Optional server-local override for regular SFT dense/pruning eval data.
    "benchmark_file": None,  # Optional shared benchmark override for both generated benchmark configs.
    "contrastive_train_file": None,  # Optional server-local override for contrastive anchor/positive/negative data.
    "contrastive_eval_file": None,  # Optional server-local override for contrastive dense/pruning eval data.
    "contrastive_anchor_eval_file": None,  # Optional alias when anchor eval differs from contrastive_eval_file.
    "original_eval_output_dir": None,  # None -> <run_root>/dense/original_decoder
    "regular_output_dir": None,  # None -> <run_root>/training/base_sft_5ep
    "contrastive_output_dir": None,  # None -> <run_root>/training/contrastive_sft_5ep
    "regular_pruning_output_dir": None,  # None -> <run_root>/pruning/base_sft
    "contrastive_pruning_output_dir": None,  # None -> <run_root>/pruning/contrastive_sft
    "generated_config_dir": None,  # None -> <run_root>/generated_configs
    "results_json": None,  # None -> <run_root>/journal_results.json
    "methods": ["magnitude", "wanda", "taylor", "2of4"],
    "eval_runs": 1,
    "top_k_exact_match": 5,
    "comparison_mode": "whitespace",
    "max_length": 256,
    "max_new_tokens": 64,
    "num_beams": 5,
    "max_new_token_hit_rate_threshold": 0.5,
    "sparsity": 0.5,
    "sparsity_levels": None,  # None -> use the single SPARSITY value.
    "pruning_scope": "transformer_linears",
    "sparsity_denominator": "prunable",
    "granularity": "global",
    "include_lm_head": False,
    "calibration_batches": 64,
    "prune_batch_size": 2,
    "prune_num_workers": 0,
    "sparsity_tolerance": 1.0e-3,
    "keep_going": True,
    "eos_retune": False,
    "eos_loss_weight": 5.0,
    "eos_retune_epochs": 1.0,
    "eos_retune_max_steps": None,
    "eos_retune_mode": "sft",
    "run_original_decoder_eval": True,
    "train_regular_sft": True,
    "train_contrastive_sft": True,
    "train_only": False,
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
    "ORIGINAL_MODEL": "original_model",
    "BASE_MODEL": "base_model",
    "CONTRASTIVE_BASE_MODEL": "contrastive_base_model",
    "SFT_TRAIN_FILE": "sft_train_file",
    "SFT_EVAL_FILE": "sft_eval_file",
    "BENCHMARK_FILE": "benchmark_file",
    "CONTRASTIVE_TRAIN_FILE": "contrastive_train_file",
    "CONTRASTIVE_EVAL_FILE": "contrastive_eval_file",
    "CONTRASTIVE_ANCHOR_EVAL_FILE": "contrastive_anchor_eval_file",
    "ORIGINAL_EVAL_OUTPUT_DIR": "original_eval_output_dir",
    "REGULAR_OUTPUT_DIR": "regular_output_dir",
    "CONTRASTIVE_OUTPUT_DIR": "contrastive_output_dir",
    "REGULAR_PRUNING_OUTPUT_DIR": "regular_pruning_output_dir",
    "CONTRASTIVE_PRUNING_OUTPUT_DIR": "contrastive_pruning_output_dir",
    "GENERATED_CONFIG_DIR": "generated_config_dir",
    "RESULTS_JSON": "results_json",
    "METHODS": "methods",
    "EVAL_RUNS": "eval_runs",
    "TOP_K_EXACT_MATCH": "top_k_exact_match",
    "COMPARISON_MODE": "comparison_mode",
    "MAX_LENGTH": "max_length",
    "MAX_NEW_TOKENS": "max_new_tokens",
    "NUM_BEAMS": "num_beams",
    "MAX_NEW_TOKEN_HIT_RATE_THRESHOLD": "max_new_token_hit_rate_threshold",
    "SPARSITY": "sparsity",
    "SPARSITY_LEVELS": "sparsity_levels",
    "PRUNING_SCOPE": "pruning_scope",
    "SPARSITY_DENOMINATOR": "sparsity_denominator",
    "GRANULARITY": "granularity",
    "INCLUDE_LM_HEAD": "include_lm_head",
    "CALIBRATION_BATCHES": "calibration_batches",
    "PRUNE_BATCH_SIZE": "prune_batch_size",
    "PRUNE_NUM_WORKERS": "prune_num_workers",
    "SPARSITY_TOLERANCE": "sparsity_tolerance",
    "KEEP_GOING": "keep_going",
    "EOS_RETUNE": "eos_retune",
    "EOS_LOSS_WEIGHT": "eos_loss_weight",
    "EOS_RETUNE_EPOCHS": "eos_retune_epochs",
    "EOS_RETUNE_MAX_STEPS": "eos_retune_max_steps",
    "EOS_RETUNE_MODE": "eos_retune_mode",
    "RUN_ORIGINAL_DECODER_EVAL": "run_original_decoder_eval",
    "TRAIN_ONLY": "train_only",
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


def parse_sparsity_levels(value: Any, fallback: Any) -> list[float]:
    if value in (None, ""):
        raw_levels = [fallback]
    elif isinstance(value, (int, float)):
        raw_levels = [value]
    elif isinstance(value, str):
        raw_levels = value.replace(",", " ").split()
    else:
        raw_levels = []
        for item in value:
            raw_levels.extend(str(item).replace(",", " ").split())
    levels: list[float] = []
    for raw_level in raw_levels:
        level = float(raw_level)
        if not 0.0 <= level <= 1.0:
            raise ValueError(f"sparsity levels must be between 0 and 1, got {level}")
        if level not in levels:
            levels.append(level)
    if not levels:
        raise ValueError("At least one sparsity level is required.")
    return levels


def sparsity_slug(level: float) -> str:
    text = f"{float(level):.6g}".replace(".", "p")
    return f"sparsity_{text}"


def sparsity_label(level: float) -> str:
    return f"{float(level):.0%}"


def methods_for_sparsity_level(methods: list[str], level: float) -> list[str]:
    """Native 2:4 is only meaningful at its fixed 50% structured target."""
    if abs(float(level) - 0.5) <= 1e-12:
        return list(methods)
    two_of_four_aliases = {"2of4", "2:4", "2_4", "2-of-4", "2_of_4", "nvidia-2of4", "nvidia_2of4", "nvidia24", "nvidia_24", "nvidia_2_4"}
    return [method for method in methods if str(method).strip().lower().replace("-", "_") not in two_of_four_aliases]


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


def require_data_path(value: str | None, config_path: str | Path, label: str, env_hint: str, *, dry_run: bool) -> str:
    resolved = resolve_path(value, config_path)
    if not resolved:
        raise ValueError(f"{label} is not configured. Set {env_hint} or update {config_path}.")
    if not dry_run and not Path(resolved).expanduser().exists():
        raise FileNotFoundError(
            f"{label} not found: {resolved}. "
            f"Set {env_hint} to the server-local dataset path, or update {config_path}."
        )
    return resolved


def first_value(*values: Any) -> str:
    for value in values:
        if value not in (None, ""):
            return str(value)
    return ""


def sft_config_with_overrides(config: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
    config = dict(config)
    if settings.get("base_model"):
        config["model_name_or_path"] = str(settings["base_model"])
    if settings.get("sft_train_file"):
        config["train_file"] = str(settings["sft_train_file"])
    if settings.get("sft_eval_file"):
        config["eval_file"] = str(settings["sft_eval_file"])
    if settings.get("benchmark_file"):
        config["benchmark_file"] = str(settings["benchmark_file"])
    return config


def contrastive_config_with_overrides(config: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
    config = dict(config)
    sft = dict(config.get("sft", {}) or {})
    if settings.get("contrastive_train_file"):
        sft["data_path"] = str(settings["contrastive_train_file"])
    if settings.get("contrastive_eval_file"):
        sft["eval_path"] = str(settings["contrastive_eval_file"])
        if not settings.get("contrastive_anchor_eval_file"):
            sft["anchor_eval_path"] = str(settings["contrastive_eval_file"])
    if settings.get("contrastive_anchor_eval_file"):
        sft["anchor_eval_path"] = str(settings["contrastive_anchor_eval_file"])
    if settings.get("benchmark_file"):
        sft["benchmark_path"] = str(settings["benchmark_file"])
    config["sft"] = sft
    return config


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


def sft_base_model_from_config(config: dict[str, Any]) -> str:
    sft = config.get("sft", {}) or {}
    run = config.get("run", {}) or {}
    return first_value(config.get("model_name_or_path"), sft.get("base_model"), run.get("base_model"))


def is_placeholder_model_path(value: Any) -> bool:
    text = str(value or "").strip()
    return not text or text.startswith("/path/to/")


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
    sft_config = read_yaml(config["sft_config"])
    settings = dict(config)
    if bool(settings.get("eos_retune")) and str(settings.get("eos_retune_mode", "sft")) != "sft":
        raise ValueError("EOS_RETUNE_MODE currently supports only 'sft' so the fixed-mask recovery isolates EOS reinforcement.")
    if float(settings.get("eos_loss_weight", 1.0)) < 1.0:
        raise ValueError("EOS_LOSS_WEIGHT must be >= 1.0.")
    settings["run_root"] = run_root
    settings["original_eval_output_dir"] = repo_path(config["original_eval_output_dir"]) if config.get("original_eval_output_dir") else run_root / "dense" / "original_decoder"
    settings["regular_output_dir"] = repo_path(config["regular_output_dir"]) if config.get("regular_output_dir") else run_root / "training" / "base_sft_5ep"
    settings["contrastive_output_dir"] = repo_path(config["contrastive_output_dir"]) if config.get("contrastive_output_dir") else run_root / "training" / "contrastive_sft_5ep"
    settings["regular_pruning_output_dir"] = repo_path(config["regular_pruning_output_dir"]) if config.get("regular_pruning_output_dir") else run_root / "pruning" / "base_sft"
    settings["contrastive_pruning_output_dir"] = repo_path(config["contrastive_pruning_output_dir"]) if config.get("contrastive_pruning_output_dir") else run_root / "pruning" / "contrastive_sft"
    settings["generated_config_dir"] = repo_path(config["generated_config_dir"]) if config.get("generated_config_dir") else run_root / "generated_configs"
    settings["results_json"] = repo_path(config["results_json"]) if config.get("results_json") else run_root / "journal_results.json"
    settings["regular_final"] = settings["regular_output_dir"] / "final"
    settings["contrastive_final"] = settings["contrastive_output_dir"] / "final"
    base_model = config.get("base_model") or config.get("original_model") or sft_base_model_from_config(sft_config)
    if is_placeholder_model_path(base_model):
        raise ValueError(
            "Set BASE_MODEL to the untuned base checkpoint, or pass it as the first argument to "
            "run_5epoch_sft_contrastive_one_shot_pruning.sh. The launcher uses that one model path "
            "for original dense eval and regular SFT training."
        )
    settings["base_model"] = str(base_model)
    settings["original_model"] = str(config.get("original_model") or settings["base_model"])
    settings["contrastive_base_model"] = config.get("contrastive_base_model") or str(settings["regular_final"])
    settings["sparsity_levels"] = parse_sparsity_levels(config.get("sparsity_levels"), config.get("sparsity", 0.5))
    settings["sparsity"] = float(settings["sparsity_levels"][0])
    settings["python"] = str(repo_path(config["python"])) if config.get("python") else sys.executable
    torchrun = config.get("torchrun")
    if torchrun:
        settings["torchrun"] = str(repo_path(torchrun))
    else:
        beside_python = Path(settings["python"]).with_name("torchrun")
        settings["torchrun"] = str(beside_python) if beside_python.exists() else "torchrun"
    return settings


def dataset_summary(settings: dict[str, Any]) -> dict[str, str]:
    sft_config = sft_config_with_overrides(read_yaml(settings["sft_config"]), settings)
    contrastive_config = contrastive_config_with_overrides(read_yaml(settings["contrastive_config"]), settings)
    sft = sft_config.get("sft", {}) or {}
    contrastive_sft = contrastive_config.get("sft", {}) or {}
    sft_inputs = sft_eval_inputs(sft_config, settings["sft_config"])
    contrastive_inputs = contrastive_eval_inputs(contrastive_config, settings["contrastive_config"])
    return {
        "regular_sft_train": resolve_path(first_value(sft_config.get("train_file"), sft.get("data_path")), settings["sft_config"]),
        "regular_sft_eval": sft_inputs["training_dataset"],
        "regular_sft_calibration": sft_inputs["calibration"],
        "contrastive_train": resolve_path(
            first_value(contrastive_config.get("train_file"), contrastive_sft.get("data_path")),
            settings["contrastive_config"],
        ),
        "contrastive_eval": contrastive_inputs["training_dataset"],
        "contrastive_calibration": contrastive_inputs["calibration"],
        "benchmark": sft_inputs["benchmark"],
        "contrastive_benchmark": contrastive_inputs["benchmark"],
    }


def command_env(settings: dict[str, Any]) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(settings["cuda_visible_devices"])
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["NCCL_DEBUG"] = str(env.get("NCCL_DEBUG") or "WARN")
    env["SYMPY_GROUND_TYPES"] = str(env.get("SYMPY_GROUND_TYPES") or "python")
    env["TORCHDYNAMO_DISABLE"] = str(env.get("TORCHDYNAMO_DISABLE") or "1")
    env["TORCH_COMPILE_DISABLE"] = str(env.get("TORCH_COMPILE_DISABLE") or "1")
    env["ACCELERATE_DYNAMO_BACKEND"] = str(env.get("ACCELERATE_DYNAMO_BACKEND") or "no")
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
    sft_config = read_yaml(settings["sft_config"])
    sft = sft_config.get("sft", {}) or {}
    train_file = first_value(settings.get("sft_train_file"), sft_config.get("train_file"), sft.get("data_path"))
    data_path = require_data_path(
        train_file,
        settings["sft_config"],
        "Regular SFT training dataset",
        "SFT_TRAIN_FILE",
        dry_run=bool(settings["dry_run"]),
    )
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
        "--data-path",
        data_path,
    ]
    if settings.get("base_model"):
        cmd.extend(["--checkpoint", str(settings["base_model"])])
    run_command(cmd, env=env, dry_run=bool(settings["dry_run"]))
    if not settings["dry_run"] and not settings["regular_final"].is_dir():
        raise FileNotFoundError(f"Regular SFT final checkpoint not found: {settings['regular_final']}")


def train_contrastive(settings: dict[str, Any], env: dict[str, str]) -> None:
    contrastive_config = read_yaml(settings["contrastive_config"])
    sft = contrastive_config.get("sft", {}) or {}
    train_file = first_value(settings.get("contrastive_train_file"), contrastive_config.get("train_file"), sft.get("data_path"))
    data_path = require_data_path(
        train_file,
        settings["contrastive_config"],
        "Contrastive SFT training dataset",
        "CONTRASTIVE_TRAIN_FILE",
        dry_run=bool(settings["dry_run"]),
    )
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
        "--data-path",
        data_path,
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
    base_checkpoint: str | Path,
    output_dir: Path,
    inputs: dict[str, str],
    methods: list[str] | None = None,
    sparsity: float | None = None,
) -> dict[str, Any]:
    benchmark_methods = list(settings["methods"] if methods is None else methods)
    target_sparsity = float(settings["sparsity"] if sparsity is None else sparsity)
    max_length = (
        int(settings["max_length"])
        if settings.get("max_length") not in (None, "")
        else int_from_config(source_config, "max_seq_length", "sft.max_length", default=128)
    )
    max_new_tokens = (
        int(settings["max_new_tokens"])
        if settings.get("max_new_tokens") not in (None, "")
        else int_from_config(source_config, "max_new_tokens", "generation.max_new_tokens", default=64)
    )
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
            "methods": benchmark_methods,
            "eval_files": eval_files,
            "run_dense_baseline": True,
            "min_dense_exact_match_accuracy": None,
            "benchmark_runs": int(settings["eval_runs"]),
            "top_k_exact_match": int(settings["top_k_exact_match"]),
            "comparison_mode": str(settings["comparison_mode"]),
            "max_new_tokens": max_new_tokens,
            "num_beams": int(settings["num_beams"]),
            "max_new_token_hit_rate_threshold": float(settings["max_new_token_hit_rate_threshold"]),
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
            "sparsity": target_sparsity,
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
        "one_shot": {"enabled": bool(benchmark_methods)},
        "retune": {
            "enabled": bool(benchmark_methods) and bool(settings["eos_retune"]),
            "config": str(settings["sft_config"]),
            "mode": str(settings["eos_retune_mode"]),
            "data_path": inputs["training_dataset"],
            "max_steps": settings.get("eos_retune_max_steps"),
            "epochs": float(settings["eos_retune_epochs"]),
            "max_seq_length": max_length,
            "keep_pruning_masks": True,
            "eos_loss_weight": float(settings["eos_loss_weight"]),
        },
        "_source_config": str(config_path),
    }


def pruning_output_dir_for_level(base_dir: Path, level: float, settings: dict[str, Any]) -> Path:
    levels = settings.get("sparsity_levels") or [settings["sparsity"]]
    if len(levels) == 1:
        return Path(base_dir)
    return Path(base_dir) / sparsity_slug(level)


def generated_config_name(base_name: str, level: float, settings: dict[str, Any]) -> str:
    levels = settings.get("sparsity_levels") or [settings["sparsity"]]
    if len(levels) == 1:
        return f"{base_name}.yaml"
    return f"{base_name}_{sparsity_slug(level)}.yaml"


def write_generated_benchmark_configs(settings: dict[str, Any]) -> tuple[Path, list[Path], list[Path]]:
    sft_config = sft_config_with_overrides(read_yaml(settings["sft_config"]), settings)
    contrastive_config = contrastive_config_with_overrides(read_yaml(settings["contrastive_config"]), settings)
    generated_dir = Path(settings["generated_config_dir"])
    generated_dir.mkdir(parents=True, exist_ok=True)
    original_path = generated_dir / "original_decoder_dense_eval.yaml"
    write_yaml(
        original_path,
        benchmark_config(
            label="original_decoder",
            source_config=sft_config,
            config_path=settings["sft_config"],
            settings=settings,
            base_checkpoint=settings["original_model"],
            output_dir=settings["original_eval_output_dir"],
            inputs=sft_eval_inputs(sft_config, settings["sft_config"]),
            methods=[],
        ),
    )
    regular_paths: list[Path] = []
    contrastive_paths: list[Path] = []
    for level in settings["sparsity_levels"]:
        level_methods = methods_for_sparsity_level(list(settings["methods"]), level)
        regular_path = generated_dir / generated_config_name("base_sft_one_shot_pruning", level, settings)
        contrastive_path = generated_dir / generated_config_name("contrastive_sft_one_shot_pruning", level, settings)
        write_yaml(
            regular_path,
            benchmark_config(
                label="base_sft",
                source_config=sft_config,
                config_path=settings["sft_config"],
                settings=settings,
                base_checkpoint=settings["regular_final"],
                output_dir=pruning_output_dir_for_level(settings["regular_pruning_output_dir"], level, settings),
                inputs=sft_eval_inputs(sft_config, settings["sft_config"]),
                methods=level_methods,
                sparsity=level,
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
                output_dir=pruning_output_dir_for_level(settings["contrastive_pruning_output_dir"], level, settings),
                inputs=contrastive_eval_inputs(contrastive_config, settings["contrastive_config"]),
                methods=level_methods,
                sparsity=level,
            ),
        )
        regular_paths.append(regular_path)
        contrastive_paths.append(contrastive_path)
    return original_path, regular_paths, contrastive_paths


def run_pruning_benchmark(
    config_path: Path,
    settings: dict[str, Any],
    env: dict[str, str],
    methods: list[str] | None = None,
) -> None:
    benchmark_methods = list(settings["methods"] if methods is None else methods)
    cmd = [
        str(settings["python"]),
        "scripts/run_pruning_benchmark.py",
        "--config",
        str(config_path),
        "--methods",
        " ".join(str(method) for method in benchmark_methods),
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


def difficulty_metrics(row: dict[str, Any]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for level in ("easy", "medium", "hard"):
        metrics[f"count_{level}"] = metric(row, f"difficulty_{level}_total_examples")
        metrics[f"em1_{level}"] = safe_float(metric(row, f"difficulty_{level}_exact_match_accuracy"))
        metrics[f"em5_{level}"] = safe_float(
            metric(
                row,
                f"difficulty_{level}_exact_match_at_5_accuracy",
                f"difficulty_{level}_top5_exact_match_accuracy",
                f"difficulty_{level}_exact_match_at_top_k_accuracy",
            )
        )
    return metrics


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


def rows_for_family(
    family: str,
    output_dir: Path,
    *,
    target_sparsity: float | None = None,
    include_dense: bool = True,
) -> list[dict[str, Any]]:
    payload = read_pruning_summary(output_dir)
    rows: list[dict[str, Any]] = []
    all_rows = (list(payload.get("dense_baseline", []) or []) if include_dense else []) + list(payload.get("results", []) or [])
    for row in all_rows:
        phase = str(row.get("phase", ""))
        method = "base_model" if phase == "dense_baseline" else str(row.get("method", ""))
        row_target_sparsity = 0.0 if phase == "dense_baseline" else target_sparsity
        if row_target_sparsity is None:
            row_target_sparsity = safe_float(metric(row, "target_whole_model_sparsity", "target_prunable_sparsity"))
        rows.append(
            {
                "model_family": family,
                "eval_name": row.get("eval_name", ""),
                "eval_file": row.get("eval_file", ""),
                "phase": phase,
                "method": method,
                "target_sparsity": safe_float(row_target_sparsity),
                "target_sparsity_label": sparsity_label(float(row_target_sparsity)) if row_target_sparsity not in ("", None) else "",
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
                **difficulty_metrics(row),
                "mean_response_loss": safe_float(metric(row, "mean_response_loss", "mean_response_loss_mean")),
                "response_perplexity": safe_float(metric(row, "response_perplexity", "response_perplexity_mean")),
                "avg_generated_tokens": safe_float(metric(row, "avg_generated_tokens", "avg_generated_tokens_mean")),
                "max_token_hit_rate": safe_float(metric(row, "reached_max_new_tokens_rate")),
                "hit_eos": metric(row, "hit_eos"),
                "eos_hit_rate": safe_float(metric(row, "eos_hit_rate")),
                "achieved_whole_model_sparsity": safe_float(metric(row, "achieved_whole_model_sparsity", "real_sparsity")),
                "achieved_prunable_sparsity": safe_float(metric(row, "achieved_prunable_sparsity")),
                "checkpoint_evaluated": row.get("checkpoint_evaluated", row.get("checkpoint_path", "")),
                "eval_output_dir": row.get("eval_output_dir", ""),
                "error": row.get("error", ""),
            }
        )
    return rows


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    return value


def sparsity_key(value: Any) -> str:
    if value in (None, ""):
        return ""
    try:
        return f"{float(value):.6g}"
    except (TypeError, ValueError):
        return str(value)


def result_completeness(rows: list[dict[str, Any]], settings: dict[str, Any]) -> dict[str, Any]:
    expected_eval_names = ["training_dataset", "benchmark"]
    expected_rows: list[dict[str, str]] = []
    if settings.get("run_original_decoder_eval", True):
        expected_rows.extend(
            {"model_family": "original_decoder", "eval_name": eval_name, "method": "base_model"}
            for eval_name in expected_eval_names
        )
    for family in ("base_sft", "contrastive_sft"):
        expected_rows.extend(
            {"model_family": family, "eval_name": eval_name, "method": "base_model"}
            for eval_name in expected_eval_names
        )
        for level in settings.get("sparsity_levels", [settings["sparsity"]]):
            level_methods = methods_for_sparsity_level(list(settings["methods"]), float(level))
            expected_rows.extend(
                {
                    "model_family": family,
                    "eval_name": eval_name,
                    "method": str(method),
                    "target_sparsity": sparsity_key(level),
                }
                for eval_name in expected_eval_names
                for method in level_methods
            )
            if settings.get("eos_retune", False):
                expected_rows.extend(
                    {
                        "model_family": family,
                        "eval_name": eval_name,
                        "method": str(method),
                        "phase": "retuned",
                        "target_sparsity": sparsity_key(level),
                    }
                    for eval_name in expected_eval_names
                    for method in level_methods
                )
    present = {
        (
            str(row.get("model_family")),
            str(row.get("eval_name")),
            str(row.get("method")),
            str(row.get("phase", "")),
            sparsity_key(row.get("target_sparsity", "")) if row.get("method") != "base_model" else "",
        )
        for row in rows
    }
    missing = [
        expected
        for expected in expected_rows
        if (
            expected["model_family"],
            expected["eval_name"],
            expected["method"],
            str(expected.get("phase", "one_shot" if expected["method"] != "base_model" else "dense_baseline")),
            expected.get("target_sparsity", "") if expected["method"] != "base_model" else "",
        )
        not in present
    ]
    return {
        "expected_eval_names": expected_eval_names,
        "expected_rows": expected_rows,
        "missing": missing,
        "complete": not missing,
    }


def write_results_json(
    settings: dict[str, Any],
    original_config: Path,
    regular_configs: list[Path],
    contrastive_configs: list[Path],
) -> None:
    original_summary = (
        read_pruning_summary(settings["original_eval_output_dir"])
        if settings.get("run_original_decoder_eval", True)
        else {}
    )
    rows: list[dict[str, Any]] = []
    if settings.get("run_original_decoder_eval", True):
        rows.extend(rows_for_family("original_decoder", settings["original_eval_output_dir"]))
    base_summaries: dict[str, Any] = {}
    contrastive_summaries: dict[str, Any] = {}
    for index, level in enumerate(settings["sparsity_levels"]):
        slug = sparsity_slug(level)
        base_dir = pruning_output_dir_for_level(settings["regular_pruning_output_dir"], level, settings)
        contrastive_dir = pruning_output_dir_for_level(settings["contrastive_pruning_output_dir"], level, settings)
        rows.extend(rows_for_family("base_sft", base_dir, target_sparsity=level, include_dense=index == 0))
        rows.extend(rows_for_family("contrastive_sft", contrastive_dir, target_sparsity=level, include_dense=index == 0))
        base_summaries[slug] = read_pruning_summary(base_dir)
        contrastive_summaries[slug] = read_pruning_summary(contrastive_dir)
    datasets = dataset_summary(settings)
    output_path = Path(settings["results_json"])
    payload = {
        "schema_version": 1,
        "run": {
            "epochs": settings["epochs"],
            "methods": settings["methods"],
            "eval_runs": settings["eval_runs"],
            "top_k_exact_match": settings["top_k_exact_match"],
            "comparison_mode": settings["comparison_mode"],
            "cuda_visible_devices": settings["cuda_visible_devices"],
            "nproc_per_node": settings["nproc_per_node"],
            "run_root": settings["run_root"],
            "config_files": {
                "original_decoder_source": settings["sft_config"],
                "base_sft": settings["sft_config"],
                "contrastive_sft": settings["contrastive_config"],
                "prune": settings["prune_config"],
            },
            "original_model": settings["original_model"],
            "base_model": settings.get("base_model"),
            "contrastive_base_model": settings["contrastive_base_model"],
            "checkpoints": {
                "original_decoder": settings["original_model"],
                "base_sft_final": settings["regular_final"],
                "contrastive_sft_final": settings["contrastive_final"],
            },
            "datasets": datasets,
            "generated_configs": {
                "original_decoder": original_config,
                "base_sft": regular_configs,
                "contrastive_sft": contrastive_configs,
            },
            "pruning": {
                "sparsity": settings["sparsity"],
                "sparsity_levels": settings["sparsity_levels"],
                "scope": settings["pruning_scope"],
                "sparsity_denominator": settings["sparsity_denominator"],
                "granularity": settings["granularity"],
                "include_lm_head": settings["include_lm_head"],
                "calibration_batches": settings["calibration_batches"],
            },
            "generation": {
                "max_length": settings["max_length"],
                "max_new_tokens": settings["max_new_tokens"],
                "num_beams": settings["num_beams"],
                "max_new_token_hit_rate_threshold": settings["max_new_token_hit_rate_threshold"],
                "top_k_exact_match": settings["top_k_exact_match"],
                "comparison_mode": settings["comparison_mode"],
            },
            "eos_reinforcement": {
                "enabled": bool(settings["eos_retune"]),
                "eos_loss_weight": settings["eos_loss_weight"],
                "retune_epochs": settings["eos_retune_epochs"],
                "retune_max_steps": settings.get("eos_retune_max_steps"),
                "retune_mode": settings["eos_retune_mode"],
                "fixed_pruning_masks": True,
            },
        },
        "checks": result_completeness(rows, settings),
        "results": rows,
        "raw_benchmark_summaries": {
            "original_decoder": original_summary,
            "base_sft": base_summaries,
            "contrastive_sft": contrastive_summaries,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(json_ready(payload), handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    csv_path = output_path.with_suffix(".csv")
    csv_fields = [
        "model_family",
        "eval_name",
        "phase",
        "method",
        "target_sparsity",
        "target_sparsity_label",
        "status",
        "em1",
        "em5",
        "em1_correct",
        "em5_correct",
        "total_examples",
        "count_easy",
        "em1_easy",
        "em5_easy",
        "count_medium",
        "em1_medium",
        "em5_medium",
        "count_hard",
        "em1_hard",
        "em5_hard",
        "mean_response_loss",
        "response_perplexity",
        "avg_generated_tokens",
        "max_token_hit_rate",
        "hit_eos",
        "eos_hit_rate",
        "achieved_whole_model_sparsity",
        "achieved_prunable_sparsity",
        "checkpoint_evaluated",
        "eval_output_dir",
        "error",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(json_ready(rows))
    print(f"\nWrote consolidated JSON results: {output_path}")
    print(f"Wrote consolidated CSV results:  {csv_path}")


def print_plan(settings: dict[str, Any]) -> None:
    datasets = dataset_summary(settings)
    print("5-epoch SFT + contrastive SFT + one-shot pruning benchmark")
    print(f"  epochs:                {settings['epochs']}")
    print(f"  methods:               {' '.join(settings['methods'])}")
    print(f"  eval runs:             {settings['eval_runs']}")
    print(f"  exact-match top-k:     {settings['top_k_exact_match']}")
    print(
        "  generation:            "
        f"max_length={settings['max_length']}, "
        f"max_new_tokens={settings['max_new_tokens']}, "
        f"num_beams={settings['num_beams']}, "
        f"max_token_hit_rate_threshold={settings['max_new_token_hit_rate_threshold']}"
    )
    print(
        "  pruning target:        "
        f"{', '.join(sparsity_label(level) for level in settings['sparsity_levels'])} "
        f"{settings['sparsity_denominator']} "
        f"{settings['pruning_scope']} ({settings['granularity']})"
    )
    print(f"  calibration batches:   {settings['calibration_batches']}")
    print(f"  run root:              {settings['run_root']}")
    print(f"  CUDA_VISIBLE_DEVICES:  {settings['cuda_visible_devices']}")
    print(f"  nproc_per_node:        {settings['nproc_per_node']}")
    print(f"  Python:                {settings['python']}")
    print(f"  torchrun:              {settings['torchrun']}")
    print(f"  original decoder:      {settings['original_model']}")
    print(f"  base SFT checkpoint:   {settings['base_model']}")
    print(f"  contrastive base:      {settings['contrastive_base_model']}")
    print(f"  regular SFT data:      {datasets['regular_sft_train']}")
    print(f"  contrastive data:      {datasets['contrastive_train']}")
    print(f"  eval dataset:          {datasets['regular_sft_eval']}")
    print(f"  benchmark dataset:     {datasets['benchmark']}")
    print(f"  results JSON:          {settings['results_json']}")
    if settings.get("eos_retune"):
        print(
            "  EOS retune:            "
            f"enabled, weight={settings['eos_loss_weight']}, "
            f"epochs={settings['eos_retune_epochs']}, max_steps={settings.get('eos_retune_max_steps')}"
        )


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

    if settings["train_only"]:
        print("\nTrain-only run complete.")
        print(f"Regular 5-epoch SFT checkpoint:      {settings['regular_final']}")
        print(f"Contrastive 5-epoch SFT checkpoint:  {settings['contrastive_final']}")
        return

    original_config, regular_configs, contrastive_configs = write_generated_benchmark_configs(settings)
    generated_config_lines = [str(original_config), *[str(path) for path in regular_configs], *[str(path) for path in contrastive_configs]]
    print("\nGenerated benchmark configs:\n  " + "\n  ".join(generated_config_lines))

    if settings["run_original_decoder_eval"]:
        run_pruning_benchmark(original_config, settings, env, methods=[])
    if settings["run_pruning_benchmarks"]:
        for level, config_path in zip(settings["sparsity_levels"], regular_configs):
            run_pruning_benchmark(config_path, settings, env, methods=methods_for_sparsity_level(list(settings["methods"]), level))
        for level, config_path in zip(settings["sparsity_levels"], contrastive_configs):
            run_pruning_benchmark(config_path, settings, env, methods=methods_for_sparsity_level(list(settings["methods"]), level))

    if settings["dry_run"]:
        print("\nDry run complete.")
        return

    write_results_json(settings, original_config, regular_configs, contrastive_configs)
    print("\nDone.")


if __name__ == "__main__":
    main()
