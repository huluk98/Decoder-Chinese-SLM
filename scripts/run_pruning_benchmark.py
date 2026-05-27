#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import copy
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METHODS = ("magnitude", "2of4", "wanda", "gradient")
DIFFICULTY_LEVELS = ("easy", "medium", "hard")
EVAL_SUMMARY_FILENAMES = (
    "prompt_response_eval_benchmark_summary.json",
    "prompt_response_eval_summary.json",
    "metrics.json",
)
HF_WEIGHT_FILENAMES = (
    "model.safetensors",
    "model.safetensors.index.json",
    "pytorch_model.bin",
    "pytorch_model.bin.index.json",
)
PRUNING_ROW_PHASES = {"one_shot", "retuned"}
PRUNING_SUMMARY_CSV_FIELDS = [
    "method",
    "phase",
    "status",
    "checkpoint",
    "checkpoint_evaluated",
    "eval_output_dir",
    "active_model_parameters",
    "active_prunable_parameters",
    "total_parameter_count",
    "real_sparsity",
    "target_sparsity_denominator",
    "achieved_prunable_sparsity",
    "method_target_note",
    "eval_name",
    "eval_file",
    "exact_match_accuracy_mean",
    "exact_match_accuracy",
    "exact_match_accuracy_std",
    "exact_match_at_top_k_accuracy_mean",
    "exact_match_at_top_k_accuracy",
    "exact_match_at_top_k_accuracy_std",
    "exact_match_at_top_k_correct",
    "exact_match_at_5_accuracy",
    "top5_exact_match_accuracy",
    "difficulty_easy_total_examples",
    "difficulty_easy_exact_match_correct",
    "difficulty_easy_exact_match_accuracy",
    "difficulty_easy_exact_match_at_top_k_correct",
    "difficulty_easy_exact_match_at_top_k_accuracy",
    "difficulty_easy_exact_match_at_5_correct",
    "difficulty_easy_exact_match_at_5_accuracy",
    "difficulty_medium_total_examples",
    "difficulty_medium_exact_match_correct",
    "difficulty_medium_exact_match_accuracy",
    "difficulty_medium_exact_match_at_top_k_correct",
    "difficulty_medium_exact_match_at_top_k_accuracy",
    "difficulty_medium_exact_match_at_5_correct",
    "difficulty_medium_exact_match_at_5_accuracy",
    "difficulty_hard_total_examples",
    "difficulty_hard_exact_match_correct",
    "difficulty_hard_exact_match_accuracy",
    "difficulty_hard_exact_match_at_top_k_correct",
    "difficulty_hard_exact_match_at_top_k_accuracy",
    "difficulty_hard_exact_match_at_5_correct",
    "difficulty_hard_exact_match_at_5_accuracy",
    "correct_examples",
    "total_examples",
    "mean_response_loss",
    "response_perplexity",
    "avg_generated_tokens",
    "pruning_report",
    "error",
]


def resolve_config_path(path: str | Path) -> Path:
    config_path = Path(path).expanduser()
    if config_path.is_dir():
        config_path = config_path / "pruning_benchmark.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Pruning benchmark config not found: {config_path}")
    return config_path.resolve()


def load_yaml(path: str | Path) -> dict[str, Any]:
    yaml_path = Path(path).expanduser()
    with yaml_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def write_yaml(path: str | Path, payload: dict[str, Any]) -> None:
    yaml_path = Path(path)
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with yaml_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)


def write_json(path: str | Path, payload: Any) -> None:
    json_path = Path(path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def method_slug(method: str) -> str:
    return "nvidia-2of4" if method == "2of4" else method


def as_path(value: Any) -> Path:
    return Path(str(value)).expanduser()


def display_path(path: Path) -> str:
    try:
        return str(path.resolve())
    except OSError:
        return str(path)


def slugify_name(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value).strip()).strip("-")
    return slug or "eval"


def eval_specs_from_benchmark(benchmark: dict[str, Any], override: str | None) -> list[dict[str, Any]]:
    if override:
        return [{"name": "eval", "slug": "eval", "path": as_path(override)}]
    eval_files = benchmark.get("eval_files")
    specs: list[dict[str, Any]] = []
    if isinstance(eval_files, dict):
        for name, path in eval_files.items():
            specs.append({"name": str(name), "slug": slugify_name(str(name)), "path": as_path(path)})
    elif isinstance(eval_files, list):
        for index, item in enumerate(eval_files, start=1):
            if isinstance(item, dict):
                path = item.get("path", item.get("file", item.get("eval_file")))
                if not path:
                    raise ValueError(f"benchmark.eval_files item {index} is missing path/file/eval_file.")
                name = str(item.get("name", Path(str(path)).stem or f"eval-{index}"))
            else:
                path = item
                name = Path(str(path)).stem or f"eval-{index}"
            specs.append({"name": name, "slug": slugify_name(name), "path": as_path(path)})
    else:
        eval_file = benchmark.get("eval_file")
        if eval_file:
            specs.append({"name": "eval", "slug": "eval", "path": as_path(eval_file)})
    if not specs:
        raise ValueError("Set benchmark.eval_file or benchmark.eval_files.")
    seen_slugs: dict[str, int] = {}
    for spec in specs:
        slug = str(spec["slug"])
        seen_slugs[slug] = seen_slugs.get(slug, 0) + 1
        if seen_slugs[slug] > 1:
            spec["slug"] = f"{slug}-{seen_slugs[slug]}"
    return specs


def eval_dir_for_phase(
    output_dir: Path,
    phase: str,
    method: str | None,
    eval_spec: dict[str, Any],
    multi_eval: bool,
) -> Path:
    eval_slug = str(eval_spec["slug"])
    if phase == "dense_baseline":
        base = output_dir / "benchmarks" / "dense_baseline"
    elif phase == "one_shot" and method is not None:
        base = output_dir / "benchmarks" / "one_shot" / method_slug(method)
    elif phase == "retuned" and method is not None:
        base = output_dir / "benchmarks" / "retuned" / method_slug(method)
    else:
        base = output_dir
    return base / eval_slug if multi_eval else base


def latest_checkpoint(output_dir: Path) -> Path:
    final = output_dir / "final"
    if final.exists():
        return final
    latest = output_dir / "latest"
    if latest.exists():
        return latest
    checkpoints = sorted([path for path in output_dir.glob("step-*") if path.is_dir()], key=lambda path: path.name)
    if not checkpoints:
        raise FileNotFoundError(f"No final, latest, or step-* checkpoint found in {output_dir}")
    return checkpoints[-1]


def has_hf_checkpoint_files(path: Path) -> bool:
    return (path / "config.json").exists() and any((path / filename).exists() for filename in HF_WEIGHT_FILENAMES)


def validate_saved_checkpoint_path(checkpoint: Path, label: str, base_checkpoint: Path | None = None) -> None:
    if not checkpoint.exists():
        raise FileNotFoundError(f"{label} checkpoint does not exist: {checkpoint}")
    if not checkpoint.is_dir():
        raise ValueError(f"{label} checkpoint must be a directory: {checkpoint}")
    if not has_hf_checkpoint_files(checkpoint):
        raise FileNotFoundError(
            f"{label} checkpoint is not a loadable Hugging Face directory: {checkpoint}. "
            "Expected config.json plus model weights."
        )
    if base_checkpoint is not None and base_checkpoint.exists() and checkpoint.resolve() == base_checkpoint.resolve():
        raise RuntimeError(f"Refusing to evaluate dense base checkpoint as {label}: {checkpoint}")


def write_checkpoint_pointer(output_dir: Path, method: str, phase: str, checkpoint: Path) -> None:
    pointer = output_dir / "checkpoint_paths"
    pointer.mkdir(parents=True, exist_ok=True)
    (pointer / f"{phase}_{method_slug(method)}.txt").write_text(str(checkpoint.resolve()) + "\n", encoding="utf-8")


def command_env(benchmark: dict[str, Any]) -> dict[str, str]:
    env = os.environ.copy()
    cuda_visible_devices = benchmark.get("cuda_visible_devices")
    if cuda_visible_devices is not None and str(cuda_visible_devices).strip():
        env["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("NCCL_DEBUG", "WARN")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("OMP_NUM_THREADS", "8")
    return env


def should_continue_on_error(args: argparse.Namespace, benchmark: dict[str, Any]) -> bool:
    if args.continue_on_error:
        return True
    if args.stop_on_error:
        return False
    return bool(benchmark.get("continue_on_error", False))


def parse_methods(value: Any) -> list[str]:
    if isinstance(value, str):
        parts = [part.strip() for part in value.replace(",", " ").split()]
    else:
        parts = [str(part).strip() for part in value]
    return [part for part in parts if part]


def parse_cuda_visible_devices(value: Any) -> list[str]:
    if value is None:
        return []
    text = str(value).strip()
    if not text or text == "-1":
        return []
    return [part.strip() for part in text.split(",") if part.strip()]


def validate_gpu_launch_config(benchmark: dict[str, Any], strict: bool) -> None:
    nproc = int(benchmark.get("nproc_per_node", 8))
    expected = int(benchmark.get("expected_gpu_count", nproc))
    visible_devices = parse_cuda_visible_devices(benchmark.get("cuda_visible_devices", os.environ.get("CUDA_VISIBLE_DEVICES")))
    problems: list[str] = []
    warnings: list[str] = []
    if nproc != expected:
        problems.append(f"benchmark.nproc_per_node must be {expected} for this suite, got {nproc}.")
    if visible_devices and len(visible_devices) != expected:
        problems.append(
            f"benchmark.cuda_visible_devices exposes {len(visible_devices)} devices, expected {expected}: "
            f"{','.join(visible_devices)}"
        )
    if not visible_devices:
        warnings.append("No CUDA_VISIBLE_DEVICES list was configured; torchrun will use whatever CUDA exposes.")
    for warning in warnings:
        print(f"Warning: {warning}", flush=True)
    if problems:
        message = "Invalid GPU launch config:\n  - " + "\n  - ".join(problems)
        if strict:
            raise ValueError(message)
        print("Warning: " + message.replace("\n", "\nWarning: "), flush=True)


def validate_pruning_report(
    report_path: Path,
    method: str,
    phase: str,
    target_sparsity: float,
    tolerance: float,
) -> None:
    report = read_pruning_report(report_path)
    if not report:
        raise FileNotFoundError(f"Missing pruning report for {method} {phase}: {report_path}")
    denominator = str(report.get("target_sparsity_denominator", "prunable")).lower()
    target_prunable = report.get("target_prunable_sparsity", target_sparsity)
    if denominator == "whole_model":
        actual_sparsity = report.get("achieved_whole_model_sparsity", report.get("model_zero_fraction"))
        expected_sparsity = report.get("target_whole_model_sparsity", report.get("target_sparsity", target_sparsity))
        if actual_sparsity is None:
            raise ValueError(f"Pruning report does not contain achieved whole-model sparsity: {report_path}")
    else:
        actual_sparsity = report.get("achieved_prunable_sparsity", report.get("mask_sparsity", report.get("sparsity")))
        expected_sparsity = target_prunable
        if actual_sparsity is None:
            raise ValueError(f"Pruning report does not contain achieved prunable sparsity: {report_path}")
    if abs(float(actual_sparsity) - float(expected_sparsity)) > float(tolerance):
        raise ValueError(
            f"{method} {phase} {denominator} sparsity check failed: actual={float(actual_sparsity):.8f}, "
            f"target={float(expected_sparsity):.8f}, tolerance={float(tolerance):.8f}"
        )
    violations = int(report.get("masked_weight_violation_count", 0) or 0)
    if violations:
        raise ValueError(f"{method} {phase} has {violations} nonzero weights under the pruning mask: {report_path}")
    active_mask_parameters = int(report.get("active_mask_parameters", 0) or 0)
    if active_mask_parameters <= 0:
        raise ValueError(f"{method} {phase} reports no active mask parameters: {report_path}")
    nonzero_parameters = report.get("nonzero_parameters")
    if nonzero_parameters is not None and int(nonzero_parameters) <= 0:
        raise ValueError(f"{method} {phase} reports zero nonzero model parameters: {report_path}")
    if "achieved_prunable_sparsity" in report and abs(float(report["achieved_prunable_sparsity"]) - float(target_prunable)) > float(tolerance):
        raise ValueError(f"{method} {phase} achieved_prunable_sparsity does not match target: {report_path}")


def validate_benchmark_paths(
    methods: list[str],
    base_checkpoint: Path,
    eval_specs: list[dict[str, Any]],
    prune_config_path: Path,
    config: dict[str, Any],
    retune: dict[str, Any],
    strict: bool,
) -> None:
    problems: list[str] = []
    warnings: list[str] = []
    if not base_checkpoint.exists():
        warnings.append(
            f"benchmark.base_checkpoint does not exist as a local path: {base_checkpoint}. "
            "If this is not a Hugging Face model id, fix the YAML before launching."
        )
    for eval_spec in eval_specs:
        eval_file = Path(eval_spec["path"])
        if not eval_file.exists():
            problems.append(f"benchmark eval file {eval_spec['name']!r} does not exist: {eval_file}")
    if not prune_config_path.exists():
        problems.append(f"benchmark.prune_config does not exist: {prune_config_path}")
    calibration_data_path = config.get("prune", {}).get("calibration_data_path")
    if any(method in {"wanda", "gradient"} for method in methods):
        if not calibration_data_path:
            problems.append("prune.calibration_data_path is required for wanda/gradient.")
        elif not as_path(calibration_data_path).exists():
            problems.append(f"prune.calibration_data_path does not exist: {calibration_data_path}")
    if bool(retune.get("enabled", True)):
        data_path = retune.get("data_path")
        if not data_path:
            problems.append("retune.data_path is required when retune.enabled=true.")
        elif not as_path(data_path).exists():
            problems.append(f"retune.data_path does not exist: {data_path}")
    for warning in warnings:
        print(f"Warning: {warning}", flush=True)
    if problems:
        message = "Invalid pruning benchmark YAML:\n  - " + "\n  - ".join(problems)
        if strict:
            raise FileNotFoundError(message)
        print("Warning: " + message.replace("\n", "\nWarning: "), flush=True)


def print_plan(
    methods: list[str],
    output_dir: Path,
    base_checkpoint: Path,
    eval_specs: list[dict[str, Any]],
    prune_config_path: Path,
    config: dict[str, Any],
    benchmark: dict[str, Any],
    retune: dict[str, Any],
    continue_on_error: bool,
) -> None:
    print("\nResolved generic pruning benchmark plan:", flush=True)
    print(f"  methods: {', '.join(methods)}", flush=True)
    print(f"  output_dir: {display_path(output_dir)}", flush=True)
    print(f"  base_checkpoint: {display_path(base_checkpoint)}", flush=True)
    print("  eval_files:", flush=True)
    for eval_spec in eval_specs:
        print(f"    {eval_spec['name']}: {display_path(Path(eval_spec['path']))}", flush=True)
    print(f"  prune_config: {display_path(prune_config_path)}", flush=True)
    print(f"  calibration_data_path: {config.get('prune', {}).get('calibration_data_path')}", flush=True)
    print(f"  target_sparsity: {config.get('prune', {}).get('sparsity', 0.5)}", flush=True)
    print(f"  pruning_scope: {config.get('prune', {}).get('scope', 'full_model')}", flush=True)
    print(f"  sparsity_denominator: {config.get('prune', {}).get('sparsity_denominator', 'prunable')}", flush=True)
    print(f"  benchmark_runs: {benchmark.get('benchmark_runs', 1)}", flush=True)
    print(f"  exact_match_top_k: {benchmark.get('top_k_exact_match', 5)}", flush=True)
    print(f"  retune.enabled: {bool(retune.get('enabled', True))}", flush=True)
    print(f"  retune.data_path: {retune.get('data_path')}", flush=True)
    print(f"  retune.max_steps: {retune.get('max_steps')}", flush=True)
    print(f"  retune.epochs: {retune.get('epochs')}", flush=True)
    print(f"  eval runner: scripts/eval_prompt_response.py", flush=True)
    print(f"  retune runner: scripts/sft.py", flush=True)
    print(f"  nproc_per_node: {benchmark.get('nproc_per_node', 8)}", flush=True)
    print(f"  cuda_visible_devices: {benchmark.get('cuda_visible_devices')}", flush=True)
    print(f"  expected_gpu_count: {benchmark.get('expected_gpu_count', benchmark.get('nproc_per_node', 8))}", flush=True)
    print(f"  continue_on_error: {continue_on_error}", flush=True)


def run_command(cmd: list[str], env: dict[str, str], dry_run: bool = False) -> None:
    print("\n$ " + " ".join(str(part) for part in cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, check=True)


def generated_prune_config(
    base_config: dict[str, Any],
    benchmark_config: dict[str, Any],
    method: str,
    checkpoint: Path,
    output_dir: Path,
    recovery_steps: int,
) -> dict[str, Any]:
    config = copy.deepcopy(base_config)
    config.setdefault("run", {})
    config.setdefault("model", {})
    config.setdefault("train", {})
    config.setdefault("prune", {})
    prune_overrides = dict(benchmark_config.get("prune", {}) or {})
    config["prune"].update(prune_overrides)
    config["prune"].update(
        {
            "base_model": str(checkpoint),
            "output_dir": str(output_dir),
            "method": method,
            "recovery_steps": int(recovery_steps),
            "overwrite": bool(config["prune"].get("overwrite", True)),
        }
    )
    return config


def run_prune(
    method: str,
    checkpoint: Path,
    output_dir: Path,
    config_path: Path,
    env: dict[str, str],
    dry_run: bool,
) -> None:
    run_command(
        [
            sys.executable,
            "scripts/prune.py",
            "--config",
            str(config_path),
            "--method",
            method,
            "--checkpoint",
            str(checkpoint),
            "--output-dir",
            str(output_dir),
        ],
        env=env,
        dry_run=dry_run,
    )


def run_eval(
    model_path: Path,
    eval_file: Path,
    output_dir: Path,
    benchmark: dict[str, Any],
    env: dict[str, str],
    dry_run: bool,
) -> None:
    nproc = int(benchmark.get("nproc_per_node", 8))
    run_command(
        [
            "torchrun",
            "--standalone",
            "--nproc_per_node",
            str(nproc),
            "scripts/eval_prompt_response.py",
            "--model-path",
            str(model_path),
            "--dataset-file",
            str(eval_file),
            "--output-dir",
            str(output_dir),
            "--max-new-tokens",
            str(int(benchmark.get("max_new_tokens", 64))),
            "--max-length",
            str(int(benchmark.get("max_length", benchmark.get("max_seq_length", 2048)))),
            "--temperature",
            "0",
            "--num-beams",
            "1",
            "--batch-size",
            str(int(benchmark.get("eval_batch_size", 16))),
            "--dtype",
            str(benchmark.get("dtype", "bf16")),
            "--benchmark-runs",
            str(int(benchmark.get("benchmark_runs", 1))),
            "--exact-match-top-k",
            str(int(benchmark.get("top_k_exact_match", benchmark.get("exact_match_top_k", 5)))),
            "--comparison-mode",
            str(benchmark.get("comparison_mode", "whitespace")),
        ],
        env=env,
        dry_run=dry_run,
    )


def dense_baseline_row(
    checkpoint: Path,
    eval_dir: Path,
    summary: dict[str, Any],
    eval_spec: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result_eval_dir = resolve_eval_result_dir(eval_dir)
    eval_name = str(eval_spec.get("name")) if eval_spec else "eval"
    eval_file = str(eval_spec.get("path")) if eval_spec else ""
    return {
        "method": "dense_sft_baseline",
        "phase": "dense_baseline",
        "status": "ok",
        "checkpoint": str(checkpoint),
        "checkpoint_evaluated": str(checkpoint),
        "eval_name": eval_name,
        "eval_file": eval_file,
        "eval_dir": str(result_eval_dir),
        "eval_output_dir": str(result_eval_dir),
        "pruning_report": "",
        "active_model_parameters": "",
        "real_sparsity": 0.0,
        "target_prunable_sparsity": 0.0,
        "achieved_prunable_sparsity": 0.0,
        "achieved_whole_model_sparsity": 0.0,
        "prunable_parameter_count": "",
        "protected_parameter_count": "",
        "total_parameter_count": "",
        "exact_match_accuracy_mean": metric(summary, "exact_match_accuracy"),
        "exact_match_accuracy": metric(summary, "exact_match_accuracy"),
        "exact_match_accuracy_std": metric_std(summary, "exact_match_accuracy"),
        "exact_match_at_top_k_accuracy_mean": metric(summary, "exact_match_at_top_k_accuracy"),
        "exact_match_at_top_k_accuracy": metric(summary, "exact_match_at_top_k_accuracy"),
        "exact_match_at_top_k_accuracy_std": metric_std(summary, "exact_match_at_top_k_accuracy"),
        "exact_match_at_top_k_correct": metric_count(summary, "exact_match_at_top_k_correct"),
        "exact_match_at_5_accuracy": metric(summary, "exact_match_at_5_accuracy"),
        "top5_exact_match_accuracy": metric(summary, "top5_exact_match_accuracy"),
        **difficulty_summary_metrics(summary),
        "correct_examples": metric_count(summary, "correct_examples", "exact_match_correct"),
        "total_examples": metric_count(summary, "total_examples"),
        "delta_vs_dense_exact_match": 0.0,
        "mean_response_loss_mean": metric(summary, "mean_response_loss"),
        "mean_response_loss": metric(summary, "mean_response_loss"),
        "mean_response_loss_std": metric_std(summary, "mean_response_loss"),
        "response_perplexity_mean": metric(summary, "response_perplexity"),
        "response_perplexity": metric(summary, "response_perplexity"),
        "response_perplexity_std": metric_std(summary, "response_perplexity"),
        "avg_generated_tokens_mean": metric(summary, "avg_generated_tokens"),
        "avg_generated_tokens": metric(summary, "avg_generated_tokens"),
        "avg_generated_tokens_std": metric_std(summary, "avg_generated_tokens"),
        "error": "",
    }


def run_retune(
    pruned_checkpoint: Path,
    masks_path: Path,
    output_dir: Path,
    retune: dict[str, Any],
    benchmark: dict[str, Any],
    env: dict[str, str],
    dry_run: bool,
) -> None:
    nproc = int(benchmark.get("nproc_per_node", 8))
    data_path = retune.get("data_path")
    if not data_path:
        raise ValueError("retune.data_path is required when retune.enabled=true.")
    cmd = [
        "torchrun",
        "--standalone",
        "--nproc_per_node",
        str(nproc),
        "scripts/sft.py",
        "--config",
        str(retune.get("config", "configs/sft_0p2b_8gpu.yaml")),
        "--mode",
        str(retune.get("mode", "sft")),
        "--checkpoint",
        str(pruned_checkpoint),
        "--data-path",
        str(as_path(data_path)),
        "--output-dir",
        str(output_dir),
    ]
    if bool(retune.get("keep_pruning_masks", True)):
        cmd.extend(["--pruning-mask", str(masks_path)])
    if retune.get("max_steps") is not None:
        cmd.extend(["--max_steps", str(int(retune["max_steps"]))])
    elif retune.get("epochs") is not None:
        cmd.extend(["--epochs", str(float(retune["epochs"]))])
    if retune.get("per_device_train_batch_size") is not None:
        cmd.extend(["--per_device_train_batch_size", str(int(retune["per_device_train_batch_size"]))])
    if retune.get("gradient_accumulation_steps") is not None:
        cmd.extend(["--gradient_accumulation_steps", str(int(retune["gradient_accumulation_steps"]))])
    if retune.get("max_seq_length") is not None:
        cmd.extend(["--max_seq_length", str(int(retune["max_seq_length"]))])
    run_command(cmd, env=env, dry_run=dry_run)


def eval_summary_path(eval_dir: Path) -> Path | None:
    for filename in EVAL_SUMMARY_FILENAMES:
        path = eval_dir / filename
        if path.exists():
            return path
    return None


def has_eval_result(eval_dir: Path) -> bool:
    return (eval_dir / "run_config.json").exists() and eval_summary_path(eval_dir) is not None


def resolve_latest_eval_marker(eval_dir: Path) -> Path | None:
    marker = eval_dir / "latest_eval_dir.txt"
    if not marker.exists():
        return None
    text = marker.read_text(encoding="utf-8").strip()
    if not text:
        return None
    marked = Path(text).expanduser()
    if not marked.is_absolute():
        marked = PROJECT_ROOT / marked
    return marked


def resolve_eval_result_dir(eval_dir: Path) -> Path:
    eval_dir = Path(eval_dir)
    if has_eval_result(eval_dir):
        return eval_dir
    marked = resolve_latest_eval_marker(eval_dir)
    if marked is not None and has_eval_result(marked):
        return marked
    if not eval_dir.exists():
        return eval_dir
    candidates = [child for child in eval_dir.iterdir() if child.is_dir() and has_eval_result(child)]
    if not candidates:
        return eval_dir
    return sorted(candidates, key=lambda path: (path.stat().st_mtime, path.name))[-1]


def read_eval_summary(eval_dir: Path) -> dict[str, Any]:
    result_dir = resolve_eval_result_dir(eval_dir)
    summary_path = eval_summary_path(result_dir)
    if summary_path is None:
        return {}
    with summary_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_pruning_report(report_path: Path | None) -> dict[str, Any]:
    if report_path is None or not report_path.exists():
        return {}
    with report_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_eval_run_config(eval_dir: Path) -> dict[str, Any]:
    path = resolve_eval_result_dir(eval_dir) / "run_config.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_eval_protocol_matches_dense(dense_eval_dir: Path, eval_dir: Path) -> None:
    dense = read_eval_run_config(dense_eval_dir)
    current = read_eval_run_config(eval_dir)
    if not dense or not current:
        raise FileNotFoundError(f"Missing run_config.json for dense/pruned eval comparison: {dense_eval_dir} vs {eval_dir}")
    checks = [
        ("dataset_file", dense.get("dataset_file"), current.get("dataset_file")),
        ("max_length", dense.get("max_length"), current.get("max_length")),
        ("generation", dense.get("generation"), current.get("generation")),
        ("tokenizer", dense.get("tokenizer"), current.get("tokenizer")),
        (
            "exact_match_comparison_mode",
            (dense.get("eval_args") or {}).get("comparison_mode"),
            (current.get("eval_args") or {}).get("comparison_mode"),
        ),
    ]
    mismatches = [f"{name}: dense={left!r} current={right!r}" for name, left, right in checks if left != right]
    if mismatches:
        raise RuntimeError("Pruned eval protocol differs from dense baseline: " + "; ".join(mismatches))


def validate_eval_checkpoint(eval_dir: Path, expected_checkpoint: Path) -> None:
    config = read_eval_run_config(eval_dir)
    checkpoint = config.get("checkpoint_path_used_for_evaluation") or config.get("model_path")
    if not checkpoint:
        raise RuntimeError(f"Eval run_config is missing checkpoint path: {eval_dir}")
    if Path(checkpoint).expanduser().resolve() != expected_checkpoint.expanduser().resolve():
        raise RuntimeError(f"Eval used {checkpoint}, expected pruned checkpoint {expected_checkpoint}")


def metric(summary: dict[str, Any], name: str) -> Any:
    if f"{name}_mean" in summary:
        return summary.get(f"{name}_mean")
    if summary.get(name) is not None:
        return summary.get(name)
    values = []
    for run_summary in summary.get("per_run_summaries", []) or []:
        value = run_summary.get(name)
        if value is None:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(numeric):
            values.append(numeric)
    if not values:
        return None
    return sum(values) / len(values)


def metric_std(summary: dict[str, Any], name: str) -> Any:
    return summary.get(f"{name}_std")


def metric_count(summary: dict[str, Any], *names: str) -> Any:
    for name in names:
        if summary.get(name) is not None:
            return summary.get(name)
        if summary.get(f"{name}_mean") is not None:
            return summary.get(f"{name}_mean")
    run_summaries = summary.get("per_run_summaries", []) or []
    for name in names:
        values = [run.get(name) for run in run_summaries if run.get(name) is not None]
        if values:
            numeric_values = [float(value) for value in values]
            mean_value = sum(numeric_values) / len(numeric_values)
            return int(mean_value) if mean_value.is_integer() else mean_value
    return None


def difficulty_summary_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for level in DIFFICULTY_LEVELS:
        prefix = f"difficulty_{level}"
        metrics[f"{prefix}_total_examples"] = metric_count(summary, f"{prefix}_total_examples")
        metrics[f"{prefix}_exact_match_correct"] = metric_count(summary, f"{prefix}_exact_match_correct")
        metrics[f"{prefix}_exact_match_accuracy"] = metric(summary, f"{prefix}_exact_match_accuracy")
        metrics[f"{prefix}_exact_match_at_top_k_correct"] = metric_count(summary, f"{prefix}_exact_match_at_top_k_correct")
        metrics[f"{prefix}_exact_match_at_top_k_accuracy"] = metric(summary, f"{prefix}_exact_match_at_top_k_accuracy")
        metrics[f"{prefix}_exact_match_at_5_correct"] = metric_count(summary, f"{prefix}_exact_match_at_5_correct")
        metrics[f"{prefix}_exact_match_at_5_accuracy"] = metric(summary, f"{prefix}_exact_match_at_5_accuracy")
        metrics[f"{prefix}_top5_exact_match_accuracy"] = metric(summary, f"{prefix}_top5_exact_match_accuracy")
    return metrics


def pruning_stat(report: dict[str, Any], *names: str) -> Any:
    reload_validation = report.get("checkpoint_reload_validation")
    sources = [reload_validation, report] if isinstance(reload_validation, dict) else [report]
    for source in sources:
        for name in names:
            if source.get(name) is not None:
                return source.get(name)
    return None


def read_pruning_stats(report_path: Path | None) -> dict[str, Any]:
    report = read_pruning_report(report_path)
    return {
        "target_prunable_sparsity": pruning_stat(report, "target_prunable_sparsity", "target_sparsity", "sparsity"),
        "target_sparsity_denominator": pruning_stat(report, "target_sparsity_denominator"),
        "target_whole_model_sparsity": pruning_stat(report, "target_whole_model_sparsity"),
        "method_target_note": pruning_stat(report, "method_target_note"),
        "achieved_prunable_sparsity": pruning_stat(report, "achieved_prunable_sparsity", "mask_sparsity", "sparsity"),
        "achieved_whole_model_sparsity": pruning_stat(report, "achieved_whole_model_sparsity", "model_zero_fraction"),
        "real_sparsity": pruning_stat(report, "model_zero_fraction", "achieved_whole_model_sparsity"),
        "active_model_parameters": pruning_stat(report, "nonzero_parameters"),
        "active_prunable_parameters": pruning_stat(report, "active_prunable_parameters", "active_mask_parameters"),
        "pruned_prunable_parameters": pruning_stat(report, "pruned_prunable_parameters", "pruned_mask_parameters"),
        "total_prunable_parameters": pruning_stat(report, "total_prunable_parameters", "mask_parameter_count"),
        "prunable_parameter_count": pruning_stat(report, "prunable_parameter_count", "mask_parameter_count"),
        "protected_parameter_count": pruning_stat(report, "protected_parameter_count", "protected_parameters"),
        "total_parameter_count": pruning_stat(report, "total_parameter_count", "total_parameters"),
        "zero_parameters": pruning_stat(report, "zero_parameters"),
        "nonzero_parameters": pruning_stat(report, "nonzero_parameters"),
        "nonzero_fraction": pruning_stat(report, "nonzero_fraction"),
        "mask_sparsity": pruning_stat(report, "mask_sparsity", "sparsity"),
        "mask_parameter_count": pruning_stat(report, "mask_parameter_count"),
        "active_mask_parameters": pruning_stat(report, "active_mask_parameters"),
        "pruned_mask_parameters": pruning_stat(report, "pruned_mask_parameters"),
        "active_mask_fraction": pruning_stat(report, "active_mask_fraction"),
        "mask_implied_active_parameters": pruning_stat(report, "mask_implied_active_parameters"),
        "mask_implied_pruned_parameters": pruning_stat(report, "mask_implied_pruned_parameters"),
        "mask_implied_active_fraction": pruning_stat(report, "mask_implied_active_fraction"),
        "mask_implied_pruned_fraction": pruning_stat(report, "mask_implied_pruned_fraction"),
        "total_parameters": pruning_stat(report, "total_parameters"),
        "masked_weight_violation_count": pruning_stat(report, "masked_weight_violation_count"),
    }


def summary_row(
    method: str,
    phase: str,
    checkpoint: Path,
    eval_dir: Path,
    status: str,
    error: str = "",
    pruning_report_path: Path | None = None,
    dense_exact_match: float | None = None,
    eval_spec: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result_eval_dir = resolve_eval_result_dir(eval_dir)
    summary = read_eval_summary(eval_dir) if status == "ok" else {}
    run_config = read_eval_run_config(eval_dir)
    pruning_stats = read_pruning_stats(pruning_report_path)
    exact_match = metric(summary, "exact_match_accuracy")
    eval_name = str(eval_spec.get("name")) if eval_spec else "eval"
    eval_file = str(eval_spec.get("path")) if eval_spec else ""
    if status == "ok" and exact_match is None:
        status = "failed"
        detail = f"Eval summary is missing exact_match_accuracy: {result_eval_dir}"
        error = f"{error}; {detail}" if error else detail
    delta = None
    if exact_match is not None and dense_exact_match is not None:
        delta = float(exact_match) - float(dense_exact_match)
    checkpoint_evaluated = run_config.get("checkpoint_path_used_for_evaluation") or run_config.get("model_path") or str(checkpoint)
    return {
        "method": method,
        "phase": phase,
        "status": status,
        "checkpoint": str(checkpoint),
        "checkpoint_evaluated": str(checkpoint_evaluated),
        "eval_name": eval_name,
        "eval_file": eval_file,
        "eval_dir": str(result_eval_dir),
        "eval_output_dir": str(result_eval_dir),
        "pruning_report": str(pruning_report_path or ""),
        **pruning_stats,
        "exact_match_accuracy_mean": metric(summary, "exact_match_accuracy"),
        "exact_match_accuracy": metric(summary, "exact_match_accuracy"),
        "exact_match_accuracy_std": metric_std(summary, "exact_match_accuracy"),
        "exact_match_at_top_k_accuracy_mean": metric(summary, "exact_match_at_top_k_accuracy"),
        "exact_match_at_top_k_accuracy": metric(summary, "exact_match_at_top_k_accuracy"),
        "exact_match_at_top_k_accuracy_std": metric_std(summary, "exact_match_at_top_k_accuracy"),
        "exact_match_at_top_k_correct": metric_count(summary, "exact_match_at_top_k_correct"),
        "exact_match_at_5_accuracy": metric(summary, "exact_match_at_5_accuracy"),
        "top5_exact_match_accuracy": metric(summary, "top5_exact_match_accuracy"),
        **difficulty_summary_metrics(summary),
        "correct_examples": metric_count(summary, "correct_examples", "exact_match_correct"),
        "total_examples": metric_count(summary, "total_examples"),
        "delta_vs_dense_exact_match": delta,
        "mean_response_loss_mean": metric(summary, "mean_response_loss"),
        "mean_response_loss": metric(summary, "mean_response_loss"),
        "mean_response_loss_std": metric_std(summary, "mean_response_loss"),
        "response_perplexity_mean": metric(summary, "response_perplexity"),
        "response_perplexity": metric(summary, "response_perplexity"),
        "response_perplexity_std": metric_std(summary, "response_perplexity"),
        "avg_generated_tokens_mean": metric(summary, "avg_generated_tokens"),
        "avg_generated_tokens": metric(summary, "avg_generated_tokens"),
        "avg_generated_tokens_std": metric_std(summary, "avg_generated_tokens"),
        "error": error,
    }


def write_summary(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    pruning_rows = [row for row in rows if row.get("phase") in PRUNING_ROW_PHASES]
    dense_rows = [row for row in rows if row.get("phase") == "dense_baseline"]
    write_json(output_dir / "pruning_benchmark_summary.json", {"results": pruning_rows, "dense_baseline": dense_rows})
    write_json(output_dir / "benchmark_summary.json", {"results": rows})
    one_shot_rows = [row for row in pruning_rows if row.get("phase") == "one_shot"]
    retuned_rows = [row for row in pruning_rows if row.get("phase") == "retuned"]
    write_json(output_dir / "dense_baseline_summary.json", {"results": dense_rows})
    write_json(output_dir / "benchmark_summary_one_shot.json", {"results": one_shot_rows})
    write_json(output_dir / "benchmark_summary_retuned.json", {"results": retuned_rows})
    csv_path = output_dir / "pruning_benchmark_summary.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    benchmark_fieldnames = [
        "method",
        "phase",
        "status",
        "checkpoint",
        "checkpoint_evaluated",
        "eval_name",
        "eval_file",
        "eval_dir",
        "eval_output_dir",
        "pruning_report",
        "active_model_parameters",
        "real_sparsity",
        "target_prunable_sparsity",
        "target_sparsity_denominator",
        "target_whole_model_sparsity",
        "achieved_prunable_sparsity",
        "achieved_whole_model_sparsity",
        "prunable_parameter_count",
        "protected_parameter_count",
        "total_parameter_count",
        "active_prunable_parameters",
        "pruned_prunable_parameters",
        "total_prunable_parameters",
        "mask_sparsity",
        "mask_parameter_count",
        "active_mask_parameters",
        "pruned_mask_parameters",
        "active_mask_fraction",
        "mask_implied_active_parameters",
        "mask_implied_pruned_parameters",
        "mask_implied_active_fraction",
        "mask_implied_pruned_fraction",
        "total_parameters",
        "nonzero_parameters",
        "zero_parameters",
        "nonzero_fraction",
        "masked_weight_violation_count",
        "exact_match_accuracy_mean",
        "exact_match_accuracy",
        "exact_match_accuracy_std",
        "exact_match_at_top_k_accuracy_mean",
        "exact_match_at_top_k_accuracy",
        "exact_match_at_top_k_accuracy_std",
        "exact_match_at_top_k_correct",
        "exact_match_at_5_accuracy",
        "top5_exact_match_accuracy",
        "difficulty_easy_total_examples",
        "difficulty_easy_exact_match_correct",
        "difficulty_easy_exact_match_accuracy",
        "difficulty_easy_exact_match_at_top_k_correct",
        "difficulty_easy_exact_match_at_top_k_accuracy",
        "difficulty_easy_exact_match_at_5_correct",
        "difficulty_easy_exact_match_at_5_accuracy",
        "difficulty_easy_top5_exact_match_accuracy",
        "difficulty_medium_total_examples",
        "difficulty_medium_exact_match_correct",
        "difficulty_medium_exact_match_accuracy",
        "difficulty_medium_exact_match_at_top_k_correct",
        "difficulty_medium_exact_match_at_top_k_accuracy",
        "difficulty_medium_exact_match_at_5_correct",
        "difficulty_medium_exact_match_at_5_accuracy",
        "difficulty_medium_top5_exact_match_accuracy",
        "difficulty_hard_total_examples",
        "difficulty_hard_exact_match_correct",
        "difficulty_hard_exact_match_accuracy",
        "difficulty_hard_exact_match_at_top_k_correct",
        "difficulty_hard_exact_match_at_top_k_accuracy",
        "difficulty_hard_exact_match_at_5_correct",
        "difficulty_hard_exact_match_at_5_accuracy",
        "difficulty_hard_top5_exact_match_accuracy",
        "correct_examples",
        "total_examples",
        "delta_vs_dense_exact_match",
        "mean_response_loss_mean",
        "mean_response_loss",
        "mean_response_loss_std",
        "response_perplexity_mean",
        "response_perplexity",
        "response_perplexity_std",
        "avg_generated_tokens_mean",
        "avg_generated_tokens",
        "avg_generated_tokens_std",
        "error",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PRUNING_SUMMARY_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(pruning_rows)
    with (output_dir / "benchmark_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=benchmark_fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "benchmark_summary_one_shot.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=benchmark_fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(one_shot_rows)
    with (output_dir / "benchmark_summary_retuned.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=benchmark_fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(retuned_rows)


def latest_checkpoint_or_default(output_dir: Path) -> Path:
    try:
        return latest_checkpoint(output_dir)
    except FileNotFoundError:
        return output_dir / "final"


def default_phase_paths(
    output_dir: Path,
    method: str,
    phase: str,
    eval_spec: dict[str, Any] | None = None,
    multi_eval: bool = False,
) -> tuple[Path, Path, Path | None]:
    slug = method_slug(method)
    if phase == "one_shot":
        checkpoint = output_dir / "one_shot" / slug
        eval_dir = eval_dir_for_phase(output_dir, phase, method, eval_spec or {"slug": "eval"}, multi_eval)
    elif phase == "retuned":
        checkpoint = latest_checkpoint_or_default(output_dir / "retuned" / slug)
        eval_dir = eval_dir_for_phase(output_dir, phase, method, eval_spec or {"slug": "eval"}, multi_eval)
    else:
        checkpoint = output_dir
        eval_dir = output_dir
    report_path = checkpoint / "pruning_report.json"
    return checkpoint, eval_dir, report_path if report_path.exists() else None


def ensure_expected_pruning_rows(
    rows: list[dict[str, Any]],
    methods: list[str],
    output_dir: Path,
    retune_enabled: bool,
    dense_exact_matches: dict[str, float | None] | float | None = None,
    eval_specs: list[dict[str, Any]] | None = None,
    multi_eval: bool = False,
    dense_exact_match: float | None = None,
) -> None:
    if dense_exact_matches is None:
        dense_exact_matches = dense_exact_match
    eval_specs = eval_specs or [{"name": "eval", "slug": "eval", "path": ""}]
    expected_phases = ["one_shot"]
    if retune_enabled:
        expected_phases.append("retuned")
    present = {
        (str(row.get("method")), str(row.get("phase")), str(row.get("eval_name", "eval")))
        for row in rows
        if row.get("phase") in PRUNING_ROW_PHASES
    }
    for method in methods:
        for phase in expected_phases:
            for eval_spec in eval_specs:
                eval_name = str(eval_spec.get("name", "eval"))
                if (method, phase, eval_name) in present:
                    continue
                checkpoint, eval_dir, report_path = default_phase_paths(output_dir, method, phase, eval_spec, multi_eval)
                dense_exact_match = (
                    dense_exact_matches.get(eval_name)
                    if isinstance(dense_exact_matches, dict)
                    else dense_exact_matches
                )
                rows.append(
                    summary_row(
                        method=method,
                        phase=phase,
                        checkpoint=checkpoint,
                        eval_dir=eval_dir,
                        status="missing",
                        error=f"{phase} row was expected by the pruning benchmark plan but no completed result was recorded.",
                        pruning_report_path=report_path,
                        dense_exact_match=dense_exact_match,
                        eval_spec=eval_spec,
                    )
                )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one-shot and SFT-retuned pruning benchmarks for all methods.")
    parser.add_argument("--config", default="configs/pruning_benchmark.yaml", help="YAML file, or a directory containing pruning_benchmark.yaml.")
    parser.add_argument("--eval-file", default=None, help="Override benchmark.eval_file with your prompt/response eval dataset.")
    parser.add_argument("--methods", default=None, help="Override benchmark.methods, e.g. 'wanda' or 'magnitude,wanda'.")
    parser.add_argument("--dry-run", action="store_true")
    errors = parser.add_mutually_exclusive_group()
    errors.add_argument("--continue-on-error", action="store_true", help="Record a failed method and continue with the next method.")
    errors.add_argument("--stop-on-error", action="store_true", help="Stop immediately when a method fails.")
    args = parser.parse_args()

    config_path = resolve_config_path(args.config)
    config = load_yaml(config_path)
    benchmark = dict(config.get("benchmark", {}) or {})
    one_shot = dict(config.get("one_shot", {}) or {})
    retune = dict(config.get("retune", {}) or {})
    methods = parse_methods(args.methods if args.methods else benchmark.get("methods", METHODS))
    unknown_methods = [method for method in methods if method not in METHODS]
    if unknown_methods:
        raise ValueError(f"Unknown pruning methods: {unknown_methods}. Expected any of {METHODS}.")

    output_dir = as_path(benchmark.get("output_dir", "runs/pruning-benchmark-0p2b"))
    generated_config_dir = output_dir / "generated_configs"
    base_checkpoint_value = benchmark.get("base_checkpoint", config.get("prune", {}).get("base_model"))
    if not base_checkpoint_value:
        raise ValueError("Set benchmark.base_checkpoint.")
    base_checkpoint = as_path(base_checkpoint_value)
    eval_specs = eval_specs_from_benchmark(benchmark, args.eval_file)
    multi_eval = len(eval_specs) > 1

    prune_config_path = as_path(benchmark.get("prune_config", "configs/prune_50.yaml"))
    base_prune_config = load_yaml(prune_config_path)
    env = command_env(benchmark)
    continue_on_error = should_continue_on_error(args, benchmark)
    rows: list[dict[str, Any]] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    validate_gpu_launch_config(benchmark, strict=not args.dry_run)
    validate_benchmark_paths(methods, base_checkpoint, eval_specs, prune_config_path, config, retune, strict=not args.dry_run)
    print_plan(
        methods=methods,
        output_dir=output_dir,
        base_checkpoint=base_checkpoint,
        eval_specs=eval_specs,
        prune_config_path=prune_config_path,
        config=config,
        benchmark=benchmark,
        retune=retune,
        continue_on_error=continue_on_error,
    )
    dense_exact_matches: dict[str, float | None] = {}
    dense_eval_dirs: dict[str, Path] = {}
    if bool(benchmark.get("run_dense_baseline", True)):
        for eval_spec in eval_specs:
            eval_name = str(eval_spec["name"])
            eval_file = Path(eval_spec["path"])
            dense_eval_dir = eval_dir_for_phase(output_dir, "dense_baseline", None, eval_spec, multi_eval)
            dense_eval_dirs[eval_name] = dense_eval_dir
            run_eval(base_checkpoint, eval_file, dense_eval_dir, benchmark=benchmark, env=env, dry_run=args.dry_run)
            if not args.dry_run:
                dense_summary = read_eval_summary(dense_eval_dir)
                if not dense_summary:
                    raise FileNotFoundError(f"Dense baseline evaluation summary missing: {dense_eval_dir}")
                dense_exact_value = metric(dense_summary, "exact_match_accuracy")
                if dense_exact_value is None:
                    raise ValueError(f"Dense baseline exact-match accuracy missing: {dense_eval_dir}")
                dense_exact_matches[eval_name] = float(dense_exact_value)
                min_dense = benchmark.get("min_dense_exact_match_accuracy", 0.01)
                if min_dense is not None and dense_exact_matches[eval_name] < float(min_dense):
                    raise RuntimeError(
                        f"Dense baseline exact-match accuracy for {eval_name} "
                        f"{dense_exact_matches[eval_name]:.6f} is below "
                        f"benchmark.min_dense_exact_match_accuracy={float(min_dense):.6f}; debug eval before pruning."
                    )
                dense_payload = {
                    "checkpoint_path": str(base_checkpoint),
                    "tokenizer_path": str(base_checkpoint),
                    "dataset_path": str(eval_file),
                    "benchmark_split": str(eval_file),
                    "eval_name": eval_name,
                    "generation_config": {
                        "max_new_tokens": int(benchmark.get("max_new_tokens", 64)),
                        "num_beams": 1,
                        "temperature": 0,
                        "do_sample": False,
                        "exact_match_top_k": int(benchmark.get("top_k_exact_match", benchmark.get("exact_match_top_k", 5))),
                    },
                    "exact_match_normalization": {"comparison_mode": benchmark.get("comparison_mode", "whitespace")},
                    **dense_summary,
                }
                dense_payload_path = output_dir / (
                    "dense_baseline_eval.json" if not multi_eval else f"dense_baseline_eval_{eval_spec['slug']}.json"
                )
                write_json(dense_payload_path, dense_payload)
                rows.append(
                    dense_baseline_row(
                        base_checkpoint,
                        dense_eval_dir,
                        dense_summary,
                        eval_spec=eval_spec,
                    )
                )
    elif not args.dry_run:
        raise RuntimeError("Dense baseline evaluation is required for pruning benchmark comparison.")

    for method in methods:
        print(f"\n=== Running pruning method: {method} ===", flush=True)
        slug = method_slug(method)
        one_shot_dir = output_dir / "one_shot" / slug
        retuned_dir = output_dir / "retuned" / slug
        one_shot_completed = False
        if bool(one_shot.get("enabled", True)):
            completed_one_shot_eval_names: set[str] = set()
            try:
                prune_config = generated_prune_config(
                    base_config=base_prune_config,
                    benchmark_config=config,
                    method=method,
                    checkpoint=base_checkpoint,
                    output_dir=one_shot_dir,
                    recovery_steps=0,
                )
                generated_one_shot_config_path = generated_config_dir / f"prune_{slug}_one_shot.yaml"
                write_yaml(generated_one_shot_config_path, prune_config)
                run_prune(method, base_checkpoint, one_shot_dir, generated_one_shot_config_path, env=env, dry_run=args.dry_run)
                if not args.dry_run:
                    validate_saved_checkpoint_path(one_shot_dir, f"{method} one-shot pruned", base_checkpoint)
                    write_checkpoint_pointer(output_dir, method, "one_shot", one_shot_dir)
                    validate_pruning_report(
                        one_shot_dir / "pruning_report.json",
                        method=method,
                        phase="one_shot",
                        target_sparsity=float(prune_config["prune"].get("sparsity", 0.5)),
                        tolerance=float(benchmark.get("sparsity_tolerance", 1e-6)),
                    )
                    if one_shot_dir.resolve() == base_checkpoint.resolve():
                        raise RuntimeError("Refusing to evaluate dense checkpoint as one-shot pruned checkpoint.")
                for eval_spec in eval_specs:
                    eval_name = str(eval_spec["name"])
                    eval_file = Path(eval_spec["path"])
                    one_shot_eval_dir = eval_dir_for_phase(output_dir, "one_shot", method, eval_spec, multi_eval)
                    run_eval(one_shot_dir, eval_file, one_shot_eval_dir, benchmark=benchmark, env=env, dry_run=args.dry_run)
                    if not args.dry_run:
                        validate_eval_checkpoint(one_shot_eval_dir, one_shot_dir)
                        validate_eval_protocol_matches_dense(dense_eval_dirs[eval_name], one_shot_eval_dir)
                    rows.append(
                        summary_row(
                            method,
                            "one_shot",
                            one_shot_dir,
                            one_shot_eval_dir,
                            "ok",
                            pruning_report_path=one_shot_dir / "pruning_report.json",
                            dense_exact_match=dense_exact_matches.get(eval_name),
                            eval_spec=eval_spec,
                        )
                    )
                    completed_one_shot_eval_names.add(eval_name)
                one_shot_completed = True
            except Exception as exc:
                for eval_spec in eval_specs:
                    eval_name = str(eval_spec.get("name", "eval"))
                    if eval_name in completed_one_shot_eval_names:
                        continue
                    one_shot_eval_dir = eval_dir_for_phase(output_dir, "one_shot", method, eval_spec, multi_eval)
                    rows.append(
                        summary_row(
                            method,
                            "one_shot",
                            one_shot_dir,
                            one_shot_eval_dir,
                            "failed",
                            error=str(exc),
                            pruning_report_path=one_shot_dir / "pruning_report.json",
                            dense_exact_match=dense_exact_matches.get(eval_name),
                            eval_spec=eval_spec,
                        )
                    )
                write_summary(output_dir, rows)
                if not continue_on_error:
                    raise
        else:
            for eval_spec in eval_specs:
                eval_name = str(eval_spec.get("name", "eval"))
                one_shot_eval_dir = eval_dir_for_phase(output_dir, "one_shot", method, eval_spec, multi_eval)
                rows.append(
                    summary_row(
                        method,
                        "one_shot",
                        one_shot_dir,
                        one_shot_eval_dir,
                        "missing",
                        error="one_shot.enabled is false; one-shot pruning result was not produced.",
                        pruning_report_path=one_shot_dir / "pruning_report.json",
                        dense_exact_match=dense_exact_matches.get(eval_name),
                        eval_spec=eval_spec,
                    )
                )

        if bool(retune.get("enabled", True)):
            if not one_shot_completed:
                for eval_spec in eval_specs:
                    eval_name = str(eval_spec.get("name", "eval"))
                    retuned_eval_dir = eval_dir_for_phase(output_dir, "retuned", method, eval_spec, multi_eval)
                    rows.append(
                        summary_row(
                            method,
                            "retuned",
                            retuned_dir / "final",
                            retuned_eval_dir,
                            "missing",
                            error=f"retune.enabled=true but retune was skipped because {method} one-shot did not complete.",
                            pruning_report_path=None,
                            dense_exact_match=dense_exact_matches.get(eval_name),
                            eval_spec=eval_spec,
                        )
                    )
                write_summary(output_dir, rows)
                continue
            completed_retuned_eval_names: set[str] = set()
            try:
                masks_path = one_shot_dir / "pruning_masks.pt"
                if not args.dry_run and not masks_path.exists():
                    raise FileNotFoundError(f"Missing pruning mask for retune: {masks_path}")
                run_retune(
                    pruned_checkpoint=one_shot_dir,
                    masks_path=masks_path,
                    output_dir=retuned_dir,
                    retune=retune,
                    benchmark=benchmark,
                    env=env,
                    dry_run=args.dry_run,
                )
                retuned_checkpoint = latest_checkpoint(retuned_dir) if not args.dry_run else retuned_dir / "final"
                retuned_report = retuned_checkpoint / "pruning_report.json"
                if not args.dry_run:
                    validate_saved_checkpoint_path(retuned_checkpoint, f"{method} retuned pruned", base_checkpoint)
                    write_checkpoint_pointer(output_dir, method, "retuned", retuned_checkpoint)
                    validate_pruning_report(
                        retuned_report,
                        method=method,
                        phase="retuned",
                        target_sparsity=float(config.get("prune", {}).get("sparsity", 0.5)),
                        tolerance=float(benchmark.get("sparsity_tolerance", 1e-6)),
                    )
                    if retuned_checkpoint.resolve() == base_checkpoint.resolve():
                        raise RuntimeError("Refusing to evaluate dense checkpoint as retuned pruned checkpoint.")
                if args.dry_run or not retuned_report.exists():
                    retuned_report = one_shot_dir / "pruning_report.json"
                for eval_spec in eval_specs:
                    eval_name = str(eval_spec["name"])
                    eval_file = Path(eval_spec["path"])
                    retuned_eval_dir = eval_dir_for_phase(output_dir, "retuned", method, eval_spec, multi_eval)
                    run_eval(retuned_checkpoint, eval_file, retuned_eval_dir, benchmark=benchmark, env=env, dry_run=args.dry_run)
                    if not args.dry_run:
                        validate_eval_checkpoint(retuned_eval_dir, retuned_checkpoint)
                        validate_eval_protocol_matches_dense(dense_eval_dirs[eval_name], retuned_eval_dir)
                    rows.append(
                        summary_row(
                            method,
                            "retuned",
                            retuned_checkpoint,
                            retuned_eval_dir,
                            "ok",
                            pruning_report_path=retuned_report,
                            dense_exact_match=dense_exact_matches.get(eval_name),
                            eval_spec=eval_spec,
                        )
                    )
                    completed_retuned_eval_names.add(eval_name)
            except Exception as exc:
                retuned_checkpoint = latest_checkpoint_or_default(retuned_dir)
                retuned_report = retuned_checkpoint / "pruning_report.json"
                for eval_spec in eval_specs:
                    eval_name = str(eval_spec.get("name", "eval"))
                    if eval_name in completed_retuned_eval_names:
                        continue
                    retuned_eval_dir = eval_dir_for_phase(output_dir, "retuned", method, eval_spec, multi_eval)
                    retuned_status = "failed" if has_eval_result(resolve_eval_result_dir(retuned_eval_dir)) else "missing"
                    rows.append(
                        summary_row(
                            method,
                            "retuned",
                            retuned_checkpoint,
                            retuned_eval_dir,
                            retuned_status,
                            error=f"retune.enabled=true but retuned checkpoint/eval was not completed: {exc}",
                            pruning_report_path=retuned_report if retuned_report.exists() else None,
                            dense_exact_match=dense_exact_matches.get(eval_name),
                            eval_spec=eval_spec,
                        )
                    )
                write_summary(output_dir, rows)
                if not continue_on_error:
                    raise
    ensure_expected_pruning_rows(
        rows=rows,
        methods=methods,
        output_dir=output_dir,
        retune_enabled=bool(retune.get("enabled", True)),
        dense_exact_matches=dense_exact_matches,
        eval_specs=eval_specs,
        multi_eval=multi_eval,
    )
    write_summary(output_dir, rows)
    print(f"\nWrote pruning benchmark summary to {output_dir / 'pruning_benchmark_summary.csv'}")
    print(f"Wrote pruning benchmark summary to {output_dir / 'pruning_benchmark_summary.json'}")


if __name__ == "__main__":
    main()
