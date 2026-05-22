#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import copy
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METHODS = ("magnitude", "2of4", "wanda", "gradient")
EVAL_SUMMARY_FILENAMES = (
    "prompt_response_eval_benchmark_summary.json",
    "prompt_response_eval_summary.json",
    "metrics.json",
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
    "exact_match_accuracy",
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


def stable_config_hash(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def method_slug(method: str) -> str:
    return "nvidia-2of4" if method == "2of4" else method


def as_path(value: Any) -> Path:
    return Path(str(value)).expanduser()


def display_path(path: Path) -> str:
    try:
        return str(path.resolve())
    except OSError:
        return str(path)


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
    eval_file: Path,
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
    if not eval_file.exists():
        problems.append(f"benchmark.eval_file does not exist: {eval_file}")
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
    eval_file: Path,
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
    print(f"  eval_file: {display_path(eval_file)}", flush=True)
    print(f"  prune_config: {display_path(prune_config_path)}", flush=True)
    print(f"  calibration_data_path: {config.get('prune', {}).get('calibration_data_path')}", flush=True)
    print(f"  target_sparsity: {config.get('prune', {}).get('sparsity', 0.5)}", flush=True)
    print(f"  sparsity_denominator: {config.get('prune', {}).get('sparsity_denominator', 'prunable')}", flush=True)
    print(f"  benchmark_runs: {benchmark.get('benchmark_runs', 1)}", flush=True)
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
        ],
        env=env,
        dry_run=dry_run,
    )


def dense_baseline_row(checkpoint: Path, eval_dir: Path, summary: dict[str, Any], comparable: bool, issues: list[str]) -> dict[str, Any]:
    issue_text = "; ".join(issues)
    if issues:
        issue_text = f"not directly comparable to CMC0.2B yet: {issue_text}"
    result_eval_dir = resolve_eval_result_dir(eval_dir)
    return {
        "method": "dense_sft_baseline",
        "phase": "dense_baseline",
        "status": "ok",
        "checkpoint": str(checkpoint),
        "checkpoint_evaluated": str(checkpoint),
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
        "cmc_comparable": comparable,
        "blocking_comparability_issues": issue_text,
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
    cmc_comparable: bool = False,
    comparability_issues: list[str] | None = None,
) -> dict[str, Any]:
    result_eval_dir = resolve_eval_result_dir(eval_dir)
    summary = read_eval_summary(eval_dir) if status == "ok" else {}
    run_config = read_eval_run_config(eval_dir)
    pruning_stats = read_pruning_stats(pruning_report_path)
    exact_match = metric(summary, "exact_match_accuracy")
    if status == "ok" and exact_match is None:
        status = "failed"
        detail = f"Eval summary is missing exact_match_accuracy: {result_eval_dir}"
        error = f"{error}; {detail}" if error else detail
    delta = None
    if exact_match is not None and dense_exact_match is not None:
        delta = float(exact_match) - float(dense_exact_match)
    issue_text = "; ".join(comparability_issues or [])
    if comparability_issues:
        issue_text = f"not directly comparable to CMC0.2B yet: {issue_text}"
    checkpoint_evaluated = run_config.get("checkpoint_path_used_for_evaluation") or run_config.get("model_path") or str(checkpoint)
    return {
        "method": method,
        "phase": phase,
        "status": status,
        "checkpoint": str(checkpoint),
        "checkpoint_evaluated": str(checkpoint_evaluated),
        "eval_dir": str(result_eval_dir),
        "eval_output_dir": str(result_eval_dir),
        "pruning_report": str(pruning_report_path or ""),
        **pruning_stats,
        "exact_match_accuracy_mean": metric(summary, "exact_match_accuracy"),
        "exact_match_accuracy": metric(summary, "exact_match_accuracy"),
        "exact_match_accuracy_std": metric_std(summary, "exact_match_accuracy"),
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
        "cmc_comparable": cmc_comparable,
        "blocking_comparability_issues": issue_text,
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
        "cmc_comparable",
        "blocking_comparability_issues",
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


def cmc_comparability_report(
    config: dict[str, Any],
    benchmark: dict[str, Any],
    methods: list[str],
    dense_checkpoint: Path,
    pruned_checkpoint_paths: list[str],
) -> dict[str, Any]:
    blocking: list[str] = []
    non_blocking: list[str] = []
    if set(methods) != set(METHODS):
        blocking.append(f"methods differ from CMC reference: {methods}")
    denominator = str(config.get("prune", {}).get("sparsity_denominator", "prunable")).lower()
    if denominator not in {"prunable", "mask", "prunable_weights", "prunable_linear", "prunable_linears"}:
        blocking.append(
            "target sparsity denominator is whole-model; this is not the CMC0.2B 50% prunable-linear protocol"
        )
    elif float(config.get("prune", {}).get("sparsity", 0.5)) != 0.5:
        blocking.append("target prunable sparsity is not 0.5")
    if bool(config.get("prune", {}).get("include_lm_head", False)):
        blocking.append("lm_head/output head pruning is enabled")
    if int(benchmark.get("benchmark_runs", 1)) != 1:
        non_blocking.append("benchmark repeats are enabled; CMC one-shot comparison should use one run per checkpoint")
    if not benchmark.get("eval_file"):
        blocking.append("benchmark eval_file is missing")
    if not config.get("prune", {}).get("calibration_data_path") and any(method in {"wanda", "gradient"} for method in methods):
        blocking.append("calibration_data_path is missing for Wanda/gradient")
    retune_enabled = bool(config.get("retune", {}).get("enabled", False))
    if retune_enabled:
        non_blocking.append("retuned phase is enabled; keep it separate from primary CMC one-shot comparison")
    return {
        "comparable": not blocking,
        "blocking_issues": blocking,
        "non_blocking_differences": non_blocking,
        "decoder_only_prunable_scope": "selected transformer attention/MLP torch.nn.Linear weights; embeddings/lm_head/norms/biases/non-Linears protected",
        "cmc_prunable_scope": "selected prunable transformer Linear weights; embeddings/output head/norms/tokenizer/non-Linears protected",
        "dense_baseline_checkpoint": str(dense_checkpoint),
        "pruned_checkpoint_paths": pruned_checkpoint_paths,
        "benchmark_config_hash": stable_config_hash(benchmark),
        "tokenizer_config_hash": stable_config_hash({"tokenizer_path": str(dense_checkpoint)}),
        "eval_config_hash": stable_config_hash(
            {
                "eval_file": benchmark.get("eval_file"),
                "max_new_tokens": benchmark.get("max_new_tokens", 64),
                "max_length": benchmark.get("max_length", benchmark.get("max_seq_length", 2048)),
                "eval_batch_size": benchmark.get("eval_batch_size", 16),
                "dtype": benchmark.get("dtype", "bf16"),
                "comparison_mode": "whitespace",
                "decoding": {"do_sample": False, "num_beams": 1, "temperature": 0},
                "response_extraction": "decoder-only prompt/response exact-match evaluator",
            }
        ),
    }


def latest_checkpoint_or_default(output_dir: Path) -> Path:
    try:
        return latest_checkpoint(output_dir)
    except FileNotFoundError:
        return output_dir / "final"


def default_phase_paths(output_dir: Path, method: str, phase: str) -> tuple[Path, Path, Path | None]:
    slug = method_slug(method)
    if phase == "one_shot":
        checkpoint = output_dir / "one_shot" / slug
        eval_dir = output_dir / "benchmarks" / "one_shot" / slug
    elif phase == "retuned":
        checkpoint = latest_checkpoint_or_default(output_dir / "retuned" / slug)
        eval_dir = output_dir / "benchmarks" / "retuned" / slug
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
    dense_exact_match: float | None,
    cmc_comparable: bool,
    comparability_issues: list[str],
) -> None:
    expected_phases = ["one_shot"]
    if retune_enabled:
        expected_phases.append("retuned")
    present = {
        (str(row.get("method")), str(row.get("phase")))
        for row in rows
        if row.get("phase") in PRUNING_ROW_PHASES
    }
    for method in methods:
        for phase in expected_phases:
            if (method, phase) in present:
                continue
            checkpoint, eval_dir, report_path = default_phase_paths(output_dir, method, phase)
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
                    cmc_comparable=cmc_comparable if phase == "one_shot" else False,
                    comparability_issues=comparability_issues
                    if phase == "one_shot"
                    else ["retuned phase is post-pruning SFT and must not be mixed into primary CMC one-shot table"],
                )
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one-shot and SFT-retuned pruning benchmarks for all methods.")
    parser.add_argument("--config", default="configs/pruning_benchmark.yaml", help="YAML file, or a directory containing pruning_benchmark.yaml.")
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
    methods = [str(method) for method in benchmark.get("methods", METHODS)]
    unknown_methods = [method for method in methods if method not in METHODS]
    if unknown_methods:
        raise ValueError(f"Unknown pruning methods: {unknown_methods}. Expected any of {METHODS}.")

    output_dir = as_path(benchmark.get("output_dir", "runs/pruning-benchmark-0p2b"))
    generated_config_dir = output_dir / "generated_configs"
    base_checkpoint_value = benchmark.get("base_checkpoint", config.get("prune", {}).get("base_model"))
    eval_file_value = benchmark.get("eval_file")
    if not base_checkpoint_value:
        raise ValueError("Set benchmark.base_checkpoint.")
    if not eval_file_value:
        raise ValueError("Set benchmark.eval_file.")
    base_checkpoint = as_path(base_checkpoint_value)
    eval_file = as_path(eval_file_value)

    prune_config_path = as_path(benchmark.get("prune_config", "configs/prune_50.yaml"))
    base_prune_config = load_yaml(prune_config_path)
    env = command_env(benchmark)
    continue_on_error = should_continue_on_error(args, benchmark)
    rows: list[dict[str, Any]] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    validate_gpu_launch_config(benchmark, strict=not args.dry_run)
    validate_benchmark_paths(methods, base_checkpoint, eval_file, prune_config_path, config, retune, strict=not args.dry_run)
    print_plan(
        methods=methods,
        output_dir=output_dir,
        base_checkpoint=base_checkpoint,
        eval_file=eval_file,
        prune_config_path=prune_config_path,
        config=config,
        benchmark=benchmark,
        retune=retune,
        continue_on_error=continue_on_error,
    )
    pruned_checkpoint_paths: list[str] = []
    comparability = cmc_comparability_report(config, benchmark, methods, base_checkpoint, pruned_checkpoint_paths)
    write_json(output_dir / "cmc_comparability_report.json", comparability)
    dense_exact_match: float | None = None
    dense_eval_dir = output_dir / "benchmarks" / "dense_baseline"
    if bool(benchmark.get("run_dense_baseline", True)):
        run_eval(base_checkpoint, eval_file, dense_eval_dir, benchmark=benchmark, env=env, dry_run=args.dry_run)
        if not args.dry_run:
            dense_summary = read_eval_summary(dense_eval_dir)
            if not dense_summary:
                raise FileNotFoundError(f"Dense baseline evaluation summary missing: {dense_eval_dir}")
            dense_exact_value = metric(dense_summary, "exact_match_accuracy")
            if dense_exact_value is None:
                raise ValueError(f"Dense baseline exact-match accuracy missing: {dense_eval_dir}")
            dense_exact_match = float(dense_exact_value)
            min_dense = benchmark.get("min_dense_exact_match_accuracy", 0.01)
            if min_dense is not None and dense_exact_match < float(min_dense):
                raise RuntimeError(
                    f"Dense baseline exact-match accuracy {dense_exact_match:.6f} is below "
                    f"benchmark.min_dense_exact_match_accuracy={float(min_dense):.6f}; debug eval before pruning."
                )
            dense_payload = {
                "checkpoint_path": str(base_checkpoint),
                "tokenizer_path": str(base_checkpoint),
                "dataset_path": str(eval_file),
                "benchmark_split": str(eval_file),
                "generation_config": {
                    "max_new_tokens": int(benchmark.get("max_new_tokens", 64)),
                    "num_beams": 1,
                    "temperature": 0,
                    "do_sample": False,
                },
                "exact_match_normalization": {"comparison_mode": "whitespace"},
                **dense_summary,
            }
            write_json(output_dir / "dense_baseline_eval.json", dense_payload)
            rows.append(
                dense_baseline_row(
                    base_checkpoint,
                    dense_eval_dir,
                    dense_summary,
                    comparable=bool(comparability["comparable"]),
                    issues=list(comparability["blocking_issues"]),
                )
            )
    elif not args.dry_run:
        raise RuntimeError("Dense baseline evaluation is required for CMC-compatible pruning comparison.")

    for method in methods:
        print(f"\n=== Running pruning method: {method} ===", flush=True)
        slug = method_slug(method)
        one_shot_dir = output_dir / "one_shot" / slug
        one_shot_eval_dir = output_dir / "benchmarks" / "one_shot" / slug
        retuned_dir = output_dir / "retuned" / slug
        retuned_eval_dir = output_dir / "benchmarks" / "retuned" / slug
        one_shot_completed = False
        if bool(one_shot.get("enabled", True)):
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
                    validate_pruning_report(
                        one_shot_dir / "pruning_report.json",
                        method=method,
                        phase="one_shot",
                        target_sparsity=float(prune_config["prune"].get("sparsity", 0.5)),
                        tolerance=float(benchmark.get("sparsity_tolerance", 1e-6)),
                    )
                    if one_shot_dir.resolve() == base_checkpoint.resolve():
                        raise RuntimeError("Refusing to evaluate dense checkpoint as one-shot pruned checkpoint.")
                run_eval(one_shot_dir, eval_file, one_shot_eval_dir, benchmark=benchmark, env=env, dry_run=args.dry_run)
                if not args.dry_run:
                    validate_eval_checkpoint(one_shot_eval_dir, one_shot_dir)
                    validate_eval_protocol_matches_dense(dense_eval_dir, one_shot_eval_dir)
                pruned_checkpoint_paths.append(str(one_shot_dir))
                rows.append(
                    summary_row(
                        method,
                        "one_shot",
                        one_shot_dir,
                        one_shot_eval_dir,
                        "ok",
                        pruning_report_path=one_shot_dir / "pruning_report.json",
                        dense_exact_match=dense_exact_match,
                        cmc_comparable=bool(comparability["comparable"]),
                        comparability_issues=list(comparability["blocking_issues"]),
                    )
                )
                one_shot_completed = True
            except Exception as exc:
                rows.append(
                    summary_row(
                        method,
                        "one_shot",
                        one_shot_dir,
                        one_shot_eval_dir,
                        "failed",
                        error=str(exc),
                        pruning_report_path=one_shot_dir / "pruning_report.json",
                        dense_exact_match=dense_exact_match,
                        cmc_comparable=bool(comparability["comparable"]),
                        comparability_issues=list(comparability["blocking_issues"]),
                    )
                )
                write_summary(output_dir, rows)
                if not continue_on_error:
                    raise
        else:
            rows.append(
                summary_row(
                    method,
                    "one_shot",
                    one_shot_dir,
                    one_shot_eval_dir,
                    "missing",
                    error="one_shot.enabled is false; one-shot pruning result was not produced.",
                    pruning_report_path=one_shot_dir / "pruning_report.json",
                    dense_exact_match=dense_exact_match,
                    cmc_comparable=bool(comparability["comparable"]),
                    comparability_issues=list(comparability["blocking_issues"]),
                )
            )

        if bool(retune.get("enabled", True)):
            if not one_shot_completed:
                rows.append(
                    summary_row(
                        method,
                        "retuned",
                        retuned_dir / "final",
                        retuned_eval_dir,
                        "missing",
                        error=f"retune.enabled=true but retune was skipped because {method} one-shot did not complete.",
                        pruning_report_path=None,
                        dense_exact_match=dense_exact_match,
                        cmc_comparable=False,
                        comparability_issues=[
                            "retuned phase is post-pruning SFT and must not be mixed into primary CMC one-shot table"
                        ],
                    )
                )
                write_summary(output_dir, rows)
                continue
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
                    validate_pruning_report(
                        retuned_report,
                        method=method,
                        phase="retuned",
                        target_sparsity=float(config.get("prune", {}).get("sparsity", 0.5)),
                        tolerance=float(benchmark.get("sparsity_tolerance", 1e-6)),
                    )
                    if retuned_checkpoint.resolve() == base_checkpoint.resolve():
                        raise RuntimeError("Refusing to evaluate dense checkpoint as retuned pruned checkpoint.")
                run_eval(retuned_checkpoint, eval_file, retuned_eval_dir, benchmark=benchmark, env=env, dry_run=args.dry_run)
                if not args.dry_run:
                    validate_eval_checkpoint(retuned_eval_dir, retuned_checkpoint)
                    validate_eval_protocol_matches_dense(dense_eval_dir, retuned_eval_dir)
                pruned_checkpoint_paths.append(str(retuned_checkpoint))
                if args.dry_run or not retuned_report.exists():
                    retuned_report = one_shot_dir / "pruning_report.json"
                rows.append(
                    summary_row(
                        method,
                        "retuned",
                        retuned_checkpoint,
                        retuned_eval_dir,
                        "ok",
                        pruning_report_path=retuned_report,
                        dense_exact_match=dense_exact_match,
                        cmc_comparable=False,
                        comparability_issues=[
                            "retuned phase is post-pruning SFT and must not be mixed into primary CMC one-shot table"
                        ],
                    )
                )
            except Exception as exc:
                retuned_checkpoint = latest_checkpoint_or_default(retuned_dir)
                retuned_report = retuned_checkpoint / "pruning_report.json"
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
                        dense_exact_match=dense_exact_match,
                        cmc_comparable=False,
                        comparability_issues=[
                            "retuned phase is post-pruning SFT and must not be mixed into primary CMC one-shot table"
                        ],
                    )
                )
                write_summary(output_dir, rows)
                if not continue_on_error:
                    raise
    comparability = cmc_comparability_report(config, benchmark, methods, base_checkpoint, pruned_checkpoint_paths)
    write_json(output_dir / "cmc_comparability_report.json", comparability)
    ensure_expected_pruning_rows(
        rows=rows,
        methods=methods,
        output_dir=output_dir,
        retune_enabled=bool(retune.get("enabled", True)),
        dense_exact_match=dense_exact_match,
        cmc_comparable=bool(comparability["comparable"]),
        comparability_issues=list(comparability["blocking_issues"]),
    )
    write_summary(output_dir, rows)
    print(f"\nWrote pruning benchmark summary to {output_dir / 'pruning_benchmark_summary.csv'}")
    print(f"Wrote pruning benchmark summary to {output_dir / 'pruning_benchmark_summary.json'}")


if __name__ == "__main__":
    main()
