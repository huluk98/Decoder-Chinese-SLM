#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import copy
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METHODS = ("magnitude", "2of4", "wanda", "gradient")


def resolve_config_path(path: str | Path) -> Path:
    config_path = Path(path).expanduser()
    if config_path.is_dir():
        config_path = config_path / "qwen25_instruct_pruning_benchmark.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Qwen pruning benchmark config not found: {config_path}")
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


def as_path(value: Any) -> Path:
    return Path(str(value)).expanduser()


def display_path(path: Path) -> str:
    try:
        return str(path.resolve())
    except OSError:
        return str(path)


def method_slug(method: str) -> str:
    return "nvidia-2of4" if method == "2of4" else method


def latest_checkpoint(output_dir: Path) -> Path:
    final = output_dir / "final"
    if final.exists():
        return final
    if output_dir.exists() and (output_dir / "config.json").exists():
        return output_dir
    raise FileNotFoundError(f"No Qwen final checkpoint found in {output_dir}")


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
    actual_sparsity = report.get("mask_sparsity", report.get("sparsity"))
    if actual_sparsity is None:
        raise ValueError(f"Pruning report does not contain mask_sparsity/sparsity: {report_path}")
    if abs(float(actual_sparsity) - float(target_sparsity)) > float(tolerance):
        raise ValueError(
            f"{method} {phase} sparsity check failed: actual={float(actual_sparsity):.8f}, "
            f"target={float(target_sparsity):.8f}, tolerance={float(tolerance):.8f}"
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


def validate_benchmark_paths(
    methods: list[str],
    base_checkpoint: Path,
    eval_file: Path,
    train_file: Path | None,
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
    if train_file is not None and not train_file.exists():
        problems.append(f"benchmark.train_file does not exist: {train_file}")
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
        message = "Invalid Qwen pruning benchmark YAML:\n  - " + "\n  - ".join(problems)
        if strict:
            raise FileNotFoundError(message)
        print("Warning: " + message.replace("\n", "\nWarning: "), flush=True)


def print_plan(
    methods: list[str],
    output_dir: Path,
    base_checkpoint: Path,
    eval_file: Path,
    train_file: Path | None,
    prune_config_path: Path,
    config: dict[str, Any],
    benchmark: dict[str, Any],
    retune: dict[str, Any],
    continue_on_error: bool,
) -> None:
    print("\nResolved Qwen2.5-Instruct pruning benchmark plan:", flush=True)
    print(f"  methods: {', '.join(methods)}", flush=True)
    print(f"  output_dir: {display_path(output_dir)}", flush=True)
    print(f"  base_checkpoint: {display_path(base_checkpoint)}", flush=True)
    print(f"  eval_file: {display_path(eval_file)}", flush=True)
    print(f"  train_file: {display_path(train_file) if train_file is not None else None}", flush=True)
    print(f"  prune_config: {display_path(prune_config_path)}", flush=True)
    print(f"  calibration_data_path: {config.get('prune', {}).get('calibration_data_path')}", flush=True)
    print(f"  target_sparsity: {config.get('prune', {}).get('sparsity', 0.5)}", flush=True)
    print(f"  benchmark_runs: {benchmark.get('benchmark_runs', 1)}", flush=True)
    print(f"  system_prompt: {benchmark.get('system_prompt')}", flush=True)
    print(f"  retune.enabled: {bool(retune.get('enabled', True))}", flush=True)
    print(f"  retune.data_path: {retune.get('data_path')}", flush=True)
    print(f"  retune.max_steps: {retune.get('max_steps')}", flush=True)
    print(f"  retune.epochs: {retune.get('epochs')}", flush=True)
    print(f"  prune runner: scripts/prune_qwen25_instruct.py", flush=True)
    print(f"  eval runner: scripts/eval_qwen25_instruct.py", flush=True)
    print(f"  retune runner: scripts/sft_qwen25_instruct.py", flush=True)
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
    benchmark = dict(benchmark_config.get("benchmark", {}) or {})
    config["model_name_or_path"] = str(checkpoint)
    config.setdefault("system_prompt", benchmark.get("system_prompt"))
    config.setdefault("max_seq_length", benchmark.get("max_seq_length", 256))
    config.setdefault("train_file", benchmark_config.get("prune", {}).get("calibration_data_path"))
    config.setdefault("prune", {})
    config["prune"].update(dict(benchmark_config.get("prune", {}) or {}))
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
    benchmark: dict[str, Any],
    env: dict[str, str],
    dry_run: bool,
) -> None:
    cmd = [
        sys.executable,
        "scripts/prune_qwen25_instruct.py",
        "--config",
        str(config_path),
        "--method",
        method,
        "--checkpoint",
        str(checkpoint),
        "--output-dir",
        str(output_dir),
    ]
    if benchmark.get("prune_dtype"):
        cmd.extend(["--dtype", str(benchmark.get("prune_dtype"))])
    run_command(cmd, env=env, dry_run=dry_run)


def run_eval(
    model_path: Path,
    eval_file: Path,
    output_dir: Path,
    benchmark: dict[str, Any],
    env: dict[str, str],
    dry_run: bool,
    train_file: Path | None = None,
) -> None:
    nproc = int(benchmark.get("nproc_per_node", 8))
    cmd = [
        "torchrun",
        "--standalone",
        "--nproc_per_node",
        str(nproc),
        "scripts/eval_qwen25_instruct.py",
        "--model-path",
        str(model_path),
        "--dataset-file",
        str(eval_file),
        "--output-dir",
        str(output_dir),
        "--max-new-tokens",
        str(int(benchmark.get("max_new_tokens", 64))),
        "--batch-size",
        str(int(benchmark.get("eval_batch_size", 16))),
        "--dtype",
        str(benchmark.get("dtype", "bf16")),
        "--benchmark-runs",
        str(int(benchmark.get("benchmark_runs", 1))),
    ]
    if benchmark.get("system_prompt"):
        cmd.extend(["--system-prompt", str(benchmark["system_prompt"])])
    if train_file is not None and train_file.exists() and train_file != eval_file:
        cmd.extend(["--train-file", str(train_file)])
    run_command(cmd, env=env, dry_run=dry_run)


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
        "scripts/sft_qwen25_instruct.py",
        "--config",
        str(retune.get("config", "configs/sft_qwen25_0p5b_instruct.yaml")),
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
        cmd.extend(["--max-steps", str(int(retune["max_steps"]))])
    elif retune.get("epochs") is not None:
        cmd.extend(["--epochs", str(float(retune["epochs"]))])
    if retune.get("per_device_train_batch_size") is not None:
        cmd.extend(["--per-device-train-batch-size", str(int(retune["per_device_train_batch_size"]))])
    if retune.get("gradient_accumulation_steps") is not None:
        cmd.extend(["--gradient-accumulation-steps", str(int(retune["gradient_accumulation_steps"]))])
    if retune.get("max_seq_length") is not None:
        cmd.extend(["--max-seq-length", str(int(retune["max_seq_length"]))])
    run_command(cmd, env=env, dry_run=dry_run)


def read_eval_summary(eval_dir: Path) -> dict[str, Any]:
    summary_path = eval_dir / "qwen25_instruct_eval_summary.json"
    if not summary_path.exists():
        return {}
    with summary_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_pruning_report(report_path: Path | None) -> dict[str, Any]:
    if report_path is None or not report_path.exists():
        return {}
    with report_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def metric(summary: dict[str, Any], name: str) -> Any:
    if f"{name}_mean" in summary:
        return summary.get(f"{name}_mean")
    return summary.get(name)


def metric_std(summary: dict[str, Any], name: str) -> Any:
    return summary.get(f"{name}_std")


def summary_row(
    method: str,
    phase: str,
    checkpoint: Path,
    eval_dir: Path,
    status: str,
    error: str = "",
    pruning_report_path: Path | None = None,
) -> dict[str, Any]:
    summary = read_eval_summary(eval_dir) if status == "ok" else {}
    pruning_report = read_pruning_report(pruning_report_path)
    return {
        "method": method,
        "phase": phase,
        "status": status,
        "checkpoint": str(checkpoint),
        "eval_dir": str(eval_dir),
        "pruning_report": str(pruning_report_path or ""),
        "mask_sparsity": pruning_report.get("mask_sparsity", pruning_report.get("sparsity")),
        "mask_parameter_count": pruning_report.get("mask_parameter_count"),
        "active_mask_parameters": pruning_report.get("active_mask_parameters"),
        "pruned_mask_parameters": pruning_report.get("pruned_mask_parameters"),
        "active_mask_fraction": pruning_report.get("active_mask_fraction"),
        "mask_implied_active_parameters": pruning_report.get("mask_implied_active_parameters"),
        "mask_implied_pruned_parameters": pruning_report.get("mask_implied_pruned_parameters"),
        "mask_implied_active_fraction": pruning_report.get("mask_implied_active_fraction"),
        "mask_implied_pruned_fraction": pruning_report.get("mask_implied_pruned_fraction"),
        "total_parameters": pruning_report.get("total_parameters"),
        "nonzero_parameters": pruning_report.get("nonzero_parameters"),
        "zero_parameters": pruning_report.get("zero_parameters"),
        "nonzero_fraction": pruning_report.get("nonzero_fraction"),
        "masked_weight_violation_count": pruning_report.get("masked_weight_violation_count"),
        "exact_match_accuracy_mean": metric(summary, "exact_match_accuracy"),
        "exact_match_accuracy_std": metric_std(summary, "exact_match_accuracy"),
        "command_exact_match_accuracy_mean": metric(summary, "command_exact_match_accuracy"),
        "command_exact_match_accuracy_std": metric_std(summary, "command_exact_match_accuracy"),
        "avg_generated_tokens_mean": metric(summary, "avg_generated_tokens"),
        "avg_generated_tokens_std": metric_std(summary, "avg_generated_tokens"),
        "error": error,
    }


def write_summary(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    write_json(output_dir / "qwen25_instruct_pruning_benchmark_summary.json", {"results": rows})
    csv_path = output_dir / "qwen25_instruct_pruning_benchmark_summary.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method",
        "phase",
        "status",
        "checkpoint",
        "eval_dir",
        "pruning_report",
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
        "exact_match_accuracy_std",
        "command_exact_match_accuracy_mean",
        "command_exact_match_accuracy_std",
        "avg_generated_tokens_mean",
        "avg_generated_tokens_std",
        "error",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Qwen2.5-Instruct one-shot and SFT-retuned pruning benchmarks.")
    parser.add_argument(
        "--config",
        default="configs/qwen25_instruct_pruning_benchmark.yaml",
        help="YAML file, or a directory containing qwen25_instruct_pruning_benchmark.yaml.",
    )
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

    output_dir = as_path(benchmark.get("output_dir", "runs/qwen25-instruct-pruning-benchmark"))
    generated_config_dir = output_dir / "generated_configs"
    base_checkpoint_value = benchmark.get("base_checkpoint", config.get("prune", {}).get("base_model"))
    eval_file_value = benchmark.get("eval_file")
    if not base_checkpoint_value:
        raise ValueError("Set benchmark.base_checkpoint.")
    if not eval_file_value:
        raise ValueError("Set benchmark.eval_file.")
    base_checkpoint = as_path(base_checkpoint_value)
    eval_file = as_path(eval_file_value)
    train_file = as_path(benchmark.get("train_file")) if benchmark.get("train_file") else None

    prune_config_path = as_path(benchmark.get("prune_config", "configs/prune_qwen25_50.yaml"))
    base_prune_config = load_yaml(prune_config_path)
    env = command_env(benchmark)
    continue_on_error = should_continue_on_error(args, benchmark)
    rows: list[dict[str, Any]] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    validate_gpu_launch_config(benchmark, strict=not args.dry_run)
    validate_benchmark_paths(
        methods,
        base_checkpoint,
        eval_file,
        train_file,
        prune_config_path,
        config,
        retune,
        strict=not args.dry_run,
    )
    print_plan(
        methods=methods,
        output_dir=output_dir,
        base_checkpoint=base_checkpoint,
        eval_file=eval_file,
        train_file=train_file,
        prune_config_path=prune_config_path,
        config=config,
        benchmark=benchmark,
        retune=retune,
        continue_on_error=continue_on_error,
    )

    for method in methods:
        print(f"\n=== Running Qwen pruning method: {method} ===", flush=True)
        slug = method_slug(method)
        one_shot_dir = output_dir / "one_shot" / slug
        one_shot_eval_dir = output_dir / "benchmarks" / "one_shot" / slug
        retuned_dir = output_dir / "retuned" / slug
        retuned_eval_dir = output_dir / "benchmarks" / "retuned" / slug
        try:
            if bool(one_shot.get("enabled", True)):
                prune_config = generated_prune_config(
                    base_config=base_prune_config,
                    benchmark_config=config,
                    method=method,
                    checkpoint=base_checkpoint,
                    output_dir=one_shot_dir,
                    recovery_steps=0,
                )
                generated_path = generated_config_dir / f"prune_qwen25_{slug}_one_shot.yaml"
                write_yaml(generated_path, prune_config)
                run_prune(
                    method,
                    base_checkpoint,
                    one_shot_dir,
                    generated_path,
                    benchmark=benchmark,
                    env=env,
                    dry_run=args.dry_run,
                )
                if not args.dry_run:
                    validate_pruning_report(
                        one_shot_dir / "pruning_report.json",
                        method=method,
                        phase="one_shot",
                        target_sparsity=float(prune_config["prune"].get("sparsity", 0.5)),
                        tolerance=float(benchmark.get("sparsity_tolerance", 1e-6)),
                    )
                run_eval(
                    one_shot_dir,
                    eval_file,
                    one_shot_eval_dir,
                    benchmark=benchmark,
                    env=env,
                    dry_run=args.dry_run,
                    train_file=train_file,
                )
                rows.append(
                    summary_row(
                        method,
                        "one_shot",
                        one_shot_dir,
                        one_shot_eval_dir,
                        "ok",
                        pruning_report_path=one_shot_dir / "pruning_report.json",
                    )
                )

            if bool(retune.get("enabled", True)):
                masks_path = one_shot_dir / "pruning_masks.pt"
                if not args.dry_run and not masks_path.exists():
                    raise FileNotFoundError(f"Missing pruning mask for Qwen retune: {masks_path}")
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
                        phase="retuned_qwen25_instruct_sft",
                        target_sparsity=float(config.get("prune", {}).get("sparsity", 0.5)),
                        tolerance=float(benchmark.get("sparsity_tolerance", 1e-6)),
                    )
                retune_train_file = as_path(retune["data_path"]) if retune.get("data_path") else train_file
                run_eval(
                    retuned_checkpoint,
                    eval_file,
                    retuned_eval_dir,
                    benchmark=benchmark,
                    env=env,
                    dry_run=args.dry_run,
                    train_file=retune_train_file,
                )
                if args.dry_run or not retuned_report.exists():
                    retuned_report = one_shot_dir / "pruning_report.json"
                rows.append(
                    summary_row(
                        method,
                        "retuned_qwen25_instruct_sft",
                        retuned_checkpoint,
                        retuned_eval_dir,
                        "ok",
                        pruning_report_path=retuned_report,
                    )
                )
        except subprocess.CalledProcessError as exc:
            rows.append(summary_row(method, "failed", output_dir, output_dir, "failed", error=str(exc)))
            write_summary(output_dir, rows)
            if not continue_on_error:
                raise
        except Exception as exc:
            rows.append(summary_row(method, "failed", output_dir, output_dir, "failed", error=str(exc)))
            write_summary(output_dir, rows)
            if not continue_on_error:
                raise
    write_summary(output_dir, rows)
    print(f"\nWrote Qwen pruning benchmark summary to {output_dir / 'qwen25_instruct_pruning_benchmark_summary.csv'}")
    print(f"Wrote Qwen pruning benchmark summary to {output_dir / 'qwen25_instruct_pruning_benchmark_summary.json'}")


if __name__ == "__main__":
    main()
