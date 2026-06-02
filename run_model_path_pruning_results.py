#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - the pruning runner itself still needs PyYAML.
    yaml = None


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_TRAINING_DATASET = "data/scenic/SCENIC_full_training_dataset.json"
DEFAULT_BENCHMARK_DATASET = "data/benchmarks/iot_instruction_benchmark_200.json"
DEFAULT_METHODS = ("wanda", "magnitude", "gradient", "nvidia")
DEFAULT_PRUNE_FAMILIES = ("base_model", "sft", "contrastive")
EVAL_NAMES = ("training_dataset", "benchmark")
METHOD_ALIASES = {
    "2:4": "2of4",
    "2-of-4": "2of4",
    "2_of_4": "2of4",
    "2of4": "2of4",
    "nvidia": "2of4",
    "nvidia-2of4": "2of4",
    "nvidia_2of4": "2of4",
    "magnitude": "magnitude",
    "wanda": "wanda",
    "gradient": "gradient",
}
FAMILY_ALIASES = {
    "base": "base_model",
    "base_model": "base_model",
    "original": "base_model",
    "original_decoder": "base_model",
    "sft": "sft",
    "regular": "sft",
    "regular_sft": "sft",
    "contrastive": "contrastive",
    "contrastive_sft": "contrastive",
}


@dataclass(frozen=True)
class ModelFamily:
    key: str
    checkpoint: str
    output_dir: Path


@dataclass(frozen=True)
class RunSettings:
    base_model_path: str
    sft_model_path: str
    contrastive_model_path: str
    training_dataset: Path
    benchmark_dataset: Path
    calibration_dataset: Path
    run_root: Path
    generated_config_dir: Path
    results_json: Path
    prune_config: Path
    methods: tuple[str, ...]
    prune_families: tuple[str, ...]
    python: str
    cuda_visible_devices: str
    nproc_per_node: int
    omp_num_threads: int
    eval_runs: int
    top_k_exact_match: int
    comparison_mode: str
    dtype: str
    max_length: int
    max_new_tokens: int
    max_new_token_hit_rate_threshold: float
    eval_batch_size: int
    sparsity: float
    pruning_scope: str
    sparsity_denominator: str
    granularity: str
    include_lm_head: bool
    calibration_batches: int
    prune_batch_size: int
    prune_num_workers: int
    sparsity_tolerance: float
    keep_going: bool
    dry_run: bool


def repo_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def resolve_executable(value: str | None) -> str:
    if not value:
        return sys.executable
    if value.startswith(("~", "/", "./", "../")):
        return str(repo_path(value))
    return value


def resolve_model_reference(value: str) -> str:
    """Keep HF model ids intact while making local paths absolute."""
    text = str(value).strip()
    if not text:
        raise ValueError("Model path cannot be empty.")
    path = Path(text).expanduser()
    project_candidate = PROJECT_ROOT / path
    looks_local = (
        path.is_absolute()
        or text.startswith(("./", "../", "~"))
        or path.exists()
        or project_candidate.exists()
    )
    if not looks_local:
        return text
    candidate = path if path.is_absolute() else project_candidate
    return str(candidate.resolve() if candidate.exists() else candidate)


def parse_words(values: Any) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        raw = values.replace(",", " ").split()
    else:
        raw = []
        for value in values:
            raw.extend(str(value).replace(",", " ").split())
    return [part.strip() for part in raw if part.strip()]


def normalize_methods(values: Any) -> tuple[str, ...]:
    methods: list[str] = []
    for value in parse_words(values):
        key = value.lower()
        if key not in METHOD_ALIASES:
            raise ValueError(f"Unknown pruning method {value!r}. Use wanda, magnitude, gradient, or nvidia.")
        method = METHOD_ALIASES[key]
        if method not in methods:
            methods.append(method)
    if not methods:
        raise ValueError("At least one pruning method is required.")
    return tuple(methods)


def normalize_families(values: Any) -> tuple[str, ...]:
    families: list[str] = []
    for value in parse_words(values):
        key = value.lower()
        if key in {"none", "no", "false", "0"}:
            continue
        if key not in FAMILY_ALIASES:
            raise ValueError(f"Unknown model family {value!r}. Use base_model, sft, or contrastive.")
        family = FAMILY_ALIASES[key]
        if family not in families:
            families.append(family)
    return tuple(families)


def display_method(method: str) -> str:
    return "nvidia" if method == "2of4" else method


def bool_from_text(value: str | bool | None) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        if yaml is None:
            json.dump(json_ready(payload), handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            return
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(json_ready(payload), handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [json_ready(item) for item in value]
    return value


def safe_float(value: Any) -> Any:
    if value in (None, ""):
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return value
    return number if math.isfinite(number) else ""


def metric(row: dict[str, Any], *names: str) -> Any:
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            return value
    return ""


def command_env(settings: RunSettings) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(settings.cuda_visible_devices)
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["NCCL_DEBUG"] = str(env.get("NCCL_DEBUG") or "WARN")
    env["PYTORCH_CUDA_ALLOC_CONF"] = str(env.get("PYTORCH_CUDA_ALLOC_CONF") or "expandable_segments:True")
    env["OMP_NUM_THREADS"] = str(settings.omp_num_threads)
    env["PYTHON"] = str(settings.python)
    python_path = Path(settings.python)
    if (python_path.is_absolute() or "/" in str(settings.python)) and python_path.parent.exists():
        env["PATH"] = str(python_path.parent) + os.pathsep + env.get("PATH", "")
    return env


def run_command(cmd: list[str], *, env: dict[str, str], dry_run: bool) -> None:
    print("\n+ " + shlex.join(str(part) for part in cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, check=True)


def model_families(settings: RunSettings) -> tuple[ModelFamily, ...]:
    return (
        ModelFamily("base_model", settings.base_model_path, settings.run_root / "base_model"),
        ModelFamily("sft", settings.sft_model_path, settings.run_root / "sft"),
        ModelFamily("contrastive", settings.contrastive_model_path, settings.run_root / "contrastive"),
    )


def family_display_name(key: str) -> str:
    if key == "base_model":
        return "base_model"
    return key


def benchmark_config_for_family(settings: RunSettings, family: ModelFamily, methods: tuple[str, ...]) -> dict[str, Any]:
    return {
        "benchmark": {
            "output_dir": str(family.output_dir),
            "base_checkpoint": family.checkpoint,
            "prune_config": str(settings.prune_config),
            "methods": list(methods),
            "eval_files": {
                "training_dataset": str(settings.training_dataset),
                "benchmark": str(settings.benchmark_dataset),
            },
            "run_dense_baseline": True,
            "min_dense_exact_match_accuracy": None,
            "benchmark_runs": int(settings.eval_runs),
            "top_k_exact_match": int(settings.top_k_exact_match),
            "comparison_mode": str(settings.comparison_mode),
            "max_new_tokens": int(settings.max_new_tokens),
            "max_new_token_hit_rate_threshold": float(settings.max_new_token_hit_rate_threshold),
            "max_length": int(settings.max_length),
            "eval_batch_size": int(settings.eval_batch_size),
            "dtype": str(settings.dtype),
            "nproc_per_node": int(settings.nproc_per_node),
            "expected_gpu_count": int(settings.nproc_per_node),
            "cuda_visible_devices": str(settings.cuda_visible_devices),
            "sparsity_tolerance": float(settings.sparsity_tolerance),
            "continue_on_error": bool(settings.keep_going),
        },
        "prune": {
            "sparsity": float(settings.sparsity),
            "scope": str(settings.pruning_scope),
            "sparsity_denominator": str(settings.sparsity_denominator),
            "granularity": str(settings.granularity),
            "include_lm_head": bool(settings.include_lm_head),
            "calibration_data_path": str(settings.calibration_dataset),
            "calibration_batches": int(settings.calibration_batches),
            "max_length": int(settings.max_length),
            "batch_size": int(settings.prune_batch_size),
            "num_workers": int(settings.prune_num_workers),
        },
        "one_shot": {"enabled": bool(methods)},
        "retune": {"enabled": False},
        "_source": "run_model_path_pruning_results.py",
        "_model_family": family.key,
    }


def write_generated_configs(settings: RunSettings) -> dict[str, Path]:
    settings.generated_config_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    prune_family_set = set(settings.prune_families)
    for family in model_families(settings):
        methods = settings.methods if family.key in prune_family_set else tuple()
        path = settings.generated_config_dir / f"{family.key}_pruning_eval.yaml"
        write_yaml(path, benchmark_config_for_family(settings, family, methods))
        paths[family.key] = path
    return paths


def run_pruning_benchmark(config_path: Path, settings: RunSettings, env: dict[str, str], methods: tuple[str, ...]) -> None:
    cmd = [
        str(settings.python),
        "scripts/run_pruning_benchmark.py",
        "--config",
        str(config_path),
        "--methods",
        " ".join(methods),
    ]
    cmd.append("--continue-on-error" if settings.keep_going else "--stop-on-error")
    if settings.dry_run:
        cmd.append("--dry-run")
    run_command(cmd, env=env, dry_run=False)


def read_pruning_summary(output_dir: Path) -> dict[str, Any]:
    summary_path = output_dir / "pruning_benchmark_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing pruning benchmark summary: {summary_path}")
    return read_json(summary_path)


def rows_for_family(family: ModelFamily) -> list[dict[str, Any]]:
    payload = read_pruning_summary(family.output_dir)
    rows: list[dict[str, Any]] = []
    all_rows = list(payload.get("dense_baseline", []) or []) + list(payload.get("results", []) or [])
    for row in all_rows:
        phase = str(row.get("phase", ""))
        internal_method = "" if phase == "dense_baseline" else str(row.get("method", ""))
        rows.append(
            {
                "model_family": family_display_name(family.key),
                "model_path": family.checkpoint,
                "eval_name": row.get("eval_name", ""),
                "eval_file": row.get("eval_file", ""),
                "phase": phase,
                "method": "dense" if phase == "dense_baseline" else display_method(internal_method),
                "internal_method": internal_method,
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
                "target_sparsity": 0.0 if phase == "dense_baseline" else safe_float(metric(row, "target_whole_model_sparsity", "target_prunable_sparsity")),
                "achieved_whole_model_sparsity": safe_float(metric(row, "achieved_whole_model_sparsity", "real_sparsity")),
                "achieved_prunable_sparsity": safe_float(metric(row, "achieved_prunable_sparsity")),
                "checkpoint_evaluated": row.get("checkpoint_evaluated", row.get("checkpoint_path", "")),
                "eval_output_dir": row.get("eval_output_dir", ""),
                "pruning_report": row.get("pruning_report", ""),
                "error": row.get("error", ""),
            }
        )
    return rows


def expected_rows(settings: RunSettings) -> list[dict[str, str]]:
    expected: list[dict[str, str]] = []
    prune_family_set = set(settings.prune_families)
    for family in model_families(settings):
        for eval_name in EVAL_NAMES:
            expected.append(
                {
                    "model_family": family_display_name(family.key),
                    "eval_name": eval_name,
                    "phase": "dense_baseline",
                    "method": "dense",
                }
            )
            if family.key in prune_family_set:
                expected.extend(
                    {
                        "model_family": family_display_name(family.key),
                        "eval_name": eval_name,
                        "phase": "one_shot",
                        "method": display_method(method),
                    }
                    for method in settings.methods
                )
    return expected


def result_completeness(rows: list[dict[str, Any]], settings: RunSettings) -> dict[str, Any]:
    present = {
        (
            str(row.get("model_family")),
            str(row.get("eval_name")),
            str(row.get("phase")),
            str(row.get("method")),
        )
        for row in rows
    }
    expected = expected_rows(settings)
    missing = [
        row
        for row in expected
        if (row["model_family"], row["eval_name"], row["phase"], row["method"]) not in present
    ]
    failed = [
        {
            "model_family": row.get("model_family", ""),
            "eval_name": row.get("eval_name", ""),
            "phase": row.get("phase", ""),
            "method": row.get("method", ""),
            "error": row.get("error", ""),
        }
        for row in rows
        if row.get("status") not in {"ok", ""}
    ]
    return {
        "expected_eval_names": list(EVAL_NAMES),
        "expected_rows": expected,
        "missing": missing,
        "failed_or_missing_status": failed,
        "complete": not missing,
        "all_status_ok": not failed and not missing,
    }


def write_results_json(settings: RunSettings, generated_configs: dict[str, Path]) -> None:
    families = model_families(settings)
    rows: list[dict[str, Any]] = []
    raw_summaries: dict[str, Any] = {}
    for family in families:
        rows.extend(rows_for_family(family))
        raw_summaries[family.key] = read_pruning_summary(family.output_dir)
    payload = {
        "schema_version": 1,
        "run": {
            "methods": [display_method(method) for method in settings.methods],
            "internal_methods": list(settings.methods),
            "prune_families": [family_display_name(family) for family in settings.prune_families],
            "eval_runs": settings.eval_runs,
            "top_k_exact_match": settings.top_k_exact_match,
            "comparison_mode": settings.comparison_mode,
            "dtype": settings.dtype,
            "cuda_visible_devices": settings.cuda_visible_devices,
            "nproc_per_node": settings.nproc_per_node,
            "run_root": settings.run_root,
            "results_json": settings.results_json,
            "generated_configs": generated_configs,
        },
        "datasets": {
            "training_dataset": settings.training_dataset,
            "benchmark": settings.benchmark_dataset,
            "calibration": settings.calibration_dataset,
        },
        "checkpoints": {
            "base_model": settings.base_model_path,
            "sft": settings.sft_model_path,
            "contrastive": settings.contrastive_model_path,
        },
        "pruning": {
            "sparsity": settings.sparsity,
            "scope": settings.pruning_scope,
            "sparsity_denominator": settings.sparsity_denominator,
            "granularity": settings.granularity,
            "include_lm_head": settings.include_lm_head,
            "calibration_batches": settings.calibration_batches,
        },
        "checks": result_completeness(rows, settings),
        "results": rows,
        "raw_benchmark_summaries": raw_summaries,
    }
    write_json(settings.results_json, payload)
    print(f"\nWrote consolidated JSON results: {settings.results_json}", flush=True)


def print_plan(settings: RunSettings) -> None:
    print("Model-path EM@1/EM@5 + 50% one-shot pruning benchmark", flush=True)
    print(f"  base model:            {settings.base_model_path}", flush=True)
    print(f"  sft model:             {settings.sft_model_path}", flush=True)
    print(f"  contrastive model:     {settings.contrastive_model_path}", flush=True)
    print(f"  training dataset:      {settings.training_dataset}", flush=True)
    print(f"  benchmark dataset:     {settings.benchmark_dataset}", flush=True)
    print(f"  calibration dataset:   {settings.calibration_dataset}", flush=True)
    print(f"  methods:               {' '.join(display_method(method) for method in settings.methods)}", flush=True)
    print(f"  internal methods:      {' '.join(settings.methods)}", flush=True)
    print(f"  prune families:        {' '.join(settings.prune_families) or 'none'}", flush=True)
    print(f"  run root:              {settings.run_root}", flush=True)
    print(f"  results JSON:          {settings.results_json}", flush=True)
    print(f"  CUDA_VISIBLE_DEVICES:  {settings.cuda_visible_devices}", flush=True)
    print(f"  nproc_per_node:        {settings.nproc_per_node}", flush=True)
    print(f"  Python:                {settings.python}", flush=True)


def run_model_path_pruning_results(settings: RunSettings) -> Path:
    os.chdir(PROJECT_ROOT)
    env = command_env(settings)
    settings.run_root.mkdir(parents=True, exist_ok=True)
    print_plan(settings)
    generated_configs = write_generated_configs(settings)
    print("\nGenerated benchmark configs:", flush=True)
    for family, path in generated_configs.items():
        print(f"  {family}: {path}", flush=True)

    prune_family_set = set(settings.prune_families)
    for family in model_families(settings):
        methods = settings.methods if family.key in prune_family_set else tuple()
        run_pruning_benchmark(generated_configs[family.key], settings, env, methods)

    if settings.dry_run:
        print("\nDry run complete. Generated configs and pruning-run dry summaries, but no consolidated results JSON.", flush=True)
        return settings.results_json

    write_results_json(settings, generated_configs)
    return settings.results_json


def build_settings(args: argparse.Namespace) -> RunSettings:
    run_root = repo_path(args.run_root)
    results_json = repo_path(args.results_json) if args.results_json else run_root / "model_path_pruning_results.json"
    training_dataset = repo_path(args.training_dataset)
    benchmark_dataset = repo_path(args.benchmark_dataset)
    calibration_dataset = repo_path(args.calibration_dataset) if args.calibration_dataset else training_dataset
    return RunSettings(
        base_model_path=resolve_model_reference(args.base_model_path),
        sft_model_path=resolve_model_reference(args.sft_model_path),
        contrastive_model_path=resolve_model_reference(args.contrastive_model_path),
        training_dataset=training_dataset,
        benchmark_dataset=benchmark_dataset,
        calibration_dataset=calibration_dataset,
        run_root=run_root,
        generated_config_dir=repo_path(args.generated_config_dir) if args.generated_config_dir else run_root / "generated_configs",
        results_json=results_json,
        prune_config=repo_path(args.prune_config),
        methods=normalize_methods(args.methods or DEFAULT_METHODS),
        prune_families=normalize_families(args.prune_families or DEFAULT_PRUNE_FAMILIES),
        python=resolve_executable(args.python),
        cuda_visible_devices=str(args.cuda_visible_devices),
        nproc_per_node=int(args.nproc_per_node),
        omp_num_threads=int(args.omp_num_threads),
        eval_runs=int(args.eval_runs),
        top_k_exact_match=int(args.top_k_exact_match),
        comparison_mode=str(args.comparison_mode),
        dtype=str(args.dtype),
        max_length=int(args.max_length),
        max_new_tokens=int(args.max_new_tokens),
        max_new_token_hit_rate_threshold=float(args.max_new_token_hit_rate_threshold),
        eval_batch_size=int(args.eval_batch_size),
        sparsity=float(args.sparsity),
        pruning_scope=str(args.pruning_scope),
        sparsity_denominator=str(args.sparsity_denominator),
        granularity=str(args.granularity),
        include_lm_head=bool_from_text(args.include_lm_head),
        calibration_batches=int(args.calibration_batches),
        prune_batch_size=int(args.prune_batch_size),
        prune_num_workers=int(args.prune_num_workers),
        sparsity_tolerance=float(args.sparsity_tolerance),
        keep_going=bool(args.keep_going),
        dry_run=bool(args.dry_run),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate base/SFT/contrastive checkpoints on training and benchmark datasets, "
            "run 50% one-shot pruning, and consolidate EM@1/EM@5 into one JSON."
        )
    )
    parser.add_argument("--base-model-path", default=os.environ.get("BASE_MODEL_PATH"), required=os.environ.get("BASE_MODEL_PATH") is None)
    parser.add_argument("--sft-model-path", default=os.environ.get("SFT_MODEL_PATH"), required=os.environ.get("SFT_MODEL_PATH") is None)
    parser.add_argument(
        "--contrastive-model-path",
        default=os.environ.get("CONTRASTIVE_MODEL_PATH"),
        required=os.environ.get("CONTRASTIVE_MODEL_PATH") is None,
    )
    parser.add_argument("--training-dataset", default=os.environ.get("TRAINING_DATASET", DEFAULT_TRAINING_DATASET))
    parser.add_argument("--benchmark-dataset", default=os.environ.get("BENCHMARK_DATASET", DEFAULT_BENCHMARK_DATASET))
    parser.add_argument("--calibration-dataset", default=os.environ.get("CALIBRATION_DATASET"))
    parser.add_argument("--run-root", default=os.environ.get("RUN_ROOT", "runs/model-path-pruning-results"))
    parser.add_argument("--generated-config-dir", default=os.environ.get("GENERATED_CONFIG_DIR"))
    parser.add_argument("--results-json", default=os.environ.get("RESULTS_JSON"))
    parser.add_argument("--prune-config", default=os.environ.get("PRUNE_CONFIG", "configs/prune_50.yaml"))
    parser.add_argument("--methods", nargs="+", default=os.environ.get("METHODS"))
    parser.add_argument("--prune-families", nargs="+", default=os.environ.get("PRUNE_FAMILIES"))
    parser.add_argument("--python", default=os.environ.get("PYTHON"))
    parser.add_argument("--cuda-visible-devices", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7"))
    parser.add_argument("--nproc-per-node", type=int, default=int(os.environ.get("NPROC_PER_NODE", "8")))
    parser.add_argument("--omp-num-threads", type=int, default=int(os.environ.get("OMP_NUM_THREADS", "8")))
    parser.add_argument("--eval-runs", type=int, default=int(os.environ.get("EVAL_RUNS", "1")))
    parser.add_argument("--top-k-exact-match", type=int, default=int(os.environ.get("TOP_K_EXACT_MATCH", "5")))
    parser.add_argument("--comparison-mode", choices=("whitespace", "normalized", "command"), default=os.environ.get("COMPARISON_MODE", "whitespace"))
    parser.add_argument("--dtype", choices=("auto", "bf16", "fp16", "fp32"), default=os.environ.get("DTYPE", "bf16"))
    parser.add_argument("--max-length", type=int, default=int(os.environ.get("MAX_LENGTH", "128")))
    parser.add_argument("--max-new-tokens", type=int, default=int(os.environ.get("MAX_NEW_TOKENS", "64")))
    parser.add_argument(
        "--max-new-token-hit-rate-threshold",
        type=float,
        default=float(os.environ.get("MAX_NEW_TOKEN_HIT_RATE_THRESHOLD", "1.01")),
        help="Pass through to eval_prompt_response.py; values above 1 keep high length-cap rates as diagnostics instead of hard errors.",
    )
    parser.add_argument("--eval-batch-size", type=int, default=int(os.environ.get("EVAL_BATCH_SIZE", "16")))
    parser.add_argument("--sparsity", type=float, default=float(os.environ.get("SPARSITY", "0.5")))
    parser.add_argument("--pruning-scope", default=os.environ.get("PRUNING_SCOPE", "transformer_linears"))
    parser.add_argument("--sparsity-denominator", default=os.environ.get("SPARSITY_DENOMINATOR", "whole_model"))
    parser.add_argument("--granularity", choices=("layer", "global"), default=os.environ.get("GRANULARITY", "layer"))
    parser.add_argument("--include-lm-head", default=os.environ.get("INCLUDE_LM_HEAD", "false"))
    parser.add_argument("--calibration-batches", type=int, default=int(os.environ.get("CALIBRATION_BATCHES", "128")))
    parser.add_argument("--prune-batch-size", type=int, default=int(os.environ.get("PRUNE_BATCH_SIZE", "2")))
    parser.add_argument("--prune-num-workers", type=int, default=int(os.environ.get("PRUNE_NUM_WORKERS", "0")))
    parser.add_argument("--sparsity-tolerance", type=float, default=float(os.environ.get("SPARSITY_TOLERANCE", "0.001")))
    parser.add_argument("--dry-run", action="store_true", default=bool_from_text(os.environ.get("DRY_RUN")))
    errors = parser.add_mutually_exclusive_group()
    errors.add_argument("--continue-on-error", dest="keep_going", action="store_true", default=not bool_from_text(os.environ.get("STOP_ON_ERROR")))
    errors.add_argument("--stop-on-error", dest="keep_going", action="store_false")
    parser.add_argument("--print-config", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = build_settings(args)
    if args.print_config:
        print(json.dumps(json_ready(settings.__dict__), indent=2, ensure_ascii=False))
        return
    run_model_path_pruning_results(settings)


if __name__ == "__main__":
    main()
