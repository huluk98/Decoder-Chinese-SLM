#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# ---------------------------------------------------------------------------
# EDIT THESE VALUES, THEN RUN:
#   bash scripts/run_sft_pruning_eval.sh
#
# Command-line args and environment variables still override these values.
# Example:
#   SCRIPT_MODEL_PATH="/models/my_dense_sft_checkpoint"
#   SCRIPT_DATA_FILE="/data/my_prompt_response_eval.json"
# ---------------------------------------------------------------------------
SCRIPT_MODEL_PATH=""
SCRIPT_DATA_FILE=""
SCRIPT_CALIBRATION_FILE=""  # optional; leave blank to reuse SCRIPT_DATA_FILE
SCRIPT_TRAIN_FILE=""        # optional; only used for leakage/split audit
SCRIPT_OUTPUT_DIR=""        # optional; blank creates runs/sft-pruning-eval-<timestamp>
SCRIPT_TEMPLATE_CONFIG="configs/prune_50.yaml"

SCRIPT_METHODS="magnitude 2of4 wanda gradient"
SCRIPT_NPROC="8"
SCRIPT_MAX_NEW_TOKENS="64"
SCRIPT_MAX_LENGTH="2048"
SCRIPT_EVAL_BATCH_SIZE="8"
SCRIPT_PRUNE_BATCH_SIZE="2"
SCRIPT_CALIBRATION_BATCHES="128"
SCRIPT_DTYPE="bf16"
SCRIPT_BENCHMARK_RUNS="1"
SCRIPT_COMPARISON_MODE="whitespace"

usage() {
  cat >&2 <<'EOF'
Run dense SFT eval, one-shot prune, reload/evaluate the pruned checkpoint, and print accuracy + pruning stats.

Usage:
  # Option A: edit SCRIPT_MODEL_PATH and SCRIPT_DATA_FILE at the top, then:
  bash scripts/run_sft_pruning_eval.sh

  # Option B: pass paths without editing:
  MODEL_PATH=/path/to/sft_or_hf_checkpoint DATA_FILE=/path/to/eval.json bash scripts/run_sft_pruning_eval.sh
  bash scripts/run_sft_pruning_eval.sh /path/to/sft_or_hf_checkpoint /path/to/eval.json

Common overrides:
  METHODS="magnitude 2of4 wanda gradient"
  CALIBRATION_FILE=/path/to/calibration_or_sft.jsonl  # defaults to DATA_FILE
  OUTPUT_DIR=runs/sft-pruning-eval
  NPROC=8
  BENCHMARK_RUNS=1
  COMPARISON_MODE=whitespace
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

MODEL_PATH="${MODEL_PATH:-${1:-${SCRIPT_MODEL_PATH}}}"
DATA_FILE="${DATA_FILE:-${EVAL_FILE:-${2:-${SCRIPT_DATA_FILE}}}}"
if [[ -z "${MODEL_PATH}" || -z "${DATA_FILE}" ]]; then
  usage
  exit 2
fi

if [[ -n "${PYTHON:-}" ]]; then
  PYTHON_BIN="${PYTHON}"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
else
  PYTHON_BIN="python"
fi

if ! "${PYTHON_BIN}" -c "import yaml" >/dev/null 2>&1; then
  echo "Python environment is missing PyYAML. Activate the training env, or set PYTHON=/path/to/env/bin/python." >&2
  exit 2
fi

resolve_file() {
  "${PYTHON_BIN}" - "$1" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1]).expanduser()
if not path.is_absolute():
    path = (Path.cwd() / path).resolve()
if not path.exists():
    raise SystemExit(f"File not found: {path}")
print(path)
PY
}

DATA_FILE="$(resolve_file "${DATA_FILE}")"
CALIBRATION_FILE="${CALIBRATION_FILE:-${SCRIPT_CALIBRATION_FILE}}"
if [[ -z "${CALIBRATION_FILE}" ]]; then
  CALIBRATION_FILE="${DATA_FILE}"
fi
CALIBRATION_FILE="$(resolve_file "${CALIBRATION_FILE}")"
TRAIN_FILE="${TRAIN_FILE:-${SCRIPT_TRAIN_FILE}}"
if [[ -n "${TRAIN_FILE}" ]]; then
  TRAIN_FILE="$(resolve_file "${TRAIN_FILE}")"
fi

METHODS_TEXT="${METHODS:-${SCRIPT_METHODS:-magnitude 2of4 wanda gradient}}"
METHODS_TEXT="${METHODS_TEXT//,/ }"
read -r -a METHOD_LIST <<<"${METHODS_TEXT}"
if [[ "${#METHOD_LIST[@]}" -eq 0 ]]; then
  echo "No pruning methods selected. Set METHODS=\"magnitude 2of4 wanda gradient\"." >&2
  exit 2
fi

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_OUTPUT_DIR}}"
if [[ -z "${OUTPUT_DIR}" ]]; then
  OUTPUT_DIR="runs/sft-pruning-eval-${STAMP}"
fi
TEMPLATE_CONFIG="${TEMPLATE_CONFIG:-${SCRIPT_TEMPLATE_CONFIG:-configs/prune_50.yaml}}"
NPROC="${NPROC:-${SCRIPT_NPROC:-8}}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-${SCRIPT_MAX_NEW_TOKENS:-64}}"
MAX_LENGTH="${MAX_LENGTH:-${SCRIPT_MAX_LENGTH:-2048}}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-${SCRIPT_EVAL_BATCH_SIZE:-8}}"
PRUNE_BATCH_SIZE="${PRUNE_BATCH_SIZE:-${SCRIPT_PRUNE_BATCH_SIZE:-2}}"
CALIBRATION_BATCHES="${CALIBRATION_BATCHES:-${SCRIPT_CALIBRATION_BATCHES:-128}}"
DTYPE="${DTYPE:-${SCRIPT_DTYPE:-bf16}}"
BENCHMARK_RUNS="${BENCHMARK_RUNS:-${SCRIPT_BENCHMARK_RUNS:-1}}"
SEED="${SEED:-42}"
DATA_SEED="${DATA_SEED:-${SEED}}"
COMPARISON_MODE="${COMPARISON_MODE:-${SCRIPT_COMPARISON_MODE:-whitespace}}"
SPARSITY="${SPARSITY:-0.5}"
SPARSITY_DENOMINATOR="${SPARSITY_DENOMINATOR:-prunable}"
GRANULARITY="${GRANULARITY:-layer}"
INCLUDE_LM_HEAD="${INCLUDE_LM_HEAD:-false}"

mkdir -p "${OUTPUT_DIR}/generated_configs" "${OUTPUT_DIR}/benchmarks/one_shot" "${OUTPUT_DIR}/one_shot"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MODEL_PATH DATA_FILE CALIBRATION_FILE OUTPUT_DIR

run_eval() {
  local checkpoint="$1"
  local out_dir="$2"
  local label="$3"
  local args=(
    --standalone
    --nproc_per_node "${NPROC}"
    scripts/eval_prompt_response.py
    --model-path "${checkpoint}"
    --dataset-file "${DATA_FILE}"
    --output-dir "${out_dir}"
    --max-new-tokens "${MAX_NEW_TOKENS}"
    --max-length "${MAX_LENGTH}"
    --temperature 0
    --num-beams 1
    --seed "${SEED}"
    --data-seed "${DATA_SEED}"
    --batch-size "${EVAL_BATCH_SIZE}"
    --dtype "${DTYPE}"
    --benchmark-runs "${BENCHMARK_RUNS}"
    --comparison-mode "${COMPARISON_MODE}"
  )
  if [[ -n "${TRAIN_FILE}" ]]; then
    args+=(--train-file "${TRAIN_FILE}")
  fi
  echo
  echo "=== SFT benchmark: ${label} ==="
  echo "checkpoint: ${checkpoint}"
  echo "eval data:  ${DATA_FILE}"
  torchrun "${args[@]}"
}

method_slug() {
  if [[ "$1" == "2of4" ]]; then
    echo "nvidia-2of4"
  else
    echo "$1"
  fi
}

write_prune_config() {
  local method="$1"
  local output_dir="$2"
  local config_path="$3"
  "${PYTHON_BIN}" - "${TEMPLATE_CONFIG}" "${config_path}" "${method}" "${MODEL_PATH}" "${output_dir}" \
    "${CALIBRATION_FILE}" "${MAX_LENGTH}" "${PRUNE_BATCH_SIZE}" "${CALIBRATION_BATCHES}" \
    "${SPARSITY}" "${SPARSITY_DENOMINATOR}" "${GRANULARITY}" "${INCLUDE_LM_HEAD}" <<'PY'
from pathlib import Path
import sys
import yaml

template, config_path, method, model_path, output_dir = sys.argv[1:6]
calibration_file, max_length, batch_size, calibration_batches = sys.argv[6:10]
sparsity, denominator, granularity, include_lm_head = sys.argv[10:14]

with Path(template).open("r", encoding="utf-8") as handle:
    config = yaml.safe_load(handle) or {}

config.setdefault("model", {})
config.setdefault("train", {})
config.setdefault("sft", {})
config.setdefault("prune", {})

config["model"]["block_size"] = int(max_length)
config["train"]["batch_size"] = int(batch_size)
config["sft"]["data_path"] = calibration_file
config["prune"].update(
    {
        "method": method,
        "base_model": model_path,
        "output_dir": output_dir,
        "calibration_data_path": calibration_file,
        "sparsity": float(sparsity),
        "sparsity_denominator": denominator,
        "granularity": granularity,
        "include_lm_head": str(include_lm_head).lower() == "true",
        "recovery_steps": 0,
        "batch_size": int(batch_size),
        "max_length": int(max_length),
        "calibration_batches": int(calibration_batches),
        "overwrite": True,
    }
)

Path(config_path).parent.mkdir(parents=True, exist_ok=True)
with Path(config_path).open("w", encoding="utf-8") as handle:
    yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)
PY
}

echo "SFT pruning eval run"
echo "  model:            ${MODEL_PATH}"
echo "  eval data:        ${DATA_FILE}"
echo "  calibration data: ${CALIBRATION_FILE}"
echo "  output:           ${OUTPUT_DIR}"
echo "  methods:          ${METHOD_LIST[*]}"
echo "  evaluator:        scripts/eval_prompt_response.py"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

DENSE_EVAL_DIR="${OUTPUT_DIR}/benchmarks/dense_sft"
run_eval "${MODEL_PATH}" "${DENSE_EVAL_DIR}" "dense SFT"

for method in "${METHOD_LIST[@]}"; do
  slug="$(method_slug "${method}")"
  pruned_dir="${OUTPUT_DIR}/one_shot/${slug}"
  eval_dir="${OUTPUT_DIR}/benchmarks/one_shot/${slug}"
  generated_config="${OUTPUT_DIR}/generated_configs/prune_${slug}.yaml"

  echo
  echo "=== Pruning: ${method} ==="
  write_prune_config "${method}" "${pruned_dir}" "${generated_config}"
  "${PYTHON_BIN}" scripts/prune.py \
    --config "${generated_config}" \
    --method "${method}" \
    --checkpoint "${MODEL_PATH}" \
    --output-dir "${pruned_dir}"

  if [[ ! -f "${pruned_dir}/pruning_report.json" ]]; then
    echo "Missing pruning report after ${method}: ${pruned_dir}/pruning_report.json" >&2
    exit 1
  fi
  run_eval "${pruned_dir}" "${eval_dir}" "${method} pruned"
done

METHODS_NORMALIZED="${METHOD_LIST[*]}" "${PYTHON_BIN}" - "${OUTPUT_DIR}" <<'PY'
from pathlib import Path
import csv
import json
import math
import os
import sys

output_dir = Path(sys.argv[1])
methods = os.environ["METHODS_NORMALIZED"].split()

summary_names = (
    "prompt_response_eval_benchmark_summary.json",
    "prompt_response_eval_summary.json",
    "metrics.json",
)


def method_slug(method: str) -> str:
    return "nvidia-2of4" if method == "2of4" else method


def result_dir(path: Path) -> Path:
    if any((path / name).exists() for name in summary_names):
        return path
    marker = path / "latest_eval_dir.txt"
    if marker.exists():
        marked = Path(marker.read_text(encoding="utf-8").strip()).expanduser()
        if not marked.is_absolute():
            marked = Path.cwd() / marked
        if any((marked / name).exists() for name in summary_names):
            return marked
    children = [child for child in path.iterdir() if child.is_dir()] if path.exists() else []
    candidates = [child for child in children if any((child / name).exists() for name in summary_names)]
    return sorted(candidates, key=lambda child: (child.stat().st_mtime, child.name))[-1] if candidates else path


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_eval(path: Path) -> dict:
    resolved = result_dir(path)
    for name in summary_names:
        summary = read_json(resolved / name)
        if summary:
            summary["_resolved_eval_dir"] = str(resolved)
            return summary
    return {"_resolved_eval_dir": str(resolved)}


def mean(values: list[float]) -> float | None:
    values = [float(value) for value in values if math.isfinite(float(value))]
    return sum(values) / len(values) if values else None


def metric(summary: dict, name: str):
    if summary.get(f"{name}_mean") is not None:
        return summary.get(f"{name}_mean")
    if summary.get(name) is not None:
        return summary.get(name)
    values = []
    for run in summary.get("per_run_summaries", []) or []:
        value = run.get(name)
        if value is not None:
            values.append(float(value))
    return mean(values)


def metric_count(summary: dict, *names: str):
    for name in names:
        if summary.get(name) is not None:
            return summary.get(name)
        if summary.get(f"{name}_mean") is not None:
            return summary.get(f"{name}_mean")
    return ""


def report_value(report: dict, *names: str):
    reload_validation = report.get("checkpoint_reload_validation")
    sources = [reload_validation, report] if isinstance(reload_validation, dict) else [report]
    for source in sources:
        for name in names:
            if source.get(name) is not None:
                return source.get(name)
    return ""


def fmt(value) -> str:
    if value in ("", None):
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


rows = []
dense_eval = read_eval(output_dir / "benchmarks" / "dense_sft")
rows.append(
    {
        "method": "dense_sft",
        "phase": "dense",
        "status": "ok" if metric(dense_eval, "exact_match_accuracy") is not None else "missing",
        "checkpoint": os.environ.get("MODEL_PATH", ""),
        "checkpoint_evaluated": dense_eval.get("checkpoint", os.environ.get("MODEL_PATH", "")),
        "eval_output_dir": dense_eval.get("_resolved_eval_dir", ""),
        "exact_match_accuracy": metric(dense_eval, "exact_match_accuracy"),
        "correct_examples": metric_count(dense_eval, "correct_examples", "exact_match_correct"),
        "total_examples": metric_count(dense_eval, "total_examples"),
        "mean_response_loss": metric(dense_eval, "mean_response_loss"),
        "response_perplexity": metric(dense_eval, "response_perplexity"),
        "avg_generated_tokens": metric(dense_eval, "avg_generated_tokens"),
        "active_model_parameters": "",
        "pruned_prunable_parameters": "",
        "active_prunable_parameters": "",
        "total_prunable_parameters": "",
        "prunable_parameter_count": "",
        "protected_parameter_count": "",
        "total_parameter_count": "",
        "real_sparsity": "",
        "achieved_prunable_sparsity": "",
        "achieved_whole_model_sparsity": "",
        "nonzero_parameters": "",
        "zero_parameters": "",
        "pruning_report": "",
    }
)

for method in methods:
    slug = method_slug(method)
    checkpoint = output_dir / "one_shot" / slug
    eval_summary = read_eval(output_dir / "benchmarks" / "one_shot" / slug)
    report_path = checkpoint / "pruning_report.json"
    report = read_json(report_path)
    exact_match_accuracy = metric(eval_summary, "exact_match_accuracy")
    nonzero_parameters = report_value(report, "nonzero_parameters")
    whole_sparsity = report_value(report, "achieved_whole_model_sparsity", "model_zero_fraction")
    prunable_sparsity = report_value(report, "achieved_prunable_sparsity", "mask_sparsity", "sparsity")
    status = "ok"
    if not report:
        status = "missing"
    elif exact_match_accuracy is None:
        status = "missing"
    rows.append(
        {
            "method": method,
            "phase": "one_shot",
            "status": status,
            "checkpoint": str(checkpoint),
            "checkpoint_evaluated": eval_summary.get("checkpoint", str(checkpoint)),
            "eval_output_dir": eval_summary.get("_resolved_eval_dir", ""),
            "exact_match_accuracy": exact_match_accuracy,
            "correct_examples": metric_count(eval_summary, "correct_examples", "exact_match_correct"),
            "total_examples": metric_count(eval_summary, "total_examples"),
            "mean_response_loss": metric(eval_summary, "mean_response_loss"),
            "response_perplexity": metric(eval_summary, "response_perplexity"),
            "avg_generated_tokens": metric(eval_summary, "avg_generated_tokens"),
            "active_model_parameters": nonzero_parameters,
            "pruned_prunable_parameters": report_value(report, "pruned_prunable_parameters", "pruned_mask_parameters"),
            "active_prunable_parameters": report_value(report, "active_prunable_parameters", "active_mask_parameters"),
            "total_prunable_parameters": report_value(report, "total_prunable_parameters", "mask_parameter_count"),
            "prunable_parameter_count": report_value(report, "prunable_parameter_count", "total_prunable_parameters", "mask_parameter_count"),
            "protected_parameter_count": report_value(report, "protected_parameter_count", "protected_parameters"),
            "total_parameter_count": report_value(report, "total_parameter_count", "total_parameters"),
            "real_sparsity": whole_sparsity,
            "achieved_prunable_sparsity": prunable_sparsity,
            "achieved_whole_model_sparsity": whole_sparsity,
            "nonzero_parameters": nonzero_parameters,
            "zero_parameters": report_value(report, "zero_parameters"),
            "pruning_report": str(report_path),
        }
    )

fields = list(rows[0].keys())
csv_path = output_dir / "sft_pruning_eval_summary.csv"
json_path = output_dir / "sft_pruning_eval_summary.json"
with csv_path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
json_path.write_text(json.dumps({"results": rows}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

print("\nSFT pruning eval summary")
print(
    "method".ljust(12),
    "phase".ljust(10),
    "accuracy".rjust(10),
    "correct/total".rjust(15),
    "pruned/total prunable".rjust(24),
    "prunable_sparsity".rjust(18),
    "real_sparsity".rjust(16),
)
for row in rows:
    correct = row.get("correct_examples", "")
    total = row.get("total_examples", "")
    pruned = row.get("pruned_prunable_parameters", "")
    total_prunable = row.get("total_prunable_parameters", "")
    print(
        str(row["method"]).ljust(12),
        str(row.get("phase", "")).ljust(10),
        fmt(row.get("exact_match_accuracy")).rjust(10),
        f"{correct}/{total}".rjust(15),
        f"{pruned}/{total_prunable}".rjust(24),
        fmt(row.get("achieved_prunable_sparsity")).rjust(18),
        fmt(row.get("real_sparsity")).rjust(16),
    )
print(f"\nWrote: {csv_path}")
print(f"Wrote: {json_path}")
PY
