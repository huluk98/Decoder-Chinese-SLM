#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

CONFIG_PATH="${CONFIG_PATH:-${1:-configs/pruning_benchmark.yaml}}"
PYTHON_BIN="${PYTHON:-python}"
MODE="${MODE:-auto}"
DRY_RUN="${DRY_RUN:-0}"

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Config file not found: ${CONFIG_PATH}" >&2
  echo "Usage: CONFIG_PATH=configs/pruning_benchmark.yaml ./scripts/run_pruning_benchmark_8way.sh" >&2
  exit 2
fi

detect_mode() {
  local config_path="$1"
  local requested="$2"
  case "${requested}" in
    generic|qwen)
      echo "${requested}"
      return
      ;;
    auto)
      ;;
    *)
      echo "Unknown MODE=${requested}. Use MODE=auto, MODE=generic, or MODE=qwen." >&2
      exit 2
      ;;
  esac

  if [[ "$(basename "${config_path}")" == *qwen* ]] || grep -Eiq 'qwen|prune_qwen25|sft_qwen25' "${config_path}"; then
    echo "qwen"
  else
    echo "generic"
  fi
}

RUN_KIND="$(detect_mode "${CONFIG_PATH}" "${MODE}")"
RUNNER="scripts/run_pruning_benchmark.py"
SUMMARY_CSV="pruning_benchmark_summary.csv"
SUMMARY_JSON="pruning_benchmark_summary.json"
if [[ "${RUN_KIND}" == "qwen" ]]; then
  RUNNER="scripts/run_qwen25_instruct_pruning_benchmark.py"
  SUMMARY_CSV="qwen25_instruct_pruning_benchmark_summary.csv"
  SUMMARY_JSON="qwen25_instruct_pruning_benchmark_summary.json"
fi

ARGS=(--config "${CONFIG_PATH}")
if [[ "${DRY_RUN}" == "1" ]]; then
  ARGS+=(--dry-run)
fi

echo "Sequential 8-way pruning benchmark"
echo "  config: ${CONFIG_PATH}"
echo "  mode:   ${RUN_KIND}"
echo "  runner: ${RUNNER}"
echo
echo "Order inside the runner:"
echo "  magnitude: one-shot prune -> eval -> retune -> eval"
echo "  2of4:      one-shot prune -> eval -> retune -> eval"
echo "  wanda:     one-shot prune -> eval -> retune -> eval"
echo "  gradient:  one-shot prune -> eval -> retune -> eval"
echo

"${PYTHON_BIN}" "${RUNNER}" "${ARGS[@]}"

echo
echo "Done. The runner wrote its 8-row benchmark summary under benchmark.output_dir:"
echo "  ${SUMMARY_CSV}"
echo "  ${SUMMARY_JSON}"

