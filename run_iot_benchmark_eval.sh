#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${CONFIG_PATH:-configs/iot_benchmark_eval.yaml}"
PYTHON_BIN="${PYTHON:-python3}"

exec "${PYTHON_BIN}" scripts/eval_iot_benchmark.py --config "${CONFIG_PATH}" "$@"
