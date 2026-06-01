#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

if [[ -n "${PYTHON:-}" ]]; then
  exec "${PYTHON}" run_5epoch_sft_contrastive_one_shot_pruning.py "$@"
elif command -v python3 >/dev/null 2>&1; then
  exec python3 run_5epoch_sft_contrastive_one_shot_pruning.py "$@"
else
  exec python run_5epoch_sft_contrastive_one_shot_pruning.py "$@"
fi
