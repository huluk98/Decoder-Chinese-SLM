#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

usage() {
  cat <<'EOF'
Usage:
  bash run_5epoch_sft_contrastive_one_shot_pruning.sh [BASE_MODEL_PATH] [extra python args]

Runs the 5-epoch regular SFT, then the 5-epoch contrastive SFT from the fresh
regular SFT checkpoint, followed by the configured dense eval and one-shot
pruning benchmark.

Environment overrides:
  BASE_MODEL            Same as first positional argument.
  ORIGINAL_MODEL        Defaults to BASE_MODEL for original decoder dense eval.
  RUN_ROOT              default: runs/5epoch-sft-contrastive-one-shot
  EPOCHS                default: 5
  TRAIN_ONLY            set to 1 to stop after the two 5-epoch training runs.
  SFT_TRAIN_FILE        optional regular SFT dataset override; default config uses data/scenic/SCENIC_full_training_dataset.json.
  SFT_EVAL_FILE         optional regular SFT eval/calibration dataset override; default config uses data/scenic/SCENIC_full_training_dataset.json.
  CONTRASTIVE_TRAIN_FILE
                        optional contrastive dataset override; default config uses data/scenic/SCENIC_full_anchor_positive_negative.json.
  CONTRASTIVE_EVAL_FILE optional contrastive eval/calibration dataset; default config uses data/scenic/SCENIC_full_training_dataset.json.
  MAX_NEW_TOKEN_HIT_RATE_THRESHOLD
                        default: 1.01, so high max-token rates are reported instead of aborting.
  BENCHMARK_FILE        optional benchmark dataset override.
  PYTHON                Python executable from the training environment.

Examples:
  PYTHON=/path/to/env/bin/python bash run_5epoch_sft_contrastive_one_shot_pruning.sh /path/to/base_model
  TRAIN_ONLY=1 bash run_5epoch_sft_contrastive_one_shot_pruning.sh /path/to/base_model
EOF
}

run_5epoch_sft_contrastive_from_base() {
  if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    return 0
  fi

  if [[ $# -gt 0 && "${1}" != -* ]]; then
    export BASE_MODEL="${1}"
    export ORIGINAL_MODEL="${ORIGINAL_MODEL:-${1}}"
    shift
  elif [[ -n "${BASE_MODEL:-}" ]]; then
    export ORIGINAL_MODEL="${ORIGINAL_MODEL:-${BASE_MODEL}}"
  fi
  export EPOCHS="${EPOCHS:-5}"

  local python_bin="${PYTHON:-}"
  if [[ -z "${python_bin}" ]]; then
    if command -v python3 >/dev/null 2>&1; then
      python_bin="python3"
    else
      python_bin="python"
    fi
  fi

  exec "${python_bin}" run_5epoch_sft_contrastive_one_shot_pruning.py "$@"
}

run_5epoch_sft_contrastive_from_base "$@"
