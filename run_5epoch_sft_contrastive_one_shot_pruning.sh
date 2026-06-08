#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

usage() {
  cat <<'EOF'
Usage:
  bash run_5epoch_sft_contrastive_one_shot_pruning.sh [COMMAND] [BASE_MODEL_PATH] [extra python args]

Runs the 5-epoch regular SFT, then the 5-epoch contrastive SFT from the fresh
regular SFT checkpoint, followed by the configured dense eval and one-shot
pruning benchmark. Defaults to 50% prunable Linear-weight pruning for magnitude,
WANDA, Taylor saliency, and NVIDIA 2:4 while keeping embeddings, norms, and
lm_head dense for EOS stability.

Commands:
  retune, eos-retune   run configured pruning target(s), then add fixed-mask EOS retune rows.
  one-shot, no-retune  run the original one-shot pruning benchmark only.
  train-only           stop after the two 5-epoch training runs.

Environment overrides:
  BASE_MODEL            Same as first positional argument.
  RUN_ROOT              default: runs/full-decoder-sft-contrastive-pruning
  EPOCHS                default: 5
  TRAIN_ONLY            set to 1 to stop after the two 5-epoch training runs.
  SFT_TRAIN_FILE        optional regular SFT dataset override; default config uses data/scenic/SCENIC_full_training_dataset.json.
  SFT_EVAL_FILE         optional regular SFT eval/calibration dataset override; default config uses data/scenic/SCENIC_full_training_dataset.json.
  CONTRASTIVE_TRAIN_FILE
                        optional contrastive dataset override; default config uses data/scenic/SCENIC_full_anchor_positive_negative.json.
  CONTRASTIVE_EVAL_FILE optional contrastive eval/calibration dataset; default config uses data/scenic/SCENIC_full_training_dataset.json.
  MAX_NEW_TOKEN_HIT_RATE_THRESHOLD
                        default: 1.01, so pruning eval records max-token hits without aborting.
                        Set to 0.5 for stricter EOS/debug failure behavior.
  METHODS               default: "magnitude wanda taylor 2of4".
  SPARSITY_LEVELS        optional whitespace/comma list such as "0.3 0.5";
                        when set, pruning/eval runs once per level after training.
  SPARSITY               default: 0.5 when SPARSITY_LEVELS is unset.
  SPARSITY_DENOMINATOR  default: prunable, so 50% means selected Linear weights rather than whole model parameters.
  GRANULARITY           default: global for magnitude/Taylor; 2:4 remains fixed per 4-weight group.
  WANDA_GRANULARITY     default: row, so WANDA prunes each Linear output row to the requested sparsity.
  RUN_DENSE_BASELINE    default: 1; set to 0 for method-only reruns that should skip dense eval.
  EOS_RETUNE            set to 1 to add masked SFT recovery rows after one-shot pruning.
  EOS_LOSS_WEIGHT       default when EOS_RETUNE=1 via wrapper: 5.0.
  EOS_RETUNE_EPOCHS     default when EOS_RETUNE=1 via wrapper: 1.0.
  BENCHMARK_FILE        optional benchmark dataset override.
  PYTHON                Python executable from the training environment.

Examples:
  PYTHON=/path/to/env/bin/python bash run_full_decoder_sft_contrastive_pruning.sh /path/to/base_model
  PYTHON=/path/to/env/bin/python bash run_5epoch_sft_contrastive_one_shot_pruning.sh /path/to/base_model
  PYTHON=/path/to/env/bin/python bash run_5epoch_sft_contrastive_one_shot_pruning.sh retune /path/to/base_model
  PYTHON=/path/to/env/bin/python bash run_5epoch_sft_contrastive_eos_reinforced_pruning.sh /path/to/base_model
  PYTHON=/path/to/env/bin/python bash run_5epoch_sft_contrastive_eos_reinforced_pruning.sh retune /path/to/base_model
  TRAIN_ONLY=1 bash run_5epoch_sft_contrastive_one_shot_pruning.sh /path/to/base_model
EOF
}

run_5epoch_sft_contrastive_from_base() {
  if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    return 0
  fi

  if [[ $# -gt 0 && "${1}" != -* ]]; then
    case "${1}" in
      retune|eos-retune)
        export EOS_RETUNE=1
        shift
        ;;
      one-shot|oneshot|no-retune)
        export EOS_RETUNE=0
        shift
        ;;
      train-only)
        export TRAIN_ONLY=1
        shift
        ;;
    esac
  fi

  if [[ $# -gt 0 && "${1}" != -* ]]; then
    export BASE_MODEL="${1}"
    export ORIGINAL_MODEL="${1}"
    shift
  elif [[ -n "${BASE_MODEL:-}" ]]; then
    export ORIGINAL_MODEL="${BASE_MODEL}"
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
  if [[ "${python_bin}" != */* ]]; then
    python_bin="$(command -v "${python_bin}")"
  fi

  exec "${python_bin}" run_5epoch_sft_contrastive_one_shot_pruning.py "$@"
}

run_5epoch_sft_contrastive_from_base "$@"
