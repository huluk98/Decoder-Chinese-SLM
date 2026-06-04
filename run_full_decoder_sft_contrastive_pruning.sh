#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

usage() {
  cat <<'EOF'
Run the full decoder-only experiment from a base checkpoint:
  1. evaluate the original decoder;
  2. train regular SFT for 5 epochs;
  3. train contrastive SFT for 5 epochs from the fresh regular SFT checkpoint;
  4. run all pruning methods on both SFT checkpoints;
  5. write EM@1 and EM@5 summaries.

Usage:
  bash run_full_decoder_sft_contrastive_pruning.sh /path/to/original_decoder_checkpoint

Common overrides:
  PYTHON=/path/to/env/bin/python
  RUN_ROOT=runs/my-full-pruning-run
  METHODS="magnitude wanda gradient 2of4"
  MAX_NEW_TOKENS=64
  MAX_NEW_TOKEN_HIT_RATE_THRESHOLD=0.5
  KEEP_GOING=1
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

base_model="${BASE_MODEL:-}"
if [[ -z "${base_model}" && $# -gt 0 && "${1}" != -* ]]; then
  base_model="${1}"
  shift
fi

if [[ -z "${base_model}" ]]; then
  usage
  exit 2
fi

export BASE_MODEL="${base_model}"
export ORIGINAL_MODEL="${ORIGINAL_MODEL:-${base_model}}"
export EPOCHS="${EPOCHS:-5}"
export METHODS="${METHODS:-magnitude wanda gradient 2of4}"
export TOP_K_EXACT_MATCH="${TOP_K_EXACT_MATCH:-5}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
export MAX_NEW_TOKEN_HIT_RATE_THRESHOLD="${MAX_NEW_TOKEN_HIT_RATE_THRESHOLD:-0.5}"
export SPARSITY="${SPARSITY:-0.5}"
export PRUNING_SCOPE="${PRUNING_SCOPE:-transformer_linears}"
export SPARSITY_DENOMINATOR="${SPARSITY_DENOMINATOR:-whole_model}"
export GRANULARITY="${GRANULARITY:-layer}"
export INCLUDE_LM_HEAD="${INCLUDE_LM_HEAD:-false}"
export RUN_ROOT="${RUN_ROOT:-runs/full-decoder-sft-contrastive-pruning}"
export KEEP_GOING="${KEEP_GOING:-1}"

echo "Full decoder SFT + contrastive SFT + pruning run"
echo "  original decoder: ${BASE_MODEL}"
echo "  epochs:           ${EPOCHS}"
echo "  methods:          ${METHODS}"
echo "  EM@K:             ${TOP_K_EXACT_MATCH}"
echo "  max_new_tokens:   ${MAX_NEW_TOKENS}"
echo "  max-token guard:  ${MAX_NEW_TOKEN_HIT_RATE_THRESHOLD}"
echo "  output root:      ${RUN_ROOT}"
echo

exec bash run_5epoch_sft_contrastive_one_shot_pruning.sh "${BASE_MODEL}" "$@"
