#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

export EOS_RETUNE="${EOS_RETUNE:-1}"
export EOS_LOSS_WEIGHT="${EOS_LOSS_WEIGHT:-5.0}"
export EOS_RETUNE_EPOCHS="${EOS_RETUNE_EPOCHS:-1.0}"
export EOS_RETUNE_MODE="${EOS_RETUNE_MODE:-sft}"

exec bash run_5epoch_sft_contrastive_one_shot_pruning.sh "$@"
