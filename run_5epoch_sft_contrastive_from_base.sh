#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

export TRAIN_ONLY="${TRAIN_ONLY:-1}"
exec bash run_5epoch_sft_contrastive_one_shot_pruning.sh "$@"
