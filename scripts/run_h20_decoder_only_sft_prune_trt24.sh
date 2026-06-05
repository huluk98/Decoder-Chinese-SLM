#!/usr/bin/env bash
set -euo pipefail

ORIGINAL_CWD="$(pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

usage() {
  cat <<'EOF'
Run the H20 decoder-only SFT -> NVIDIA 2:4 -> ONNX -> TensorRT FP16 baseline.

Only --base_model is required. Dataset and output paths are baked in below and
can be overridden by CLI flags or environment variables when needed.

Usage:
  bash scripts/run_h20_decoder_only_sft_prune_trt24.sh --base_model /PATH/TO/BASE_MODEL

Optional overrides:
  --train_jsonl PATH       default: data/scenic/SCENIC_full_training_dataset.json
  --iot200_jsonl PATH      default: data/benchmarks/iot_instruction_benchmark_200.json
  --output_dir PATH        default: runs/h20_decoder_only_trt24
  --gpus IDS              default: 0,1,2,3,4,5,6,7
  --epochs N              default: 5
  --seq_len N             default: 64
  --batch_size N          default: 16 per GPU for SFT
  --eval_batch_size N     default: 16 per GPU for PyTorch EM eval
  --warmup_iters N        default: 100
  --measure_iters N       default: 1000
  --env_help              print H20/TensorRT environment setup help and exit

Useful environment toggles:
  PYTHON=/path/to/python
  TRUST_REMOTE_CODE=1
  PROMPT_FORMAT=raw|legacy|chat-template
  COMPARISON_MODE=whitespace|normalized|command
  SKIP_TRAIN=1 SKIP_PRUNE=1 SKIP_EXPORT=1 SKIP_BUILD=1 SKIP_EVAL=1 SKIP_LATENCY=1
  RUN_INT8=1 INT8_CALIB_CACHE=/path/to/real.cache
EOF
}

print_env_setup_help() {
  cat <<'EOF'
H20 benchmark environment setup checklist:

1. Run this on the H20/NVIDIA GPU machine, not on a CPU-only laptop/login shell.
   First verify the driver can see GPUs:

     nvidia-smi

2. Activate the Python env you will use for training:

     conda activate chatlm-decoder

3. Install/update the Python runtime packages:

     python -m pip install --upgrade -r requirements.txt --extra-index-url https://pypi.nvidia.com

   Or install only the TensorRT/ONNX additions:

     python -m pip install --upgrade "onnx>=1.16" "onnxruntime-gpu>=1.18" "cuda-python>=12.2" "tensorrt>=10.0" --extra-index-url https://pypi.nvidia.com

4. Make sure native TensorRT tools are on PATH. `trtexec` is not a normal pip
   script; it comes from the NVIDIA TensorRT SDK/container. Verify:

     trtexec --version

5. Verify the final env:

     python - <<'PY'
     import torch, onnx, onnxruntime as ort, tensorrt as trt
     import cuda.cudart
     print("torch cuda:", torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count())
     print("onnx:", onnx.__version__)
     print("ort:", ort.__version__, ort.get_available_providers())
     print("trt:", trt.__version__)
     PY

If torch.cuda.is_available() is false after this, fix the machine/container GPU
visibility first: driver, CUDA container flags, scheduler GPU allocation, or
CUDA_VISIBLE_DEVICES.
EOF
}

resolve_input_path() {
  local value="$1"
  if [[ "${value}" == /* ]]; then
    printf '%s\n' "${value}"
  elif [[ -e "${ORIGINAL_CWD}/${value}" ]]; then
    (cd "${ORIGINAL_CWD}" && printf '%s\n' "$(pwd)/${value}")
  elif [[ -e "${PROJECT_ROOT}/${value}" ]]; then
    (cd "${PROJECT_ROOT}" && printf '%s\n' "$(pwd)/${value}")
  else
    printf '%s\n' "${value}"
  fi
}

resolve_repo_path() {
  local value="$1"
  if [[ "${value}" == /* ]]; then
    printf '%s\n' "${value}"
  elif [[ -e "${ORIGINAL_CWD}/${value}" ]]; then
    (cd "${ORIGINAL_CWD}" && printf '%s\n' "$(pwd)/${value}")
  else
    printf '%s\n' "${PROJECT_ROOT}/${value}"
  fi
}

BASE_MODEL="${BASE_MODEL:-}"
TRAIN_JSONL="${TRAIN_JSONL:-data/scenic/SCENIC_full_training_dataset.json}"
IOT200_JSONL="${IOT200_JSONL:-data/benchmarks/iot_instruction_benchmark_200.json}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/h20_decoder_only_trt24}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
EPOCHS="${EPOCHS:-5}"
SEQ_LEN="${SEQ_LEN:-64}"
BATCH_SIZE="${BATCH_SIZE:-16}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
MEASURE_ITERS="${MEASURE_ITERS:-1000}"
WARMUP_ITERS="${WARMUP_ITERS:-100}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base_model|--base-model)
      BASE_MODEL="$(resolve_input_path "$2")"
      shift 2
      ;;
    --train_jsonl|--train-jsonl)
      TRAIN_JSONL="$2"
      shift 2
      ;;
    --iot200_jsonl|--iot200-jsonl)
      IOT200_JSONL="$2"
      shift 2
      ;;
    --output_dir|--output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --gpus)
      GPUS="$2"
      shift 2
      ;;
    --epochs)
      EPOCHS="$2"
      shift 2
      ;;
    --seq_len|--seq-len)
      SEQ_LEN="$2"
      shift 2
      ;;
    --batch_size|--batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --eval_batch_size|--eval-batch-size)
      EVAL_BATCH_SIZE="$2"
      shift 2
      ;;
    --measure_iters|--measure-iters)
      MEASURE_ITERS="$2"
      shift 2
      ;;
    --warmup_iters|--warmup-iters)
      WARMUP_ITERS="$2"
      shift 2
      ;;
    --env_help|--env-help)
      print_env_setup_help
      exit 0
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    -*)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
    *)
      if [[ -z "${BASE_MODEL}" ]]; then
        BASE_MODEL="$(resolve_input_path "$1")"
        shift
      else
        echo "Unexpected positional argument: $1" >&2
        exit 2
      fi
      ;;
  esac
done

if [[ -z "${BASE_MODEL}" ]]; then
  usage >&2
  exit 2
fi

PYTHON_BIN="${PYTHON:-python3}"
TRAIN_JSONL="$(resolve_repo_path "${TRAIN_JSONL}")"
IOT200_JSONL="$(resolve_repo_path "${IOT200_JSONL}")"
OUTPUT_DIR="$(resolve_repo_path "${OUTPUT_DIR}")"

if [[ ! -f "${TRAIN_JSONL}" ]]; then
  echo "Training data not found: ${TRAIN_JSONL}" >&2
  exit 1
fi
if [[ ! -f "${IOT200_JSONL}" ]]; then
  echo "IoT200 benchmark data not found: ${IOT200_JSONL}" >&2
  exit 1
fi

IFS=',' read -r -a GPU_ARRAY <<< "${GPUS}"
NPROC_PER_NODE="${NPROC_PER_NODE:-${#GPU_ARRAY[@]}}"
LATENCY_GPU="${LATENCY_GPU:-${GPU_ARRAY[0]}}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
LEARNING_RATE="${LEARNING_RATE:-2.0e-5}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
WORKSPACE_MIB="${WORKSPACE_MIB:-4096}"
PROMPT_FORMAT="${PROMPT_FORMAT:-raw}"
COMPARISON_MODE="${COMPARISON_MODE:-whitespace}"
MAX_NEW_TOKEN_HIT_RATE_THRESHOLD="${MAX_NEW_TOKEN_HIT_RATE_THRESHOLD:-0.5}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-0}"
RUN_INT8="${RUN_INT8:-0}"
INT8_CALIB_CACHE="${INT8_CALIB_CACHE:-}"

export CUDA_VISIBLE_DEVICES="${GPUS}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

DENSE_CKPT="${OUTPUT_DIR}/checkpoints/dense_sft_fp16"
DENSE_TRAIN_DIR="${OUTPUT_DIR}/checkpoints/_dense_sft_training"
PRUNED_CKPT="${OUTPUT_DIR}/checkpoints/nvidia_2_4_sft_fp16"
DENSE_ONNX_DIR="${OUTPUT_DIR}/onnx/dense_sft_fp16"
PRUNED_ONNX_DIR="${OUTPUT_DIR}/onnx/nvidia_2_4_sft_fp16"
DENSE_ONNX="${DENSE_ONNX_DIR}/model.onnx"
PRUNED_ONNX="${PRUNED_ONNX_DIR}/model.onnx"
DENSE_ENGINE="${OUTPUT_DIR}/engines/dense_sft_fp16_seq64.plan"
PRUNED_ENGINE="${OUTPUT_DIR}/engines/nvidia_2_4_sft_fp16_seq64.plan"

mkdir -p \
  "${OUTPUT_DIR}/env" \
  "${OUTPUT_DIR}/logs" \
  "${OUTPUT_DIR}/reports" \
  "${OUTPUT_DIR}/results" \
  "${OUTPUT_DIR}/generated_configs" \
  "${OUTPUT_DIR}/engines"

TRUST_ARGS=()
if [[ "${TRUST_REMOTE_CODE}" == "1" ]]; then
  TRUST_ARGS=(--trust-remote-code)
fi

SYSTEM_PROMPT_ARGS=()
if [[ -n "${SYSTEM_PROMPT:-}" ]]; then
  SYSTEM_PROMPT_ARGS=(--system-prompt "${SYSTEM_PROMPT}")
fi

run_logged() {
  local log_path="$1"
  shift
  echo
  echo "[$(date -Is)] $*" | tee -a "${log_path}"
  "$@" 2>&1 | tee -a "${log_path}"
}

write_env_report() {
  local txt="${OUTPUT_DIR}/env/env_report.txt"
  local json="${OUTPUT_DIR}/env/env_report.json"
  {
    echo "=== nvidia-smi ==="
    if command -v nvidia-smi >/dev/null 2>&1; then nvidia-smi; else echo "nvidia-smi not found"; fi
    echo
    echo "=== nvcc --version ==="
    if command -v nvcc >/dev/null 2>&1; then nvcc --version; else echo "nvcc not found"; fi
    echo
    echo "=== python ==="
    "${PYTHON_BIN}" --version
    echo
    echo "=== torch / TensorRT / ONNX Runtime ==="
    "${PYTHON_BIN}" - <<'PY'
import json
import shutil
import subprocess
import sys

def module_version(name):
    try:
        module = __import__(name)
        return getattr(module, "__version__", "unknown")
    except Exception as exc:
        return f"unavailable ({exc})"

print("python:", sys.version.replace("\n", " "))
try:
    import torch
    print("torch:", torch.__version__)
    print("torch.version.cuda:", torch.version.cuda)
    print("torch.cuda.is_available():", torch.cuda.is_available())
    print("torch.cuda.device_count():", torch.cuda.device_count())
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            print(f"torch.cuda.get_device_name({index}):", torch.cuda.get_device_name(index))
except Exception as exc:
    print("torch import failed:", repr(exc))
print("tensorrt:", module_version("tensorrt"))
try:
    import onnxruntime as ort
    print("onnxruntime:", ort.__version__)
    print("onnxruntime providers:", ort.get_available_providers())
except Exception as exc:
    print("onnxruntime import failed:", repr(exc))
if shutil.which("trtexec"):
    try:
        print("trtexec:", subprocess.check_output(["trtexec", "--version"], text=True, stderr=subprocess.STDOUT).strip())
    except Exception as exc:
        print("trtexec --version failed:", repr(exc))
else:
    print("trtexec: not found")
PY
    echo
    echo "=== trtexec --version ==="
    if command -v trtexec >/dev/null 2>&1; then trtexec --version; else echo "trtexec not found"; fi
  } | tee "${txt}"

  "${PYTHON_BIN}" - "${json}" <<'PY'
import json
import shutil
import subprocess
import sys
from pathlib import Path

output = Path(sys.argv[1])
report = {
    "python_version": sys.version,
    "trtexec_path": shutil.which("trtexec"),
}
try:
    import torch
    report["torch_version"] = torch.__version__
    report["torch_cuda_version"] = torch.version.cuda
    report["torch_cuda_available"] = bool(torch.cuda.is_available())
    report["torch_cuda_device_count"] = int(torch.cuda.device_count())
    report["torch_cuda_device_names"] = [
        torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
    ] if torch.cuda.is_available() else []
except Exception as exc:
    report["torch_error"] = repr(exc)
    report["torch_cuda_available"] = False
    report["torch_cuda_device_count"] = 0
    report["torch_cuda_device_names"] = []

try:
    import tensorrt as trt
    report["tensorrt_version"] = getattr(trt, "__version__", "unknown")
except Exception as exc:
    report["tensorrt_error"] = repr(exc)
    report["tensorrt_version"] = None

try:
    import onnxruntime as ort
    report["onnxruntime_version"] = ort.__version__
    report["onnxruntime_available_providers"] = ort.get_available_providers()
except Exception as exc:
    report["onnxruntime_error"] = repr(exc)
    report["onnxruntime_version"] = None
    report["onnxruntime_available_providers"] = []

if report["trtexec_path"]:
    try:
        report["trtexec_version"] = subprocess.check_output(
            ["trtexec", "--version"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except Exception as exc:
        report["trtexec_version_error"] = repr(exc)
else:
    report["trtexec_version"] = None

missing = []
if report.get("tensorrt_version") is None:
    missing.append("tensorrt python module")
if report.get("onnxruntime_version") is None:
    missing.append("onnxruntime-gpu python module")
if not report.get("trtexec_path"):
    missing.append("trtexec binary")
report["missing_runtime_requirements"] = missing

output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
if not report.get("torch_cuda_available"):
    raise SystemExit(
        "CUDA GPU is not available. Refusing to run H20 GPU benchmark.\n"
        "Run on the H20 GPU machine/container and check `nvidia-smi`, scheduler GPU allocation, "
        "and CUDA_VISIBLE_DEVICES. For install help: "
        "bash scripts/run_h20_decoder_only_sft_prune_trt24.sh --env-help"
    )
if int(report.get("torch_cuda_device_count") or 0) < 1:
    raise SystemExit(
        "No visible CUDA devices. Check CUDA_VISIBLE_DEVICES/--gpus and scheduler/container GPU visibility. "
        "For install help: bash scripts/run_h20_decoder_only_sft_prune_trt24.sh --env-help"
    )
if missing:
    raise SystemExit(
        "Missing runtime requirements: "
        + ", ".join(missing)
        + ". Install Python deps with `python -m pip install --upgrade -r requirements.txt "
        "--extra-index-url https://pypi.nvidia.com`; install TensorRT SDK/container for trtexec."
    )
PY
}

write_sft_config() {
  local path="${OUTPUT_DIR}/generated_configs/sft_5epoch_seq64_fp16.yaml"
  cat > "${path}" <<EOF
model_name_or_path: "${BASE_MODEL}"
train_file: "${TRAIN_JSONL}"
eval_file: "${TRAIN_JSONL}"
benchmark_file: "${IOT200_JSONL}"
output_dir: "${DENSE_TRAIN_DIR}"

max_seq_length: ${SEQ_LEN}
max_new_tokens: ${MAX_NEW_TOKENS}
benchmark_runs: 1
top_k_exact_match: 5

num_train_epochs: ${EPOCHS}
learning_rate: ${LEARNING_RATE}
warmup_ratio: ${WARMUP_RATIO}
weight_decay: ${WEIGHT_DECAY}
lr_scheduler_type: cosine
max_grad_norm: 1.0

bf16: false
fp16: true
load_in_training_dtype: true
tf32: true
attn_implementation: eager
sdp_flash: false
sdp_mem_efficient: false
sdp_math: true

per_device_train_batch_size: ${BATCH_SIZE}
per_device_eval_batch_size: ${EVAL_BATCH_SIZE}
gradient_accumulation_steps: ${GRAD_ACCUM_STEPS}

gradient_checkpointing: false
flash_attention: false
torch_compile: false

logging_steps: 10
save_strategy: epoch
eval_strategy: none
save_final_only: true
save_total_limit: 1

dataloader_num_workers: 8
dataloader_pin_memory: true
persistent_workers: true
group_by_length: true
remove_unused_columns: false

generation:
  max_new_tokens: ${MAX_NEW_TOKENS}
  do_sample: false
  num_beams: 1
EOF
  printf '%s\n' "${path}"
}

ensure_fp16_checkpoint() {
  local model_dir="$1"
  echo "[fp16] ensuring checkpoint weights save as torch.float16: ${model_dir}"
  "${PYTHON_BIN}" - "${model_dir}" "${TRUST_REMOTE_CODE}" <<'PY'
import gc
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

path = Path(sys.argv[1]).expanduser()
trust = sys.argv[2] == "1"
model = AutoModelForCausalLM.from_pretrained(
    str(path),
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    trust_remote_code=trust,
)
model.save_pretrained(path, safe_serialization=True)
tokenizer = AutoTokenizer.from_pretrained(str(path), trust_remote_code=trust)
tokenizer.save_pretrained(path)
del model
gc.collect()
PY
}

train_sft() {
  local config_path
  config_path="$(write_sft_config)"
  echo
  echo "[train] 5-epoch FP16 SFT on ${NPROC_PER_NODE} GPU(s)"
  echo "  base model: ${BASE_MODEL}"
  echo "  train data: ${TRAIN_JSONL}"
  echo "  output:     ${DENSE_CKPT}"
  if [[ "${SKIP_TRAIN:-0}" == "1" ]]; then
    echo "[skip] train; expecting existing checkpoint at ${DENSE_CKPT}"
    return
  fi
  CUDA_VISIBLE_DEVICES="${GPUS}" torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" scripts/train.py \
    --config "${config_path}"
  if [[ ! -d "${DENSE_TRAIN_DIR}/final" || ! -f "${DENSE_TRAIN_DIR}/final/config.json" ]]; then
    echo "SFT final checkpoint not found: ${DENSE_TRAIN_DIR}/final" >&2
    exit 1
  fi
  rm -rf "${DENSE_CKPT}"
  mkdir -p "${DENSE_CKPT}"
  cp -R "${DENSE_TRAIN_DIR}/final/." "${DENSE_CKPT}/"
  ensure_fp16_checkpoint "${DENSE_CKPT}"
}

write_prune_config() {
  local path="${OUTPUT_DIR}/generated_configs/prune_nvidia_2_4_seq64.yaml"
  cat > "${path}" <<EOF
run:
  seed: 42
model:
  block_size: ${SEQ_LEN}
train:
  batch_size: 2
prune:
  base_model: "${DENSE_CKPT}"
  output_dir: "${PRUNED_CKPT}"
  method: 2of4
  sparsity: 0.5
  scope: transformer_linears
  sparsity_denominator: prunable
  granularity: global
  include_lm_head: false
  attn_implementation: eager
  calibration_data_path: "${TRAIN_JSONL}"
  calibration_batches: 128
  max_length: ${SEQ_LEN}
  batch_size: 2
  num_workers: 0
  recovery_steps: 0
  overwrite: true
EOF
  printf '%s\n' "${path}"
}

prune_nvidia_2_4() {
  local config_path
  config_path="$(write_prune_config)"
  echo
  echo "[prune] NVIDIA 2:4 structured pruning"
  echo "  dense:  ${DENSE_CKPT}"
  echo "  pruned: ${PRUNED_CKPT}"
  if [[ "${SKIP_PRUNE:-0}" == "1" ]]; then
    echo "[skip] prune; expecting existing checkpoint at ${PRUNED_CKPT}"
    return
  fi
  "${PYTHON_BIN}" scripts/prune.py \
    --config "${config_path}" \
    --method 2of4 \
    --checkpoint "${DENSE_CKPT}" \
    --output-dir "${PRUNED_CKPT}"
  ensure_fp16_checkpoint "${PRUNED_CKPT}"
}

verify_2_4_sparsity() {
  echo
  echo "[verify] exact NVIDIA 2:4 sparsity report"
  "${PYTHON_BIN}" - "${PRUNED_CKPT}" "${OUTPUT_DIR}/reports/sparsity_2_4_report.json" "${TRUST_REMOTE_CODE}" <<'PY'
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

model_path = Path(sys.argv[1]).expanduser()
output_path = Path(sys.argv[2]).expanduser()
trust = sys.argv[3] == "1"
model = AutoModelForCausalLM.from_pretrained(
    str(model_path),
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    trust_remote_code=trust,
)

per_layer = []
total_checked_weights = 0
total_blocks = 0
exact_blocks = 0
eligible_blocks = 0
total_zero_count = 0
non_compliant_layers = []

for name, module in model.named_modules():
    if not isinstance(module, torch.nn.Linear):
        continue
    weight = module.weight.detach().cpu()
    if weight.ndim != 2:
        continue
    out_features, in_features = weight.shape
    trim = (in_features // 4) * 4
    trailing = int(out_features * (in_features - trim))
    if trim == 0:
        layer = {
            "layer": name,
            "shape": [int(out_features), int(in_features)],
            "total_blocks": 0,
            "exact_2_zero_blocks": 0,
            "at_least_2_zero_blocks": 0,
            "exact_2_zero_block_pct": 0.0,
            "tensorrt_eligible_block_pct": 0.0,
            "total_zero_count": int((weight == 0).sum().item()),
            "total_sparsity_pct": float((weight == 0).float().mean().item() * 100.0) if weight.numel() else 0.0,
            "trailing_weights_not_checked": trailing,
            "compliant": False,
        }
        per_layer.append(layer)
        non_compliant_layers.append(name)
        continue
    grouped = weight[:, :trim].reshape(out_features, trim // 4, 4)
    zero_counts = (grouped == 0).sum(dim=2)
    blocks = int(zero_counts.numel())
    exact = int((zero_counts == 2).sum().item())
    eligible = int((zero_counts >= 2).sum().item())
    zeros = int((weight == 0).sum().item())
    checked = int(out_features * trim)
    compliant = exact == blocks and trailing == 0
    layer = {
        "layer": name,
        "shape": [int(out_features), int(in_features)],
        "checked_weights": checked,
        "total_blocks": blocks,
        "exact_2_zero_blocks": exact,
        "at_least_2_zero_blocks": eligible,
        "exact_2_zero_block_pct": 100.0 * exact / float(blocks or 1),
        "tensorrt_eligible_block_pct": 100.0 * eligible / float(blocks or 1),
        "total_zero_count": zeros,
        "total_sparsity_pct": 100.0 * zeros / float(weight.numel() or 1),
        "trailing_weights_not_checked": trailing,
        "compliant": compliant,
    }
    per_layer.append(layer)
    total_checked_weights += checked
    total_blocks += blocks
    exact_blocks += exact
    eligible_blocks += eligible
    total_zero_count += zeros
    if not compliant:
        non_compliant_layers.append(name)

report = {
    "checkpoint": str(model_path),
    "check": "contiguous groups of 4 along Linear dim=1 / input-reduction K dimension",
    "total_checked_weights": total_checked_weights,
    "total_blocks": total_blocks,
    "exact_2_zero_blocks": exact_blocks,
    "at_least_2_zero_blocks": eligible_blocks,
    "exact_2_zero_block_pct": 100.0 * exact_blocks / float(total_blocks or 1),
    "tensorrt_eligible_block_pct": 100.0 * eligible_blocks / float(total_blocks or 1),
    "total_zero_count": total_zero_count,
    "total_sparsity_pct": 100.0 * total_zero_count / float(total_checked_weights or 1),
    "per_layer": per_layer,
    "non_compliant_layers": non_compliant_layers,
}
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
if non_compliant_layers:
    raise SystemExit(f"Found non-compliant Linear layers: {non_compliant_layers[:10]}")
PY
}

run_pytorch_eval() {
  local variant="$1"
  local model_path="$2"
  local dataset_name="$3"
  local dataset_path="$4"
  local out_root="${OUTPUT_DIR}/eval/${variant}"
  local run_dir="${out_root}/${dataset_name}_run"
  local pred_out="${out_root}/${dataset_name}_predictions.jsonl"
  local metrics_out="${out_root}/${dataset_name}_metrics.json"
  mkdir -p "${out_root}"
  rm -rf "${run_dir}"
  echo
  echo "[eval:pytorch] ${variant} on ${dataset_name} with ${NPROC_PER_NODE} GPU shard(s)"
  CUDA_VISIBLE_DEVICES="${GPUS}" torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" scripts/eval_prompt_response.py \
    --model-path "${model_path}" \
    --dataset-file "${dataset_path}" \
    --output-dir "${run_dir}" \
    --no-unique-output-dir \
    --device auto \
    --dtype fp16 \
    --batch-size "${EVAL_BATCH_SIZE}" \
    --max-length "${SEQ_LEN}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --exact-match-top-k 5 \
    --comparison-mode "${COMPARISON_MODE}" \
    --max-new-token-hit-rate-threshold "${MAX_NEW_TOKEN_HIT_RATE_THRESHOLD}" \
    --no-fail-on-leakage
  cp "${run_dir}/prompt_response_eval_predictions.jsonl" "${pred_out}"
  if [[ -f "${run_dir}/prompt_response_eval_summary.json" ]]; then
    cp "${run_dir}/prompt_response_eval_summary.json" "${metrics_out}"
  else
    cp "${run_dir}/metrics.json" "${metrics_out}"
  fi
}

export_onnx_model() {
  local label="$1"
  local model_path="$2"
  local onnx_dir="$3"
  local model_onnx="$4"
  echo
  echo "[onnx] export ${label}"
  if [[ "${SKIP_EXPORT:-0}" != "1" ]]; then
    "${PYTHON_BIN}" scripts/export_decoder_onnx.py \
      --model-path "${model_path}" \
      --onnx-dir "${onnx_dir}" \
      --dtype fp16 \
      --opset 18 \
      --seq-len "${SEQ_LEN}" \
      --batch-size 1 \
      --attn-implementation eager \
      --no-export-cache \
      "${TRUST_ARGS[@]}" \
      --overwrite
    cp "${onnx_dir}/model_decoder_nocache.onnx" "${model_onnx}"
  else
    echo "[skip] export; expecting ${model_onnx}"
  fi
}

inspect_onnx() {
  local onnx_path="$1"
  local report_path="$2"
  echo "[onnx] inspect ${onnx_path}"
  "${PYTHON_BIN}" - "${onnx_path}" "${report_path}" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

import onnx

onnx_path = Path(sys.argv[1]).expanduser()
report_path = Path(sys.argv[2]).expanduser()
onnx.checker.check_model(str(onnx_path))
model = onnx.load(str(onnx_path), load_external_data=False)
dtype_name = onnx.TensorProto.DataType.Name

def shape_of(value):
    dims = []
    shape = value.type.tensor_type.shape
    for dim in shape.dim:
        if dim.dim_param:
            dims.append(dim.dim_param)
        elif dim.HasField("dim_value"):
            dims.append(int(dim.dim_value))
        else:
            dims.append("?")
    return dims

input_shapes = {value.name: shape_of(value) for value in model.graph.input}
output_shapes = {value.name: shape_of(value) for value in model.graph.output}
input_dtypes = {value.name: dtype_name(value.type.tensor_type.elem_type) for value in model.graph.input}
output_dtypes = {value.name: dtype_name(value.type.tensor_type.elem_type) for value in model.graph.output}
initializer_dtypes = Counter(dtype_name(tensor.data_type) for tensor in model.graph.initializer)
floating_initializer_total = sum(count for dtype, count in initializer_dtypes.items() if dtype in {"FLOAT", "FLOAT16", "BFLOAT16", "DOUBLE"})
floating_initializer_fp16 = initializer_dtypes.get("FLOAT16", 0)
dynamic_axes = {
    name: [index for index, dim in enumerate(shape) if isinstance(dim, str) or dim == "?"]
    for name, shape in {**input_shapes, **output_shapes}.items()
}
dynamic_axes = {name: axes for name, axes in dynamic_axes.items() if axes}
report = {
    "onnx_path": str(onnx_path),
    "exists": onnx_path.exists(),
    "input_names": [value.name for value in model.graph.input],
    "output_names": [value.name for value in model.graph.output],
    "input_shapes": input_shapes,
    "output_shapes": output_shapes,
    "input_dtypes": input_dtypes,
    "output_dtypes": output_dtypes,
    "dynamic_axes": dynamic_axes,
    "initializer_dtypes": dict(initializer_dtypes),
    "floating_initializer_total": int(floating_initializer_total),
    "floating_initializer_fp16": int(floating_initializer_fp16),
    "weights_are_fp16": bool(floating_initializer_total and floating_initializer_fp16 == floating_initializer_total),
    "onnx_model_size_mb": onnx_path.stat().st_size / 1024**2,
}
report_path.parent.mkdir(parents=True, exist_ok=True)
report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
if not report["weights_are_fp16"]:
    print("[warning] Not all floating ONNX initializers are FLOAT16. See", report_path)
PY
}

shape_arg_for_trtexec() {
  local onnx_path="$1"
  "${PYTHON_BIN}" - "${onnx_path}" "${SEQ_LEN}" <<'PY'
import sys
from pathlib import Path
import onnx

onnx_path = Path(sys.argv[1]).expanduser()
seq_len = int(sys.argv[2])
model = onnx.load(str(onnx_path), load_external_data=False)
specs = []
for value in model.graph.input:
    name = value.name
    dims = value.type.tensor_type.shape.dim
    rank = len(dims)
    if rank == 2:
        specs.append(f"{name}:1x{seq_len}")
    else:
        shape = []
        for index, dim in enumerate(dims):
            if index == 0:
                shape.append("1")
            elif dim.HasField("dim_value") and int(dim.dim_value) > 0:
                shape.append(str(int(dim.dim_value)))
            else:
                shape.append(str(seq_len if index == 1 else 1))
        specs.append(f"{name}:{'x'.join(shape)}")
print("--shapes=" + ",".join(specs))
PY
}

build_trt_engine() {
  local label="$1"
  local onnx_path="$2"
  local engine_path="$3"
  local sparsity_mode="$4"
  local log_path="$5"
  mkdir -p "$(dirname "${engine_path}")"
  rm -f "${engine_path}"
  echo
  echo "[trt] build ${label} (${sparsity_mode})"
  if [[ "${SKIP_BUILD:-0}" == "1" ]]; then
    echo "[skip] build; expecting ${engine_path}"
    return
  fi
  if command -v trtexec >/dev/null 2>&1; then
    local shape_arg
    shape_arg="$(shape_arg_for_trtexec "${onnx_path}")"
    trtexec \
      --onnx="${onnx_path}" \
      --saveEngine="${engine_path}" \
      --fp16 \
      --sparsity="${sparsity_mode}" \
      "${shape_arg}" \
      --memPoolSize="workspace:${WORKSPACE_MIB}" \
      --verbose \
      --profilingVerbosity=detailed \
      2>&1 | tee "${log_path}"
  else
    echo "[warning] trtexec not found; falling back to scripts/build_trt_engines.py" | tee "${log_path}"
    local tmp_dir="${OUTPUT_DIR}/engines/_${label}_python_build"
    local sparse_args=()
    if [[ "${sparsity_mode}" == "enable" ]]; then
      sparse_args=(--sparse-weights)
    fi
    "${PYTHON_BIN}" scripts/build_trt_engines.py \
      --onnx "${onnx_path}" \
      --output-dir "${tmp_dir}" \
      --precision fp16 \
      --model-path "${DENSE_CKPT}" \
      --min_seq_len "${SEQ_LEN}" \
      --opt_seq_len "${SEQ_LEN}" \
      --max_seq_len "${SEQ_LEN}" \
      --batch_size 1 \
      --workspace-gb "$(( WORKSPACE_MIB / 1024 ))" \
      "${sparse_args[@]}" \
      "${TRUST_ARGS[@]}" \
      --overwrite \
      2>&1 | tee -a "${log_path}"
    cp "${tmp_dir}/model_fp16.engine" "${engine_path}"
  fi
  if [[ ! -f "${engine_path}" ]]; then
    echo "TensorRT engine was not created: ${engine_path}" >&2
    exit 1
  fi
}

parse_sparse_tactics_log() {
  "${PYTHON_BIN}" - "${OUTPUT_DIR}/logs/build_nvidia_2_4_sparse_fp16.log" "${OUTPUT_DIR}/reports/trt_sparse_tactics_report.json" <<'PY'
import json
import re
import sys
from pathlib import Path

log_path = Path(sys.argv[1]).expanduser()
report_path = Path(sys.argv[2]).expanduser()
text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
lines = [line for line in text.splitlines() if re.search(r"spars|sparse tactic|eligible|chose", line, re.I)]
eligible = any(re.search(r"eligible.*sparse|sparse.*eligible", line, re.I) for line in lines)
selected = any(re.search(r"(chose|selected|using).{0,80}sparse", line, re.I) for line in lines)
if any(re.search(r"0\s+sparse\s+tactics|sparse tactics:\s*0", line, re.I) for line in lines):
    selected = False
report = {
    "log_path": str(log_path),
    "sparse_tactics_eligible": bool(eligible),
    "sparse_tactics_selected": bool(selected),
    "evidence_lines": lines[:200],
}
report_path.parent.mkdir(parents=True, exist_ok=True)
report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
PY
}

write_unavailable_latency() {
  local path="$1"
  local runtime="$2"
  local reason="$3"
  "${PYTHON_BIN}" - "${path}" "${runtime}" "${reason}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1]).expanduser()
payload = {"available": False, "runtime": sys.argv[2], "error": sys.argv[3]}
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY
}

benchmark_runtime() {
  CUDA_VISIBLE_DEVICES="${LATENCY_GPU}" "${PYTHON_BIN}" - "$@" <<'PY'
import argparse
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

def percentile(values, q):
    if not values:
        return None
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lower = math.floor(pos)
    upper = math.ceil(pos)
    if lower == upper:
        return ordered[int(pos)]
    return ordered[lower] * (upper - pos) + ordered[upper] * (pos - lower)

def common_stats(latencies):
    total = sum(latencies) / 1000.0
    return {
        "mean_latency_ms": statistics.mean(latencies) if latencies else None,
        "median_latency_ms": statistics.median(latencies) if latencies else None,
        "p95_latency_ms": percentile(latencies, 0.95),
        "p99_latency_ms": percentile(latencies, 0.99),
        "throughput_qps": (len(latencies) / total) if total > 0 else None,
    }

def file_mb(path):
    if not path:
        return None
    p = Path(path).expanduser()
    return p.stat().st_size / 1024**2 if p.exists() else None

parser = argparse.ArgumentParser()
parser.add_argument("--runtime", required=True, choices=["pytorch", "onnx", "trt"])
parser.add_argument("--model-path", default=None)
parser.add_argument("--onnx-path", default=None)
parser.add_argument("--engine-path", default=None)
parser.add_argument("--providers", default="")
parser.add_argument("--output-json", required=True)
parser.add_argument("--label", required=True)
parser.add_argument("--architecture", default="Decoder-only")
parser.add_argument("--precision", default="FP16")
parser.add_argument("--sparsity", default="Dense")
parser.add_argument("--seq-len", type=int, default=64)
parser.add_argument("--batch-size", type=int, default=1)
parser.add_argument("--warmup-iters", type=int, default=100)
parser.add_argument("--measure-iters", type=int, default=1000)
parser.add_argument("--trust-remote-code", action="store_true")
args = parser.parse_args()

result = {
    "available": False,
    "label": args.label,
    "architecture": args.architecture,
    "runtime": args.runtime,
    "precision": args.precision,
    "sparsity": args.sparsity,
    "seq_len": int(args.seq_len),
    "batch_size": int(args.batch_size),
    "model_path": args.model_path,
    "onnx_path": args.onnx_path,
    "engine_path": args.engine_path,
    "onnx_size_mb": file_mb(args.onnx_path),
    "engine_size_mb": file_mb(args.engine_path),
}

try:
    if args.runtime == "pytorch":
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available for PyTorch benchmark.")
        device = torch.device("cuda:0")
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=args.trust_remote_code)
        token_id = getattr(tokenizer, "bos_token_id", None) or getattr(tokenizer, "eos_token_id", None) or 1
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=args.trust_remote_code,
        ).to(device)
        model.eval()
        input_ids = torch.full((args.batch_size, args.seq_len), int(token_id), dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)
        with torch.no_grad():
            for _ in range(args.warmup_iters):
                model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(device)
            latencies = []
            for _ in range(args.measure_iters):
                torch.cuda.synchronize()
                start = time.perf_counter()
                model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
                torch.cuda.synchronize()
                latencies.append((time.perf_counter() - start) * 1000.0)
        result.update(common_stats(latencies))
        result.update({
            "available": True,
            "provider": "torch.cuda",
            "gpu": torch.cuda.get_device_name(0),
            "cuda_version": torch.version.cuda,
            "peak_gpu_memory_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
            "model_type": getattr(model.config, "model_type", None),
        })

    elif args.runtime == "onnx":
        import numpy as np
        import onnxruntime as ort

        providers = [part for part in args.providers.split(",") if part]
        available = ort.get_available_providers()
        missing = [provider for provider in providers if provider not in available]
        if missing:
            raise RuntimeError(f"Requested ORT providers unavailable: {missing}; available={available}")
        session = ort.InferenceSession(args.onnx_path, providers=providers)
        feed = {}
        for item in session.get_inputs():
            dtype = np.int64 if "int64" in item.type else np.int32
            shape = []
            for index, dim in enumerate(item.shape):
                if index == 0:
                    shape.append(args.batch_size)
                elif index == 1:
                    shape.append(args.seq_len)
                elif isinstance(dim, int) and dim > 0:
                    shape.append(dim)
                else:
                    shape.append(1)
            feed[item.name] = np.ones(tuple(shape), dtype=dtype)
        for _ in range(args.warmup_iters):
            session.run(None, feed)
        latencies = []
        for _ in range(args.measure_iters):
            start = time.perf_counter()
            session.run(None, feed)
            latencies.append((time.perf_counter() - start) * 1000.0)
        result.update(common_stats(latencies))
        result.update({
            "available": True,
            "provider": ",".join(session.get_providers()),
            "onnxruntime_available_providers": available,
            "peak_gpu_memory_mb": None,
        })

    elif args.runtime == "trt":
        import numpy as np
        import tensorrt as trt
        from trt_edge_common import TensorRTEngineRunner

        runner = TensorRTEngineRunner(args.engine_path)
        try:
            feed = {}
            for name in runner.input_names:
                feed[name] = np.ones((args.batch_size, args.seq_len), dtype=np.int32)
            for _ in range(args.warmup_iters):
                runner.infer(feed)
            free_start, total_memory = runner.cuda.mem_info()
            min_free = free_start
            latencies = []
            for _ in range(args.measure_iters):
                start = time.perf_counter()
                runner.infer(feed)
                latencies.append((time.perf_counter() - start) * 1000.0)
                free_now, _ = runner.cuda.mem_info()
                min_free = min(min_free, free_now)
            result.update(common_stats(latencies))
            result.update({
                "available": True,
                "provider": "native TensorRT",
                "tensorrt_version": getattr(trt, "__version__", "unknown"),
                "peak_gpu_memory_mb": (total_memory - min_free) / 1024**2 if total_memory else None,
            })
        finally:
            runner.close()

except Exception as exc:
    result["error"] = repr(exc)

Path(args.output_json).expanduser().parent.mkdir(parents=True, exist_ok=True)
Path(args.output_json).expanduser().write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
if not result.get("available"):
    raise SystemExit(f"Benchmark unavailable for {args.label}: {result.get('error')}")
PY
}

run_latency_benchmarks() {
  echo
  echo "[latency] batch-1 seq-${SEQ_LEN} single-GPU runtime benchmarks on CUDA_VISIBLE_DEVICES=${LATENCY_GPU}"
  if [[ "${SKIP_LATENCY:-0}" == "1" ]]; then
    echo "[skip] latency"
    return
  fi

  benchmark_runtime \
    --runtime pytorch \
    --model-path "${DENSE_CKPT}" \
    --output-json "${OUTPUT_DIR}/results/latency_dense_pytorch_fp16.json" \
    --label "dense_sft_fp16_pytorch" \
    --precision FP16 \
    --sparsity Dense \
    --seq-len "${SEQ_LEN}" \
    --batch-size 1 \
    --warmup-iters "${WARMUP_ITERS}" \
    --measure-iters "${MEASURE_ITERS}" \
    "${TRUST_ARGS[@]}"

  benchmark_runtime \
    --runtime pytorch \
    --model-path "${PRUNED_CKPT}" \
    --output-json "${OUTPUT_DIR}/results/latency_nvidia_2_4_pytorch_fp16.json" \
    --label "nvidia_2_4_sft_fp16_pytorch" \
    --precision FP16 \
    --sparsity "NVIDIA 2:4" \
    --seq-len "${SEQ_LEN}" \
    --batch-size 1 \
    --warmup-iters "${WARMUP_ITERS}" \
    --measure-iters "${MEASURE_ITERS}" \
    "${TRUST_ARGS[@]}"

  if "${PYTHON_BIN}" - <<'PY'
import onnxruntime as ort
raise SystemExit(0 if "CUDAExecutionProvider" in ort.get_available_providers() else 1)
PY
  then
    benchmark_runtime \
      --runtime onnx \
      --onnx-path "${DENSE_ONNX}" \
      --providers CUDAExecutionProvider \
      --output-json "${OUTPUT_DIR}/results/latency_dense_ort_cuda_fp16.json" \
      --label "dense_sft_fp16_onnxruntime_cuda" \
      --precision FP16 \
      --sparsity Dense \
      --seq-len "${SEQ_LEN}" \
      --batch-size 1 \
      --warmup-iters "${WARMUP_ITERS}" \
      --measure-iters "${MEASURE_ITERS}"
  else
    write_unavailable_latency "${OUTPUT_DIR}/results/latency_dense_ort_cuda_fp16.json" "ONNX Runtime CUDA" "CUDAExecutionProvider is unavailable; CPU ONNX is not used for GPU speedup claims."
  fi

  if "${PYTHON_BIN}" - <<'PY'
import onnxruntime as ort
raise SystemExit(0 if "TensorrtExecutionProvider" in ort.get_available_providers() else 1)
PY
  then
    benchmark_runtime \
      --runtime onnx \
      --onnx-path "${DENSE_ONNX}" \
      --providers TensorrtExecutionProvider,CUDAExecutionProvider \
      --output-json "${OUTPUT_DIR}/results/latency_dense_ort_trt_fp16.json" \
      --label "dense_sft_fp16_onnxruntime_tensorrt" \
      --precision FP16 \
      --sparsity Dense \
      --seq-len "${SEQ_LEN}" \
      --batch-size 1 \
      --warmup-iters "${WARMUP_ITERS}" \
      --measure-iters "${MEASURE_ITERS}"
  else
    write_unavailable_latency "${OUTPUT_DIR}/results/latency_dense_ort_trt_fp16.json" "ONNX Runtime TensorRT EP" "TensorrtExecutionProvider is unavailable; native TensorRT path is used."
  fi

  benchmark_runtime \
    --runtime trt \
    --engine-path "${DENSE_ENGINE}" \
    --onnx-path "${DENSE_ONNX}" \
    --output-json "${OUTPUT_DIR}/results/latency_dense_native_trt_fp16.json" \
    --label "dense_sft_fp16_native_tensorrt" \
    --precision FP16 \
    --sparsity Dense \
    --seq-len "${SEQ_LEN}" \
    --batch-size 1 \
    --warmup-iters "${WARMUP_ITERS}" \
    --measure-iters "${MEASURE_ITERS}"

  benchmark_runtime \
    --runtime trt \
    --engine-path "${PRUNED_ENGINE}" \
    --onnx-path "${PRUNED_ONNX}" \
    --output-json "${OUTPUT_DIR}/results/latency_nvidia_2_4_native_trt_fp16.json" \
    --label "nvidia_2_4_sft_fp16_native_tensorrt" \
    --precision FP16 \
    --sparsity "NVIDIA 2:4" \
    --seq-len "${SEQ_LEN}" \
    --batch-size 1 \
    --warmup-iters "${WARMUP_ITERS}" \
    --measure-iters "${MEASURE_ITERS}"
}

run_trt_eval() {
  local variant="$1"
  local model_path="$2"
  local engine_path="$3"
  local onnx_path="$4"
  local dataset_name="$5"
  local dataset_path="$6"
  local out_dir="${OUTPUT_DIR}/eval_trt/${variant}/${dataset_name}"
  echo
  echo "[eval:trt] ${variant} on ${dataset_name}"
  CUDA_VISIBLE_DEVICES="${LATENCY_GPU}" "${PYTHON_BIN}" scripts/eval_trt_prompt_response.py \
    --engine "${engine_path}" \
    --model-path "${model_path}" \
    --dataset "${dataset_path}" \
    --output-dir "${out_dir}" \
    --precision fp16 \
    --variant "${variant}" \
    --runtime "native TensorRT" \
    --onnx-path "${onnx_path}" \
    --batch-size 1 \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --max-seq-len "${SEQ_LEN}" \
    --exact-match-top-k 5 \
    --comparison-mode "${COMPARISON_MODE}" \
    --prompt-format "${PROMPT_FORMAT}" \
    "${SYSTEM_PROMPT_ARGS[@]}" \
    "${TRUST_ARGS[@]}" \
    --overwrite
}

write_int8_status() {
  if [[ "${RUN_INT8}" != "1" ]]; then
    "${PYTHON_BIN}" - "${OUTPUT_DIR}/reports/int8_status.json" <<'PY'
import json, sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({
  "int8_enabled": False,
  "status": "skipped",
  "reason": "RUN_INT8 is not set. No INT8 rows are included."
}, indent=2) + "\n")
PY
    return
  fi

  if [[ -z "${INT8_CALIB_CACHE}" || ! -f "${INT8_CALIB_CACHE}" ]]; then
    "${PYTHON_BIN}" - "${OUTPUT_DIR}/reports/int8_status.json" <<'PY'
import json, os, sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({
  "int8_enabled": True,
  "status": "skipped",
  "reason": "RUN_INT8=1 but INT8_CALIB_CACHE does not point to a real calibration cache. Random INT8 ranges are intentionally not used."
}, indent=2) + "\n")
PY
    return
  fi

  echo "[int8] Real INT8 calibration cache detected: ${INT8_CALIB_CACHE}"
  echo "[int8] This script records INT8 readiness but leaves paper EM rows disabled unless you add calibrated/QDQ eval support."
  "${PYTHON_BIN}" - "${OUTPUT_DIR}/reports/int8_status.json" "${INT8_CALIB_CACHE}" <<'PY'
import json, sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({
  "int8_enabled": True,
  "status": "calibration_cache_available",
  "calibration_cache": sys.argv[2],
  "note": "Build calibrated INT8 engines only when the cache/dataloader matches the exported ONNX input distribution."
}, indent=2) + "\n")
PY
}

write_final_results() {
  "${PYTHON_BIN}" - "${OUTPUT_DIR}" "${SEQ_LEN}" <<'PY'
import csv
import json
import math
import sys
from pathlib import Path

out = Path(sys.argv[1]).expanduser()
seq_len = int(sys.argv[2])

def read_json(path):
    path = Path(path)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}

env = read_json(out / "env/env_report.json")
sparse = read_json(out / "reports/trt_sparse_tactics_report.json")
int8 = read_json(out / "reports/int8_status.json")
sparsity = read_json(out / "reports/sparsity_2_4_report.json")

def metric_pair(path):
    data = read_json(path)
    em1 = data.get("exact_match_accuracy")
    em5 = data.get("exact_match_at_5_accuracy", data.get("exact_match_at_top_k_accuracy"))
    return em1, em5

dense_iot = metric_pair(out / "eval/dense_sft_fp16/iot200_metrics.json")
dense_train = metric_pair(out / "eval/dense_sft_fp16/train_metrics.json")
pruned_iot = metric_pair(out / "eval/nvidia_2_4_sft_fp16/iot200_metrics.json")
pruned_train = metric_pair(out / "eval/nvidia_2_4_sft_fp16/train_metrics.json")
trt_dense_iot = metric_pair(out / "eval_trt/dense_sft_fp16/iot200/prompt_response_eval_summary.json")
trt_dense_train = metric_pair(out / "eval_trt/dense_sft_fp16/train/prompt_response_eval_summary.json")
trt_pruned_iot = metric_pair(out / "eval_trt/nvidia_2_4_sft_fp16/iot200/prompt_response_eval_summary.json")
trt_pruned_train = metric_pair(out / "eval_trt/nvidia_2_4_sft_fp16/train/prompt_response_eval_summary.json")

def latency(path):
    return read_json(out / "results" / path)

def fmt(value):
    if value is None or value == "" or (isinstance(value, float) and math.isnan(value)):
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return value

def row(model, runtime, precision, sparsity_label, latency_data, em_iot=("", ""), em_train=("", ""), sparse_selected=""):
    return {
        "Model": model,
        "Architecture": latency_data.get("architecture") or "Decoder-only",
        "Runtime": runtime,
        "Precision": precision,
        "Sparsity": sparsity_label,
        "Seq. Len.": seq_len,
        "Batch Size": latency_data.get("batch_size", 1),
        "Latency": fmt(latency_data.get("mean_latency_ms")),
        "Median Lat.": fmt(latency_data.get("median_latency_ms")),
        "P95 Lat.": fmt(latency_data.get("p95_latency_ms")),
        "P99 Lat.": fmt(latency_data.get("p99_latency_ms")),
        "Throughput QPS": fmt(latency_data.get("throughput_qps")),
        "Memory": fmt(latency_data.get("peak_gpu_memory_mb")),
        "ONNX MB": fmt(latency_data.get("onnx_size_mb")),
        "Engine MB": fmt(latency_data.get("engine_size_mb")),
        "EM@1 IoT200": fmt(em_iot[0]),
        "EM@5 IoT200": fmt(em_iot[1]),
        "EM@1 Train": fmt(em_train[0]),
        "EM@5 Train": fmt(em_train[1]),
        "GPU": latency_data.get("gpu") or ", ".join(env.get("torch_cuda_device_names", [])),
        "Provider": latency_data.get("provider", ""),
        "Sparse Tactics Selected": sparse_selected,
        "Speedup vs Dense TRT FP16": "",
    }

rows = [
    row("dense_sft_fp16", "PyTorch", "FP16", "Dense", latency("latency_dense_pytorch_fp16.json"), dense_iot, dense_train, "false"),
    row("dense_sft_fp16", "ONNX Runtime CUDA", "FP16", "Dense", latency("latency_dense_ort_cuda_fp16.json"), ("", ""), ("", ""), "false"),
    row("dense_sft_fp16", "ONNX Runtime TensorRT EP", "FP16", "Dense", latency("latency_dense_ort_trt_fp16.json"), ("", ""), ("", ""), "false"),
    row("dense_sft_fp16", "native TensorRT", "FP16", "Dense", latency("latency_dense_native_trt_fp16.json"), trt_dense_iot, trt_dense_train, "false"),
    row("nvidia_2_4_sft_fp16", "PyTorch", "FP16", "NVIDIA 2:4", latency("latency_nvidia_2_4_pytorch_fp16.json"), pruned_iot, pruned_train, "false"),
    row("nvidia_2_4_sft_fp16", "native TensorRT", "FP16", "NVIDIA 2:4", latency("latency_nvidia_2_4_native_trt_fp16.json"), trt_pruned_iot, trt_pruned_train, str(bool(sparse.get("sparse_tactics_selected"))).lower()),
]

dense_trt = latency("latency_dense_native_trt_fp16.json")
pruned_trt = latency("latency_nvidia_2_4_native_trt_fp16.json")
dense_ms = dense_trt.get("mean_latency_ms")
pruned_ms = pruned_trt.get("mean_latency_ms")
speedup = None
throughput_gain = None
if dense_ms and pruned_ms and float(pruned_ms) > 0:
    speedup = float(dense_ms) / float(pruned_ms)
    rows[-1]["Speedup vs Dense TRT FP16"] = fmt(speedup)
    dense_qps = dense_trt.get("throughput_qps")
    pruned_qps = pruned_trt.get("throughput_qps")
    if dense_qps and float(dense_qps) > 0 and pruned_qps:
        throughput_gain = float(pruned_qps) / float(dense_qps)

fields = [
    "Model", "Architecture", "Runtime", "Precision", "Sparsity", "Seq. Len.",
    "Batch Size", "Latency", "Median Lat.", "P95 Lat.", "P99 Lat.",
    "Throughput QPS", "Memory", "ONNX MB", "Engine MB", "EM@1 IoT200",
    "EM@5 IoT200", "EM@1 Train", "EM@5 Train", "GPU", "Provider",
    "Sparse Tactics Selected", "Speedup vs Dense TRT FP16",
]
csv_path = out / "results/final_metrics.csv"
json_path = out / "results/final_metrics.json"
with csv_path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    for item in rows:
        writer.writerow({field: item.get(field, "") for field in fields})

payload = {
    "rows": rows,
    "main_speedup_dense_trt_fp16_over_nvidia_2_4_trt_fp16": speedup,
    "throughput_gain_nvidia_2_4_trt_fp16_over_dense_trt_fp16": throughput_gain,
    "sparse_tactics_selected": bool(sparse.get("sparse_tactics_selected")),
    "sparse_tactics_report": str(out / "reports/trt_sparse_tactics_report.json"),
    "int8_status": int8,
    "sparsity_report_summary": {
        "exact_2_zero_block_pct": sparsity.get("exact_2_zero_block_pct"),
        "tensorrt_eligible_block_pct": sparsity.get("tensorrt_eligible_block_pct"),
        "non_compliant_layers": sparsity.get("non_compliant_layers", []),
    },
}
json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

ort_providers = env.get("onnxruntime_available_providers", [])
cpu_only_onnx = ort_providers == ["CPUExecutionProvider"] or ("CUDAExecutionProvider" not in ort_providers and "TensorrtExecutionProvider" not in ort_providers)
speedup_claim = bool(sparse.get("sparse_tactics_selected")) and speedup is not None and speedup > 1.0
acc_delta_iot = None
if pruned_iot[0] is not None and dense_iot[0] is not None:
    acc_delta_iot = float(pruned_iot[0]) - float(dense_iot[0])

summary = f"""# H20 Decoder-only SFT + NVIDIA 2:4 TensorRT Summary

- GPU benchmark ran on CUDA: {bool(env.get("torch_cuda_available"))}; visible GPUs: {env.get("torch_cuda_device_names", [])}
- ONNX Runtime providers: {ort_providers}
- CPU ONNX fallback happened: {cpu_only_onnx}. CPU ONNX is excluded from the main speedup claim.
- TensorRT sparse tactics selected: {bool(sparse.get("sparse_tactics_selected"))}
- NVIDIA 2:4 exact block compliance: {sparsity.get("exact_2_zero_block_pct")}%; TensorRT-eligible blocks: {sparsity.get("tensorrt_eligible_block_pct")}%
- Real NVIDIA 2:4 speedup claim allowed: {speedup_claim}
- Dense TRT FP16 mean latency / NVIDIA 2:4 TRT FP16 mean latency: {speedup}
- NVIDIA 2:4 TRT FP16 QPS / dense TRT FP16 QPS: {throughput_gain}
- Dense PyTorch EM@1/EM@5 IoT200: {dense_iot[0]} / {dense_iot[1]}
- NVIDIA 2:4 PyTorch EM@1/EM@5 IoT200: {pruned_iot[0]} / {pruned_iot[1]}
- Dense PyTorch EM@1/EM@5 train: {dense_train[0]} / {dense_train[1]}
- NVIDIA 2:4 PyTorch EM@1/EM@5 train: {pruned_train[0]} / {pruned_train[1]}
- IoT200 EM@1 delta after pruning: {acc_delta_iot}
- INT8 status: {int8.get("status")}; {int8.get("reason", int8.get("note", ""))}

## Limitations

- Native TensorRT accuracy uses the repository no-cache autoregressive evaluator. If TensorRT generation fails in this environment, rely on the saved error reports and compare PyTorch/ONNX logits before making accuracy claims.
- ONNX Runtime CPUExecutionProvider rows are portability evidence only and are not part of GPU speedup claims.
- Sparse hardware speedup should be claimed only when sparse tactics were selected and the measured TensorRT latency improves.
"""
(out / "SUMMARY.md").write_text(summary, encoding="utf-8")
print(f"Wrote final CSV: {csv_path}")
print(f"Wrote final JSON: {json_path}")
print(f"Wrote summary: {out / 'SUMMARY.md'}")
PY
}

echo "H20 decoder-only SFT + NVIDIA 2:4 + TensorRT FP16 baseline"
echo "  base model:       ${BASE_MODEL}"
echo "  train data:       ${TRAIN_JSONL}"
echo "  IoT200 data:      ${IOT200_JSONL}"
echo "  output dir:       ${OUTPUT_DIR}"
echo "  GPUs:             ${GPUS}"
echo "  epochs:           ${EPOCHS}"
echo "  seq_len:          ${SEQ_LEN}"
echo "  SFT batch/GPU:    ${BATCH_SIZE}"
echo "  eval batch/GPU:   ${EVAL_BATCH_SIZE}"
echo "  latency iters:    warmup=${WARMUP_ITERS} measure=${MEASURE_ITERS}"

write_env_report
train_sft
prune_nvidia_2_4
verify_2_4_sparsity

if [[ "${SKIP_EVAL:-0}" != "1" ]]; then
  run_pytorch_eval "dense_sft_fp16" "${DENSE_CKPT}" "iot200" "${IOT200_JSONL}"
  run_pytorch_eval "dense_sft_fp16" "${DENSE_CKPT}" "train" "${TRAIN_JSONL}"
  run_pytorch_eval "nvidia_2_4_sft_fp16" "${PRUNED_CKPT}" "iot200" "${IOT200_JSONL}"
  run_pytorch_eval "nvidia_2_4_sft_fp16" "${PRUNED_CKPT}" "train" "${TRAIN_JSONL}"
else
  echo "[skip] PyTorch EM eval"
fi

export_onnx_model "dense_sft_fp16" "${DENSE_CKPT}" "${DENSE_ONNX_DIR}" "${DENSE_ONNX}"
export_onnx_model "nvidia_2_4_sft_fp16" "${PRUNED_CKPT}" "${PRUNED_ONNX_DIR}" "${PRUNED_ONNX}"
inspect_onnx "${DENSE_ONNX}" "${OUTPUT_DIR}/reports/onnx_inspection_dense_sft_fp16.json"
inspect_onnx "${PRUNED_ONNX}" "${OUTPUT_DIR}/reports/onnx_inspection_nvidia_2_4_sft_fp16.json"

build_trt_engine "dense_sft_fp16" "${DENSE_ONNX}" "${DENSE_ENGINE}" "disable" "${OUTPUT_DIR}/logs/build_dense_fp16.log"
build_trt_engine "nvidia_2_4_sft_fp16" "${PRUNED_ONNX}" "${PRUNED_ENGINE}" "enable" "${OUTPUT_DIR}/logs/build_nvidia_2_4_sparse_fp16.log"
parse_sparse_tactics_log
write_int8_status

if [[ "${SKIP_EVAL:-0}" != "1" ]]; then
  run_trt_eval "dense_sft_fp16" "${DENSE_CKPT}" "${DENSE_ENGINE}" "${DENSE_ONNX}" "iot200" "${IOT200_JSONL}"
  run_trt_eval "nvidia_2_4_sft_fp16" "${PRUNED_CKPT}" "${PRUNED_ENGINE}" "${PRUNED_ONNX}" "iot200" "${IOT200_JSONL}"
  run_trt_eval "dense_sft_fp16" "${DENSE_CKPT}" "${DENSE_ENGINE}" "${DENSE_ONNX}" "train" "${TRAIN_JSONL}"
  run_trt_eval "nvidia_2_4_sft_fp16" "${PRUNED_CKPT}" "${PRUNED_ENGINE}" "${PRUNED_ONNX}" "train" "${TRAIN_JSONL}"
else
  echo "[skip] TensorRT EM eval"
fi

run_latency_benchmarks
write_final_results

echo
echo "Finished."
echo "Shell script:"
echo "  ${PROJECT_ROOT}/scripts/run_h20_decoder_only_sft_prune_trt24.sh"
echo "Example command:"
echo "  bash scripts/run_h20_decoder_only_sft_prune_trt24.sh --base_model /PATH/TO/BASE_MODEL"
echo "Final outputs:"
echo "  ${OUTPUT_DIR}/results/final_metrics.json"
echo "  ${OUTPUT_DIR}/results/final_metrics.csv"
echo "  ${OUTPUT_DIR}/SUMMARY.md"
