#!/usr/bin/env bash
set -euo pipefail

ORIGINAL_CWD="$(pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

usage() {
  cat <<'EOF'
Run the H20 decoder-only SFT -> ONNX FP16/INT8 GPU latency comparison.

Only --base_model is required. Dataset and output paths are included below and
can be overridden by CLI flags or environment variables when needed.

This script intentionally does not use TensorRT, trtexec, or NVIDIA 2:4 sparse
tactics. It compares:
  - PyTorch FP16 CUDA forward latency
  - ONNX Runtime CUDA FP16 forward latency
  - ONNX Runtime CUDA INT8 forward latency, using QDQ quantization

Usage:
  bash scripts/run_h20_decoder_only_sft_prune_trt24.sh --base_model /PATH/TO/BASE_MODEL

Optional overrides:
  --train_jsonl PATH       default: data/scenic/SCENIC_full_training_dataset.json
  --iot200_jsonl PATH      default: data/benchmarks/iot_instruction_benchmark_200.json
  --output_dir PATH        default: runs/h20_decoder_only_onnx_int8_latency
  --gpus IDS              default: keep CUDA_VISIBLE_DEVICES, else auto-detect
  --epochs N              default: 5
  --seq_len N             default: 64
  --batch_size N          default: 16 per GPU for SFT
  --eval_batch_size N     accepted for compatibility; latency uses batch size 1
  --warmup_iters N        default: 100
  --measure_iters N       default: 1000
  --env_help              print GPU/ONNX Runtime setup help and exit

Useful environment toggles:
  PYTHON=/path/to/python     default: python
  TRUST_REMOTE_CODE=1
  SKIP_TRAIN=1 SKIP_EXPORT=1 SKIP_QUANTIZE=1 SKIP_LATENCY=1
  CALIB_SAMPLES=128
  PROMPT_FORMAT=raw|legacy|chat-template
EOF
}

print_env_setup_help() {
  cat <<'EOF'
H20 ONNX GPU benchmark environment checklist:

1. Run this on the H20/NVIDIA GPU machine or a GPU-enabled container:

     nvidia-smi

2. Activate the Python env you will use:

     conda activate chatlm-decoder

3. Install/update the required Python runtime packages:

     python -m pip install --upgrade -r requirements.txt

   Or install only the ONNX additions:

     python -m pip install --upgrade "onnx>=1.16" "onnxruntime-gpu>=1.18"

4. Verify CUDA and ONNX Runtime CUDA provider:

     python - <<'PY'
     import sys, torch, onnx, onnxruntime as ort
     print("python exe:", sys.executable)
     print("torch cuda:", torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count())
     print("onnx:", onnx.__version__)
     print("ort:", ort.__version__, ort.get_available_providers())
     PY

If torch.cuda.is_available() is false, package installs will not fix it. Fix the
GPU allocation/container first: driver visibility, Docker --gpus all, scheduler
GPU request, or CUDA_VISIBLE_DEVICES.

If CUDAExecutionProvider is missing from ONNX Runtime, install onnxruntime-gpu
in the active env and make sure its CUDA/cuDNN requirements match the machine.
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
OUTPUT_DIR="${OUTPUT_DIR:-runs/h20_decoder_only_onnx_int8_latency}"
ORIGINAL_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
GPUS="${GPUS:-${ORIGINAL_CUDA_VISIBLE_DEVICES}}"
EPOCHS="${EPOCHS:-5}"
SEQ_LEN="${SEQ_LEN:-64}"
BATCH_SIZE="${BATCH_SIZE:-16}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
MEASURE_ITERS="${MEASURE_ITERS:-1000}"
WARMUP_ITERS="${WARMUP_ITERS:-100}"

detect_gpu_mask() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    local count
    if ! count="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"; then
      count="0"
    fi
    if [[ "${count}" =~ ^[0-9]+$ && "${count}" -gt 0 ]]; then
      seq -s, 0 $((count - 1))
      return
    fi
  fi
  echo "0"
}

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

PYTHON_BIN="${PYTHON:-python}"
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

if [[ -z "${GPUS}" ]]; then
  GPUS="$(detect_gpu_mask)"
fi
if [[ -n "${GPUS}" ]]; then
  export CUDA_VISIBLE_DEVICES="${GPUS}"
  IFS=',' read -r -a GPU_ARRAY <<< "${GPUS}"
else
  GPU_ARRAY=(0)
fi
NPROC_PER_NODE="${NPROC_PER_NODE:-${#GPU_ARRAY[@]}}"
export NPROC_PER_NODE
LATENCY_GPU="${LATENCY_GPU:-${GPU_ARRAY[0]}}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
LEARNING_RATE="${LEARNING_RATE:-2.0e-5}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
CALIB_SAMPLES="${CALIB_SAMPLES:-128}"
PROMPT_FORMAT="${PROMPT_FORMAT:-raw}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-0}"

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

DENSE_CKPT="${OUTPUT_DIR}/checkpoints/dense_sft_fp16"
DENSE_TRAIN_DIR="${OUTPUT_DIR}/checkpoints/_dense_sft_training"
ONNX_FP16_DIR="${OUTPUT_DIR}/onnx/dense_sft_fp16"
ONNX_INT8_DIR="${OUTPUT_DIR}/onnx/dense_sft_int8"
ONNX_FP16="${ONNX_FP16_DIR}/model_decoder_nocache.onnx"
ONNX_INT8="${ONNX_INT8_DIR}/model.int8.onnx"

mkdir -p \
  "${OUTPUT_DIR}/checkpoints" \
  "${OUTPUT_DIR}/env" \
  "${OUTPUT_DIR}/logs" \
  "${OUTPUT_DIR}/onnx" \
  "${OUTPUT_DIR}/reports" \
  "${OUTPUT_DIR}/results" \
  "${OUTPUT_DIR}/generated_configs"

TRUST_ARGS=()
if [[ "${TRUST_REMOTE_CODE}" == "1" ]]; then
  TRUST_ARGS=(--trust-remote-code)
fi

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
    if command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
      command -v "${PYTHON_BIN}"
    else
      echo "${PYTHON_BIN} not found on PATH"
    fi
    "${PYTHON_BIN}" --version
    echo
    echo "=== CUDA visibility ==="
    echo "original CUDA_VISIBLE_DEVICES: ${ORIGINAL_CUDA_VISIBLE_DEVICES:-<unset>}"
    echo "effective CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"
    echo "requested GPUS mask: ${GPUS:-<unset>}"
    echo "NPROC_PER_NODE: ${NPROC_PER_NODE}"
    echo "LATENCY_GPU: ${LATENCY_GPU}"
    echo
    echo "=== torch / ONNX / ONNX Runtime ==="
    "${PYTHON_BIN}" - <<'PY'
import sys
print("python executable:", sys.executable)
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
try:
    import onnx
    print("onnx:", onnx.__version__)
except Exception as exc:
    print("onnx import failed:", repr(exc))
try:
    import onnxruntime as ort
    print("onnxruntime:", ort.__version__)
    print("onnxruntime providers:", ort.get_available_providers())
except Exception as exc:
    print("onnxruntime import failed:", repr(exc))
PY
  } | tee "${txt}"

  "${PYTHON_BIN}" - "${json}" <<'PY'
import json
import os
import sys
from pathlib import Path

output = Path(sys.argv[1])
report = {
    "python_executable": sys.executable,
    "python_version": sys.version,
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "nvidia_visible_devices": os.environ.get("NVIDIA_VISIBLE_DEVICES"),
    "nproc_per_node": os.environ.get("NPROC_PER_NODE"),
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
    import onnx
    report["onnx_version"] = onnx.__version__
except Exception as exc:
    report["onnx_error"] = repr(exc)
    report["onnx_version"] = None

try:
    import onnxruntime as ort
    report["onnxruntime_version"] = ort.__version__
    report["onnxruntime_available_providers"] = ort.get_available_providers()
except Exception as exc:
    report["onnxruntime_error"] = repr(exc)
    report["onnxruntime_version"] = None
    report["onnxruntime_available_providers"] = []

missing = []
if report.get("onnx_version") is None:
    missing.append("onnx python module")
if report.get("onnxruntime_version") is None:
    missing.append("onnxruntime-gpu python module")
elif "CUDAExecutionProvider" not in report.get("onnxruntime_available_providers", []):
    missing.append("onnxruntime CUDAExecutionProvider")
report["missing_runtime_requirements"] = missing

output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
if not report.get("torch_cuda_available"):
    raise SystemExit(
        "CUDA GPU is not available. Refusing to run H20 GPU latency benchmark.\n"
        f"Python executable: {report.get('python_executable')}\n"
        f"Torch version: {report.get('torch_version')}; torch CUDA: {report.get('torch_cuda_version')}\n"
        "If scenic-ED works, run this script with the same interpreter, for example "
        "`PYTHON=python bash scripts/run_h20_decoder_only_sft_prune_trt24.sh --base_model ...`.\n"
        "Run on the H20 GPU machine/container and check `nvidia-smi`, scheduler GPU allocation, "
        "Docker --gpus all, and CUDA_VISIBLE_DEVICES."
    )
if int(report.get("torch_cuda_device_count") or 0) < 1:
    raise SystemExit("No visible CUDA devices. Check CUDA_VISIBLE_DEVICES/--gpus.")
if missing:
    raise SystemExit(
        "Missing runtime requirements: "
        + ", ".join(missing)
        + ". Install with `python -m pip install --upgrade onnx onnxruntime-gpu` in the active env."
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
  torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" scripts/train.py \
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

export_onnx_fp16() {
  echo
  echo "[onnx] export dense SFT FP16"
  if [[ "${SKIP_EXPORT:-0}" == "1" ]]; then
    echo "[skip] export; expecting ${ONNX_FP16}"
    return
  fi
  "${PYTHON_BIN}" scripts/export_decoder_onnx.py \
    --model-path "${DENSE_CKPT}" \
    --onnx-dir "${ONNX_FP16_DIR}" \
    --dtype fp16 \
    --opset 18 \
    --seq-len "${SEQ_LEN}" \
    --batch-size 1 \
    --attn-implementation eager \
    --no-export-cache \
    "${TRUST_ARGS[@]}" \
    --overwrite
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
    for dim in value.type.tensor_type.shape.dim:
        if dim.dim_param:
            dims.append(dim.dim_param)
        elif dim.HasField("dim_value"):
            dims.append(int(dim.dim_value))
        else:
            dims.append("?")
    return dims

initializer_dtypes = Counter(dtype_name(tensor.data_type) for tensor in model.graph.initializer)
floating_initializer_total = sum(
    count for dtype, count in initializer_dtypes.items()
    if dtype in {"FLOAT", "FLOAT16", "BFLOAT16", "DOUBLE"}
)
report = {
    "onnx_path": str(onnx_path),
    "exists": onnx_path.exists(),
    "input_names": [value.name for value in model.graph.input],
    "output_names": [value.name for value in model.graph.output],
    "input_shapes": {value.name: shape_of(value) for value in model.graph.input},
    "output_shapes": {value.name: shape_of(value) for value in model.graph.output},
    "input_dtypes": {value.name: dtype_name(value.type.tensor_type.elem_type) for value in model.graph.input},
    "output_dtypes": {value.name: dtype_name(value.type.tensor_type.elem_type) for value in model.graph.output},
    "initializer_dtypes": dict(initializer_dtypes),
    "floating_initializer_total": int(floating_initializer_total),
    "floating_initializer_fp16": int(initializer_dtypes.get("FLOAT16", 0)),
    "weights_are_fp16": bool(
        floating_initializer_total and initializer_dtypes.get("FLOAT16", 0) == floating_initializer_total
    ),
    "onnx_model_size_mb": onnx_path.stat().st_size / 1024**2,
}
report_path.parent.mkdir(parents=True, exist_ok=True)
report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
PY
}

quantize_onnx_int8() {
  echo
  echo "[onnx] quantize dense SFT ONNX to INT8 QDQ"
  if [[ "${SKIP_QUANTIZE:-0}" == "1" ]]; then
    echo "[skip] quantize; expecting ${ONNX_INT8}"
    return
  fi
  mkdir -p "${ONNX_INT8_DIR}"
  "${PYTHON_BIN}" - \
    "${ONNX_FP16}" \
    "${ONNX_INT8}" \
    "${DENSE_CKPT}" \
    "${TRAIN_JSONL}" \
    "${SEQ_LEN}" \
    "${CALIB_SAMPLES}" \
    "${PROMPT_FORMAT}" \
    "${TRUST_REMOTE_CODE}" \
    "${OUTPUT_DIR}/reports/onnx_int8_quantization_report.json" <<'PY'
import csv
import json
import sys
from pathlib import Path

import numpy as np
import onnx
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_dynamic, quantize_static
from transformers import AutoTokenizer

input_path = Path(sys.argv[1]).expanduser()
output_path = Path(sys.argv[2]).expanduser()
model_path = Path(sys.argv[3]).expanduser()
calib_path = Path(sys.argv[4]).expanduser()
seq_len = int(sys.argv[5])
calib_samples = int(sys.argv[6])
prompt_format = sys.argv[7]
trust = sys.argv[8] == "1"
report_path = Path(sys.argv[9]).expanduser()

model = onnx.load(str(input_path), load_external_data=False)
input_infos = []
dtype_name = onnx.TensorProto.DataType.Name
for value in model.graph.input:
    dtype = dtype_name(value.type.tensor_type.elem_type)
    input_infos.append({"name": value.name, "dtype": dtype})

def read_records(path):
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            for key in ("data", "records", "items", "examples", "train", "validation", "test"):
                if isinstance(payload.get(key), list):
                    return payload[key]
            return [payload]
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    if suffix == ".txt":
        return [{"prompt": line.strip()} for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    raise ValueError(f"Unsupported calibration data extension: {path.suffix}")

def clean(value):
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value).strip()

def prompt_from_record(record):
    for key in ("prompt", "instruction", "input", "question", "query", "text", "command"):
        text = clean(record.get(key))
        if text:
            return text
    messages = record.get("messages") or record.get("conversations")
    if isinstance(messages, list):
        parts = []
        for turn in messages:
            if not isinstance(turn, dict):
                continue
            role = clean(turn.get("role") or turn.get("from") or turn.get("speaker")).lower()
            content = clean(turn.get("content") or turn.get("value") or turn.get("text"))
            if role in {"user", "human", "instruction", "prompt"} and content:
                parts.append(content)
        if parts:
            return "\n".join(parts)
    return ""

def apply_format(tokenizer, prompt):
    if prompt_format == "legacy":
        return f"<|user|>\n{prompt}\n<|assistant|>\n"
    if prompt_format == "chat-template":
        return str(tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        ))
    return prompt

tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=trust)
if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
    tokenizer.pad_token = tokenizer.eos_token

records = read_records(calib_path)
prompts = [prompt_from_record(record) for record in records]
prompts = [prompt for prompt in prompts if prompt][:calib_samples]
if not prompts:
    fallback = getattr(tokenizer, "bos_token", None) or getattr(tokenizer, "eos_token", None) or "hello"
    prompts = [fallback]

class Reader(CalibrationDataReader):
    def __init__(self):
        self.items = []
        for prompt in prompts:
            formatted = apply_format(tokenizer, prompt)
            encoded = tokenizer(
                formatted,
                return_tensors="np",
                truncation=True,
                padding="max_length",
                max_length=seq_len,
            )
            feed = {}
            for info in input_infos:
                name = info["name"]
                dtype = np.int64 if info["dtype"] == "INT64" else np.int32
                if name in encoded:
                    feed[name] = np.ascontiguousarray(encoded[name].astype(dtype))
                elif name == "attention_mask":
                    feed[name] = np.ascontiguousarray(encoded["attention_mask"].astype(dtype))
                else:
                    feed[name] = np.ones((1, seq_len), dtype=dtype)
            self.items.append(feed)
        self.index = 0

    def get_next(self):
        if self.index >= len(self.items):
            return None
        item = self.items[self.index]
        self.index += 1
        return item

output_path.parent.mkdir(parents=True, exist_ok=True)
report = {
    "input_onnx": str(input_path),
    "output_onnx": str(output_path),
    "calibration_file": str(calib_path),
    "calibration_samples": len(prompts),
    "quantization": "static_qdq_int8",
}

try:
    quantize_static(
        str(input_path),
        str(output_path),
        Reader(),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        op_types_to_quantize=["MatMul", "Gemm"],
        use_external_data_format=True,
    )
except TypeError:
    quantize_static(
        str(input_path),
        str(output_path),
        Reader(),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        op_types_to_quantize=["MatMul", "Gemm"],
    )
except Exception as exc:
    report["static_qdq_error"] = repr(exc)
    report["quantization"] = "dynamic_weight_int8_fallback"
    quantize_dynamic(
        str(input_path),
        str(output_path),
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["MatMul", "Gemm"],
        use_external_data_format=True,
    )

report["output_exists"] = output_path.exists()
report["output_size_mb"] = output_path.stat().st_size / 1024**2 if output_path.exists() else None
report_path.parent.mkdir(parents=True, exist_ok=True)
report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
PY
}

benchmark_runtime() {
  CUDA_VISIBLE_DEVICES="${LATENCY_GPU}" "${PYTHON_BIN}" - "$@" <<'PY'
import argparse
import json
import math
import statistics
import time
from pathlib import Path

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

def stats(latencies):
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
parser.add_argument("--runtime", required=True, choices=["pytorch", "onnx"])
parser.add_argument("--model-path", default=None)
parser.add_argument("--onnx-path", default=None)
parser.add_argument("--output-json", required=True)
parser.add_argument("--label", required=True)
parser.add_argument("--precision", required=True)
parser.add_argument("--seq-len", type=int, default=64)
parser.add_argument("--batch-size", type=int, default=1)
parser.add_argument("--warmup-iters", type=int, default=100)
parser.add_argument("--measure-iters", type=int, default=1000)
parser.add_argument("--trust-remote-code", action="store_true")
args = parser.parse_args()

result = {
    "available": False,
    "label": args.label,
    "runtime": args.runtime,
    "precision": args.precision,
    "seq_len": int(args.seq_len),
    "batch_size": int(args.batch_size),
    "model_path": args.model_path,
    "onnx_path": args.onnx_path,
    "onnx_size_mb": file_mb(args.onnx_path),
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
        result.update(stats(latencies))
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

        available = ort.get_available_providers()
        if "CUDAExecutionProvider" not in available:
            raise RuntimeError(f"CUDAExecutionProvider unavailable; providers={available}")
        so = ort.SessionOptions()
        try:
            so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        except Exception:
            pass
        session = ort.InferenceSession(str(args.onnx_path), sess_options=so, providers=["CUDAExecutionProvider"])
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
        result.update(stats(latencies))
        result.update({
            "available": True,
            "provider": ",".join(session.get_providers()),
            "onnxruntime_available_providers": available,
            "peak_gpu_memory_mb": None,
        })

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
  echo "[latency] batch-1 seq-${SEQ_LEN} single-GPU benchmarks on CUDA_VISIBLE_DEVICES=${LATENCY_GPU}"
  if [[ "${SKIP_LATENCY:-0}" == "1" ]]; then
    echo "[skip] latency"
    return
  fi

  benchmark_runtime \
    --runtime pytorch \
    --model-path "${DENSE_CKPT}" \
    --output-json "${OUTPUT_DIR}/results/latency_pytorch_fp16.json" \
    --label "pytorch_fp16_cuda" \
    --precision FP16 \
    --seq-len "${SEQ_LEN}" \
    --batch-size 1 \
    --warmup-iters "${WARMUP_ITERS}" \
    --measure-iters "${MEASURE_ITERS}" \
    "${TRUST_ARGS[@]}"

  benchmark_runtime \
    --runtime onnx \
    --onnx-path "${ONNX_FP16}" \
    --output-json "${OUTPUT_DIR}/results/latency_onnxruntime_cuda_fp16.json" \
    --label "onnxruntime_cuda_fp16" \
    --precision FP16 \
    --seq-len "${SEQ_LEN}" \
    --batch-size 1 \
    --warmup-iters "${WARMUP_ITERS}" \
    --measure-iters "${MEASURE_ITERS}"

  benchmark_runtime \
    --runtime onnx \
    --onnx-path "${ONNX_INT8}" \
    --output-json "${OUTPUT_DIR}/results/latency_onnxruntime_cuda_int8.json" \
    --label "onnxruntime_cuda_int8" \
    --precision INT8 \
    --seq-len "${SEQ_LEN}" \
    --batch-size 1 \
    --warmup-iters "${WARMUP_ITERS}" \
    --measure-iters "${MEASURE_ITERS}"
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

def fmt(value):
    if value is None or value == "" or (isinstance(value, float) and math.isnan(value)):
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return value

env = read_json(out / "env/env_report.json")
quant = read_json(out / "reports/onnx_int8_quantization_report.json")

latency_files = [
    ("PyTorch", "FP16", "latency_pytorch_fp16.json"),
    ("ONNX Runtime CUDA", "FP16", "latency_onnxruntime_cuda_fp16.json"),
    ("ONNX Runtime CUDA", "INT8", "latency_onnxruntime_cuda_int8.json"),
]

latencies = {name: read_json(out / "results" / filename) for _runtime, _precision, filename in latency_files for name in [filename]}
pytorch = latencies.get("latency_pytorch_fp16.json", {})
onnx_fp16 = latencies.get("latency_onnxruntime_cuda_fp16.json", {})

def speedup(numerator_ms, denominator_ms):
    try:
        if numerator_ms is None or denominator_ms is None or float(denominator_ms) <= 0:
            return None
        return float(numerator_ms) / float(denominator_ms)
    except Exception:
        return None

rows = []
for runtime, precision, filename in latency_files:
    data = latencies.get(filename, {})
    mean_ms = data.get("mean_latency_ms")
    row = {
        "Runtime": runtime,
        "Precision": precision,
        "Seq. Len.": seq_len,
        "Batch Size": data.get("batch_size", 1),
        "Mean Latency ms": fmt(mean_ms),
        "Median Latency ms": fmt(data.get("median_latency_ms")),
        "P95 Latency ms": fmt(data.get("p95_latency_ms")),
        "P99 Latency ms": fmt(data.get("p99_latency_ms")),
        "Throughput QPS": fmt(data.get("throughput_qps")),
        "Peak GPU Memory MB": fmt(data.get("peak_gpu_memory_mb")),
        "ONNX MB": fmt(data.get("onnx_size_mb")),
        "Provider": data.get("provider", ""),
        "GPU": data.get("gpu") or ", ".join(env.get("torch_cuda_device_names", [])),
        "Available": str(bool(data.get("available"))).lower(),
        "Error": data.get("error", ""),
        "Speedup vs PyTorch FP16": fmt(speedup(pytorch.get("mean_latency_ms"), mean_ms)),
        "Speedup vs ONNX FP16": fmt(speedup(onnx_fp16.get("mean_latency_ms"), mean_ms)),
    }
    rows.append(row)

fields = [
    "Runtime", "Precision", "Seq. Len.", "Batch Size", "Mean Latency ms",
    "Median Latency ms", "P95 Latency ms", "P99 Latency ms", "Throughput QPS",
    "Peak GPU Memory MB", "ONNX MB", "Provider", "GPU", "Available", "Error",
    "Speedup vs PyTorch FP16", "Speedup vs ONNX FP16",
]
csv_path = out / "results/final_latency_comparison.csv"
json_path = out / "results/final_latency_comparison.json"
with csv_path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field, "") for field in fields})

payload = {
    "rows": rows,
    "quantization_report": quant,
    "environment": env,
    "notes": [
        "Latency is one forward pass at batch size 1 and the configured sequence length.",
        "ONNX Runtime sessions request CUDAExecutionProvider and disable CPU EP fallback when supported.",
        "INT8 is ONNX Runtime QDQ quantization, not TensorRT INT8.",
    ],
}
json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

int8_data = latencies.get("latency_onnxruntime_cuda_int8.json", {})
int8_speedup_pytorch = speedup(pytorch.get("mean_latency_ms"), int8_data.get("mean_latency_ms"))
int8_speedup_onnx = speedup(onnx_fp16.get("mean_latency_ms"), int8_data.get("mean_latency_ms"))
summary = f"""# H20 ONNX FP16 vs INT8 Latency Summary

- GPU benchmark ran on CUDA: {bool(env.get("torch_cuda_available"))}; visible GPUs: {env.get("torch_cuda_device_names", [])}
- ONNX Runtime providers: {env.get("onnxruntime_available_providers", [])}
- Sequence length: {seq_len}; batch size: 1
- INT8 quantization mode: {quant.get("quantization")}
- PyTorch FP16 mean latency: {pytorch.get("mean_latency_ms")} ms
- ONNX Runtime CUDA FP16 mean latency: {onnx_fp16.get("mean_latency_ms")} ms
- ONNX Runtime CUDA INT8 mean latency: {int8_data.get("mean_latency_ms")} ms
- ONNX INT8 speedup vs PyTorch FP16: {int8_speedup_pytorch}
- ONNX INT8 speedup vs ONNX FP16: {int8_speedup_onnx}

## Limitations

- This is forward-pass latency, not autoregressive end-to-end generation latency.
- INT8 uses ONNX Runtime quantization and CUDAExecutionProvider, not TensorRT.
- If ONNX INT8 is unavailable or slower, inspect `reports/onnx_int8_quantization_report.json` and the INT8 latency JSON for unsupported/fallback ops.
"""
(out / "SUMMARY.md").write_text(summary, encoding="utf-8")
print(f"Wrote final CSV: {csv_path}")
print(f"Wrote final JSON: {json_path}")
print(f"Wrote summary: {out / 'SUMMARY.md'}")
PY
}

echo "H20 decoder-only ONNX FP16/INT8 latency comparison"
echo "  base model:       ${BASE_MODEL}"
echo "  train data:       ${TRAIN_JSONL}"
echo "  calibration data: ${TRAIN_JSONL}"
echo "  IoT200 data:      ${IOT200_JSONL}"
echo "  output dir:       ${OUTPUT_DIR}"
echo "  GPUs:             ${GPUS}"
echo "  epochs:           ${EPOCHS}"
echo "  seq_len:          ${SEQ_LEN}"
echo "  SFT batch/GPU:    ${BATCH_SIZE}"
echo "  latency iters:    warmup=${WARMUP_ITERS} measure=${MEASURE_ITERS}"

write_env_report
train_sft
export_onnx_fp16
inspect_onnx "${ONNX_FP16}" "${OUTPUT_DIR}/reports/onnx_inspection_fp16.json"
quantize_onnx_int8
inspect_onnx "${ONNX_INT8}" "${OUTPUT_DIR}/reports/onnx_inspection_int8.json"
run_latency_benchmarks
write_final_results

echo
echo "Finished."
echo "Shell script:"
echo "  ${PROJECT_ROOT}/scripts/run_h20_decoder_only_sft_prune_trt24.sh"
echo "Example command:"
echo "  bash scripts/run_h20_decoder_only_sft_prune_trt24.sh --base_model /PATH/TO/BASE_MODEL"
echo "Final outputs:"
echo "  ${OUTPUT_DIR}/results/final_latency_comparison.json"
echo "  ${OUTPUT_DIR}/results/final_latency_comparison.csv"
echo "  ${OUTPUT_DIR}/SUMMARY.md"
