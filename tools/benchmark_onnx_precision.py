#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import logging
import math
import os
import platform
import socket
import statistics
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from chatlm_decoder.tokenizer import prepare_decoder_tokenizer

try:
    from trt_edge_common import apply_prompt_format, prompt_and_reference, read_records, setup_logging
except Exception:  # pragma: no cover - fallback only for unusual standalone use.
    apply_prompt_format = None
    prompt_and_reference = None
    read_records = None

    def setup_logging(verbose: bool = False) -> None:
        logging.basicConfig(
            level=logging.DEBUG if verbose else logging.INFO,
            format="%(asctime)s %(levelname)s %(message)s",
            datefmt="%H:%M:%S",
        )


TABLE_COLUMNS = [
    ("precision", "Precision"),
    ("onnx_model_path", "ONNX model path"),
    ("onnx_file_size_mb", "ONNX size MB"),
    ("execution_provider", "Execution Provider"),
    ("session_providers", "Session providers"),
    ("hardware_device", "Hardware/device"),
    ("mean_latency_ms", "Mean ms"),
    ("median_latency_ms", "Median ms"),
    ("p90_latency_ms", "p90 ms"),
    ("p95_latency_ms", "p95 ms"),
    ("throughput_samples_per_sec", "Samples/s"),
    ("peak_host_rss_mb", "Peak host RSS MB"),
    ("peak_device_memory_mb", "Device memory MB"),
    ("average_power_w", "Avg power W"),
    ("energy_per_inference_mj", "Energy/inference mJ"),
    ("accuracy_or_drift_vs_fp32", "Accuracy/drift vs FP32"),
    ("mean_abs_error", "Mean abs error"),
    ("max_abs_error", "Max abs error"),
    ("cosine_similarity", "Cosine sim"),
    ("speedup_vs_fp32", "Speedup vs FP32"),
    ("size_reduction_vs_fp32", "Size reduction vs FP32"),
    ("notes", "Notes/warnings"),
]


@dataclass
class InputMeta:
    name: str
    shape: list[int | str | None]
    np_dtype: Any
    elem_type: int


@dataclass
class PrecisionArtifact:
    precision: str
    path: Path
    notes: list[str]
    metadata: dict[str, Any]


class RequiredPackageError(RuntimeError):
    pass


def import_required(module_name: str, feature: str) -> Any:
    try:
        __import__(module_name)
        return sys.modules[module_name]
    except ImportError as exc:
        raise RequiredPackageError(
            f"Missing required package '{module_name}' for {feature}. Install it and rerun this script."
        ) from exc


def import_optional(module_name: str) -> Any | None:
    try:
        __import__(module_name)
        return sys.modules[module_name]
    except ImportError:
        return None


def preflight_required_packages(args: argparse.Namespace) -> None:
    requested: list[tuple[str, str, str, str]] = [
        ("numpy", "numpy", "numpy", "input generation and drift metrics"),
        ("onnx", "onnx", "onnx", "model validation and input metadata"),
        ("onnxruntime", "onnxruntime or onnxruntime-gpu", "onnxruntime", "ONNX Runtime benchmarking"),
        ("psutil", "psutil", "psutil", "peak host RSS memory measurement"),
    ]
    if not args.fp16_onnx and not args.skip_fp16_conversion:
        requested.append(
            ("onnxconverter_common", "onnxconverter-common", "onnxconverter-common", "FP16 ONNX conversion")
        )
    if not args.int8_onnx and not args.skip_int8_quantization:
        requested.append(
            ("onnxruntime.quantization", "onnxruntime or onnxruntime-gpu", "onnxruntime", "INT8 ONNX quantization")
        )
    missing: list[tuple[str, str, str]] = []
    for import_name, package_name, pip_name, feature in requested:
        try:
            __import__(import_name)
        except ImportError:
            missing.append((package_name, pip_name, feature))
    if missing:
        lines = ["Missing required package(s) for the requested benchmark:"]
        for package_name, _pip_name, feature in missing:
            lines.append(f"  - {package_name}: needed for {feature}")
        unique_packages = sorted({pip_name for _package_name, pip_name, _feature in missing})
        lines.append("Install them, for example:")
        lines.append(f"  python -m pip install {' '.join(unique_packages)}")
        if any(package_name == "onnxruntime or onnxruntime-gpu" for package_name, _pip_name, _feature in missing):
            lines.append("Use onnxruntime-gpu instead of onnxruntime when you need CUDAExecutionProvider.")
        raise RequiredPackageError("\n".join(lines))


def expand_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    return path


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=json_default) + "\n", encoding="utf-8")


def format_value(value: Any, digits: int = 3) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return str(value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "N/A"
    if abs(number) >= 100:
        return f"{number:.2f}"
    if abs(number) >= 10:
        return f"{number:.3f}"
    return f"{number:.{digits}f}"


def percentile(values: list[float], q: float) -> float:
    if not values:
        return math.nan
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = (len(sorted_values) - 1) * q
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight)


def parse_input_shapes(values: list[str] | None) -> tuple[tuple[int, ...] | None, dict[str, tuple[int, ...]]]:
    if not values:
        return None, {}
    default_shape: tuple[int, ...] | None = None
    per_name: dict[str, tuple[int, ...]] = {}
    for raw in values:
        text = raw.strip()
        if not text:
            continue
        if "=" in text:
            name, shape_text = text.split("=", 1)
            per_name[name.strip()] = tuple(int(part.strip()) for part in shape_text.split(",") if part.strip())
        else:
            default_shape = tuple(int(part.strip()) for part in text.split(",") if part.strip())
    return default_shape, per_name


def numpy_dtype_from_onnx_elem_type(onnx: Any, np: Any, elem_type: int) -> Any:
    tensor_proto = onnx.TensorProto
    mapping = {
        tensor_proto.FLOAT: np.float32,
        tensor_proto.FLOAT16: np.float16,
        tensor_proto.DOUBLE: np.float64,
        tensor_proto.INT8: np.int8,
        tensor_proto.UINT8: np.uint8,
        tensor_proto.INT16: np.int16,
        tensor_proto.UINT16: np.uint16,
        tensor_proto.INT32: np.int32,
        tensor_proto.UINT32: np.uint32,
        tensor_proto.INT64: np.int64,
        tensor_proto.UINT64: np.uint64,
        tensor_proto.BOOL: np.bool_,
    }
    return mapping.get(elem_type, np.float32)


def load_onnx_input_metas(path: Path, onnx: Any, np: Any) -> list[InputMeta]:
    model = onnx.load(str(path), load_external_data=False)
    initializer_names = {initializer.name for initializer in model.graph.initializer}
    metas: list[InputMeta] = []
    for value_info in model.graph.input:
        if value_info.name in initializer_names:
            continue
        tensor_type = value_info.type.tensor_type
        if not tensor_type.HasField("elem_type"):
            continue
        dims: list[int | str | None] = []
        if tensor_type.HasField("shape"):
            for dim in tensor_type.shape.dim:
                if dim.HasField("dim_value"):
                    dims.append(int(dim.dim_value))
                elif dim.dim_param:
                    dims.append(str(dim.dim_param))
                else:
                    dims.append(None)
        metas.append(
            InputMeta(
                name=value_info.name,
                shape=dims,
                np_dtype=numpy_dtype_from_onnx_elem_type(onnx, np, int(tensor_type.elem_type)),
                elem_type=int(tensor_type.elem_type),
            )
        )
    if not metas:
        raise RuntimeError(f"Could not discover tensor inputs in ONNX graph: {path}")
    return metas


def concrete_shape(meta: InputMeta, args: argparse.Namespace) -> tuple[int, ...]:
    default_shape, per_name = parse_input_shapes(args.input_shape)
    if meta.name in per_name:
        return per_name[meta.name]
    if default_shape is not None and len(default_shape) == len(meta.shape):
        return default_shape
    shape: list[int] = []
    for axis, dim in enumerate(meta.shape):
        if isinstance(dim, int) and dim > 0:
            shape.append(int(dim))
        elif axis == 0:
            shape.append(int(args.batch_size))
        else:
            shape.append(int(args.max_seq_len))
    if not shape:
        shape = [int(args.batch_size)]
    return tuple(shape)


def is_integer_dtype(np: Any, dtype: Any) -> bool:
    return bool(np.issubdtype(np.dtype(dtype), np.integer))


def is_float_dtype(np: Any, dtype: Any) -> bool:
    return bool(np.issubdtype(np.dtype(dtype), np.floating))


class InputFactory:
    def __init__(self, args: argparse.Namespace, metas: list[InputMeta], np: Any) -> None:
        self.args = args
        self.metas = metas
        self.np = np
        self.default_shape, self.per_name_shapes = parse_input_shapes(args.input_shape)
        self.records: list[dict[str, Any]] = []
        self.tokenizer: Any | None = None
        self.source_kind = "dummy"
        self.notes: list[str] = []
        self._load_real_prompt_source()

    def _load_real_prompt_source(self) -> None:
        if not self.args.dataset or not self.args.model_path:
            self.notes.append(
                "No --dataset/--model-path pair supplied; using deterministic dummy inputs and marking accuracy as N/A_dummy_inputs."
            )
            return
        if read_records is None or prompt_and_reference is None or apply_prompt_format is None:
            self.notes.append("Repo dataset helpers were unavailable; using deterministic dummy inputs.")
            return
        transformers = import_optional("transformers")
        if transformers is None:
            self.notes.append("Package transformers is missing; using deterministic dummy inputs instead of real prompts.")
            return
        dataset_path = expand_path(self.args.dataset)
        if not dataset_path.exists():
            self.notes.append(f"Dataset not found ({dataset_path}); using deterministic dummy inputs.")
            return
        max_needed = max(
            int(self.args.calibration_samples),
            int(self.args.batch_size) * max(int(self.args.runs), int(self.args.warmup), int(self.args.drift_samples), 1),
        )
        self.records = read_records(dataset_path, limit=max_needed)
        if not self.records:
            self.notes.append(f"Dataset produced zero records ({dataset_path}); using deterministic dummy inputs.")
            return
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.args.model_path,
            trust_remote_code=bool(self.args.trust_remote_code),
        )
        prepare_decoder_tokenizer(self.tokenizer)
        if getattr(self.tokenizer, "pad_token_id", None) is None and getattr(self.tokenizer, "eos_token_id", None) is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if getattr(self.tokenizer, "pad_token_id", None) is None:
            self.notes.append("Tokenizer has no pad/eos token; using deterministic dummy inputs.")
            self.tokenizer = None
            return
        self.source_kind = "dataset"
        self.notes.append(f"Using repo prompt dataset helpers with {len(self.records)} record(s) from {dataset_path}.")

    def batch_size_for_inputs(self, inputs: dict[str, Any]) -> int:
        if not inputs:
            return int(self.args.batch_size)
        first = next(iter(inputs.values()))
        return int(first.shape[0]) if getattr(first, "ndim", 0) > 0 else int(self.args.batch_size)

    def make_batch(self, batch_index: int) -> dict[str, Any]:
        if self.source_kind == "dataset" and self.tokenizer is not None:
            return self._make_dataset_batch(batch_index)
        return self._make_dummy_batch(batch_index)

    def make_batches(self, count: int) -> list[dict[str, Any]]:
        return [self.make_batch(index) for index in range(max(1, int(count)))]

    def _make_dataset_batch(self, batch_index: int) -> dict[str, Any]:
        assert self.tokenizer is not None
        prompts: list[str] = []
        for offset in range(int(self.args.batch_size)):
            record = self.records[(batch_index * int(self.args.batch_size) + offset) % len(self.records)]
            prompt, _reference = prompt_and_reference(record, prompt_field=self.args.prompt_field)
            prompts.append(apply_prompt_format(self.tokenizer, prompt, self.args.prompt_format, self.args.system_prompt))
        encoded = self.tokenizer(
            prompts,
            return_tensors="np",
            padding="max_length",
            truncation=True,
            max_length=int(self.args.max_seq_len),
            add_special_tokens=bool(self.args.add_special_tokens),
        )
        batch: dict[str, Any] = {}
        for meta in self.metas:
            lower_name = meta.name.lower()
            if "input_ids" in lower_name and "input_ids" in encoded:
                batch[meta.name] = self.np.ascontiguousarray(encoded["input_ids"].astype(meta.np_dtype))
            elif "attention_mask" in lower_name and "attention_mask" in encoded:
                batch[meta.name] = self.np.ascontiguousarray(encoded["attention_mask"].astype(meta.np_dtype))
            else:
                batch[meta.name] = self._dummy_array(meta, batch_index)
        return batch

    def _make_dummy_batch(self, batch_index: int) -> dict[str, Any]:
        return {meta.name: self._dummy_array(meta, batch_index) for meta in self.metas}

    def _dummy_array(self, meta: InputMeta, batch_index: int) -> Any:
        shape = concrete_shape(meta, self.args)
        rng = self.np.random.default_rng(int(self.args.seed) + batch_index * 997 + sum(ord(c) for c in meta.name))
        lower_name = meta.name.lower()
        dtype = self.np.dtype(meta.np_dtype)
        if "attention_mask" in lower_name:
            return self.np.ones(shape, dtype=dtype)
        if is_integer_dtype(self.np, dtype):
            if "input_ids" in lower_name or "token" in lower_name:
                high = 32000
            else:
                high = 16
            return self.np.ascontiguousarray(rng.integers(0, high, size=shape).astype(dtype))
        if is_float_dtype(self.np, dtype):
            return self.np.ascontiguousarray(rng.standard_normal(size=shape).astype(dtype))
        if self.np.issubdtype(dtype, self.np.bool_):
            return self.np.ones(shape, dtype=dtype)
        return self.np.zeros(shape, dtype=dtype)


class CalibrationReader:
    def __init__(self, input_factory: InputFactory, sample_count: int, batch_size: int) -> None:
        self.input_factory = input_factory
        self.sample_count = max(1, int(sample_count))
        self.batch_size = max(1, int(batch_size))
        self.batch_count = max(1, math.ceil(self.sample_count / float(self.batch_size)))
        self.index = 0

    def get_next(self) -> dict[str, Any] | None:
        if self.index >= self.batch_count:
            return None
        batch = self.input_factory.make_batch(self.index)
        self.index += 1
        return batch

    def rewind(self) -> None:
        self.index = 0


class PeakRSSMonitor:
    def __init__(self, psutil: Any, interval_s: float = 0.005) -> None:
        self.process = psutil.Process(os.getpid())
        self.interval_s = interval_s
        self.peak = 0
        self._running = False
        self._thread: threading.Thread | None = None

    def _sample_loop(self) -> None:
        while self._running:
            self.peak = max(self.peak, int(self.process.memory_info().rss))
            time.sleep(self.interval_s)

    def start(self) -> None:
        self.peak = int(self.process.memory_info().rss)
        self._running = True
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self) -> float:
        self.peak = max(self.peak, int(self.process.memory_info().rss))
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        return self.peak / 1024.0**2


class NvmlMemoryMonitor:
    def __init__(self, pynvml: Any, device_index: int = 0, interval_s: float = 0.01) -> None:
        self.pynvml = pynvml
        self.device_index = int(device_index)
        self.interval_s = interval_s
        self.handle = None
        self.peak_used = 0
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self.pynvml.nvmlInit()
        self.handle = self.pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
        self.peak_used = int(self.pynvml.nvmlDeviceGetMemoryInfo(self.handle).used)
        self._running = True
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def _sample_loop(self) -> None:
        while self._running:
            try:
                self.peak_used = max(self.peak_used, int(self.pynvml.nvmlDeviceGetMemoryInfo(self.handle).used))
            except Exception:
                pass
            time.sleep(self.interval_s)

    def stop(self) -> float:
        try:
            if self.handle is not None:
                self.peak_used = max(self.peak_used, int(self.pynvml.nvmlDeviceGetMemoryInfo(self.handle).used))
        finally:
            self._running = False
            if self._thread is not None:
                self._thread.join(timeout=1.0)
            try:
                self.pynvml.nvmlShutdown()
            except Exception:
                pass
        return self.peak_used / 1024.0**2


def convert_fp16(fp32_path: Path, output_dir: Path) -> PrecisionArtifact | None:
    onnx = import_required("onnx", "FP16 conversion validation")
    try:
        from onnxconverter_common.float16 import convert_float_to_float16
    except ImportError as exc:
        raise RequiredPackageError(
            "Missing required package 'onnxconverter-common' for FP16 conversion. "
            "Install it or pass --fp16-onnx/--skip-fp16-conversion."
        ) from exc

    output_path = output_dir / "model_fp16.onnx"
    notes: list[str] = []
    logging.info("Converting FP32 ONNX to FP16: %s", output_path)
    model = onnx.load(str(fp32_path))
    try:
        converted = convert_float_to_float16(model, keep_io_types=True)
        notes.append("FP16 conversion used keep_io_types=True.")
    except Exception as exc:
        notes.append(f"keep_io_types=True failed ({exc}); retried with keep_io_types=False.")
        converted = convert_float_to_float16(model, keep_io_types=False)
    onnx.save_model(converted, str(output_path))
    onnx.checker.check_model(str(output_path))
    return PrecisionArtifact("FP16", output_path, notes, {"conversion": "onnxconverter_common.float16"})


def quant_format_value(name: str, quantization: Any) -> Any:
    if name == "qdq":
        return quantization.QuantFormat.QDQ
    if name == "qoperator":
        return quantization.QuantFormat.QOperator
    raise ValueError(f"Unknown quant format: {name}")


def quantize_int8(fp32_path: Path, output_dir: Path, args: argparse.Namespace, input_factory: InputFactory) -> PrecisionArtifact | None:
    onnx = import_required("onnx", "INT8 validation")
    try:
        import importlib

        quantization = importlib.import_module("onnxruntime.quantization")
    except ImportError as exc:
        raise RequiredPackageError(
            "Missing onnxruntime.quantization for INT8 quantization. "
            "Install onnxruntime or pass --int8-onnx/--skip-int8-quantization."
        ) from exc

    output_path = output_dir / "model_int8.onnx"
    notes: list[str] = []
    metadata: dict[str, Any] = {
        "quantization_mode": args.quantization_mode,
        "quantization_format": args.quant_format,
        "calibration_samples": int(args.calibration_samples),
        "activation_type": "QInt8",
        "weight_type": "QInt8",
    }
    logging.info("Quantizing FP32 ONNX to INT8 (%s): %s", args.quantization_mode, output_path)
    if args.quantization_mode == "dynamic":
        quantization.quantize_dynamic(
            str(fp32_path),
            str(output_path),
            weight_type=quantization.QuantType.QInt8,
        )
        notes.append("INT8 dynamic quantization used quantize_dynamic with QInt8 weights.")
    else:
        model_input = fp32_path
        preprocessed_path = output_dir / "model_fp32_quant_preprocessed.onnx"
        try:
            try:
                from onnxruntime.quantization.shape_inference import quant_pre_process
            except ImportError:
                quant_pre_process = quantization.quant_pre_process
            quant_pre_process(str(fp32_path), str(preprocessed_path))
            model_input = preprocessed_path
            notes.append("Ran quant_pre_process before static quantization.")
        except Exception as exc:
            notes.append(f"quant_pre_process unavailable or failed ({exc}); quantizing original FP32 graph.")
        reader = CalibrationReader(
            input_factory=input_factory,
            sample_count=int(args.calibration_samples),
            batch_size=int(args.batch_size),
        )
        quantization.quantize_static(
            str(model_input),
            str(output_path),
            calibration_data_reader=reader,
            quant_format=quant_format_value(args.quant_format, quantization),
            activation_type=quantization.QuantType.QInt8,
            weight_type=quantization.QuantType.QInt8,
        )
        notes.append(
            f"INT8 static quantization used {args.quant_format.upper()} format with "
            f"{int(args.calibration_samples)} calibration sample(s)."
        )
    onnx.checker.check_model(str(output_path))
    return PrecisionArtifact("INT8", output_path, notes, metadata)


def file_size_mb(path: Path) -> float:
    return path.stat().st_size / 1024.0**2


def available_provider_names(ort: Any) -> list[str]:
    try:
        return list(ort.get_available_providers())
    except Exception:
        return []


def make_session(ort: Any, model_path: Path, provider: str, args: argparse.Namespace, output_dir: Path, precision: str) -> Any:
    options = ort.SessionOptions()
    if args.num_threads:
        options.intra_op_num_threads = int(args.num_threads)
        options.inter_op_num_threads = max(1, int(args.num_threads))
    if args.profile_ort:
        options.enable_profiling = True
        safe_provider = provider.replace("ExecutionProvider", "").replace("/", "_")
        options.profile_file_prefix = str(output_dir / f"ort_profile_{precision.lower()}_{safe_provider.lower()}")
    return ort.InferenceSession(str(model_path), sess_options=options, providers=[provider])


def provider_iobinding_device(provider: str) -> str | None:
    mapping = {
        "CUDAExecutionProvider": "cuda",
        "ROCMExecutionProvider": "cuda",
        "DmlExecutionProvider": "dml",
    }
    return mapping.get(provider)


def run_session_once(session: Any, inputs: dict[str, Any], provider: str, use_iobinding: bool) -> tuple[list[Any], bool, str | None]:
    if not use_iobinding:
        return list(session.run(None, inputs)), False, None
    device_type = provider_iobinding_device(provider)
    if device_type is None:
        return list(session.run(None, inputs)), False, f"I/O Binding device mapping is unavailable for {provider}."
    try:
        binding = session.io_binding()
        for name, array in inputs.items():
            binding.bind_cpu_input(name, array)
        for output in session.get_outputs():
            binding.bind_output(output.name, device_type=device_type, device_id=0)
        session.run_with_iobinding(binding)
        return list(binding.copy_outputs_to_cpu()), True, None
    except Exception as exc:
        return list(session.run(None, inputs)), False, f"I/O Binding failed for {provider}: {exc}"


def numeric_outputs(np: Any, outputs: list[Any]) -> list[Any]:
    arrays = []
    for output in outputs:
        if hasattr(output, "dtype") and np.issubdtype(output.dtype, np.number):
            arrays.append(output)
    return arrays


def compute_drift(np: Any, baseline_outputs: list[list[Any]], candidate_outputs: list[list[Any]]) -> dict[str, Any]:
    if not baseline_outputs or not candidate_outputs:
        return {
            "accuracy_or_drift_vs_fp32": "N/A_no_fp32_baseline",
            "mean_abs_error": None,
            "max_abs_error": None,
            "cosine_similarity": None,
        }
    total_abs = 0.0
    total_count = 0
    max_abs = 0.0
    dot = 0.0
    base_norm = 0.0
    cand_norm = 0.0
    for base_group, cand_group in zip(baseline_outputs, candidate_outputs):
        base_arrays = numeric_outputs(np, base_group)
        cand_arrays = numeric_outputs(np, cand_group)
        if len(base_arrays) != len(cand_arrays):
            return {
                "accuracy_or_drift_vs_fp32": "N/A_output_count_mismatch",
                "mean_abs_error": None,
                "max_abs_error": None,
                "cosine_similarity": None,
            }
        for base, cand in zip(base_arrays, cand_arrays):
            if tuple(base.shape) != tuple(cand.shape):
                return {
                    "accuracy_or_drift_vs_fp32": "N/A_output_shape_mismatch",
                    "mean_abs_error": None,
                    "max_abs_error": None,
                    "cosine_similarity": None,
                }
            base_flat = base.astype(np.float32, copy=False).ravel()
            cand_flat = cand.astype(np.float32, copy=False).ravel()
            diff = np.abs(base_flat - cand_flat)
            total_abs += float(diff.sum())
            total_count += int(diff.size)
            max_abs = max(max_abs, float(diff.max()) if diff.size else 0.0)
            dot += float(np.dot(base_flat, cand_flat))
            base_norm += float(np.dot(base_flat, base_flat))
            cand_norm += float(np.dot(cand_flat, cand_flat))
    if total_count == 0:
        return {
            "accuracy_or_drift_vs_fp32": "N/A_no_numeric_outputs",
            "mean_abs_error": None,
            "max_abs_error": None,
            "cosine_similarity": None,
        }
    cosine = dot / math.sqrt(base_norm * cand_norm) if base_norm > 0 and cand_norm > 0 else None
    mean_abs = total_abs / float(total_count)
    return {
        "accuracy_or_drift_vs_fp32": f"MAE={mean_abs:.6g}; max={max_abs:.6g}; cos={format_value(cosine, 6)}",
        "mean_abs_error": mean_abs,
        "max_abs_error": max_abs,
        "cosine_similarity": cosine,
    }


def read_power_log(args: argparse.Namespace) -> tuple[list[tuple[float, float]] | None, str | None]:
    if not args.power_log:
        return None, "energy unavailable: no --power-log supplied"
    path = expand_path(args.power_log)
    if not path.exists():
        return None, f"energy unavailable: --power-log not found ({path})"
    samples: list[tuple[float, float]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if args.timestamp_column not in (reader.fieldnames or []) or args.power_column not in (reader.fieldnames or []):
            return None, (
                f"energy unavailable: power log must contain columns "
                f"{args.timestamp_column!r} and {args.power_column!r}"
            )
        for row in reader:
            try:
                timestamp = float(row[args.timestamp_column])
                power = float(row[args.power_column])
            except (TypeError, ValueError):
                continue
            if math.isfinite(timestamp) and math.isfinite(power):
                samples.append((timestamp, power))
    samples.sort(key=lambda item: item[0])
    if len(samples) < 2:
        return None, "energy unavailable: power log has fewer than two valid samples"
    return samples, None


def interpolate_power(samples: list[tuple[float, float]], timestamp: float) -> float | None:
    if timestamp < samples[0][0] or timestamp > samples[-1][0]:
        return None
    for index in range(1, len(samples)):
        left_t, left_p = samples[index - 1]
        right_t, right_p = samples[index]
        if left_t <= timestamp <= right_t:
            if right_t == left_t:
                return float(right_p)
            weight = (timestamp - left_t) / (right_t - left_t)
            return float(left_p * (1.0 - weight) + right_p * weight)
    return None


def energy_from_power_log(
    samples: list[tuple[float, float]] | None,
    start_epoch_s: float,
    end_epoch_s: float,
    inference_count: int,
) -> tuple[float | None, float | None, str | None]:
    if samples is None:
        return None, None, None
    duration = max(0.0, end_epoch_s - start_epoch_s)
    if duration <= 0 or inference_count <= 0:
        return None, None, "energy unavailable: benchmark window was empty"
    if samples[0][0] > 1.0e8:
        start = start_epoch_s
        end = end_epoch_s
        mode_note = "power log interpreted as epoch seconds"
    else:
        start = 0.0
        end = duration
        mode_note = "power log interpreted as relative seconds"
    start_power = interpolate_power(samples, start)
    end_power = interpolate_power(samples, end)
    if start_power is None or end_power is None:
        return None, None, "energy unavailable: power log does not overlap the benchmark window"
    window_points = [(start, start_power)]
    window_points.extend((timestamp, power) for timestamp, power in samples if start < timestamp < end)
    window_points.append((end, end_power))
    if len(window_points) < 2:
        return None, None, "energy unavailable: fewer than two power samples in benchmark window"
    total_energy_j = 0.0
    for (left_t, left_p), (right_t, right_p) in zip(window_points, window_points[1:]):
        total_energy_j += 0.5 * (left_p + right_p) * (right_t - left_t)
    avg_power_w = total_energy_j / duration
    energy_mj = (total_energy_j / float(inference_count)) * 1000.0
    return avg_power_w, energy_mj, mode_note


def benchmark_artifact(
    artifact: PrecisionArtifact,
    provider: str,
    args: argparse.Namespace,
    ort: Any,
    np: Any,
    psutil: Any,
    input_factory: InputFactory,
    output_dir: Path,
    power_samples: list[tuple[float, float]] | None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, list[list[Any]] | None, str | None]:
    available = available_provider_names(ort)
    if provider not in available:
        return None, {
            "precision": artifact.precision,
            "onnx_model_path": str(artifact.path),
            "execution_provider": provider,
            "reason": f"Provider is not available in this ONNX Runtime install. Available: {available}",
        }, None, None

    notes = list(artifact.notes)
    if artifact.precision == "FP16" and provider == "CPUExecutionProvider":
        notes.append("FP16 on CPU may be emulated or unsupported depending on ONNX Runtime kernels.")
    if power_samples is None:
        notes.append("energy unavailable: no usable --power-log supplied")
    if input_factory.source_kind == "dummy":
        notes.append("accuracy=N/A_dummy_inputs; drift was measured on deterministic dummy inputs")

    try:
        session = make_session(ort, artifact.path, provider, args, output_dir, artifact.precision)
    except Exception as exc:
        return None, {
            "precision": artifact.precision,
            "onnx_model_path": str(artifact.path),
            "execution_provider": provider,
            "reason": f"Failed to create ONNX Runtime session: {exc}",
        }, None, None

    session_providers = list(session.get_providers())
    if not session_providers or session_providers[0] != provider:
        notes.append(f"provider fallback suspected: session providers={session_providers}")

    warmup_batches = input_factory.make_batches(int(args.warmup))
    timed_batches = input_factory.make_batches(int(args.runs))
    drift_batches = input_factory.make_batches(int(args.drift_samples))
    use_iobinding = bool(provider != "CPUExecutionProvider" and not args.disable_iobinding)
    iobinding_used = False
    iobinding_note_recorded = False

    try:
        for batch in warmup_batches[: int(args.warmup)]:
            _outputs, used_binding, binding_note = run_session_once(session, batch, provider, use_iobinding)
            iobinding_used = iobinding_used or used_binding
            if binding_note and not iobinding_note_recorded:
                notes.append(binding_note)
                iobinding_note_recorded = True
                use_iobinding = False
    except Exception as exc:
        return None, {
            "precision": artifact.precision,
            "onnx_model_path": str(artifact.path),
            "execution_provider": provider,
            "reason": f"Warmup inference failed: {exc}",
        }, None, None

    nvml_monitor: NvmlMemoryMonitor | None = None
    pynvml = import_optional("pynvml") if provider == "CUDAExecutionProvider" else None
    if pynvml is not None:
        try:
            nvml_monitor = NvmlMemoryMonitor(pynvml, device_index=int(args.device_index))
            nvml_monitor.start()
        except Exception as exc:
            notes.append(f"pynvml GPU memory sampling unavailable: {exc}")
            nvml_monitor = None
    elif provider == "CUDAExecutionProvider":
        notes.append("pynvml unavailable; device memory is N/A")

    rss_monitor = PeakRSSMonitor(psutil)
    rss_monitor.start()
    latencies_ms: list[float] = []
    timed_start_epoch = time.time()
    try:
        for batch in timed_batches[: int(args.runs)]:
            start_ns = time.perf_counter_ns()
            _outputs, used_binding, binding_note = run_session_once(session, batch, provider, use_iobinding)
            elapsed_ms = (time.perf_counter_ns() - start_ns) / 1.0e6
            latencies_ms.append(float(elapsed_ms))
            iobinding_used = iobinding_used or used_binding
            if binding_note and not iobinding_note_recorded:
                notes.append(binding_note)
                iobinding_note_recorded = True
                use_iobinding = False
    except Exception as exc:
        rss_monitor.stop()
        if nvml_monitor is not None:
            nvml_monitor.stop()
        return None, {
            "precision": artifact.precision,
            "onnx_model_path": str(artifact.path),
            "execution_provider": provider,
            "reason": f"Timed inference failed: {exc}",
        }, None, None
    timed_end_epoch = time.time()
    peak_host_rss_mb = rss_monitor.stop()
    peak_device_memory_mb = nvml_monitor.stop() if nvml_monitor is not None else None

    if provider != "CPUExecutionProvider" and args.disable_iobinding:
        notes.append("I/O Binding disabled by --disable-iobinding.")
    elif provider != "CPUExecutionProvider" and iobinding_used:
        notes.append("I/O Binding used for at least one timed or warmup inference.")
    elif provider != "CPUExecutionProvider":
        notes.append("I/O Binding was not used; regular session.run path measured.")

    drift_outputs: list[list[Any]] = []
    try:
        for batch in drift_batches[: int(args.drift_samples)]:
            outputs, _used_binding, _binding_note = run_session_once(session, batch, provider, False)
            drift_outputs.append(outputs)
    except Exception as exc:
        notes.append(f"drift output collection failed: {exc}")
        drift_outputs = []

    batch_size = input_factory.batch_size_for_inputs(timed_batches[0]) if timed_batches else int(args.batch_size)
    inference_count = max(1, int(args.runs) * max(1, batch_size))
    total_time_s = max(0.0, timed_end_epoch - timed_start_epoch)
    avg_power_w, energy_mj, power_note = energy_from_power_log(
        power_samples,
        timed_start_epoch,
        timed_end_epoch,
        inference_count=inference_count,
    )
    if power_note:
        notes.append(power_note)
    if avg_power_w is None:
        avg_power_w_value: float | str | None = "N/A"
        energy_mj_value: float | str | None = "N/A"
    else:
        avg_power_w_value = avg_power_w
        energy_mj_value = energy_mj

    profile_path = None
    if args.profile_ort:
        try:
            profile_path = session.end_profiling()
            notes.append(f"ORT profile saved: {profile_path}")
        except Exception as exc:
            notes.append(f"ORT profiling enabled but end_profiling failed: {exc}")

    row = {
        "precision": artifact.precision,
        "onnx_model_path": str(artifact.path),
        "onnx_file_size_mb": file_size_mb(artifact.path),
        "execution_provider": provider,
        "session_providers": ",".join(session_providers),
        "hardware_device": args.device_name or platform.platform(),
        "mean_latency_ms": statistics.mean(latencies_ms) if latencies_ms else None,
        "median_latency_ms": statistics.median(latencies_ms) if latencies_ms else None,
        "p90_latency_ms": percentile(latencies_ms, 0.90),
        "p95_latency_ms": percentile(latencies_ms, 0.95),
        "throughput_samples_per_sec": (inference_count / total_time_s) if total_time_s > 0 else None,
        "peak_host_rss_mb": peak_host_rss_mb,
        "peak_device_memory_mb": peak_device_memory_mb,
        "average_power_w": avg_power_w_value,
        "energy_per_inference_mj": energy_mj_value,
        "accuracy_or_drift_vs_fp32": "pending",
        "mean_abs_error": None,
        "max_abs_error": None,
        "cosine_similarity": None,
        "speedup_vs_fp32": None,
        "size_reduction_vs_fp32": None,
        "notes": "; ".join(dict.fromkeys(note for note in notes if note)),
        "_profile_path": profile_path,
        "_latencies_ms": latencies_ms,
    }
    return row, None, drift_outputs, profile_path


def update_relative_metrics(rows: list[dict[str, Any]], fp32_size_mb: float) -> None:
    fp32_latency_by_provider: dict[str, float] = {}
    for row in rows:
        if row.get("precision") == "FP32" and isinstance(row.get("mean_latency_ms"), (int, float)):
            fp32_latency_by_provider[str(row["execution_provider"])] = float(row["mean_latency_ms"])
    for row in rows:
        provider = str(row.get("execution_provider"))
        mean_latency = row.get("mean_latency_ms")
        baseline_latency = fp32_latency_by_provider.get(provider)
        if baseline_latency and isinstance(mean_latency, (int, float)) and float(mean_latency) > 0:
            row["speedup_vs_fp32"] = baseline_latency / float(mean_latency)
        if fp32_size_mb > 0 and isinstance(row.get("onnx_file_size_mb"), (int, float)):
            row["size_reduction_vs_fp32"] = 100.0 * (1.0 - float(row["onnx_file_size_mb"]) / fp32_size_mb)


def write_csv_table(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [key for key, _label in TABLE_COLUMNS]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def markdown_escape(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def write_markdown_table(path: Path, rows: list[dict[str, Any]]) -> None:
    labels = [label for _key, label in TABLE_COLUMNS]
    keys = [key for key, _label in TABLE_COLUMNS]
    table_rows = [[markdown_escape(format_value(row.get(key))) for key in keys] for row in rows]
    widths = [len(label) for label in labels]
    for row in table_rows:
        widths = [max(width, len(cell)) for width, cell in zip(widths, row)]
    header = "| " + " | ".join(label.ljust(width) for label, width in zip(labels, widths)) + " |"
    sep = "| " + " | ".join("-" * width for width in widths) + " |"
    body = ["| " + " | ".join(cell.ljust(width) for cell, width in zip(row, widths)) + " |" for row in table_rows]
    path.write_text("\n".join([header, sep, *body]) + "\n", encoding="utf-8")


def latex_escape(value: Any) -> str:
    text = str(value).replace("\n", " ")
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def write_latex_table(path: Path, rows: list[dict[str, Any]]) -> None:
    labels = [label for _key, label in TABLE_COLUMNS]
    keys = [key for key, _label in TABLE_COLUMNS]
    column_spec = "l" * len(keys)
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2pt}",
        r"\begin{tabular}{" + column_spec + r"}",
        r"\hline",
        " & ".join(latex_escape(label) for label in labels) + r" \\",
        r"\hline",
    ]
    for row in rows:
        lines.append(" & ".join(latex_escape(format_value(row.get(key))) for key in keys) + r" \\")
    lines.extend([r"\hline", r"\end{tabular}", r"\end{table*}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_summary(path: Path, rows: list[dict[str, Any]], failures: list[dict[str, Any]], args: argparse.Namespace) -> None:
    successful = [row for row in rows if isinstance(row.get("mean_latency_ms"), (int, float))]
    providers = sorted({str(row.get("execution_provider")) for row in rows})
    best_latency = min(successful, key=lambda row: float(row["mean_latency_ms"])) if successful else None
    smallest = min(rows, key=lambda row: float(row["onnx_file_size_mb"])) if rows else None
    lines = [
        "ONNX Precision Benchmark Summary",
        f"Generated at: {dt.datetime.utcnow().replace(microsecond=0).isoformat()}Z",
        f"Hardware/device label: {args.device_name or platform.platform()}",
        f"Providers tested successfully: {', '.join(providers) if providers else 'none'}",
    ]
    if best_latency is not None:
        lines.append(
            "Best latency: "
            f"{best_latency['precision']} / {best_latency['execution_provider']} "
            f"mean={format_value(best_latency['mean_latency_ms'])} ms"
        )
    else:
        lines.append("Best latency: N/A")
    if smallest is not None:
        lines.append(
            "Smallest model: "
            f"{smallest['precision']} size={format_value(smallest['onnx_file_size_mb'])} MB "
            f"path={smallest['onnx_model_path']}"
        )
    else:
        lines.append("Smallest model: N/A")
    for provider in providers:
        fp32 = next((row for row in rows if row["precision"] == "FP32" and row["execution_provider"] == provider), None)
        if fp32 is None:
            continue
        fp32_latency = float(fp32["mean_latency_ms"])
        for precision in ("FP16", "INT8"):
            candidate = next(
                (row for row in rows if row["precision"] == precision and row["execution_provider"] == provider),
                None,
            )
            if candidate is None or not isinstance(candidate.get("mean_latency_ms"), (int, float)):
                lines.append(f"{precision} latency improvement on {provider}: not measured")
                continue
            improved = float(candidate["mean_latency_ms"]) < fp32_latency
            lines.append(
                f"{precision} latency improvement on {provider}: "
                f"{'yes' if improved else 'no'} "
                f"(speedup={format_value(candidate.get('speedup_vs_fp32'))}x)"
            )
    if args.device_name:
        lines.append(
            "Actual edge hardware: user supplied a device label; the script does not independently verify hardware class."
        )
    else:
        lines.append("Actual edge hardware: not indicated; use --device-name to label Raspberry Pi, Jetson, Snapdragon, etc.")
    any_energy = any(isinstance(row.get("energy_per_inference_mj"), (int, float)) for row in rows)
    lines.append("Energy measurement: measured from --power-log" if any_energy else "Energy measurement: unavailable (N/A)")
    if failures:
        lines.append(f"Failed precision/provider combinations: {len(failures)}; see onnx_precision_benchmark_failures.json")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def hardware_metadata(args: argparse.Namespace, ort: Any, psutil: Any) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "generated_at_utc": dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "command": sys.argv,
        "device_name": args.device_name,
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "hostname": socket.gethostname(),
        "python": sys.version,
        "cpu_count_logical": os.cpu_count(),
        "onnxruntime_version": getattr(ort, "__version__", None),
        "onnxruntime_available_providers": available_provider_names(ort),
        "requested_providers": args.providers,
    }
    try:
        vm = psutil.virtual_memory()
        metadata["host_memory_total_mb"] = vm.total / 1024.0**2
    except Exception:
        pass
    pynvml = import_optional("pynvml")
    if pynvml is not None:
        try:
            pynvml.nvmlInit()
            count = int(pynvml.nvmlDeviceGetCount())
            devices = []
            for index in range(count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(index)
                name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode("utf-8", errors="replace")
                memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                devices.append({"index": index, "name": name, "memory_total_mb": memory.total / 1024.0**2})
            metadata["nvidia_devices"] = devices
            metadata["nvidia_driver_version"] = pynvml.nvmlSystemGetDriverVersion()
            pynvml.nvmlShutdown()
        except Exception as exc:
            metadata["nvidia_metadata_error"] = str(exc)
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark FP32, FP16, and INT8 ONNX Runtime inference across precision/provider combinations."
    )
    parser.add_argument("--fp32-onnx", required=True, help="Path to the FP32 baseline ONNX model.")
    parser.add_argument("--output-dir", required=True, help="Directory for benchmark tables, metadata, and converted models.")
    parser.add_argument("--providers", nargs="+", default=["CPUExecutionProvider"], help="ONNX Runtime Execution Providers to try.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--calibration-samples", type=int, default=128)
    parser.add_argument("--table-formats", nargs="+", choices=("csv", "markdown", "latex"), default=["csv", "markdown", "latex"])
    parser.add_argument("--fp16-onnx", default=None, help="Existing FP16 ONNX model. If omitted, conversion is attempted.")
    parser.add_argument("--int8-onnx", default=None, help="Existing INT8 ONNX model. If omitted, quantization is attempted.")
    parser.add_argument("--skip-fp16-conversion", action="store_true")
    parser.add_argument("--skip-int8-quantization", action="store_true")
    parser.add_argument("--quantization-mode", choices=("static", "dynamic"), default="static")
    parser.add_argument("--quant-format", choices=("qdq", "qoperator"), default="qdq")
    parser.add_argument("--power-log", default=None, help="CSV containing real measured power samples.")
    parser.add_argument("--power-column", default="power_w")
    parser.add_argument("--timestamp-column", default="timestamp_s")
    parser.add_argument("--device-name", default=None, help="Human-readable hardware label for the benchmark table.")
    parser.add_argument(
        "--input-shape",
        nargs="+",
        default=None,
        help="Fallback shape override, e.g. '1,64' or 'input_ids=1,64 attention_mask=1,64'.",
    )
    parser.add_argument("--num-threads", type=int, default=None, help="Set ORT intra/inter op thread count.")
    parser.add_argument("--disable-iobinding", action="store_true")
    parser.add_argument("--profile-ort", action="store_true", help="Enable ONNX Runtime profiling for each successful session.")
    parser.add_argument("--dataset", default=None, help="Optional SCENIC/JSON/JSONL/CSV/TXT prompt dataset for real inputs.")
    parser.add_argument("--model-path", default=None, help="Optional tokenizer/model path used with --dataset.")
    parser.add_argument("--prompt-field", default=None)
    parser.add_argument("--prompt-format", choices=("raw", "legacy", "chat-template"), default="raw")
    parser.add_argument("--system-prompt", default=None)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--add-special-tokens", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--drift-samples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device-index", type=int, default=0, help="CUDA/NVML device index for optional GPU memory sampling.")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)
    if args.batch_size <= 0 or args.runs <= 0:
        raise ValueError("--batch-size and --runs must be positive.")
    fp32_path = expand_path(args.fp32_onnx)
    if not fp32_path.exists():
        raise FileNotFoundError(f"FP32 ONNX model not found: {fp32_path}")
    output_dir = expand_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    preflight_required_packages(args)
    np = import_required("numpy", "input generation and drift metrics")
    onnx = import_required("onnx", "model validation and input metadata")
    ort = import_required("onnxruntime", "ONNX Runtime benchmarking")
    psutil = import_required("psutil", "peak host RSS memory measurement")
    if import_optional("pandas") is None:
        logging.info("Optional package pandas is not installed; using the built-in CSV writer.")
    if import_optional("tabulate") is None:
        logging.info("Optional package tabulate is not installed; using the built-in Markdown table writer.")

    onnx.checker.check_model(str(fp32_path))
    input_metas = load_onnx_input_metas(fp32_path, onnx, np)
    input_factory = InputFactory(args, input_metas, np)
    for note in input_factory.notes:
        logging.warning(note) if "dummy" in note.lower() or "missing" in note.lower() else logging.info(note)

    artifacts: list[PrecisionArtifact] = [PrecisionArtifact("FP32", fp32_path, ["FP32 baseline model."], {"source": "user"})]
    conversion_metadata: dict[str, Any] = {"FP32": {"path": str(fp32_path), "size_mb": file_size_mb(fp32_path)}}
    if args.fp16_onnx:
        fp16_path = expand_path(args.fp16_onnx)
        if not fp16_path.exists():
            raise FileNotFoundError(f"FP16 ONNX model not found: {fp16_path}")
        onnx.checker.check_model(str(fp16_path))
        artifacts.append(PrecisionArtifact("FP16", fp16_path, ["Using user-supplied FP16 ONNX."], {"source": "user"}))
    elif not args.skip_fp16_conversion:
        fp16_artifact = convert_fp16(fp32_path, output_dir)
        if fp16_artifact is not None:
            artifacts.append(fp16_artifact)
    else:
        logging.warning("Skipping FP16 conversion; FP16 rows will not be benchmarked.")

    if args.int8_onnx:
        int8_path = expand_path(args.int8_onnx)
        if not int8_path.exists():
            raise FileNotFoundError(f"INT8 ONNX model not found: {int8_path}")
        onnx.checker.check_model(str(int8_path))
        artifacts.append(PrecisionArtifact("INT8", int8_path, ["Using user-supplied INT8 ONNX."], {"source": "user"}))
    elif not args.skip_int8_quantization:
        int8_artifact = quantize_int8(fp32_path, output_dir, args, input_factory)
        if int8_artifact is not None:
            artifacts.append(int8_artifact)
    else:
        logging.warning("Skipping INT8 quantization; INT8 rows will not be benchmarked.")

    for artifact in artifacts:
        conversion_metadata[artifact.precision] = {
            "path": str(artifact.path),
            "size_mb": file_size_mb(artifact.path),
            "notes": artifact.notes,
            **artifact.metadata,
        }

    power_samples, power_warning = read_power_log(args)
    if power_warning:
        logging.warning(power_warning)

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    fp32_drift_by_provider: dict[str, list[list[Any]]] = {}

    for artifact in artifacts:
        for provider in args.providers:
            logging.info("Benchmarking %s on %s", artifact.precision, provider)
            row, failure, drift_outputs, _profile = benchmark_artifact(
                artifact,
                provider,
                args,
                ort,
                np,
                psutil,
                input_factory,
                output_dir,
                power_samples,
            )
            if failure is not None:
                logging.warning("%s / %s failed: %s", artifact.precision, provider, failure["reason"])
                failures.append(failure)
                continue
            assert row is not None
            if artifact.precision == "FP32":
                fp32_drift_by_provider[provider] = drift_outputs or []
                row.update(
                    {
                        "accuracy_or_drift_vs_fp32": "baseline",
                        "mean_abs_error": 0.0,
                        "max_abs_error": 0.0,
                        "cosine_similarity": 1.0,
                    }
                )
            else:
                baseline_outputs = fp32_drift_by_provider.get(provider) or []
                drift = compute_drift(np, baseline_outputs, drift_outputs or [])
                row.update(drift)
            rows.append(row)

    update_relative_metrics(rows, fp32_size_mb=file_size_mb(fp32_path))

    public_rows = [{key: row.get(key) for key, _label in TABLE_COLUMNS} for row in rows]
    if "csv" in args.table_formats:
        write_csv_table(output_dir / "onnx_precision_benchmark.csv", public_rows)
    if "markdown" in args.table_formats:
        write_markdown_table(output_dir / "onnx_precision_benchmark.md", public_rows)
    if "latex" in args.table_formats:
        write_latex_table(output_dir / "onnx_precision_benchmark.tex", public_rows)

    write_json(
        output_dir / "benchmark_config.json",
        {
            "args": vars(args),
            "input_metadata": [
                {"name": meta.name, "shape": meta.shape, "dtype": str(np.dtype(meta.np_dtype))}
                for meta in input_metas
            ],
            "input_source": input_factory.source_kind,
            "input_notes": input_factory.notes,
            "precision_artifacts": conversion_metadata,
        },
    )
    write_json(output_dir / "hardware_metadata.json", hardware_metadata(args, ort, psutil))
    write_json(output_dir / "onnx_precision_benchmark_failures.json", failures)
    write_summary(output_dir / "onnx_precision_benchmark_summary.txt", public_rows, failures, args)

    if not rows:
        raise RuntimeError("No precision/provider combinations completed successfully. See failure report in output-dir.")
    print(f"Wrote ONNX precision benchmark outputs to: {output_dir}")
    for suffix in ("csv", "md", "tex"):
        path = output_dir / f"onnx_precision_benchmark.{suffix}"
        if path.exists():
            print(f"  {path}")
    print(f"  {output_dir / 'benchmark_config.json'}")
    print(f"  {output_dir / 'hardware_metadata.json'}")
    print(f"  {output_dir / 'onnx_precision_benchmark_summary.txt'}")
    if failures:
        print(f"  {output_dir / 'onnx_precision_benchmark_failures.json'}")


if __name__ == "__main__":
    try:
        main()
    except RequiredPackageError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc
