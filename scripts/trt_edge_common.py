#!/usr/bin/env python
from __future__ import annotations

import csv
import importlib
import json
import logging
import re
import unicodedata
from pathlib import Path
from typing import Any


JSON_LIST_KEYS = ("data", "records", "items", "examples", "eval", "validation", "test", "train")
PROMPT_FIELDS = ("prompt", "instruction", "question", "input", "query", "text", "command", "x")
REFERENCE_FIELDS = ("response", "responses", "output", "answer", "completion", "target", "label", "y")
CONTENT_FIELDS = ("content", "value", "text", "message")
MESSAGE_FIELDS = ("messages", "conversations")
ZERO_WIDTH_PATTERN = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060\ufeff]")


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


def import_required(module_name: str, feature: str) -> Any:
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise RuntimeError(
            f"Missing required package/import '{module_name}'. Install or enable it for: {feature}."
        ) from exc


def expand_path(path: str | Path) -> Path:
    return Path(path).expanduser()


def ensure_output_path(path: Path, overwrite: bool = False, kind: str = "file") -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing {kind}: {path}. Pass --overwrite to replace it.")
    path.parent.mkdir(parents=True, exist_ok=True)


def coerce_json_records(payload: Any, path: Path) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict):
        records = None
        for key in JSON_LIST_KEYS:
            value = payload.get(key)
            if isinstance(value, list):
                records = value
                break
        if records is None:
            records = [payload]
    else:
        raise ValueError(f"{path} must contain a JSON object, JSON list, or wrapper with one of {JSON_LIST_KEYS}.")

    clean_records: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"{path}: record {index} must be a JSON object.")
        clean_records.append(record)
    return clean_records


def read_records(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    data_path = expand_path(path)
    suffix = data_path.suffix.lower()
    records: list[dict[str, Any]]
    if suffix == ".json":
        records = coerce_json_records(json.loads(data_path.read_text(encoding="utf-8")), data_path)
    elif suffix == ".jsonl":
        records = []
        with data_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError(f"{data_path}:{line_number} must be a JSON object.")
                records.append(record)
    elif suffix == ".csv":
        with data_path.open("r", encoding="utf-8", newline="") as handle:
            records = list(csv.DictReader(handle))
    elif suffix == ".txt":
        with data_path.open("r", encoding="utf-8") as handle:
            records = [{"text": line.strip()} for line in handle if line.strip()]
    else:
        raise ValueError(f"Unsupported dataset extension {suffix}. Use .json, .jsonl, .csv, or .txt.")
    return records[: int(limit)] if limit is not None else records


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        text = "\n".join(clean_text(item) for item in value)
    elif isinstance(value, dict):
        for field in CONTENT_FIELDS + PROMPT_FIELDS + REFERENCE_FIELDS:
            if field in value:
                text = clean_text(value.get(field))
                break
        else:
            text = json.dumps(value, ensure_ascii=False)
    else:
        text = str(value)
    text = text.replace("\ufeff", "").replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.strip() for line in text.splitlines()).strip()


def first_text(record: dict[str, Any], fields: tuple[str, ...]) -> str:
    for field in fields:
        text = clean_text(record.get(field))
        if text:
            return text
    return ""


def message_role(turn: dict[str, Any]) -> str:
    return clean_text(turn.get("role") or turn.get("from") or turn.get("speaker")).lower()


def message_content(turn: dict[str, Any]) -> str:
    for field in CONTENT_FIELDS:
        text = clean_text(turn.get(field))
        if text:
            return text
    return ""


def iter_messages(record: dict[str, Any]) -> list[dict[str, Any]]:
    for field in MESSAGE_FIELDS:
        messages = record.get(field)
        if isinstance(messages, list):
            return [turn for turn in messages if isinstance(turn, dict)]
    return []


def prompt_and_reference(record: dict[str, Any], prompt_field: str | None = None) -> tuple[str, str | None]:
    if prompt_field:
        prompt = clean_text(record.get(prompt_field))
        if not prompt:
            raise KeyError(f"Prompt field '{prompt_field}' is empty or missing. Available fields: {', '.join(record)}")
    else:
        messages = iter_messages(record)
        prompt = ""
        if messages:
            assistant_indices = [
                index
                for index, turn in enumerate(messages)
                if message_role(turn) in {"assistant", "gpt", "bot", "model"} and message_content(turn)
            ]
            stop = assistant_indices[-1] if assistant_indices else len(messages)
            user_parts = [
                message_content(turn)
                for turn in messages[:stop]
                if message_role(turn) in {"user", "human", "instruction", "prompt"} and message_content(turn)
            ]
            prompt = "\n".join(user_parts).strip()
        if not prompt:
            prompt = first_text(record, PROMPT_FIELDS)
            if record.get("instruction") and record.get("input") and clean_text(record["instruction"]) != clean_text(record["input"]):
                prompt = f"{clean_text(record['instruction'])}\n{clean_text(record['input'])}".strip()
        if not prompt:
            raise KeyError(f"Could not find prompt text. Available fields: {', '.join(record)}")

    reference = first_text(record, REFERENCE_FIELDS)
    return prompt, reference or None


def apply_prompt_format(tokenizer: Any, prompt: str, prompt_format: str, system_prompt: str | None = None) -> str:
    if prompt_format == "raw":
        return prompt
    if prompt_format == "legacy":
        return f"<|user|>\n{prompt}\n<|assistant|>\n"
    if prompt_format == "chat-template":
        if not hasattr(tokenizer, "apply_chat_template"):
            raise AttributeError("Tokenizer does not provide apply_chat_template; use --prompt-format raw or legacy.")
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return str(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
    raise ValueError(f"Unknown prompt format: {prompt_format}")


def normalize_exact(text: str, tokenizer: Any | None = None) -> str:
    text = unicodedata.normalize("NFKC", str(text))
    text = ZERO_WIDTH_PATTERN.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    special_tokens = {
        "<|eos|>",
        "<|endoftext|>",
        "</s>",
    }
    if tokenizer is not None:
        for attr in ("bos_token", "eos_token", "pad_token", "unk_token"):
            token = getattr(tokenizer, attr, None)
            if token:
                special_tokens.add(str(token))
    for token in sorted(special_tokens, key=len, reverse=True):
        text = text.replace(token, "")
    return " ".join(text.strip().split())


def infer_precision_from_engine(engine_path: str | Path) -> str:
    name = Path(engine_path).name.lower()
    for precision in ("fp16", "int8", "int4", "fp32"):
        if precision in name:
            return precision
    return Path(engine_path).stem


def format_table(headers: list[str], rows: list[list[Any]]) -> str:
    text_rows = [[str(item) for item in row] for row in rows]
    widths = [len(header) for header in headers]
    for row in text_rows:
        widths = [max(width, len(cell)) for width, cell in zip(widths, row)]
    line = " | ".join(header.ljust(width) for header, width in zip(headers, widths))
    sep = "-+-".join("-" * width for width in widths)
    body = [" | ".join(cell.ljust(width) for cell, width in zip(row, widths)) for row in text_rows]
    return "\n".join([line, sep, *body])


class CudaRuntime:
    def __init__(self) -> None:
        try:
            self.cudart = importlib.import_module("cuda.cudart")
        except ImportError as exc:
            raise RuntimeError(
                "Missing required package/import 'cuda.cudart'. Install cuda-python for TensorRT device "
                "memory allocation and INT8 calibration."
            ) from exc

    def check(self, result: Any, action: str) -> Any:
        if not isinstance(result, tuple):
            return result
        err = result[0]
        success = getattr(self.cudart.cudaError_t, "cudaSuccess", 0)
        if err != success:
            name = self.cudart.cudaGetErrorName(err)
            label = name[1].decode("utf-8") if isinstance(name, tuple) and isinstance(name[1], bytes) else str(err)
            raise RuntimeError(f"CUDA failure during {action}: {label}")
        if len(result) == 1:
            return None
        if len(result) == 2:
            return result[1]
        return result[1:]

    def malloc(self, nbytes: int) -> int:
        return int(self.check(self.cudart.cudaMalloc(int(nbytes)), f"cudaMalloc({nbytes})"))

    def free(self, ptr: int) -> None:
        if ptr:
            self.check(self.cudart.cudaFree(ptr), "cudaFree")

    def memcpy_htod_async(self, dst: int, src: Any, nbytes: int, stream: Any) -> None:
        self.check(
            self.cudart.cudaMemcpyAsync(
                int(dst),
                int(src.ctypes.data),
                int(nbytes),
                self.cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                stream,
            ),
            "cudaMemcpyAsync host-to-device",
        )

    def memcpy_dtoh_async(self, dst: Any, src: int, nbytes: int, stream: Any) -> None:
        self.check(
            self.cudart.cudaMemcpyAsync(
                int(dst.ctypes.data),
                int(src),
                int(nbytes),
                self.cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                stream,
            ),
            "cudaMemcpyAsync device-to-host",
        )

    def create_stream(self) -> Any:
        return self.check(self.cudart.cudaStreamCreate(), "cudaStreamCreate")

    def destroy_stream(self, stream: Any) -> None:
        self.check(self.cudart.cudaStreamDestroy(stream), "cudaStreamDestroy")

    def sync_stream(self, stream: Any) -> None:
        self.check(self.cudart.cudaStreamSynchronize(stream), "cudaStreamSynchronize")

    def mem_info(self) -> tuple[int, int]:
        free_bytes, total_bytes = self.check(self.cudart.cudaMemGetInfo(), "cudaMemGetInfo")
        return int(free_bytes), int(total_bytes)


class TensorRTEngineRunner:
    def __init__(self, engine_path: str | Path, verbose: bool = False) -> None:
        self.engine_path = expand_path(engine_path)
        self.trt = import_required("tensorrt", "TensorRT engine inference")
        self.cuda = CudaRuntime()
        logger_level = self.trt.Logger.VERBOSE if verbose else self.trt.Logger.WARNING
        self.logger = self.trt.Logger(logger_level)
        runtime = self.trt.Runtime(self.logger)
        engine_bytes = self.engine_path.read_bytes()
        self.engine = runtime.deserialize_cuda_engine(engine_bytes)
        if self.engine is None:
            raise RuntimeError(f"TensorRT failed to deserialize engine: {self.engine_path}")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("TensorRT failed to create execution context.")
        self.stream = self.cuda.create_stream()
        self.device_buffers: dict[str, int] = {}
        self.buffer_sizes: dict[str, int] = {}
        self.input_names, self.output_names = self._collect_io_names()

    def _collect_io_names(self) -> tuple[list[str], list[str]]:
        inputs: list[str] = []
        outputs: list[str] = []
        if hasattr(self.engine, "num_io_tensors"):
            for index in range(int(self.engine.num_io_tensors)):
                name = self.engine.get_tensor_name(index)
                mode = self.engine.get_tensor_mode(name)
                if mode == self.trt.TensorIOMode.INPUT:
                    inputs.append(name)
                else:
                    outputs.append(name)
        else:
            for index in range(int(self.engine.num_bindings)):
                name = self.engine.get_binding_name(index)
                if self.engine.binding_is_input(index):
                    inputs.append(name)
                else:
                    outputs.append(name)
        return inputs, outputs

    def close(self) -> None:
        for ptr in list(self.device_buffers.values()):
            self.cuda.free(ptr)
        self.device_buffers.clear()
        self.buffer_sizes.clear()
        if getattr(self, "stream", None) is not None:
            self.cuda.destroy_stream(self.stream)
            self.stream = None

    def __del__(self) -> None:  # pragma: no cover - best-effort cleanup.
        try:
            self.close()
        except Exception:
            pass

    def trt_dtype_to_numpy(self, dtype: Any) -> Any:
        np = import_required("numpy", "TensorRT input/output arrays")
        if hasattr(self.trt, "nptype"):
            return self.trt.nptype(dtype)
        mapping = {
            self.trt.float32: np.float32,
            self.trt.float16: np.float16,
            self.trt.int32: np.int32,
            self.trt.int8: np.int8,
            self.trt.bool: np.bool_,
        }
        if hasattr(self.trt, "int64"):
            mapping[self.trt.int64] = np.int64
        if dtype not in mapping:
            raise TypeError(f"No NumPy dtype mapping for TensorRT dtype {dtype}.")
        return mapping[dtype]

    def tensor_dtype(self, name: str) -> Any:
        if hasattr(self.engine, "get_tensor_dtype"):
            return self.engine.get_tensor_dtype(name)
        return self.engine.get_binding_dtype(self.engine.get_binding_index(name))

    def tensor_shape(self, name: str) -> tuple[int, ...]:
        if hasattr(self.context, "get_tensor_shape"):
            return tuple(int(dim) for dim in self.context.get_tensor_shape(name))
        return tuple(int(dim) for dim in self.context.get_binding_shape(self.engine.get_binding_index(name)))

    def set_input_shape(self, name: str, shape: tuple[int, ...]) -> None:
        if hasattr(self.context, "set_input_shape"):
            result = self.context.set_input_shape(name, shape)
            if result is False:
                raise RuntimeError(f"TensorRT rejected shape for {name}: {shape}")
            return
        index = self.engine.get_binding_index(name)
        result = self.context.set_binding_shape(index, shape)
        if result is False:
            raise RuntimeError(f"TensorRT rejected binding shape for {name}: {shape}")

    def set_tensor_address(self, name: str, ptr: int) -> None:
        if hasattr(self.context, "set_tensor_address"):
            result = self.context.set_tensor_address(name, int(ptr))
            if result is False:
                raise RuntimeError(f"TensorRT rejected tensor address for {name}.")
            return
        raise RuntimeError("This runner requires TensorRT execute_async_v3/set_tensor_address support.")

    def ensure_device_buffer(self, name: str, nbytes: int) -> int:
        current = self.device_buffers.get(name)
        if current is not None and self.buffer_sizes.get(name, 0) >= nbytes:
            return current
        if current is not None:
            self.cuda.free(current)
        ptr = self.cuda.malloc(nbytes)
        self.device_buffers[name] = ptr
        self.buffer_sizes[name] = int(nbytes)
        return ptr

    def infer(self, inputs: dict[str, Any]) -> dict[str, Any]:
        np = import_required("numpy", "TensorRT inference arrays")
        missing = [name for name in self.input_names if name not in inputs]
        if missing:
            raise KeyError(f"Missing TensorRT inputs: {missing}. Engine expects: {self.input_names}")

        host_inputs: dict[str, Any] = {}
        for name in self.input_names:
            expected_dtype = self.trt_dtype_to_numpy(self.tensor_dtype(name))
            array = np.ascontiguousarray(inputs[name], dtype=expected_dtype)
            host_inputs[name] = array
            self.set_input_shape(name, tuple(int(dim) for dim in array.shape))

        host_outputs: dict[str, Any] = {}
        for name, array in host_inputs.items():
            ptr = self.ensure_device_buffer(name, int(array.nbytes))
            self.cuda.memcpy_htod_async(ptr, array, int(array.nbytes), self.stream)
            self.set_tensor_address(name, ptr)

        for name in self.output_names:
            shape = self.tensor_shape(name)
            if any(dim < 0 for dim in shape):
                raise RuntimeError(f"TensorRT output shape for {name} is still dynamic after inputs were set: {shape}")
            dtype = self.trt_dtype_to_numpy(self.tensor_dtype(name))
            array = np.empty(shape, dtype=dtype)
            ptr = self.ensure_device_buffer(name, int(array.nbytes))
            self.set_tensor_address(name, ptr)
            host_outputs[name] = array

        if not hasattr(self.context, "execute_async_v3"):
            raise RuntimeError("This runner requires TensorRT 10-style execute_async_v3.")
        try:
            ok = self.context.execute_async_v3(stream_handle=self.stream)
        except TypeError:
            ok = self.context.execute_async_v3(self.stream)
        if not ok:
            raise RuntimeError("TensorRT execute_async_v3 returned false.")

        for name, array in host_outputs.items():
            self.cuda.memcpy_dtoh_async(array, self.device_buffers[name], int(array.nbytes), self.stream)
        self.cuda.sync_stream(self.stream)
        return host_outputs
