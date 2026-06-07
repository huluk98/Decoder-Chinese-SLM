#!/usr/bin/env python
from __future__ import annotations

import argparse
import inspect
import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[0]
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from chatlm_decoder.tokenizer import prepare_decoder_tokenizer, strip_unused_decoder_model_kwargs
from trt_edge_common import (
    CudaRuntime,
    apply_prompt_format,
    ensure_output_path,
    expand_path,
    import_required,
    prompt_and_reference,
    read_records,
    setup_logging,
)


INT4_UNSUPPORTED_MESSAGE = (
    "INT4 TensorRT build is not supported in this environment. "
    "Use ModelOpt/TensorRT-LLM weight-only INT4 path or upgrade TensorRT/ModelOpt."
)


def bytes_from_gib(value: float) -> int:
    return int(float(value) * 1024 * 1024 * 1024)


def trt_logger(trt: Any, verbose: bool) -> Any:
    return trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.INFO)


def cuda_version_and_gpu() -> tuple[str, str]:
    try:
        cuda = CudaRuntime()
        cudart = cuda.cudart
        runtime_version = cuda.check(cudart.cudaRuntimeGetVersion(), "cudaRuntimeGetVersion")
        cuda_version = f"{runtime_version // 1000}.{(runtime_version % 1000) // 10}"
        device = cuda.check(cudart.cudaGetDevice(), "cudaGetDevice")
        props = cuda.check(cudart.cudaGetDeviceProperties(device), "cudaGetDeviceProperties")
        raw_name = getattr(props, "name", b"unknown")
        if isinstance(raw_name, bytes):
            gpu_name = raw_name.split(b"\0", 1)[0].decode("utf-8", errors="replace")
        else:
            gpu_name = str(raw_name).split("\0", 1)[0]
        return cuda_version, gpu_name
    except Exception as exc:
        return f"unavailable ({exc})", "unavailable"


def print_environment(trt: Any) -> None:
    cuda_version, gpu_name = cuda_version_and_gpu()
    logging.info("TensorRT version: %s", getattr(trt, "__version__", "unknown"))
    logging.info("CUDA runtime version: %s", cuda_version)
    logging.info("GPU: %s", gpu_name)


def parse_onnx_network(trt: Any, onnx_path: Path, verbose: bool) -> tuple[Any, Any, Any, Any]:
    logger = trt_logger(trt, verbose)
    builder = trt.Builder(logger)
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)
    logging.info("Parsing ONNX: %s", onnx_path)
    if not parser.parse(onnx_path.read_bytes()):
        errors = []
        for index in range(parser.num_errors):
            errors.append(str(parser.get_error(index)))
        raise RuntimeError("TensorRT ONNX parser failed:\n" + "\n".join(errors))
    config = builder.create_builder_config()
    return builder, network, parser, config


def set_workspace(trt: Any, config: Any, workspace_bytes: int) -> None:
    if hasattr(config, "set_memory_pool_limit") and hasattr(trt, "MemoryPoolType"):
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_bytes))
        logging.info("Workspace memory pool limit: %.2f GiB", workspace_bytes / 1024**3)
    elif hasattr(config, "max_workspace_size"):
        config.max_workspace_size = int(workspace_bytes)
        logging.info("Legacy max_workspace_size: %.2f GiB", workspace_bytes / 1024**3)
    else:
        logging.warning("Could not set TensorRT workspace limit with this TensorRT Python API.")


def input_names(network: Any) -> list[str]:
    return [network.get_input(index).name for index in range(network.num_inputs)]


def positive_dim(dim: int, fallback: int) -> int:
    return int(dim) if int(dim) > 0 else int(fallback)


def profile_shapes_for_input(name: str, shape: tuple[int, ...], args: argparse.Namespace) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    batch = int(args.batch_size)
    min_seq = int(args.min_seq_len)
    opt_seq = int(args.opt_seq_len)
    max_seq = int(args.max_seq_len)
    if len(shape) == 2:
        if name == "attention_mask" and "past" in name:
            return (batch, min_seq), (batch, opt_seq), (batch, max_seq)
        return (batch, min_seq), (batch, opt_seq), (batch, max_seq)
    if len(shape) == 4:
        base = [positive_dim(dim, 1) for dim in shape]
        base[0] = batch
        seq_axis = 2 if len(base) > 2 else len(base) - 1
        min_shape = list(base)
        opt_shape = list(base)
        max_shape = list(base)
        min_shape[seq_axis] = max(1, min_seq)
        opt_shape[seq_axis] = max(1, opt_seq)
        max_shape[seq_axis] = max(1, max_seq)
        return tuple(min_shape), tuple(opt_shape), tuple(max_shape)
    base = [positive_dim(dim, 1) for dim in shape]
    if base:
        base[0] = batch
    return tuple(base), tuple(base), tuple(base)


def create_profile(builder: Any, network: Any, args: argparse.Namespace) -> Any:
    profile = builder.create_optimization_profile()
    for index in range(network.num_inputs):
        tensor = network.get_input(index)
        shape = tuple(int(dim) for dim in tensor.shape)
        if any(dim < 0 for dim in shape):
            min_shape, opt_shape, max_shape = profile_shapes_for_input(tensor.name, shape, args)
            logging.info("Profile %s min=%s opt=%s max=%s", tensor.name, min_shape, opt_shape, max_shape)
            result = profile.set_shape(tensor.name, min_shape, opt_shape, max_shape)
            if result is False:
                raise RuntimeError(f"TensorRT rejected optimization profile for {tensor.name}.")
        else:
            logging.info("Static input %s shape=%s dtype=%s", tensor.name, shape, tensor.dtype)
    return profile


def set_precision_flags(trt: Any, builder: Any, config: Any, precision: str) -> None:
    if precision in {"fp16", "int8"}:
        if hasattr(trt.BuilderFlag, "FP16"):
            if not getattr(builder, "platform_has_fast_fp16", True):
                logging.warning("Builder reports platform_has_fast_fp16=False; FP16 may be slow or unavailable.")
            config.set_flag(trt.BuilderFlag.FP16)
            logging.info("Enabled TensorRT FP16 flag.")
    if precision == "int8":
        if not hasattr(trt.BuilderFlag, "INT8"):
            raise RuntimeError("This TensorRT Python package does not expose BuilderFlag.INT8.")
        if not getattr(builder, "platform_has_fast_int8", True):
            logging.warning("Builder reports platform_has_fast_int8=False; INT8 may be slow or unavailable.")
        config.set_flag(trt.BuilderFlag.INT8)
        logging.info("Enabled TensorRT INT8 flag.")


def set_sparse_weights_flag(trt: Any, config: Any, enabled: bool) -> None:
    if not enabled:
        return
    if not hasattr(trt.BuilderFlag, "SPARSE_WEIGHTS"):
        raise RuntimeError("This TensorRT Python package does not expose BuilderFlag.SPARSE_WEIGHTS.")
    config.set_flag(trt.BuilderFlag.SPARSE_WEIGHTS)
    logging.info("Enabled TensorRT sparse weights flag for NVIDIA 2:4 weights.")


class JsonPromptCalibrator:
    def __init__(
        self,
        trt: Any,
        tokenizer: Any,
        records: list[dict[str, Any]],
        input_names_: list[str],
        cache_path: Path,
        batch_size: int,
        seq_len: int,
        prompt_format: str,
        system_prompt: str | None,
        overwrite_cache: bool,
    ) -> None:
        self.trt = trt
        self.np = import_required("numpy", "INT8 calibration arrays")
        self.cuda = CudaRuntime()
        self.tokenizer = tokenizer
        self.records = records
        self.input_names = input_names_
        self.cache_path = cache_path
        self.batch_size = int(batch_size)
        self.seq_len = int(seq_len)
        self.prompt_format = prompt_format
        self.system_prompt = system_prompt
        self.overwrite_cache = bool(overwrite_cache)
        self.batch_index = 0
        self.device_buffers: dict[str, int] = {}
        self.buffer_sizes: dict[str, int] = {}
        self.batches = self._make_batches()

    def _make_batches(self) -> list[dict[str, Any]]:
        if "input_ids" not in self.input_names or "attention_mask" not in self.input_names:
            raise RuntimeError(
                f"INT8 calibration currently expects input_ids and attention_mask inputs. Found: {self.input_names}"
            )
        batches = []
        for record in self.records:
            prompt, _ = prompt_and_reference(record)
            prompt = apply_prompt_format(self.tokenizer, prompt, self.prompt_format, self.system_prompt)
            encoded = self.tokenizer(
                prompt,
                return_tensors="np",
                truncation=True,
                padding="max_length",
                max_length=self.seq_len,
            )
            input_ids = self.np.ascontiguousarray(encoded["input_ids"].astype(self.np.int32))
            attention_mask = self.np.ascontiguousarray(encoded["attention_mask"].astype(self.np.int32))
            batches.append({"input_ids": input_ids, "attention_mask": attention_mask})
        if not batches:
            raise ValueError("INT8 calibration dataset produced zero usable prompts.")
        logging.info("Prepared %d INT8 calibration batches at seq_len=%d.", len(batches), self.seq_len)
        return batches

    def get_batch_size(self) -> int:
        return self.batch_size

    def read_calibration_cache(self) -> bytes | None:
        if self.cache_path.exists() and not self.overwrite_cache:
            logging.info("Reusing INT8 calibration cache: %s", self.cache_path)
            return self.cache_path.read_bytes()
        return None

    def write_calibration_cache(self, cache: bytes) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_bytes(cache)
        logging.info("Wrote INT8 calibration cache: %s", self.cache_path)

    def _ensure_buffer(self, name: str, nbytes: int) -> int:
        current = self.device_buffers.get(name)
        if current is not None and self.buffer_sizes.get(name, 0) >= nbytes:
            return current
        if current is not None:
            self.cuda.free(current)
        ptr = self.cuda.malloc(nbytes)
        self.device_buffers[name] = ptr
        self.buffer_sizes[name] = nbytes
        return ptr

    def get_batch(self, names: list[str]) -> list[int] | None:
        if self.batch_index >= len(self.batches):
            return None
        batch = self.batches[self.batch_index]
        self.batch_index += 1
        ptrs: list[int] = []
        for name in names:
            if name not in batch:
                raise RuntimeError(f"INT8 calibrator cannot provide TensorRT input '{name}'. Available: {list(batch)}")
            array = self.np.ascontiguousarray(batch[name])
            ptr = self._ensure_buffer(name, int(array.nbytes))
            self.cuda.check(
                self.cuda.cudart.cudaMemcpy(
                    int(ptr),
                    int(array.ctypes.data),
                    int(array.nbytes),
                    self.cuda.cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                ),
                "cudaMemcpy calibration batch",
            )
            ptrs.append(ptr)
        return ptrs

    def free(self) -> None:
        for ptr in list(self.device_buffers.values()):
            self.cuda.free(ptr)
        self.device_buffers.clear()
        self.buffer_sizes.clear()


def make_int8_calibrator(trt: Any, network: Any, args: argparse.Namespace) -> Any:
    if not args.calib_json:
        raise ValueError("--calib_json is required for --precision int8.")
    transformers = import_required("transformers", "INT8 calibration tokenizer loading")
    tokenizer_path = args.tokenizer_path or args.model_path
    if not tokenizer_path:
        raise ValueError("--model-path or --tokenizer-path is required for INT8 calibration tokenization.")
    tokenizer = prepare_decoder_tokenizer(
        transformers.AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=bool(args.trust_remote_code))
    )
    records = read_records(args.calib_json, limit=int(args.calib_samples))
    names = input_names(network)
    cache_path = Path(args.output_dir).expanduser() / "model_int8_calib.cache"

    base_class = getattr(trt, "IInt8EntropyCalibrator2", None) or getattr(trt, "IInt8MinMaxCalibrator", None)
    if base_class is None:
        raise RuntimeError("TensorRT Python package does not expose an INT8 calibrator base class.")

    class Calibrator(base_class):
        def __init__(self) -> None:
            base_class.__init__(self)
            self.impl = JsonPromptCalibrator(
                trt=trt,
                tokenizer=tokenizer,
                records=records,
                input_names_=names,
                cache_path=cache_path,
                batch_size=int(args.batch_size),
                seq_len=int(args.calib_seq_len or args.opt_seq_len),
                prompt_format=args.prompt_format,
                system_prompt=args.system_prompt,
                overwrite_cache=bool(args.overwrite),
            )

        def get_batch_size(self) -> int:
            return self.impl.get_batch_size()

        def get_batch(self, names: list[str]) -> list[int] | None:
            return self.impl.get_batch(names)

        def read_calibration_cache(self) -> bytes | None:
            return self.impl.read_calibration_cache()

        def write_calibration_cache(self, cache: bytes) -> None:
            self.impl.write_calibration_cache(cache)

    return Calibrator()


def build_engine(args: argparse.Namespace) -> Path:
    trt = import_required("tensorrt", "building TensorRT engines")
    print_environment(trt)
    onnx_path = expand_path(args.onnx)
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX file does not exist: {onnx_path}")

    output_dir = Path(args.output_dir).expanduser()
    engine_path = output_dir / f"model_{args.precision}.engine"
    ensure_output_path(engine_path, overwrite=args.overwrite, kind="TensorRT engine")

    builder, network, _parser, config = parse_onnx_network(trt, onnx_path, args.verbose)
    set_workspace(trt, config, bytes_from_gib(args.workspace_gb))
    profile = create_profile(builder, network, args)
    config.add_optimization_profile(profile)
    set_precision_flags(trt, builder, config, args.precision)
    set_sparse_weights_flag(trt, config, bool(args.sparse_weights))

    calibrator = None
    if args.precision == "int8":
        calibrator = make_int8_calibrator(trt, network, args)
        if hasattr(config, "set_calibration_profile"):
            config.set_calibration_profile(profile)
        config.int8_calibrator = calibrator

    logging.info("Building TensorRT %s engine. This can take several minutes on Jetson.", args.precision)
    serialized = None
    if hasattr(builder, "build_serialized_network"):
        serialized = builder.build_serialized_network(network, config)
    else:
        engine = builder.build_engine(network, config)
        if engine is not None:
            serialized = engine.serialize()
    if calibrator is not None and hasattr(calibrator, "impl"):
        calibrator.impl.free()
    if serialized is None:
        raise RuntimeError("TensorRT engine build failed; builder returned None. Check parser/build logs above.")

    output_dir.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(bytes(serialized))
    size_mb = engine_path.stat().st_size / 1024**2
    logging.info("Wrote TensorRT engine: %s (%.2f MB)", engine_path, size_mb)
    return engine_path


def run_forward_loop_for_modelopt(model: Any, tokenizer: Any, records: list[dict[str, Any]], args: argparse.Namespace) -> Any:
    torch = import_required("torch", "ModelOpt INT4 calibration forward loop")
    device = next(model.parameters()).device

    def loop(model_inner: Any) -> None:
        model_inner.eval()
        with torch.no_grad():
            for record in records[: int(args.calib_samples)]:
                prompt, _ = prompt_and_reference(record)
                prompt = apply_prompt_format(tokenizer, prompt, args.prompt_format, args.system_prompt)
                encoded = tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=int(args.opt_seq_len),
                ).to(device)
                strip_unused_decoder_model_kwargs(encoded)
                model_inner(**encoded, use_cache=False)

    return loop


def attempt_int4_modelopt(args: argparse.Namespace) -> Path:
    if not args.model_path:
        logging.error("--model-path is required for INT4 weight-only quantization because ONNX alone is insufficient.")
        raise RuntimeError(INT4_UNSUPPORTED_MESSAGE)
    if not args.calib_json:
        logging.error("--calib_json is required for ModelOpt/AWQ-style INT4 calibration.")
        raise RuntimeError(INT4_UNSUPPORTED_MESSAGE)

    try:
        torch = import_required("torch", "ModelOpt INT4 quantization")
        transformers = import_required("transformers", "ModelOpt INT4 Hugging Face model loading")
        mtq = import_required("modelopt.torch.quantization", "ModelOpt INT4 weight-only quantization")
        mte = import_required("modelopt.torch.export", "ModelOpt TensorRT-LLM export")
    except RuntimeError as exc:
        logging.error("%s", exc)
        raise RuntimeError(INT4_UNSUPPORTED_MESSAGE) from exc

    config = None
    for name in ("INT4_AWQ_CFG", "INT4_WEIGHT_ONLY_CFG", "W4A16_AWQ_BETA_CFG"):
        config = getattr(mtq, name, None)
        if config is not None:
            logging.info("Using ModelOpt quantization config: %s", name)
            break
    if config is None or not hasattr(mtq, "quantize"):
        logging.error("Installed modelopt.torch.quantization does not expose a known INT4 AWQ/weight-only config.")
        raise RuntimeError(INT4_UNSUPPORTED_MESSAGE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    tokenizer = prepare_decoder_tokenizer(
        transformers.AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=bool(args.trust_remote_code))
    )
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        trust_remote_code=bool(args.trust_remote_code),
    ).to(device)
    model.eval()
    records = read_records(args.calib_json, limit=int(args.calib_samples))
    forward_loop = run_forward_loop_for_modelopt(model, tokenizer, records, args)
    logging.info("Running ModelOpt INT4 quantization calibration.")
    model = mtq.quantize(model, config, forward_loop)

    artifact_dir = Path(args.output_dir).expanduser() / "model_int4_modelopt"
    if artifact_dir.exists() and args.overwrite:
        shutil.rmtree(artifact_dir)
    elif artifact_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing INT4 artifact directory: {artifact_dir}")
    artifact_dir.mkdir(parents=True, exist_ok=True)

    export_fn = getattr(mte, "export_tensorrt_llm_checkpoint", None)
    if export_fn is None:
        logging.error("ModelOpt export_tensorrt_llm_checkpoint is unavailable in this installation.")
        raise RuntimeError(INT4_UNSUPPORTED_MESSAGE)
    logging.info("Exporting ModelOpt INT4 TensorRT-LLM checkpoint artifact to %s", artifact_dir)
    export_errors: list[str] = []
    export_attempts = [
        lambda: export_fn(model, "gpt", dtype=dtype, export_dir=str(artifact_dir)),
        lambda: export_fn(model, model_type="gpt", dtype=dtype, export_dir=str(artifact_dir)),
        lambda: export_fn(model, export_dir=str(artifact_dir), dtype=dtype),
        lambda: export_fn(model, str(artifact_dir)),
    ]
    signature = None
    try:
        signature = str(inspect.signature(export_fn))
    except Exception:
        signature = "unavailable"
    for attempt in export_attempts:
        try:
            attempt()
            break
        except TypeError as exc:
            export_errors.append(str(exc))
    else:
        logging.error("Could not call ModelOpt export_tensorrt_llm_checkpoint. Signature: %s Errors: %s", signature, export_errors)
        raise RuntimeError(INT4_UNSUPPORTED_MESSAGE)

    trtllm_build = shutil.which(args.trtllm_build_cmd)
    if not trtllm_build:
        logging.error("Exported INT4 artifact, but '%s' is not installed to build a TensorRT-LLM engine.", args.trtllm_build_cmd)
        raise RuntimeError(INT4_UNSUPPORTED_MESSAGE)

    engine_dir = Path(args.output_dir).expanduser() / "model_int4_trtllm_engine"
    if engine_dir.exists() and args.overwrite:
        shutil.rmtree(engine_dir)
    engine_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        trtllm_build,
        "--checkpoint_dir",
        str(artifact_dir),
        "--output_dir",
        str(engine_dir),
        "--max_batch_size",
        str(args.batch_size),
        "--max_input_len",
        str(args.max_seq_len),
        "--max_seq_len",
        str(args.max_seq_len + args.max_new_tokens),
    ]
    logging.info("Running TensorRT-LLM build: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)
    candidates = sorted(engine_dir.glob("*.engine"))
    if not candidates:
        logging.error("TensorRT-LLM build finished but did not produce a .engine file in %s.", engine_dir)
        raise RuntimeError(INT4_UNSUPPORTED_MESSAGE)
    output_engine = Path(args.output_dir).expanduser() / "model_int4.engine"
    ensure_output_path(output_engine, overwrite=args.overwrite, kind="TensorRT INT4 engine")
    shutil.copyfile(candidates[0], output_engine)
    logging.info("Wrote INT4 TensorRT engine: %s (%.2f MB)", output_engine, output_engine.stat().st_size / 1024**2)
    return output_engine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build TensorRT engines from decoder ONNX exports.")
    parser.add_argument("--onnx", default="outputs/onnx/model_decoder_nocache.onnx", help="Path to decoder ONNX file.")
    parser.add_argument("--output-dir", default="outputs/trt", help="Directory for TensorRT engines and calibration cache.")
    parser.add_argument("--precision", required=True, choices=("fp16", "int8", "int4"))
    parser.add_argument("--model-path", default=None, help="HF model/tokenizer path, required for INT8 tokenization and INT4 ModelOpt.")
    parser.add_argument("--tokenizer-path", default=None, help="Optional tokenizer path for INT8 calibration.")
    parser.add_argument("--calib_json", "--calibration-json", dest="calib_json", default=None, help="Representative JSON/JSONL/CSV/TXT prompts.")
    parser.add_argument("--calib-samples", type=int, default=128)
    parser.add_argument("--calib-seq-len", type=int, default=None)
    parser.add_argument("--min_seq_len", type=int, default=1)
    parser.add_argument("--opt_seq_len", type=int, default=64)
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--workspace-gb", type=float, default=4.0)
    parser.add_argument(
        "--sparse-weights",
        action="store_true",
        help="Enable TensorRT SPARSE_WEIGHTS. Use with validated NVIDIA 2:4 pruned checkpoints.",
    )
    parser.add_argument("--prompt-format", choices=("raw", "legacy", "chat-template"), default="raw")
    parser.add_argument("--system-prompt", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=64, help="INT4 TensorRT-LLM max generation budget.")
    parser.add_argument("--trtllm-build-cmd", default="trtllm-build")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)
    if args.batch_size != 1:
        logging.warning("Edge profile defaults to batch_size=1; requested batch_size=%d.", args.batch_size)
    if args.precision == "int4":
        try:
            attempt_int4_modelopt(args)
        except Exception as exc:
            logging.error("%s", exc)
            print(INT4_UNSUPPORTED_MESSAGE)
            raise SystemExit(2) from exc
        return
    try:
        build_engine(args)
    except Exception:
        if args.precision == "int8":
            logging.error("INT8 build failed. Check calibration data, tokenizer path, input names, and TensorRT parser logs.")
        raise


if __name__ == "__main__":
    main()
