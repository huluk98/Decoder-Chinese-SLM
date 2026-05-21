#!/usr/bin/env python
from __future__ import annotations

import argparse
import collections
import inspect
import logging
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[0]
sys.path.insert(0, str(SCRIPT_DIR))

from trt_edge_common import ensure_output_path, import_required, setup_logging


class NoCacheDecoderWrapper:
    def __init__(self, torch: Any, model: Any) -> None:
        self.torch = torch
        self.module = self._build(torch, model)

    @staticmethod
    def _build(torch: Any, model: Any) -> Any:
        class _Wrapper(torch.nn.Module):
            def __init__(self, inner: Any) -> None:
                super().__init__()
                self.inner = inner

            def forward(self, input_ids: Any, attention_mask: Any) -> Any:
                outputs = self.inner(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    return_dict=True,
                )
                return outputs.logits

        return _Wrapper(model)


class CacheDecoderWrapper:
    def __init__(self, torch: Any, model: Any, layer_value_counts: list[int]) -> None:
        self.torch = torch
        self.module = self._build(torch, model, layer_value_counts)

    @staticmethod
    def _build(torch: Any, model: Any, layer_value_counts: list[int]) -> Any:
        class _Wrapper(torch.nn.Module):
            def __init__(self, inner: Any, counts: list[int]) -> None:
                super().__init__()
                self.inner = inner
                self.counts = counts

            def forward(self, input_ids: Any, attention_mask: Any, *flat_past: Any) -> tuple[Any, ...]:
                offset = 0
                past_key_values = []
                for count in self.counts:
                    past_key_values.append(tuple(flat_past[offset : offset + count]))
                    offset += count
                outputs = self.inner(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    past_key_values=tuple(past_key_values),
                    use_cache=True,
                    return_dict=True,
                )
                flat_present = []
                for layer in outputs.past_key_values:
                    flat_present.extend(list(layer))
                return (outputs.logits, *flat_present)

        return _Wrapper(model, layer_value_counts)


def select_device(torch: Any, requested: str) -> Any:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def dtype_for(torch: Any, name: str, device: Any) -> Any:
    if name == "auto":
        return "auto"
    if name == "fp16":
        return torch.float16 if device.type == "cuda" else torch.float32
    if name == "bf16":
        return torch.bfloat16 if device.type == "cuda" else torch.float32
    if name == "fp32":
        return torch.float32
    raise ValueError(f"Unknown dtype: {name}")


def configure_tokenizer(tokenizer: Any) -> None:
    if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token_id", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token


def dummy_token_ids(torch: Any, tokenizer: Any, batch_size: int, seq_len: int, device: Any) -> Any:
    vocab_size = int(getattr(tokenizer, "vocab_size", 32000) or 32000)
    token_id = int(getattr(tokenizer, "bos_token_id", None) or getattr(tokenizer, "eos_token_id", None) or 1)
    token_id = max(0, min(token_id, max(vocab_size - 1, 0)))
    return torch.full((int(batch_size), int(seq_len)), token_id, dtype=torch.int32, device=device)


def export_with_compatible_kwargs(torch: Any, wrapper: Any, args: tuple[Any, ...], output_path: Path, kwargs: dict[str, Any]) -> None:
    signature = inspect.signature(torch.onnx.export)
    export_kwargs = dict(kwargs)
    if "external_data" in signature.parameters:
        export_kwargs["external_data"] = True
    elif "use_external_data_format" in signature.parameters:
        export_kwargs["use_external_data_format"] = True
    unsupported = [name for name in ("dynamo",) if name in export_kwargs and name not in signature.parameters]
    for name in unsupported:
        export_kwargs.pop(name, None)
    torch.onnx.export(wrapper, args, str(output_path), **export_kwargs)


def shape_from_value_info(value: Any) -> list[str | int]:
    tensor_type = value.type.tensor_type
    if not tensor_type.HasField("shape"):
        return []
    dims: list[str | int] = []
    for dim in tensor_type.shape.dim:
        if dim.dim_param:
            dims.append(dim.dim_param)
        elif dim.HasField("dim_value"):
            dims.append(int(dim.dim_value))
        else:
            dims.append("?")
    return dims


def summarize_onnx(path: Path) -> None:
    onnx = import_required("onnx", "ONNX graph validation and summary")
    logging.info("Running ONNX checker for %s", path)
    onnx.checker.check_model(str(path))
    model = onnx.load(str(path), load_external_data=False)
    dtype_name = onnx.TensorProto.DataType.Name
    logging.info("ONNX opsets: %s", {item.domain or "ai.onnx": int(item.version) for item in model.opset_import})

    logging.info("Inputs:")
    for value in model.graph.input:
        elem_type = value.type.tensor_type.elem_type
        logging.info("  %s dtype=%s shape=%s", value.name, dtype_name(elem_type), shape_from_value_info(value))

    logging.info("Outputs:")
    for value in model.graph.output:
        elem_type = value.type.tensor_type.elem_type
        logging.info("  %s dtype=%s shape=%s", value.name, dtype_name(elem_type), shape_from_value_info(value))

    counts = collections.Counter(node.op_type for node in model.graph.node)
    top_ops = ", ".join(f"{name}:{count}" for name, count in counts.most_common(30))
    logging.info("Operator summary: %s", top_ops)


def maybe_simplify(path: Path) -> None:
    onnx = import_required("onnx", "ONNX simplification")
    onnxsim = import_required("onnxsim", "optional graph simplification requested by --simplify")
    logging.info("Simplifying ONNX graph with onnxsim: %s", path)
    model = onnx.load(str(path))
    simplified, ok = onnxsim.simplify(model)
    if not ok:
        raise RuntimeError("onnxsim.simplify reported validation failure; leaving original graph untouched.")
    onnx.save_model(simplified, str(path))
    summarize_onnx(path)


def sequence_axis_for_cache_tensor(tensor: Any, seq_len: int) -> int:
    shape = list(tensor.shape)
    for axis, dim in enumerate(shape):
        if axis > 0 and int(dim) == int(seq_len):
            return axis
    return 2 if len(shape) >= 3 else max(0, len(shape) - 1)


def export_nocache(args: argparse.Namespace, torch: Any, model: Any, tokenizer: Any, device: Any) -> Path:
    output_path = Path(args.onnx_dir).expanduser() / "model_decoder_nocache.onnx"
    ensure_output_path(output_path, overwrite=args.overwrite, kind="ONNX file")
    wrapper = NoCacheDecoderWrapper(torch, model).module.eval()
    input_ids = dummy_token_ids(torch, tokenizer, args.batch_size, args.seq_len, device)
    attention_mask = torch.ones((args.batch_size, args.seq_len), dtype=torch.int32, device=device)

    with torch.no_grad():
        logits = wrapper(input_ids, attention_mask)
    logging.info("No-cache dry run logits shape: %s", tuple(logits.shape))

    dynamic_axes = {
        "input_ids": {0: "batch", 1: "seq_len"},
        "attention_mask": {0: "batch", 1: "seq_len"},
        "logits": {0: "batch", 1: "seq_len"},
    }
    logging.info("Exporting no-cache decoder ONNX to %s", output_path)
    export_with_compatible_kwargs(
        torch,
        wrapper,
        (input_ids, attention_mask),
        output_path,
        {
            "input_names": ["input_ids", "attention_mask"],
            "output_names": ["logits"],
            "dynamic_axes": dynamic_axes,
            "opset_version": int(args.opset),
            "do_constant_folding": True,
            "dynamo": bool(args.dynamo),
        },
    )
    summarize_onnx(output_path)
    if args.simplify:
        maybe_simplify(output_path)
    return output_path


def flatten_past(past_key_values: Any, past_seq_len: int) -> tuple[list[Any], list[int], list[str], dict[str, dict[int, str]]]:
    flat: list[Any] = []
    counts: list[int] = []
    names: list[str] = []
    dynamic_axes: dict[str, dict[int, str]] = {}
    for layer_index, layer in enumerate(past_key_values):
        layer_values = list(layer)
        counts.append(len(layer_values))
        for value_index, tensor in enumerate(layer_values):
            suffix = "key" if value_index == 0 else "value" if value_index == 1 else f"value_{value_index}"
            name = f"past_key_values.{layer_index}.{suffix}"
            names.append(name)
            flat.append(tensor)
            axis = sequence_axis_for_cache_tensor(tensor, int(past_seq_len))
            dynamic_axes[name] = {0: "batch", axis: "past_seq_len"}
    return flat, counts, names, dynamic_axes


def export_cache(args: argparse.Namespace, torch: Any, model: Any, tokenizer: Any, device: Any) -> Path | None:
    output_path = Path(args.onnx_dir).expanduser() / "model_decoder_cache.onnx"
    ensure_output_path(output_path, overwrite=args.overwrite, kind="ONNX file")

    past_seq_len = max(1, int(args.past_seq_len))
    prefill_ids = dummy_token_ids(torch, tokenizer, args.batch_size, past_seq_len, device)
    prefill_mask = torch.ones((args.batch_size, past_seq_len), dtype=torch.int32, device=device)
    logging.info("Attempting cached export by first inferring past_key_values from a %d-token prefill.", past_seq_len)
    with torch.no_grad():
        prefill_outputs = model(
            input_ids=prefill_ids,
            attention_mask=prefill_mask,
            use_cache=True,
            return_dict=True,
        )
    if not getattr(prefill_outputs, "past_key_values", None):
        raise RuntimeError("Model did not return past_key_values; cached export is not available for this architecture.")

    flat_past, layer_value_counts, past_names, past_dynamic_axes = flatten_past(
        prefill_outputs.past_key_values,
        past_seq_len=past_seq_len,
    )
    input_ids = dummy_token_ids(torch, tokenizer, args.batch_size, 1, device)
    attention_mask = torch.ones((args.batch_size, past_seq_len + 1), dtype=torch.int32, device=device)
    wrapper = CacheDecoderWrapper(torch, model, layer_value_counts).module.eval()
    with torch.no_grad():
        outputs = wrapper(input_ids, attention_mask, *flat_past)
    logging.info("Cached dry run returned %d tensors; logits shape=%s", len(outputs), tuple(outputs[0].shape))

    present_names: list[str] = []
    output_dynamic_axes: dict[str, dict[int, str]] = {"logits": {0: "batch", 1: "decode_seq_len"}}
    present_index = 0
    for layer_index, count in enumerate(layer_value_counts):
        for value_index in range(count):
            suffix = "key" if value_index == 0 else "value" if value_index == 1 else f"value_{value_index}"
            name = f"present_key_values.{layer_index}.{suffix}"
            present_names.append(name)
            tensor = outputs[1 + present_index]
            axis = sequence_axis_for_cache_tensor(tensor, past_seq_len + 1)
            output_dynamic_axes[name] = {0: "batch", axis: "total_seq_len"}
            present_index += 1

    dynamic_axes = {
        "input_ids": {0: "batch", 1: "decode_seq_len"},
        "attention_mask": {0: "batch", 1: "total_seq_len"},
        **past_dynamic_axes,
        **output_dynamic_axes,
    }
    logging.info("Exporting cached decoder ONNX to %s", output_path)
    export_with_compatible_kwargs(
        torch,
        wrapper,
        (input_ids, attention_mask, *flat_past),
        output_path,
        {
            "input_names": ["input_ids", "attention_mask", *past_names],
            "output_names": ["logits", *present_names],
            "dynamic_axes": dynamic_axes,
            "opset_version": int(args.opset),
            "do_constant_folding": True,
            "dynamo": bool(args.dynamo),
        },
    )
    summarize_onnx(output_path)
    if args.simplify:
        maybe_simplify(output_path)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a Hugging Face decoder-only CausalLM to TensorRT-oriented ONNX.")
    parser.add_argument("--model-path", required=True, help="Local Hugging Face model/checkpoint directory.")
    parser.add_argument("--onnx-dir", default="outputs/onnx", help="Directory for exported ONNX files.")
    parser.add_argument("--opset", type=int, default=18, help="ONNX opset. TensorRT 10 generally supports opset 18+.")
    parser.add_argument("--seq-len", type=int, default=64, help="Dummy sequence length for no-cache export.")
    parser.add_argument("--past-seq-len", type=int, default=8, help="Dummy past length for cached decode export.")
    parser.add_argument("--batch-size", type=int, default=1, help="Edge deployment batch size; defaults to 1.")
    parser.add_argument("--device", default="auto", help="auto, cuda, cuda:0, or cpu.")
    parser.add_argument("--dtype", default="auto", choices=("auto", "fp16", "bf16", "fp32"))
    parser.add_argument("--trust-remote-code", action="store_true", help="Allow custom HF model/tokenizer code.")
    parser.add_argument("--export-cache", action=argparse.BooleanOptionalAction, default=True, help="Attempt cached decode ONNX export.")
    parser.add_argument("--dynamo", action="store_true", help="Use torch.onnx.export(..., dynamo=True) when supported.")
    parser.add_argument("--simplify", action="store_true", help="Optionally run onnxsim after checker validation.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing ONNX files.")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)
    if int(args.opset) < 18:
        raise ValueError("Use --opset 18 or newer unless you have confirmed your local TensorRT parser requires older.")

    torch = import_required("torch", "ONNX export")
    transformers = import_required("transformers", "loading Hugging Face AutoModelForCausalLM and AutoTokenizer")
    AutoModelForCausalLM = transformers.AutoModelForCausalLM
    AutoTokenizer = transformers.AutoTokenizer

    device = select_device(torch, args.device)
    dtype = dtype_for(torch, args.dtype, device)
    logging.info("Loading tokenizer from %s", args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=bool(args.trust_remote_code))
    configure_tokenizer(tokenizer)
    logging.info("Loading model from %s on %s dtype=%s", args.model_path, device, dtype)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        trust_remote_code=bool(args.trust_remote_code),
    ).to(device)
    model.eval()

    Path(args.onnx_dir).expanduser().mkdir(parents=True, exist_ok=True)
    nocache_path = export_nocache(args, torch, model, tokenizer, device)
    logging.info("No-cache ONNX export complete: %s", nocache_path)

    if args.export_cache:
        try:
            cache_path = export_cache(args, torch, model, tokenizer, device)
            logging.info("Cached ONNX export complete: %s", cache_path)
        except Exception as exc:
            logging.warning(
                "Cached ONNX export failed. The no-cache engine remains usable but slower for autoregressive decode. "
                "Reason: %s",
                exc,
            )


if __name__ == "__main__":
    main()
