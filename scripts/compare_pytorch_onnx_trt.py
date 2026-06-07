#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
PROJECT_ROOT = SCRIPT_DIR.parents[0]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from chatlm_decoder.tokenizer import move_batch_to_device, prepare_decoder_tokenizer

from trt_edge_common import (
    TensorRTEngineRunner,
    apply_prompt_format,
    ensure_output_path,
    expand_path,
    import_required,
    normalize_exact,
    setup_logging,
)


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


def load_tokenizer(args: argparse.Namespace) -> Any:
    transformers = import_required("transformers", "comparison tokenizer loading")
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=bool(args.trust_remote_code))
    return prepare_decoder_tokenizer(tokenizer)


def output_logits(outputs: dict[str, Any]) -> Any:
    if "logits" in outputs:
        return outputs["logits"]
    for name, value in outputs.items():
        if "logits" in name.lower():
            return value
    return next(iter(outputs.values()))


def encode_np(tokenizer: Any, prompt: str, max_seq_len: int, add_special_tokens: bool) -> dict[str, Any]:
    np = import_required("numpy", "comparison input arrays")
    encoded = tokenizer(
        prompt,
        return_tensors="np",
        truncation=True,
        max_length=int(max_seq_len),
        add_special_tokens=bool(add_special_tokens),
    )
    return {
        "input_ids": np.ascontiguousarray(encoded["input_ids"].astype(np.int32)),
        "attention_mask": np.ascontiguousarray(encoded["attention_mask"].astype(np.int32)),
    }


def run_pytorch(args: argparse.Namespace, prompt: str, tokenizer: Any) -> tuple[Any, Any, str]:
    torch = import_required("torch", "PyTorch comparison baseline")
    transformers = import_required("transformers", "PyTorch comparison model/tokenizer loading")
    device = select_device(torch, args.device)
    dtype = dtype_for(torch, args.dtype, device)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        trust_remote_code=bool(args.trust_remote_code),
    ).to(device)
    model.eval()

    encoded = move_batch_to_device(tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=int(args.max_seq_len),
        add_special_tokens=bool(args.add_special_tokens),
    ), device)
    with torch.no_grad():
        outputs = model(**encoded, use_cache=False, return_dict=True)
        generated_ids = model.generate(
            **encoded,
            max_new_tokens=int(args.max_new_tokens),
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    input_len = int(encoded["input_ids"].shape[-1])
    decoded = tokenizer.decode(generated_ids[0][input_len:], skip_special_tokens=True).strip()
    return outputs.logits.detach().float().cpu().numpy(), encoded, decoded


def run_onnx(args: argparse.Namespace, inputs: dict[str, Any]) -> tuple[Any | None, str | None]:
    if not args.onnx:
        return None, None
    try:
        ort = import_required("onnxruntime", "ONNX Runtime comparison")
    except RuntimeError as exc:
        return None, str(exc)
    onnx_path = expand_path(args.onnx)
    if not onnx_path.exists():
        return None, f"ONNX file does not exist: {onnx_path}"
    session = ort.InferenceSession(str(onnx_path), providers=args.ort_providers.split(","))
    feed = {}
    for item in session.get_inputs():
        if item.name in inputs:
            feed[item.name] = inputs[item.name]
    try:
        outputs = session.run(None, feed)
    except Exception as exc:
        return None, f"ONNX Runtime inference failed with int32 inputs: {exc}"
    return outputs[0], None


def greedy_generate_onnx(
    args: argparse.Namespace,
    tokenizer: Any,
    prompt: str,
) -> tuple[str | None, str | None]:
    if not args.onnx:
        return None, None
    try:
        np = import_required("numpy", "ONNX Runtime generation arrays")
        ort = import_required("onnxruntime", "ONNX Runtime generation")
    except RuntimeError as exc:
        return None, str(exc)
    onnx_path = expand_path(args.onnx)
    if not onnx_path.exists():
        return None, f"ONNX file does not exist: {onnx_path}"
    session = ort.InferenceSession(str(onnx_path), providers=args.ort_providers.split(","))
    input_names = {item.name for item in session.get_inputs()}
    encoded = encode_np(tokenizer, prompt, int(args.max_seq_len), bool(args.add_special_tokens))
    context = encoded["input_ids"][0].astype("int64").tolist()
    generated: list[int] = []
    for _ in range(int(args.max_new_tokens)):
        window = context[-int(args.max_seq_len) :]
        input_ids = np.asarray([window], dtype=np.int32)
        attention_mask = np.ones_like(input_ids, dtype=np.int32)
        feed = {}
        if "input_ids" in input_names:
            feed["input_ids"] = input_ids
        if "attention_mask" in input_names:
            feed["attention_mask"] = attention_mask
        try:
            outputs = session.run(None, feed)
        except Exception as exc:
            return None, f"ONNX Runtime generation failed with int32 inputs: {exc}"
        next_token = int(np.argmax(outputs[0][0, -1, :]))
        generated.append(next_token)
        context.append(next_token)
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        if eos_token_id is not None and next_token == int(eos_token_id):
            break
    return tokenizer.decode(generated, skip_special_tokens=True).strip(), None


def run_trt(args: argparse.Namespace, inputs: dict[str, Any]) -> tuple[Any | None, str | None]:
    if not args.engine:
        return None, None
    engine_path = expand_path(args.engine)
    if not engine_path.exists():
        return None, f"TensorRT engine does not exist: {engine_path}"
    try:
        runner = TensorRTEngineRunner(engine_path, verbose=bool(args.verbose))
        outputs = runner.infer(inputs)
        logits = output_logits(outputs)
        runner.close()
        return logits, None
    except Exception as exc:
        return None, f"TensorRT inference failed: {exc}"


def greedy_generate_with_runner(
    runner: TensorRTEngineRunner,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int,
    max_seq_len: int,
    add_special_tokens: bool,
) -> str:
    np = import_required("numpy", "TensorRT generation arrays")
    encoded = encode_np(tokenizer, prompt, max_seq_len, add_special_tokens)
    context = encoded["input_ids"][0].astype("int64").tolist()
    generated: list[int] = []
    for _ in range(int(max_new_tokens)):
        window = context[-int(max_seq_len) :]
        input_ids = np.asarray([window], dtype=np.int32)
        attention_mask = np.ones_like(input_ids, dtype=np.int32)
        outputs = runner.infer({"input_ids": input_ids, "attention_mask": attention_mask})
        next_token = int(np.argmax(output_logits(outputs)[0, -1, :]))
        generated.append(next_token)
        context.append(next_token)
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        if eos_token_id is not None and next_token == int(eos_token_id):
            break
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def compare_logits(label: str, baseline: Any, candidate: Any | None, error: str | None) -> dict[str, Any]:
    np = import_required("numpy", "logits comparison")
    if error:
        return {"name": label, "available": False, "error": error}
    if candidate is None:
        return {"name": label, "available": False, "error": "not requested"}
    result: dict[str, Any] = {
        "name": label,
        "available": True,
        "logits_shape": list(candidate.shape),
        "baseline_shape": list(baseline.shape),
    }
    if tuple(candidate.shape) == tuple(baseline.shape):
        diff = np.abs(candidate.astype(np.float32) - baseline.astype(np.float32))
        result["max_abs_diff"] = float(diff.max()) if diff.size else 0.0
        base_argmax = int(np.argmax(baseline[0, -1, :]))
        cand_argmax = int(np.argmax(candidate[0, -1, :]))
        result["baseline_argmax_token"] = base_argmax
        result["candidate_argmax_token"] = cand_argmax
        result["argmax_agreement"] = base_argmax == cand_argmax
    else:
        result["max_abs_diff"] = None
        result["argmax_agreement"] = False
        result["error"] = "shape mismatch"
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare PyTorch, ONNX Runtime, and TensorRT decoder logits/generation.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--onnx", default="outputs/onnx/model_decoder_nocache.onnx")
    parser.add_argument("--engine", default="outputs/trt/model_fp16.engine")
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--prompt-format", choices=("raw", "legacy", "chat-template"), default="raw")
    parser.add_argument("--system-prompt", default=None)
    parser.add_argument("--add-special-tokens", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto", choices=("auto", "fp16", "bf16", "fp32"))
    parser.add_argument("--ort-providers", default="CPUExecutionProvider")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)
    raw_prompt = args.prompt
    tokenizer = load_tokenizer(args)
    prompt = apply_prompt_format(tokenizer, raw_prompt, args.prompt_format, args.system_prompt)
    torch_logits, _encoded_pt, pytorch_text = run_pytorch(args, prompt, tokenizer)

    np_inputs = encode_np(tokenizer, prompt, int(args.max_seq_len), bool(args.add_special_tokens))
    onnx_logits, onnx_error = run_onnx(args, np_inputs)
    trt_logits, trt_error = run_trt(args, np_inputs)
    onnx_text, onnx_text_error = greedy_generate_onnx(args, tokenizer, prompt)
    comparisons = [
        compare_logits("onnxruntime", torch_logits, onnx_logits, onnx_error),
        compare_logits("tensorrt", torch_logits, trt_logits, trt_error),
    ]

    trt_text = None
    if args.engine and not trt_error:
        runner = TensorRTEngineRunner(args.engine, verbose=bool(args.verbose))
        try:
            trt_text = greedy_generate_with_runner(
                runner,
                tokenizer,
                prompt,
                max_new_tokens=int(args.max_new_tokens),
                max_seq_len=int(args.max_seq_len),
                add_special_tokens=bool(args.add_special_tokens),
            )
        finally:
            runner.close()

    report = {
        "prompt": raw_prompt,
        "formatted_prompt": prompt,
        "pytorch": {
            "logits_shape": list(torch_logits.shape),
            "generated_text": pytorch_text,
            "normalized_generated_text": normalize_exact(pytorch_text, tokenizer),
        },
        "onnxruntime_generated_text": onnx_text,
        "onnxruntime_generation_error": onnx_text_error,
        "onnxruntime_normalized_generated_text": normalize_exact(onnx_text, tokenizer) if onnx_text is not None else None,
        "comparisons": comparisons,
        "tensorrt_generated_text": trt_text,
        "tensorrt_normalized_generated_text": normalize_exact(trt_text, tokenizer) if trt_text is not None else None,
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.output_json:
        output_path = Path(args.output_json).expanduser()
        ensure_output_path(output_path, overwrite=bool(args.overwrite), kind="comparison JSON")
        output_path.write_text(text + "\n", encoding="utf-8")
        logging.info("Wrote comparison report: %s", output_path)


if __name__ == "__main__":
    main()
