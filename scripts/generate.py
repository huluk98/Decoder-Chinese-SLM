#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


DEFAULT_TEXT_FIELDS = ("prompt", "instruction", "question", "input", "text", "query")


def select_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def dtype_for(name: str, device: torch.device) -> torch.dtype | str:
    if name == "auto":
        return "auto"
    if name == "bf16":
        return torch.bfloat16 if device.type == "cuda" else torch.float32
    if name == "fp16":
        return torch.float16 if device.type == "cuda" else torch.float32
    if name == "fp32":
        return torch.float32
    raise ValueError(f"Unknown dtype: {name}")


def read_dataset(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        records = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError(f"{path}:{line_number} must be a JSON object.")
                records.append(record)
        return records
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            for key in ("data", "records", "items", "examples"):
                if isinstance(payload.get(key), list):
                    payload = payload[key]
                    break
        if not isinstance(payload, list):
            raise ValueError(f"{path} must contain a JSON list, or an object with data/records/items/examples.")
        if not all(isinstance(item, dict) for item in payload):
            raise ValueError(f"{path} must contain JSON objects.")
        return list(payload)
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    if suffix == ".txt":
        with path.open("r", encoding="utf-8") as handle:
            return [{"id": idx, "text": line.strip()} for idx, line in enumerate(handle) if line.strip()]
    raise ValueError(f"Unsupported dataset file extension: {suffix}. Use .jsonl, .json, .csv, or .txt.")


def prompt_from_record(record: dict[str, Any], text_field: str | None) -> str:
    fields = (text_field,) if text_field else DEFAULT_TEXT_FIELDS
    for field in fields:
        value = record.get(field)
        if value is not None and str(value).strip():
            return str(value).strip()
    available = ", ".join(record.keys())
    raise KeyError(f"Could not find prompt text. Use --text-field. Available fields: {available}")


def format_prompt(prompt: str, chat_format: bool) -> str:
    if not chat_format:
        return prompt
    return f"<|user|>\n{prompt}\n<|assistant|>\n"


def generate_text(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: torch.device,
    prompt: str,
    chat_format: bool,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    formatted_prompt = format_prompt(prompt, chat_format=chat_format)
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if temperature > 0:
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = top_p
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            **generation_kwargs,
        )
    input_length = int(inputs["input_ids"].shape[-1])
    completion_ids = output_ids[0][input_length:]
    return tokenizer.decode(completion_ids, skip_special_tokens=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text from a trained checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint directory, for example runs/h20-8gpu-llama-0p2b-deepspeed/latest.")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--prompt", default=None, help="Single prompt text.")
    input_group.add_argument("--dataset-file", default=None, help="Local .jsonl, .json, .csv, or .txt file of prompts.")
    parser.add_argument("--text-field", default=None, help="Prompt field name for JSON/JSONL/CSV files. Defaults to prompt/instruction/question/input/text/query.")
    parser.add_argument("--id-field", default=None, help="Optional id field to copy into batch generation outputs.")
    parser.add_argument("--output-file", default=None, help="Where to write JSONL batch generations. Defaults to runs/local_generations.jsonl for --dataset-file.")
    parser.add_argument("--device", default="auto", help="auto, cuda, cuda:0, cpu, or mps.")
    parser.add_argument("--dtype", default="auto", choices=("auto", "bf16", "fp16", "fp32"))
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--no-chat-format", action="store_true", help="Do not wrap prompts in <|user|>/<|assistant|> tokens.")
    args = parser.parse_args()

    device = select_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.checkpoint, torch_dtype=dtype_for(args.dtype, device)).to(device)
    model.eval()

    chat_format = not args.no_chat_format
    if args.prompt is not None:
        completion = generate_text(
            model=model,
            tokenizer=tokenizer,
            device=device,
            prompt=args.prompt,
            chat_format=chat_format,
            max_new_tokens=int(args.max_new_tokens),
            temperature=float(args.temperature),
            top_p=float(args.top_p),
        )
        print(completion)
        return

    dataset_path = Path(args.dataset_file).expanduser()
    records = read_dataset(dataset_path)
    output_path = Path(args.output_file or "runs/local_generations.jsonl").expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for idx, record in enumerate(records):
            prompt = prompt_from_record(record, args.text_field)
            completion = generate_text(
                model=model,
                tokenizer=tokenizer,
                device=device,
                prompt=prompt,
                chat_format=chat_format,
                max_new_tokens=int(args.max_new_tokens),
                temperature=float(args.temperature),
                top_p=float(args.top_p),
            )
            result = {
                "index": idx,
                "prompt": prompt,
                "completion": completion,
            }
            if args.id_field and args.id_field in record:
                result["id"] = record[args.id_field]
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            print(f"[{idx + 1}/{len(records)}] {completion[:120]}")
    print(f"Wrote generations to {output_path}")


if __name__ == "__main__":
    main()
