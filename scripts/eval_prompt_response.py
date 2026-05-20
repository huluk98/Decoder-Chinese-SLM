#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from chatlm_decoder.sft_data import EOS_TOKEN, normalize_sft_record

JSON_LIST_KEYS = ("data", "records", "items", "examples", "eval", "validation", "test")
CHAT_MARKERS = (
    "<|user|>",
    "<|assistant|>",
    "<|system|>",
    "<|eos|>",
    "assistant:",
    "Assistant:",
    "ASSISTANT:",
    "回答:",
    "回答：",
    "答案:",
    "答案：",
    "输出:",
    "输出：",
)
ZERO_WIDTH_PATTERN = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060\ufeff]")
FENCE_PATTERN = re.compile(r"^\s*```(?:json|python|text|txt|bash|sh)?\s*|\s*```\s*$", re.IGNORECASE)
TRAILING_STOP_PATTERN = re.compile(r"[\s。．.；;，,]+$")
STRUCTURED_SPACE_PATTERN = re.compile(r"\s*([(),:=\[\]{}])\s*")
QUOTE_TRANSLATION = str.maketrans(
    {
        "“": '"',
        "”": '"',
        "„": '"',
        "‘": "'",
        "’": "'",
        "‚": "'",
        "（": "(",
        "）": ")",
        "，": ",",
        "：": ":",
        "；": ";",
        "［": "[",
        "］": "]",
        "｛": "{",
        "｝": "}",
        "＝": "=",
    }
)


def select_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def setup_distributed_eval(requested_device: str) -> tuple[torch.device, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("Multi-GPU eval requires CUDA.")
        torch.cuda.set_device(local_rank)
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        return torch.device("cuda", local_rank), rank, local_rank, world_size
    return select_device(requested_device), rank, local_rank, world_size


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


def coerce_records(payload: Any, path: Path) -> list[dict[str, Any]]:
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


def read_prompt_response_records(path: str | Path) -> list[dict[str, Any]]:
    data_path = Path(path).expanduser()
    suffix = data_path.suffix.lower()
    if suffix == ".json":
        records = coerce_records(json.loads(data_path.read_text(encoding="utf-8")), data_path)
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
    else:
        raise ValueError(f"Unsupported eval file extension: {suffix}. Use .json, .jsonl, or .csv.")

    for index, record in enumerate(records):
        try:
            normalize_sft_record(record)
        except ValueError as exc:
            raise ValueError(
                f"{data_path}: record {index} must normalize into prompt/response text. "
                "Use fields like instruction+response, prompt+response, question+answer, "
                f"or messages/conversations. Original error: {exc}"
            ) from exc
    return records


def tokenize_example(tokenizer: Any, record: dict[str, Any], max_length: int) -> dict[str, Any]:
    prompt_text, response_text = normalize_sft_record(record)
    full_text = prompt_text + response_text
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False, truncation=True, max_length=max_length)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False, truncation=True, max_length=max_length)["input_ids"]
    labels = list(full_ids)
    prompt_len = min(len(prompt_ids), len(labels))
    labels[:prompt_len] = [-100] * prompt_len
    return {
        "record": record,
        "prompt_text": prompt_text,
        "response_text": response_text,
        "full_text": full_text,
        "input_ids": [int(token_id) for token_id in full_ids],
        "labels": [int(token_id) for token_id in labels],
    }


def pad_batch(examples: list[dict[str, Any]], pad_token_id: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    max_len = max(len(example["input_ids"]) for example in examples)
    input_ids = torch.full((len(examples), max_len), int(pad_token_id), dtype=torch.long)
    attention_mask = torch.zeros((len(examples), max_len), dtype=torch.long)
    labels = torch.full((len(examples), max_len), -100, dtype=torch.long)
    for row, example in enumerate(examples):
        ids = torch.tensor(example["input_ids"], dtype=torch.long)
        row_labels = torch.tensor(example["labels"], dtype=torch.long)
        input_ids[row, : ids.numel()] = ids
        attention_mask[row, : ids.numel()] = 1
        labels[row, : row_labels.numel()] = row_labels
    return input_ids, attention_mask, labels


@torch.no_grad()
def score_batch(
    model: AutoModelForCausalLM,
    examples: list[dict[str, Any]],
    pad_token_id: int,
    device: torch.device,
) -> list[dict[str, float | int]]:
    input_ids, attention_mask, labels = pad_batch(examples, pad_token_id=pad_token_id)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    labels = labels.to(device)
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    results: list[dict[str, float | int]] = []
    for row in range(shift_labels.shape[0]):
        row_labels = shift_labels[row]
        token_count = int((row_labels != -100).sum().detach().cpu())
        if token_count == 0:
            results.append({"loss": math.nan, "token_count": 0, "loss_sum": 0.0})
            continue
        loss_sum = F.cross_entropy(
            shift_logits[row].view(-1, shift_logits.shape[-1]).float(),
            row_labels.view(-1),
            ignore_index=-100,
            reduction="sum",
        )
        loss_value = float((loss_sum / token_count).detach().cpu())
        results.append(
            {
                "loss": loss_value,
                "token_count": token_count,
                "loss_sum": float(loss_sum.detach().cpu()),
            }
        )
    return results


@torch.no_grad()
def generate_completion(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt_text: str,
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    num_beams: int,
) -> tuple[str, int]:
    inputs = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).to(device)
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "num_beams": int(num_beams),
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if temperature > 0:
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = top_p
    output_ids = model.generate(**inputs, **generation_kwargs)
    completion_ids = output_ids[0][int(inputs["input_ids"].shape[-1]) :]
    completion = tokenizer.decode(completion_ids, skip_special_tokens=False).strip()
    return completion, int(completion_ids.numel())


def normalize_for_exact_match(text: str, tokenizer: Any | None = None) -> str:
    text = unicodedata.normalize("NFKC", str(text))
    text = text.translate(QUOTE_TRANSLATION)
    text = ZERO_WIDTH_PATTERN.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    for special_token in (
        EOS_TOKEN,
        getattr(tokenizer, "bos_token", None),
        getattr(tokenizer, "eos_token", None),
        getattr(tokenizer, "pad_token", None),
        getattr(tokenizer, "unk_token", None),
    ):
        if special_token:
            text = text.replace(str(special_token), "")
    text = FENCE_PATTERN.sub("", text).strip()
    for marker in CHAT_MARKERS:
        if text.startswith(marker):
            text = text[len(marker) :].strip()
    for marker in ("<|user|>", "<|system|>", "\nUser:", "\n用户:"):
        if marker in text:
            text = text.split(marker, 1)[0].strip()
    text = strip_wrapping_quotes(text)
    text = TRAILING_STOP_PATTERN.sub("", text)
    return " ".join(text.split()).strip()


def strip_wrapping_quotes(text: str) -> str:
    text = text.strip()
    changed = True
    while changed and len(text) >= 2:
        changed = False
        pairs = (("`", "`"), ('"', '"'), ("'", "'"))
        for left, right in pairs:
            if text.startswith(left) and text.endswith(right):
                text = text[1:-1].strip()
                changed = True
    return text


def canonicalize_command_response(text: str) -> str:
    text = normalize_for_exact_match(text)
    text = STRUCTURED_SPACE_PATTERN.sub(r"\1", text)
    text = text.replace('"', "'")
    return text.strip()


def mismatch_reason(generated: str, target: str) -> str:
    if generated == target:
        return "match"
    if generated.lower() == target.lower():
        return "case_only_difference"
    if generated.replace(" ", "") == target.replace(" ", ""):
        return "whitespace_only_difference"
    if generated in target or target in generated:
        return "one_side_contains_the_other"
    return "different_text"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a local SFT prompt/response dataset.")
    parser.add_argument("--model-path", "--checkpoint", dest="checkpoint", required=True)
    parser.add_argument(
        "--dataset-file",
        required=True,
        help="Local .json, .jsonl, or .csv file with instruction+response, prompt+response, question+answer, or messages.",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default="auto", help="auto, cuda, cuda:0, cpu, or mps.")
    parser.add_argument("--dtype", default="bf16", choices=("auto", "bf16", "fp16", "fp32"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--exact-match",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate every row and report normalized exact-match accuracy. Use --no-exact-match for loss-only eval.",
    )
    parser.add_argument("--generate-samples", type=int, default=0, help="Generate completions for the first N rows.")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument(
        "--comparison-mode",
        choices=("normalized", "command"),
        default="command",
        help="normalized compares cleaned text; command additionally canonicalizes punctuation, quote style, and spaces around command syntax.",
    )
    args = parser.parse_args()

    device, rank, local_rank, world_size = setup_distributed_eval(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=False)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint,
        torch_dtype=dtype_for(args.dtype, device),
        trust_remote_code=False,
    ).to(device)
    model.eval()

    max_length = args.max_length
    if max_length is None:
        config_max = getattr(model.config, "max_position_embeddings", None)
        tokenizer_max = getattr(tokenizer, "model_max_length", None)
        max_length = int(config_max or tokenizer_max or 2048)

    dataset_path = Path(args.dataset_file).expanduser()
    records = read_prompt_response_records(dataset_path)
    if args.limit is not None:
        records = records[: int(args.limit)]
    if not records:
        raise ValueError(f"No eval records found in {dataset_path}.")
    examples = [tokenize_example(tokenizer, record, max_length=max_length) for record in records]

    output_dir = Path(args.output_dir or Path("runs") / "eval" / f"{dataset_path.stem}_prompt_response").expanduser()
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"Prompt/response eval runtime: world_size={world_size} "
            f"rank={rank} local_rank={local_rank} device={device} "
            f"total_examples={len(records)} shard_examples={(len(records) + world_size - 1 - rank) // world_size}"
        )
    if world_size > 1:
        dist.barrier()

    total_loss_sum = 0.0
    total_tokens = 0
    exact_correct = 0
    generated_token_lengths: list[int] = []
    results: list[dict[str, Any]] = []
    batch_size = max(1, int(args.batch_size))
    indexed_examples = [(index, example) for index, example in enumerate(examples) if index % world_size == rank]
    progress = tqdm(
        range(0, len(indexed_examples), batch_size),
        desc=f"prompt-response-eval-rank{rank}",
        disable=(rank != 0),
    )
    for start in progress:
        batch_pairs = indexed_examples[start : start + batch_size]
        batch = [example for _, example in batch_pairs]
        scores = score_batch(model=model, examples=batch, pad_token_id=int(pad_token_id), device=device)
        for offset, score in enumerate(scores):
            index = batch_pairs[offset][0]
            token_count = int(score["token_count"])
            loss_sum = float(score["loss_sum"])
            total_tokens += token_count
            total_loss_sum += loss_sum
            result = {
                "index": index,
                "prompt": batch[offset]["prompt_text"],
                "response": batch[offset]["response_text"],
                "loss": score["loss"],
                "perplexity": math.exp(float(score["loss"])) if token_count > 0 else math.nan,
                "response_token_count": token_count,
            }
            if bool(args.exact_match) or index < int(args.generate_samples):
                generated, generated_token_count = generate_completion(
                    model=model,
                    tokenizer=tokenizer,
                    prompt_text=batch[offset]["prompt_text"],
                    device=device,
                    max_new_tokens=int(args.max_new_tokens),
                    temperature=float(args.temperature),
                    top_p=float(args.top_p),
                    num_beams=int(args.num_beams),
                )
                normalized_generated = normalize_for_exact_match(generated, tokenizer=tokenizer)
                normalized_target = normalize_for_exact_match(batch[offset]["response_text"], tokenizer=tokenizer)
                if args.comparison_mode == "command":
                    comparison_generated = canonicalize_command_response(normalized_generated)
                    comparison_target = canonicalize_command_response(normalized_target)
                else:
                    comparison_generated = normalized_generated
                    comparison_target = normalized_target
                result["generated"] = generated
                result["normalized_generated"] = normalized_generated
                result["normalized_target"] = normalized_target
                result["comparison_generated"] = comparison_generated
                result["comparison_target"] = comparison_target
                result["comparison_mode"] = args.comparison_mode
                result["generated_token_count"] = generated_token_count
                if bool(args.exact_match):
                    is_exact = comparison_generated == comparison_target
                    exact_correct += int(is_exact)
                    generated_token_lengths.append(generated_token_count)
                    result["exact_match"] = is_exact
                    result["mismatch_reason"] = mismatch_reason(comparison_generated, comparison_target)
            results.append(result)

    local_payload = {
        "total_loss_sum": total_loss_sum,
        "total_tokens": total_tokens,
        "exact_correct": exact_correct,
        "generated_token_lengths": generated_token_lengths,
        "results": results,
    }
    if world_size > 1:
        gathered: list[dict[str, Any] | None] | None = [None for _ in range(world_size)] if rank == 0 else None
        dist.gather_object(local_payload, gathered, dst=0)
    else:
        gathered = [local_payload]

    if rank != 0:
        if world_size > 1:
            dist.barrier()
            dist.destroy_process_group()
        return

    payloads = [payload for payload in gathered or [] if payload is not None]
    total_loss_sum = sum(float(payload["total_loss_sum"]) for payload in payloads)
    total_tokens = sum(int(payload["total_tokens"]) for payload in payloads)
    exact_correct = sum(int(payload["exact_correct"]) for payload in payloads)
    generated_token_lengths = [
        int(length)
        for payload in payloads
        for length in payload["generated_token_lengths"]
    ]
    results = sorted(
        [result for payload in payloads for result in payload["results"]],
        key=lambda row: int(row["index"]),
    )

    mean_loss = total_loss_sum / total_tokens if total_tokens else math.nan
    avg_generated_tokens = (
        sum(generated_token_lengths) / len(generated_token_lengths) if generated_token_lengths else math.nan
    )
    summary = {
        "checkpoint": args.checkpoint,
        "dataset_file": str(dataset_path),
        "total_examples": len(records),
        "response_tokens": total_tokens,
        "mean_response_loss": mean_loss,
        "response_perplexity": math.exp(mean_loss) if total_tokens else math.nan,
        "exact_match_accuracy": exact_correct / len(records) if args.exact_match and records else None,
        "exact_match_correct": exact_correct if args.exact_match else None,
        "avg_generated_tokens": avg_generated_tokens if args.exact_match else None,
        "max_new_tokens": int(args.max_new_tokens),
        "comparison_mode": args.comparison_mode if args.exact_match else None,
        "max_length": max_length,
        "batch_size": batch_size,
    }

    with (output_dir / "prompt_response_eval_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    with (output_dir / "prompt_response_eval_predictions.jsonl").open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(
        f"Prompt/response eval loss={summary['mean_response_loss']:.4f} "
        f"ppl={summary['response_perplexity']:.4f} "
        f"examples={summary['total_examples']} tokens={summary['response_tokens']}"
    )
    if args.exact_match:
        print(
            f"Exact-match accuracy={summary['exact_match_accuracy']:.4f} "
            f"({summary['exact_match_correct']}/{summary['total_examples']}) "
            f"avg_generated_tokens={summary['avg_generated_tokens']:.2f}"
        )
        if summary["avg_generated_tokens"] is not None and summary["avg_generated_tokens"] >= 0.9 * int(args.max_new_tokens):
            print("[warning] Average generated length is close to max_new_tokens; the model may not be stopping cleanly.")
    print(f"Wrote results to {output_dir}")
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
