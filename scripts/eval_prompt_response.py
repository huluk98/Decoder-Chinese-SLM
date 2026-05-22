#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import random
import re
import shutil
import statistics
import subprocess
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    import numpy as np
except Exception:  # pragma: no cover - numpy is optional here.
    np = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from chatlm_decoder.command_eval import canonicalize_command_response
from chatlm_decoder.sft_data import EOS_TOKEN, normalize_sft_record

JSON_LIST_KEYS = ("data", "records", "items", "examples", "eval", "validation", "test")
ANCHOR_ID_FIELDS = ("anchor_id", "semantic_anchor_id", "group_id", "cluster_id", "intent_id", "anchor")
RECORD_ID_FIELDS = ("id", "uid", "uuid", "example_id", "sample_id", "index")
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


def git_commit_hash() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def set_eval_seeds(seed: int, deterministic: bool) -> dict[str, Any]:
    seed = int(seed)
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)
    return {
        "python_random_seed": seed,
        "numpy_seed": seed if np is not None else None,
        "torch_cpu_seed": seed,
        "torch_cuda_seed": seed if torch.cuda.is_available() else None,
        "deterministic_algorithms_enabled": bool(torch.are_deterministic_algorithms_enabled()),
    }


def safe_slug(value: str, max_length: int = 80) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-._")
    return (slug or "run")[:max_length]


def stable_short_hash(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:10]


def checkpoint_step_name(path: str | Path) -> str:
    checkpoint_path = Path(path).expanduser()
    for candidate in (checkpoint_path, checkpoint_path.resolve() if checkpoint_path.exists() else checkpoint_path):
        match = re.search(r"(?:^|/)(step-[0-9]+|epoch-[0-9]+|latest)(?:$|/)", str(candidate))
        if match:
            return match.group(1)
    return "checkpoint"


def checkpoint_file_timestamps(path: str | Path) -> dict[str, Any]:
    checkpoint_path = Path(path).expanduser()
    if not checkpoint_path.exists():
        return {"exists": False, "path": str(checkpoint_path)}
    files: dict[str, Any] = {}
    candidates = [
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "model.safetensors",
        "pytorch_model.bin",
        "trainer_state.pt",
        "run_config.json",
        "checkpoint_manifest.json",
    ]
    for name in candidates:
        candidate = checkpoint_path / name
        if candidate.exists():
            stat = candidate.stat()
            files[name] = {
                "size_bytes": int(stat.st_size),
                "mtime": dt.datetime.fromtimestamp(stat.st_mtime, tz=dt.timezone.utc).isoformat(),
            }
    return {
        "exists": True,
        "path": str(checkpoint_path),
        "resolved_path": str(checkpoint_path.resolve()),
        "files": files,
    }


def parameter_count(model: torch.nn.Module) -> int:
    return sum(int(parameter.numel()) for parameter in model.parameters())


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


def first_nonempty_field(record: dict[str, Any], fields: tuple[str, ...]) -> str:
    for field in fields:
        value = record.get(field)
        if value is not None and str(value).strip():
            return normalize_whitespace_exact(str(value))
    return ""


def record_id(record: dict[str, Any], fallback: int) -> str:
    explicit = first_nonempty_field(record, RECORD_ID_FIELDS)
    return explicit or str(fallback)


def split_record_signature(record: dict[str, Any], index: int) -> dict[str, str]:
    prompt, response = normalize_sft_record(record)
    prompt_norm = normalize_whitespace_exact(prompt)
    response_norm = normalize_whitespace_exact(response)
    pair_norm = f"{prompt_norm}\u241f{response_norm}"
    return {
        "id": record_id(record, index),
        "prompt": prompt_norm,
        "response": response_norm,
        "pair": pair_norm,
        "anchor_id": first_nonempty_field(record, ANCHOR_ID_FIELDS),
    }


def duplicate_count(values: list[str]) -> int:
    seen: set[str] = set()
    duplicates = 0
    for value in values:
        if value in seen:
            duplicates += 1
        else:
            seen.add(value)
    return duplicates


def split_stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    signatures = [split_record_signature(record, index) for index, record in enumerate(records)]
    prompts = [signature["prompt"] for signature in signatures]
    responses = [signature["response"] for signature in signatures]
    pairs = [signature["pair"] for signature in signatures]
    anchors = [signature["anchor_id"] for signature in signatures if signature["anchor_id"]]
    return {
        "num_samples": len(records),
        "unique_prompts": len(set(prompts)),
        "unique_responses": len(set(responses)),
        "unique_prompt_response_pairs": len(set(pairs)),
        "unique_anchor_ids": len(set(anchors)),
        "duplicate_prompt_count": duplicate_count(prompts),
        "duplicate_response_count": duplicate_count(responses),
        "duplicate_prompt_response_pair_count": duplicate_count(pairs),
        "duplicate_anchor_id_count": duplicate_count(anchors),
        "_sets": {
            "prompts": set(prompts),
            "responses": set(responses),
            "pairs": set(pairs),
            "anchors": set(anchors),
        },
    }


def sample_values(values: set[str], limit: int = 20) -> list[str]:
    return sorted(values)[:limit]


def audit_splits(
    dataset_file: Path,
    train_file: str | None = None,
    validation_file: str | None = None,
    test_file: str | None = None,
) -> dict[str, Any]:
    split_paths: dict[str, Path] = {"test": Path(test_file).expanduser() if test_file else dataset_file}
    if train_file:
        split_paths["train"] = Path(train_file).expanduser()
    if validation_file:
        split_paths["validation"] = Path(validation_file).expanduser()

    split_records: dict[str, list[dict[str, Any]]] = {}
    for split, path in split_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"{split} split file does not exist: {path}")
        split_records[split] = read_prompt_response_records(path)

    stats = {split: split_stats(records) for split, records in split_records.items()}
    overlaps: dict[str, Any] = {}
    suspicious: list[dict[str, Any]] = []
    split_names = sorted(stats)
    for left_index, left in enumerate(split_names):
        for right in split_names[left_index + 1 :]:
            left_sets = stats[left]["_sets"]
            right_sets = stats[right]["_sets"]
            pair_key = f"{left}_vs_{right}"
            overlap = {
                "prompt_overlap_count": len(left_sets["prompts"] & right_sets["prompts"]),
                "response_overlap_count": len(left_sets["responses"] & right_sets["responses"]),
                "prompt_response_pair_overlap_count": len(left_sets["pairs"] & right_sets["pairs"]),
                "anchor_id_overlap_count": len(left_sets["anchors"] & right_sets["anchors"]),
                "prompt_overlap_examples": sample_values(left_sets["prompts"] & right_sets["prompts"]),
                "anchor_id_overlap_examples": sample_values(left_sets["anchors"] & right_sets["anchors"]),
            }
            overlaps[pair_key] = overlap
            for field in ("prompt_overlap_count", "prompt_response_pair_overlap_count", "anchor_id_overlap_count"):
                if overlap[field] > 0:
                    suspicious.append({"split_pair": pair_key, "issue": field, "count": overlap[field]})

    public_stats = {}
    for split, split_stat in stats.items():
        public_stats[split] = {key: value for key, value in split_stat.items() if key != "_sets"}
    return {
        "split_paths": {split: str(path) for split, path in split_paths.items()},
        "splits": public_stats,
        "overlaps": overlaps,
        "suspicious": suspicious,
        "has_suspicious_leakage": bool(suspicious),
        "notes": [
            "Response overlap is logged but not treated as leakage because this task is many-to-one.",
            "Prompt, prompt-response pair, or anchor_id overlap across train/validation/test is treated as suspicious.",
        ],
    }


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


def label_token_audit(examples: list[dict[str, Any]], tokenizer: Any) -> dict[str, Any]:
    labels = [token_id for example in examples for token_id in example["labels"]]
    valid_labels = [token_id for token_id in labels if token_id != -100]
    return {
        "total_label_positions": len(labels),
        "ignored_label_positions": sum(1 for token_id in labels if token_id == -100),
        "supervised_label_positions": len(valid_labels),
        "label_min": min(labels) if labels else None,
        "label_max": max(labels) if labels else None,
        "supervised_label_min": min(valid_labels) if valid_labels else None,
        "supervised_label_max": max(valid_labels) if valid_labels else None,
        "labels_less_than_minus_100": sum(1 for token_id in labels if token_id < -100),
        "supervised_contains_pad_token": (
            int(tokenizer.pad_token_id) in valid_labels if tokenizer.pad_token_id is not None else False
        ),
        "supervised_contains_eos_token": (
            int(tokenizer.eos_token_id) in valid_labels if tokenizer.eos_token_id is not None else False
        ),
        "supervised_contains_bos_token": (
            int(tokenizer.bos_token_id) in valid_labels if getattr(tokenizer, "bos_token_id", None) is not None else False
        ),
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
    do_sample: bool,
    temperature: float,
    top_p: float,
    top_k: int,
    num_beams: int,
    repetition_penalty: float,
) -> tuple[str, int]:
    inputs = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).to(device)
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": bool(do_sample),
        "num_beams": int(num_beams),
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "repetition_penalty": float(repetition_penalty),
    }
    if do_sample:
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = top_p
        generation_kwargs["top_k"] = int(top_k)
    output_ids = model.generate(**inputs, **generation_kwargs)
    completion_ids = output_ids[0][int(inputs["input_ids"].shape[-1]) :]
    completion = tokenizer.decode(completion_ids, skip_special_tokens=False).strip()
    return completion, int(completion_ids.numel())


@torch.no_grad()
def generate_completions_batch(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt_texts: list[str],
    device: torch.device,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    top_k: int,
    num_beams: int,
    repetition_penalty: float,
) -> list[tuple[str, int]]:
    if not prompt_texts:
        return []
    previous_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        inputs = tokenizer(prompt_texts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
    finally:
        tokenizer.padding_side = previous_padding_side
    prompt_width = int(inputs["input_ids"].shape[-1])
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": bool(do_sample),
        "num_beams": int(num_beams),
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "repetition_penalty": float(repetition_penalty),
    }
    if do_sample:
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = top_p
        generation_kwargs["top_k"] = int(top_k)
    output_ids = model.generate(**inputs, **generation_kwargs)
    completions: list[tuple[str, int]] = []
    for row in range(int(output_ids.shape[0])):
        completion_ids = output_ids[row, prompt_width:]
        completion = tokenizer.decode(completion_ids, skip_special_tokens=False).strip()
        completions.append((completion, int(completion_ids.numel())))
    return completions


def normalize_whitespace_exact(text: str, tokenizer: Any | None = None) -> str:
    text = unicodedata.normalize("NFKC", str(text))
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
    return " ".join(text.split()).strip()


def normalize_for_exact_match(text: str, tokenizer: Any | None = None) -> str:
    text = normalize_whitespace_exact(text, tokenizer=tokenizer)
    text = text.translate(QUOTE_TRANSLATION)
    text = TRAILING_STOP_PATTERN.sub("", text)
    return " ".join(text.split()).strip()


def comparison_text(text: str, tokenizer: Any | None, comparison_mode: str) -> str:
    if comparison_mode == "whitespace":
        return normalize_whitespace_exact(text, tokenizer=tokenizer)
    normalized = normalize_for_exact_match(text, tokenizer=tokenizer)
    if comparison_mode == "command":
        return canonicalize_command_response(normalized)
    return normalized


def invalid_structured_output(text: str, comparison: str) -> bool:
    if not comparison:
        return True
    return any(marker in str(text) for marker in ("<|user|>", "<|system|>"))


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


def mean_std(values: list[float]) -> tuple[float, float]:
    finite_values = [float(value) for value in values if math.isfinite(float(value))]
    if not finite_values:
        return math.nan, math.nan
    if len(finite_values) == 1:
        return finite_values[0], 0.0
    return statistics.mean(finite_values), statistics.stdev(finite_values)


def benchmark_metric_summary(run_summaries: list[dict[str, Any]], key: str) -> dict[str, float | str]:
    values = [
        float(summary[key])
        for summary in run_summaries
        if summary.get(key) is not None and math.isfinite(float(summary[key]))
    ]
    mean_value, std_value = mean_std(values)
    return {
        f"{key}_mean": mean_value,
        f"{key}_std": std_value,
        f"{key}_mean_pm_std": f"{mean_value:.6f} ± {std_value:.6f}",
    }


def resolve_output_dirs(args: argparse.Namespace, dataset_path: Path) -> tuple[Path, Path]:
    output_root = Path(args.output_dir or Path("runs") / "eval" / f"{dataset_path.stem}_prompt_response").expanduser()
    if not args.unique_output_dir:
        return output_root, output_root
    checkpoint_slug = safe_slug(Path(args.checkpoint).name or stable_short_hash(args.checkpoint), max_length=40)
    dataset_slug = safe_slug(dataset_path.stem, max_length=40)
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    step = checkpoint_step_name(args.checkpoint)
    run_name = f"{dataset_slug}_{checkpoint_slug}_{step}_seed{int(args.seed)}_{timestamp}"
    return output_root, output_root / run_name


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_prediction_files(output_dir: Path, results: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "prompt_response_eval_summary.json", summary)
    write_json(output_dir / "metrics.json", summary)
    with (output_dir / "prompt_response_eval_predictions.jsonl").open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
    with (output_dir / "prediction_debug.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "id",
            "prompt",
            "raw_prediction",
            "normalized_prediction",
            "raw_label",
            "normalized_label",
            "exact_match",
            "generated_length",
            "label_length",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "id": result.get("id", result.get("index")),
                    "prompt": result.get("prompt", ""),
                    "raw_prediction": result.get("raw_prediction", ""),
                    "normalized_prediction": result.get("normalized_prediction", ""),
                    "raw_label": result.get("raw_label", result.get("response", "")),
                    "normalized_label": result.get("normalized_label", ""),
                    "exact_match": result.get("exact_match", ""),
                    "generated_length": result.get("generated_length", result.get("generated_token_count", "")),
                    "label_length": result.get("label_length", result.get("response_token_count", "")),
                }
            )


def write_eval_run_config(
    output_dir: Path,
    args: argparse.Namespace,
    model: torch.nn.Module,
    tokenizer: Any,
    dataset_path: Path,
    split_audit: dict[str, Any],
    label_audit: dict[str, Any],
    seed_info: dict[str, Any],
    world_size: int,
    local_rank: int,
    device: torch.device,
    max_length: int,
) -> None:
    payload = {
        "script": "scripts/eval_prompt_response.py",
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git_commit": git_commit_hash(),
        "model_path": args.checkpoint,
        "checkpoint_path_used_for_evaluation": args.checkpoint,
        "tokenizer_path": args.checkpoint,
        "checkpoint_files": checkpoint_file_timestamps(args.checkpoint),
        "dataset_file": str(dataset_path),
        "train_file": args.train_file,
        "validation_file": args.validation_file,
        "test_file": args.test_file or str(dataset_path),
        "output_dir": str(output_dir),
        "model_class": model.__class__.__name__,
        "model_type": getattr(model.config, "model_type", None),
        "parameter_count": parameter_count(model),
        "world_size": int(world_size),
        "local_rank": int(local_rank),
        "device": str(device),
        "seed_info": seed_info,
        "max_length": int(max_length),
        "generation": {
            "do_sample": bool(args.do_sample),
            "num_beams": int(args.num_beams),
            "temperature": None if not args.do_sample else float(args.temperature),
            "top_p": None if not args.do_sample else float(args.top_p),
            "top_k": None if not args.do_sample else int(args.top_k),
            "max_new_tokens": int(args.max_new_tokens),
            "repetition_penalty": float(args.repetition_penalty),
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
        },
        "eval_args": vars(args),
        "split_audit_summary": {
            "has_suspicious_leakage": split_audit.get("has_suspicious_leakage"),
            "suspicious": split_audit.get("suspicious", []),
        },
        "label_audit": label_audit,
        "tokenizer": {
            "class": tokenizer.__class__.__name__,
            "vocab_size": len(tokenizer),
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        },
    }
    write_json(output_dir / "run_config.json", payload)


def mirror_summary_to_root(output_root: Path, output_dir: Path, summary: dict[str, Any], split_audit: dict[str, Any]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "latest_eval_dir.txt").write_text(str(output_dir) + "\n", encoding="utf-8")
    write_json(output_root / "prompt_response_eval_summary.json", summary)
    write_json(output_root / "metrics.json", summary)
    write_json(output_root / "split_audit.json", split_audit)


def load_cached_prediction_map(path: str | Path) -> dict[int, str]:
    cache_path = Path(path).expanduser()
    if not cache_path.exists():
        raise FileNotFoundError(f"Cached predictions file does not exist: {cache_path}")
    predictions: dict[int, str] = {}
    if cache_path.suffix.lower() == ".csv":
        with cache_path.open("r", encoding="utf-8", newline="") as handle:
            for row_index, row in enumerate(csv.DictReader(handle)):
                index = int(row.get("index") or row.get("id") or row_index)
                predictions[index] = str(
                    row.get("raw_prediction")
                    or row.get("generated")
                    or row.get("prediction")
                    or row.get("normalized_prediction")
                    or ""
                )
    else:
        with cache_path.open("r", encoding="utf-8") as handle:
            for row_index, line in enumerate(handle):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"{cache_path}:{row_index + 1} must contain a JSON object.")
                index = int(row.get("index", row_index))
                predictions[index] = str(
                    row.get("raw_prediction")
                    or row.get("generated")
                    or row.get("prediction")
                    or row.get("normalized_prediction")
                    or ""
                )
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a local SFT prompt/response dataset.")
    parser.add_argument("--model-path", "--checkpoint", dest="checkpoint", required=True)
    parser.add_argument(
        "--dataset-file",
        required=True,
        help="Local .json, .jsonl, or .csv file with instruction+response, prompt+response, question+answer, or messages.",
    )
    parser.add_argument("--train-file", default=None, help="Optional train split file for leakage/split audit.")
    parser.add_argument("--validation-file", default=None, help="Optional validation split file for leakage/split audit.")
    parser.add_argument("--test-file", default=None, help="Optional explicit test split file. Defaults to --dataset-file.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--unique-output-dir",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Create a timestamped child directory under --output-dir. Parent summary files are still updated for launch-script compatibility.",
    )
    parser.add_argument("--device", default="auto", help="auto, cuda, cuda:0, cpu, or mps.")
    parser.add_argument("--dtype", default="bf16", choices=("auto", "bf16", "fp16", "fp32"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-seed", "--data_seed", type=int, default=None)
    parser.add_argument(
        "--deterministic-eval",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Seed Python/NumPy/PyTorch and request deterministic algorithms where available.",
    )
    parser.add_argument(
        "--fail-on-leakage",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail if train/validation/test share exact prompts, prompt-response pairs, or anchor ids.",
    )
    parser.add_argument(
        "--use-cached-predictions",
        action="store_true",
        help="Explicitly allow scoring an existing prediction file instead of generating fresh predictions.",
    )
    parser.add_argument("--cached-predictions-file", default=None)
    parser.add_argument(
        "--exact-match",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate every row and report normalized exact-match accuracy. Use --no-exact-match for loss-only eval.",
    )
    parser.add_argument("--generate-samples", type=int, default=0, help="Generate completions for the first N rows.")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--do-sample", action="store_true", help="Enable sampling. Default exact-match eval is greedy.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument(
        "--benchmark-runs",
        type=int,
        default=1,
        help="Repeat full generation eval N times and report mean ± sample std across runs.",
    )
    parser.add_argument(
        "--comparison-mode",
        choices=("whitespace", "normalized", "command"),
        default="whitespace",
        help="whitespace is the main strict metric; normalized strips light formatting; command canonicalizes smart-home wording variants.",
    )
    args = parser.parse_args()

    if args.use_cached_predictions and not args.cached_predictions_file:
        raise ValueError("--use-cached-predictions requires --cached-predictions-file.")
    if not args.use_cached_predictions and args.cached_predictions_file:
        raise ValueError("--cached-predictions-file was provided without --use-cached-predictions.")

    seed_info = set_eval_seeds(args.seed, deterministic=bool(args.deterministic_eval))
    device, rank, local_rank, world_size = setup_distributed_eval(args.device)
    checkpoint_path = Path(args.checkpoint).expanduser()
    local_checkpoint_like = (
        checkpoint_path.is_absolute()
        or args.checkpoint.startswith(("./", "../"))
        or ("/" in args.checkpoint and checkpoint_path.parent.exists())
    )
    if local_checkpoint_like and not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint path does not exist: {checkpoint_path}. "
            "If SFT training just finished, point --model-path at the run's final checkpoint directory."
        )
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
    label_audit = label_token_audit(examples, tokenizer)
    if int(label_audit["labels_less_than_minus_100"]) > 0:
        raise ValueError(f"Tokenized labels contain values less than -100: {label_audit}")
    cached_predictions = load_cached_prediction_map(args.cached_predictions_file) if args.use_cached_predictions else {}
    if args.use_cached_predictions:
        missing_cached = [index for index in range(len(records)) if index not in cached_predictions]
        if missing_cached:
            raise ValueError(
                f"Cached predictions are missing {len(missing_cached)} eval rows. "
                f"First missing indices: {missing_cached[:20]}"
            )

    output_root, output_dir = resolve_output_dirs(args, dataset_path)
    if (
        not args.unique_output_dir
        and not args.use_cached_predictions
        and (output_dir / "prompt_response_eval_predictions.jsonl").exists()
    ):
        raise FileExistsError(
            f"Refusing to overwrite existing predictions without a unique run directory: {output_dir}. "
            "Use the default --unique-output-dir behavior, choose a fresh --output-dir, or pass "
            "--use-cached-predictions with --cached-predictions-file intentionally."
        )
    split_audit = audit_splits(
        dataset_file=Path(args.test_file).expanduser() if args.test_file else dataset_path,
        train_file=args.train_file,
        validation_file=args.validation_file,
        test_file=args.test_file,
    )
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(output_dir / "split_audit.json", split_audit)
        write_eval_run_config(
            output_dir=output_dir,
            args=args,
            model=model,
            tokenizer=tokenizer,
            dataset_path=dataset_path,
            split_audit=split_audit,
            label_audit=label_audit,
            seed_info={**seed_info, "data_seed": args.data_seed},
            world_size=world_size,
            local_rank=local_rank,
            device=device,
            max_length=max_length,
        )
    if args.fail_on_leakage and split_audit.get("has_suspicious_leakage"):
        raise RuntimeError(
            "Split audit found suspicious leakage. See split_audit.json. "
            f"Issues: {split_audit.get('suspicious')}"
        )

    benchmark_runs = max(1, int(args.benchmark_runs))
    if rank == 0:
        print(
            f"Prompt/response eval runtime: world_size={world_size} "
            f"rank={rank} local_rank={local_rank} device={device} "
            f"total_examples={len(records)} shard_examples={(len(records) + world_size - 1 - rank) // world_size} "
            f"benchmark_runs={benchmark_runs}\n"
            f"  model_path={args.checkpoint}\n"
            f"  checkpoint_path_used_for_evaluation={args.checkpoint}\n"
            f"  tokenizer_path={args.checkpoint}\n"
            f"  model_class={model.__class__.__name__} model_type={getattr(model.config, 'model_type', None)}\n"
            f"  parameter_count={parameter_count(model):,}\n"
            f"  git_commit={git_commit_hash()}\n"
            f"  seed={args.seed} data_seed={args.data_seed} deterministic={seed_info['deterministic_algorithms_enabled']}\n"
            f"  comparison_mode={args.comparison_mode}\n"
            f"  use_cached_predictions={bool(args.use_cached_predictions)}\n"
            f"  output_dir={output_dir}"
        )
        print(
            "Generation settings: "
            f"do_sample={bool(args.do_sample)} num_beams={int(args.num_beams)} "
            f"temperature={None if not args.do_sample else float(args.temperature)} "
            f"top_p={None if not args.do_sample else float(args.top_p)} "
            f"top_k={None if not args.do_sample else int(args.top_k)} "
            f"max_new_tokens={int(args.max_new_tokens)} repetition_penalty={float(args.repetition_penalty)} "
            f"eos_token_id={tokenizer.eos_token_id} pad_token_id={tokenizer.pad_token_id}"
        )
        print(
            "Label audit: "
            f"ignored={label_audit['ignored_label_positions']} "
            f"supervised={label_audit['supervised_label_positions']} "
            f"valid_min/max={label_audit['supervised_label_min']}/{label_audit['supervised_label_max']} "
            f"contains_pad={label_audit['supervised_contains_pad_token']} "
            f"contains_eos={label_audit['supervised_contains_eos_token']} "
            f"less_than_-100={label_audit['labels_less_than_minus_100']}"
        )
    if world_size > 1:
        dist.barrier()

    batch_size = max(1, int(args.batch_size))
    indexed_examples = [(index, example) for index, example in enumerate(examples) if index % world_size == rank]
    run_summaries: list[dict[str, Any]] = []

    for run_index in range(benchmark_runs):
        run_start = time.perf_counter()
        run_output_dir = output_dir if benchmark_runs == 1 else output_dir / f"run_{run_index + 1:02d}"
        if rank == 0:
            source = f"cached predictions from {args.cached_predictions_file}" if args.use_cached_predictions else "fresh generation"
            print(
                f"[run {run_index + 1}/{benchmark_runs}] Prediction source: {source}; writing: "
                f"{run_output_dir / 'prompt_response_eval_predictions.jsonl'}"
            )
        total_loss_sum = 0.0
        total_tokens = 0
        exact_correct = 0
        generated_token_lengths: list[int] = []
        results: list[dict[str, Any]] = []
        progress = tqdm(
            range(0, len(indexed_examples), batch_size),
            desc=f"prompt-response-eval-run{run_index + 1:02d}-rank{rank}",
            disable=(rank != 0),
        )
        for start in progress:
            batch_pairs = indexed_examples[start : start + batch_size]
            batch = [example for _, example in batch_pairs]
            scores = score_batch(model=model, examples=batch, pad_token_id=int(pad_token_id), device=device)
            generation_by_index: dict[int, tuple[str, int, str]] = {}
            needs_generation = [
                (offset, index, batch[offset])
                for offset, (index, _example) in enumerate(batch_pairs)
                if bool(args.exact_match) or index < int(args.generate_samples)
            ]
            if needs_generation:
                if args.use_cached_predictions:
                    for _offset, index, _example in needs_generation:
                        generated = cached_predictions[index]
                        generated_token_count = len(tokenizer(generated, add_special_tokens=False)["input_ids"])
                        generation_by_index[index] = (generated, generated_token_count, "")
                else:
                    try:
                        generated_batch = generate_completions_batch(
                            model=model,
                            tokenizer=tokenizer,
                            prompt_texts=[example["prompt_text"] for _offset, _index, example in needs_generation],
                            device=device,
                            max_new_tokens=int(args.max_new_tokens),
                            do_sample=bool(args.do_sample),
                            temperature=float(args.temperature),
                            top_p=float(args.top_p),
                            top_k=int(args.top_k),
                            num_beams=int(args.num_beams),
                            repetition_penalty=float(args.repetition_penalty),
                        )
                        for (_offset, index, _example), (generated, generated_token_count) in zip(
                            needs_generation, generated_batch
                        ):
                            generation_by_index[index] = (generated, generated_token_count, "")
                    except Exception as exc:
                        for _offset, index, _example in needs_generation:
                            generation_by_index[index] = ("", 0, repr(exc))
            for offset, score in enumerate(scores):
                index = batch_pairs[offset][0]
                token_count = int(score["token_count"])
                loss_sum = float(score["loss_sum"])
                total_tokens += token_count
                total_loss_sum += loss_sum
                result = {
                    "benchmark_run": run_index + 1,
                    "id": record_id(batch[offset]["record"], index),
                    "index": index,
                    "prompt": batch[offset]["prompt_text"],
                    "response": batch[offset]["response_text"],
                    "raw_label": batch[offset]["response_text"],
                    "loss": score["loss"],
                    "perplexity": math.exp(float(score["loss"])) if token_count > 0 else math.nan,
                    "response_token_count": token_count,
                    "label_length": token_count,
                }
                if bool(args.exact_match) or index < int(args.generate_samples):
                    generated, generated_token_count, generation_error = generation_by_index.get(index, ("", 0, ""))
                    normalized_generated = normalize_whitespace_exact(generated, tokenizer=tokenizer)
                    normalized_target = normalize_whitespace_exact(batch[offset]["response_text"], tokenizer=tokenizer)
                    comparison_generated = comparison_text(generated, tokenizer=tokenizer, comparison_mode=args.comparison_mode)
                    comparison_target = comparison_text(
                        batch[offset]["response_text"],
                        tokenizer=tokenizer,
                        comparison_mode=args.comparison_mode,
                    )
                    result["generated"] = generated
                    result["raw_prediction"] = generated
                    result["normalized_generated"] = normalized_generated
                    result["normalized_prediction"] = comparison_generated
                    result["normalized_target"] = normalized_target
                    result["normalized_label"] = comparison_target
                    result["comparison_generated"] = comparison_generated
                    result["comparison_target"] = comparison_target
                    result["comparison_mode"] = args.comparison_mode
                    result["generated_token_count"] = generated_token_count
                    result["generated_length"] = generated_token_count
                    result["empty_prediction"] = not bool(normalized_generated)
                    result["invalid_structured_output"] = invalid_structured_output(generated, comparison_generated)
                    result["generation_error"] = generation_error
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
            continue

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
        label_lengths = [int(result.get("label_length", 0)) for result in results]
        empty_predictions = sum(1 for result in results if result.get("empty_prediction"))
        invalid_outputs = sum(1 for result in results if result.get("invalid_structured_output"))
        generation_errors = sum(1 for result in results if result.get("generation_error"))
        avg_label_length = sum(label_lengths) / max(1, len(label_lengths))
        incorrect = len(records) - exact_correct if args.exact_match else None
        summary = {
            "benchmark_run": run_index + 1,
            "benchmark_runs": benchmark_runs,
            "checkpoint": args.checkpoint,
            "dataset_file": str(dataset_path),
            "total_examples": len(records),
            "correct_examples": exact_correct if args.exact_match else None,
            "incorrect_examples": incorrect,
            "response_tokens": total_tokens,
            "mean_response_loss": mean_loss,
            "response_perplexity": math.exp(mean_loss) if total_tokens else math.nan,
            "exact_match_accuracy": exact_correct / len(records) if args.exact_match and records else None,
            "exact_match_correct": exact_correct if args.exact_match else None,
            "empty_predictions": empty_predictions if args.exact_match else None,
            "invalid_structured_outputs": invalid_outputs if args.exact_match else None,
            "generation_errors": generation_errors if args.exact_match else None,
            "avg_generated_tokens": avg_generated_tokens if args.exact_match else None,
            "avg_label_tokens": avg_label_length,
            "max_new_tokens": int(args.max_new_tokens),
            "comparison_mode": args.comparison_mode if args.exact_match else None,
            "max_length": max_length,
            "batch_size": batch_size,
            "label_audit": label_audit,
            "prediction_file": str((output_dir if benchmark_runs == 1 else output_dir / f"run_{run_index + 1:02d}") / "prompt_response_eval_predictions.jsonl"),
            "prediction_debug_file": str((output_dir if benchmark_runs == 1 else output_dir / f"run_{run_index + 1:02d}") / "prediction_debug.csv"),
            "eval_wall_seconds": time.perf_counter() - run_start,
        }
        run_summaries.append(summary)

        write_prediction_files(run_output_dir, results, summary)

        print(
            f"[run {run_index + 1}/{benchmark_runs}] Prompt/response eval loss={summary['mean_response_loss']:.4f} "
            f"ppl={summary['response_perplexity']:.4f} "
            f"examples={summary['total_examples']} tokens={summary['response_tokens']}"
        )
        if args.exact_match:
            print(
                f"[run {run_index + 1}/{benchmark_runs}] Exact-match accuracy={summary['exact_match_accuracy']:.4f} "
                f"({summary['exact_match_correct']}/{summary['total_examples']}) "
                f"avg_generated_tokens={summary['avg_generated_tokens']:.2f}"
            )
            if summary["avg_generated_tokens"] is not None and summary["avg_generated_tokens"] >= 0.9 * int(args.max_new_tokens):
                print("[warning] Average generated length is close to max_new_tokens; the model may not be stopping cleanly.")

    if rank != 0:
        if world_size > 1:
            dist.barrier()
            dist.destroy_process_group()
        return

    if benchmark_runs > 1:
        benchmark_summary: dict[str, Any] = {
            "checkpoint": args.checkpoint,
            "dataset_file": str(dataset_path),
            "benchmark_runs": benchmark_runs,
            "total_examples": len(records),
            "max_new_tokens": int(args.max_new_tokens),
            "comparison_mode": args.comparison_mode if args.exact_match else None,
            "max_length": max_length,
            "batch_size": batch_size,
            "output_dir": str(output_dir),
            "split_audit_file": str(output_dir / "split_audit.json"),
            "run_config_file": str(output_dir / "run_config.json"),
            "label_audit": label_audit,
            "per_run_summaries": run_summaries,
        }
        for key in (
            "mean_response_loss",
            "response_perplexity",
            "avg_generated_tokens",
            "avg_label_tokens",
            "empty_predictions",
            "invalid_structured_outputs",
            "generation_errors",
            "eval_wall_seconds",
        ):
            benchmark_summary.update(benchmark_metric_summary(run_summaries, key))
        if args.exact_match:
            benchmark_summary.update(benchmark_metric_summary(run_summaries, "exact_match_accuracy"))
            correct_values = [
                float(summary["exact_match_correct"])
                for summary in run_summaries
                if summary.get("exact_match_correct") is not None
            ]
            correct_mean, correct_std = mean_std(correct_values)
            benchmark_summary["exact_match_correct_mean"] = correct_mean
            benchmark_summary["exact_match_correct_std"] = correct_std

        write_json(output_dir / "prompt_response_eval_benchmark_summary.json", benchmark_summary)
        write_json(output_dir / "prompt_response_eval_summary.json", benchmark_summary)
        write_json(output_dir / "metrics.json", benchmark_summary)
        last_run_dir = output_dir / f"run_{benchmark_runs:02d}"
        for filename in ("prompt_response_eval_predictions.jsonl", "prediction_debug.csv"):
            source = last_run_dir / filename
            if source.exists():
                shutil.copyfile(source, output_dir / filename)
        mirror_summary_to_root(output_root, output_dir, benchmark_summary, split_audit)

        print(
            "Prompt/response benchmark summary:\n"
            f"  loss={benchmark_summary['mean_response_loss_mean_pm_std']}\n"
            f"  ppl={benchmark_summary['response_perplexity_mean_pm_std']}\n"
            f"  avg_generated_tokens={benchmark_summary['avg_generated_tokens_mean_pm_std']}"
        )
        if args.exact_match:
            print(f"  exact_match_accuracy={benchmark_summary['exact_match_accuracy_mean_pm_std']}")
        print(f"Wrote benchmark results to {output_dir}")
    else:
        if run_summaries:
            mirror_summary_to_root(output_root, output_dir, run_summaries[-1], split_audit)
        print(f"Wrote results to {output_dir}")

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
