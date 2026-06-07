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
except Exception:  # pragma: no cover
    np = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from chatlm_decoder.command_eval import canonicalize_command_response
from chatlm_decoder.qwen25_instruct_data import (
    DEFAULT_SYSTEM_PROMPT,
    assert_no_legacy_tokens,
    extract_instruction_response,
    format_qwen_sft_example,
    qwen25_instruct_collate,
    qwen_prompt_text,
    read_records,
)
from chatlm_decoder.tokenizer import move_batch_to_device, prepare_decoder_tokenizer

ANCHOR_ID_FIELDS = ("anchor_id", "semantic_anchor_id", "group_id", "cluster_id", "intent_id", "anchor")
RECORD_ID_FIELDS = ("id", "uid", "uuid", "example_id", "sample_id", "index")
ZERO_WIDTH_PATTERN = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060\ufeff]")
FENCE_PATTERN = re.compile(r"^\s*```(?:json|text|txt|bash|sh|python)?\s*|\s*```\s*$", re.IGNORECASE)
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


def setup_distributed_eval(requested_device: str) -> tuple[torch.device, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("Multi-GPU Qwen eval requires CUDA.")
        torch.cuda.set_device(local_rank)
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        return torch.device("cuda", local_rank), rank, local_rank, world_size
    if requested_device != "auto":
        return torch.device(requested_device), rank, local_rank, world_size
    if torch.cuda.is_available():
        return torch.device("cuda"), rank, local_rank, world_size
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps"), rank, local_rank, world_size
    return torch.device("cpu"), rank, local_rank, world_size


def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


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


def configure_cuda(tf32: bool = True) -> None:
    if not torch.cuda.is_available():
        return
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = bool(tf32)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high" if tf32 else "highest")


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


def safe_slug(value: str, max_length: int = 80) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-._")
    return (slug or "run")[:max_length]


def stable_short_hash(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:10]


def checkpoint_step_name(path: str | Path) -> str:
    checkpoint_path = Path(path).expanduser()
    for candidate in (checkpoint_path, checkpoint_path.resolve() if checkpoint_path.exists() else checkpoint_path):
        match = re.search(r"(?:^|/)(step-[0-9]+|epoch-[0-9]+|final|latest)(?:$|/)", str(candidate))
        if match:
            return match.group(1)
    return "checkpoint"


def checkpoint_file_timestamps(path: str | Path) -> dict[str, Any]:
    checkpoint_path = Path(path).expanduser()
    if not checkpoint_path.exists():
        return {"exists": False, "path": str(checkpoint_path)}
    files: dict[str, Any] = {}
    for candidate in checkpoint_path.iterdir():
        if candidate.name.endswith((".json", ".safetensors", ".bin", ".pt", ".model")) or candidate.name in {
            "tokenizer.json",
            "merges.txt",
            "vocab.json",
        }:
            stat = candidate.stat()
            files[candidate.name] = {
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


def configure_tokenizer(tokenizer: Any) -> None:
    prepare_decoder_tokenizer(tokenizer)
    if not hasattr(tokenizer, "apply_chat_template"):
        raise AttributeError("Qwen2.5-Instruct tokenizer must provide apply_chat_template.")
    if tokenizer.eos_token is None:
        raise ValueError("Qwen2.5-Instruct tokenizer eos_token must not be None.")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


def strip_wrapping_quotes(text: str) -> str:
    text = text.strip()
    changed = True
    while changed and len(text) >= 2:
        changed = False
        for left, right in (("`", "`"), ('"', '"'), ("'", "'")):
            if text.startswith(left) and text.endswith(right):
                text = text[1:-1].strip()
                changed = True
    return text


def normalize_qwen_text(text: str, tokenizer: Any | None = None) -> str:
    text = unicodedata.normalize("NFKC", str(text))
    text = ZERO_WIDTH_PATTERN.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    for special_token in (
        getattr(tokenizer, "bos_token", None),
        getattr(tokenizer, "eos_token", None),
        getattr(tokenizer, "pad_token", None),
        getattr(tokenizer, "unk_token", None),
    ):
        if special_token:
            text = text.replace(str(special_token), "")
    text = FENCE_PATTERN.sub("", text).strip()
    text = strip_wrapping_quotes(text)
    text = text.translate(QUOTE_TRANSLATION)
    text = TRAILING_STOP_PATTERN.sub("", text).strip()
    return " ".join(text.split()).strip()


def mismatch_reason(prediction: str, label: str, command_prediction: str, command_label: str) -> str:
    if prediction == label:
        return "match"
    if not prediction:
        return "empty_prediction"
    if command_prediction and command_prediction == command_label:
        return "command_mode_match_only"
    if prediction.lower() == label.lower():
        return "case_only_difference"
    if prediction.replace(" ", "") == label.replace(" ", ""):
        return "whitespace_only_difference"
    if prediction in label or label in prediction:
        return "one_side_contains_the_other"
    return "different_text"


def first_nonempty_field(record: dict[str, Any], fields: tuple[str, ...]) -> str:
    for field in fields:
        value = record.get(field)
        if value is not None and str(value).strip():
            return normalize_qwen_text(str(value))
    return ""


def record_id(record: dict[str, Any], fallback: int) -> str:
    return first_nonempty_field(record, RECORD_ID_FIELDS) or str(fallback)


def read_eval_records(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    records = list(read_records(path))
    if limit is not None:
        records = records[: int(limit)]
    for index, record in enumerate(records):
        try:
            extract_instruction_response(record)
        except ValueError as exc:
            raise ValueError(
                f"{path}: record {index} must contain prompt/response, instruction/response, "
                f"question/answer, input/output, or messages/conversations. Original error: {exc}"
            ) from exc
    return records


def split_signature(record: dict[str, Any], index: int) -> dict[str, str]:
    instruction, response = extract_instruction_response(record)
    prompt = normalize_qwen_text(instruction)
    label = normalize_qwen_text(response)
    return {
        "id": record_id(record, index),
        "prompt": prompt,
        "response": label,
        "pair": f"{prompt}\u241f{label}",
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
    signatures = [split_signature(record, index) for index, record in enumerate(records)]
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


def audit_splits(dataset_file: Path, train_file: str | None) -> dict[str, Any]:
    split_paths: dict[str, Path] = {"eval": dataset_file}
    if train_file:
        split_paths["train"] = Path(train_file).expanduser()
    split_records = {name: read_eval_records(path) for name, path in split_paths.items()}
    stats = {name: split_stats(records) for name, records in split_records.items()}
    overlaps: dict[str, Any] = {}
    suspicious: list[dict[str, Any]] = []
    names = sorted(stats)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            left_sets = stats[left]["_sets"]
            right_sets = stats[right]["_sets"]
            key = f"{left}_vs_{right}"
            overlap = {
                "prompt_overlap_count": len(left_sets["prompts"] & right_sets["prompts"]),
                "response_overlap_count": len(left_sets["responses"] & right_sets["responses"]),
                "prompt_response_pair_overlap_count": len(left_sets["pairs"] & right_sets["pairs"]),
                "anchor_id_overlap_count": len(left_sets["anchors"] & right_sets["anchors"]),
                "prompt_overlap_examples": sample_values(left_sets["prompts"] & right_sets["prompts"]),
                "anchor_id_overlap_examples": sample_values(left_sets["anchors"] & right_sets["anchors"]),
            }
            overlaps[key] = overlap
            for field in ("prompt_overlap_count", "prompt_response_pair_overlap_count", "anchor_id_overlap_count"):
                if overlap[field] > 0:
                    suspicious.append({"split_pair": key, "issue": field, "count": overlap[field]})
    return {
        "split_paths": {name: str(path) for name, path in split_paths.items()},
        "splits": {name: {key: value for key, value in stat.items() if key != "_sets"} for name, stat in stats.items()},
        "overlaps": overlaps,
        "suspicious": suspicious,
        "has_suspicious_leakage": bool(suspicious),
        "notes": [
            "Response overlap is logged but not treated as leakage because this task is many-to-one.",
            "Prompt, exact prompt-response pair, or anchor_id overlap is suspicious.",
        ],
    }


def make_eval_examples(tokenizer: Any, records: list[dict[str, Any]], system_prompt: str) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        formatted = format_qwen_sft_example(tokenizer, record, system_prompt)
        instruction = formatted["instruction"]
        response = formatted["response"]
        prompt_text = formatted["prompt_text"]
        assert_no_legacy_tokens(prompt_text, "Qwen eval prompt")
        examples.append(
            {
                "index": index,
                "id": record_id(record, index),
                "record": record,
                "instruction": instruction,
                "prompt_text": prompt_text,
                "response_text": response,
                "response_with_eos": formatted["response_with_eos"],
                "full_text": formatted["full_text"],
                "prompt_token_count": len(tokenizer(prompt_text, add_special_tokens=False)["input_ids"]),
            }
        )
    return examples


@torch.no_grad()
def score_batch(
    model: torch.nn.Module,
    tokenizer: Any,
    examples: list[dict[str, Any]],
    device: torch.device,
    max_length: int,
) -> list[dict[str, float | int]]:
    batch = qwen25_instruct_collate(examples, tokenizer=tokenizer, max_seq_length=int(max_length))
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    scores: list[dict[str, float | int]] = []
    for row in range(shift_labels.shape[0]):
        row_labels = shift_labels[row]
        token_count = int((row_labels != -100).sum().detach().cpu())
        if token_count == 0:
            scores.append({"loss": math.nan, "token_count": 0, "loss_sum": 0.0})
            continue
        loss_sum = F.cross_entropy(
            shift_logits[row].view(-1, shift_logits.shape[-1]).float(),
            row_labels.view(-1),
            ignore_index=-100,
            reduction="sum",
        )
        loss_value = float((loss_sum / token_count).detach().cpu())
        scores.append(
            {
                "loss": loss_value,
                "token_count": token_count,
                "loss_sum": float(loss_sum.detach().cpu()),
            }
        )
    return scores


@torch.no_grad()
def generate_batch(
    model: torch.nn.Module,
    tokenizer: Any,
    prompts: list[str],
    device: torch.device,
    max_new_tokens: int,
) -> list[tuple[str, int, bool]]:
    previous_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        inputs = move_batch_to_device(
            tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False),
            device,
        )
    finally:
        tokenizer.padding_side = previous_padding_side
    input_length = int(inputs["input_ids"].shape[1])
    output_ids = model.generate(
        **inputs,
        do_sample=False,
        num_beams=1,
        max_new_tokens=int(max_new_tokens),
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    generations: list[tuple[str, int, bool]] = []
    for row in range(output_ids.shape[0]):
        continuation = output_ids[row, input_length:]
        token_ids = continuation.detach().cpu().tolist()
        trimmed_ids: list[int] = []
        hit_eos = False
        for token_id in token_ids:
            if tokenizer.eos_token_id is not None and int(token_id) == int(tokenizer.eos_token_id):
                hit_eos = True
                break
            if (
                tokenizer.pad_token_id is not None
                and tokenizer.pad_token_id != tokenizer.eos_token_id
                and int(token_id) == int(tokenizer.pad_token_id)
            ):
                break
            trimmed_ids.append(int(token_id))
        raw = tokenizer.decode(trimmed_ids, skip_special_tokens=True)
        generations.append((raw, len(trimmed_ids), (not hit_eos) and len(token_ids) >= int(max_new_tokens)))
    return generations


def mean_std(values: list[float]) -> tuple[float, float]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return math.nan, math.nan
    if len(finite) == 1:
        return finite[0], 0.0
    return statistics.mean(finite), statistics.stdev(finite)


def metric_mean_std(run_summaries: list[dict[str, Any]], key: str) -> dict[str, float | str]:
    values = [float(summary[key]) for summary in run_summaries if summary.get(key) is not None]
    mean_value, std_value = mean_std(values)
    return {
        f"{key}_mean": mean_value,
        f"{key}_std": std_value,
        f"{key}_mean_pm_std": f"{mean_value:.6f} ± {std_value:.6f}",
    }


def resolve_output_dirs(args: argparse.Namespace, dataset_path: Path) -> tuple[Path, Path]:
    output_root = Path(args.output_dir or Path("runs") / "eval" / f"{dataset_path.stem}_qwen25_instruct").expanduser()
    if not args.unique_output_dir:
        return output_root, output_root
    checkpoint_slug = safe_slug(Path(args.model_path).name or stable_short_hash(args.model_path), max_length=40)
    dataset_slug = safe_slug(dataset_path.stem, max_length=40)
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    step = checkpoint_step_name(args.model_path)
    return output_root, output_root / f"{dataset_slug}_{checkpoint_slug}_{step}_seed{int(args.seed)}_{timestamp}"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_prediction_files(output_dir: Path, results: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "qwen25_instruct_eval_summary.json", summary)
    write_json(output_dir / "metrics.json", summary)
    with (output_dir / "qwen25_instruct_predictions.jsonl").open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
    with (output_dir / "qwen25_instruct_prediction_debug.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "index",
            "prompt",
            "raw_prediction",
            "normalized_prediction",
            "raw_label",
            "normalized_label",
            "exact_match",
            "command_exact_match",
            "generated_token_count",
            "response_loss",
            "response_token_count",
            "reached_max_new_tokens",
            "empty_prediction",
            "mismatch_reason",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow({field: result.get(field, "") for field in fieldnames})
    failed = [result for result in results if not bool(result.get("exact_match"))]
    write_json(output_dir / "failed_examples_20.json", failed[:20])
    generation_samples = [
        {
            "raw_input_example": result.get("record", {}),
            "formatted_prompt": result.get("prompt", ""),
            "tokenized_prompt_length": result.get("tokenized_prompt_length"),
            "gold_response": result.get("raw_label", ""),
            "raw_generated_text": result.get("raw_prediction", ""),
            "extracted_response": result.get("raw_prediction", ""),
            "normalized_prediction": result.get("normalized_prediction", ""),
            "normalized_gold": result.get("normalized_label", ""),
            "exact_match": result.get("exact_match"),
            "response_loss": result.get("response_loss"),
        }
        for result in results[:50]
    ]
    write_json(output_dir / "generation_samples.json", generation_samples)
    write_json(output_dir / "exact_match_failure_cases.json", failed[:50])


def mirror_summary_to_root(output_root: Path, output_dir: Path, summary: dict[str, Any], split_audit: dict[str, Any]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "latest_eval_dir.txt").write_text(str(output_dir) + "\n", encoding="utf-8")
    write_json(output_root / "qwen25_instruct_eval_summary.json", summary)
    write_json(output_root / "metrics.json", summary)
    write_json(output_root / "split_audit.json", split_audit)


def write_run_config(
    output_dir: Path,
    args: argparse.Namespace,
    model: torch.nn.Module,
    tokenizer: Any,
    dataset_path: Path,
    split_audit: dict[str, Any],
    seed_info: dict[str, Any],
    world_size: int,
    local_rank: int,
    device: torch.device,
) -> None:
    payload = {
        "script": "scripts/eval_qwen25_instruct.py",
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git_commit": git_commit_hash(),
        "model_path": args.model_path,
        "checkpoint_path_used_for_evaluation": args.model_path,
        "tokenizer_path": args.model_path,
        "checkpoint_files": checkpoint_file_timestamps(args.model_path),
        "dataset_file": args.dataset_file,
        "train_file": args.train_file,
        "output_dir": str(output_dir),
        "model_class": model.__class__.__name__,
        "model_type": getattr(model.config, "model_type", None),
        "parameter_count": parameter_count(model),
        "world_size": int(world_size),
        "local_rank": int(local_rank),
        "device": str(device),
        "seed_info": seed_info,
        "uses_qwen_apply_chat_template": True,
        "system_prompt": args.system_prompt,
        "generation": {
            "do_sample": False,
            "num_beams": 1,
            "temperature": None,
            "top_p": None,
            "top_k": None,
            "max_new_tokens": int(args.max_new_tokens),
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
        },
        "scoring": {
            "max_length": int(args.max_length),
            "label_masking": "prompt tokens are -100; assistant response tokens only",
        },
        "eval_args": vars(args),
        "split_audit_summary": {
            "has_suspicious_leakage": split_audit.get("has_suspicious_leakage"),
            "suspicious": split_audit.get("suspicious", []),
        },
        "tokenizer": {
            "class": tokenizer.__class__.__name__,
            "vocab_size": len(tokenizer),
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        },
    }
    write_json(output_dir / "run_config.json", payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Qwen2.5-Instruct SFT checkpoints with Qwen chat templates.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dataset-file", required=True)
    parser.add_argument("--train-file", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--unique-output-dir", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="bf16", choices=("auto", "bf16", "fp16", "fp32"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic-eval", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--benchmark-runs", type=int, default=1)
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--fail-on-leakage", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_info = set_eval_seeds(args.seed, deterministic=bool(args.deterministic_eval))
    device, rank, local_rank, world_size = setup_distributed_eval(args.device)
    configure_cuda(tf32=True)

    model_path = Path(args.model_path).expanduser()
    local_model_like = (
        model_path.is_absolute()
        or args.model_path.startswith(("./", "../"))
        or ("/" in args.model_path and model_path.parent.exists())
    )
    if local_model_like and not model_path.exists():
        raise FileNotFoundError(f"Qwen SFT checkpoint path does not exist: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=False)
    configure_tokenizer(tokenizer)
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype_for(args.dtype, device),
        trust_remote_code=False,
    ).to(device)
    model.eval()

    dataset_path = Path(args.dataset_file).expanduser()
    records = read_eval_records(dataset_path, limit=args.limit)
    if not records:
        raise ValueError(f"No Qwen eval records found in {dataset_path}.")
    examples = make_eval_examples(tokenizer, records, args.system_prompt)
    split_audit = audit_splits(dataset_path, args.train_file)
    if args.fail_on_leakage and split_audit.get("has_suspicious_leakage"):
        raise RuntimeError(f"Split audit found suspicious leakage: {split_audit.get('suspicious')}")

    output_root, output_dir = resolve_output_dirs(args, dataset_path)
    if not args.unique_output_dir and (output_dir / "qwen25_instruct_predictions.jsonl").exists():
        raise FileExistsError(
            f"Refusing to overwrite existing Qwen predictions without a unique run directory: {output_dir}."
        )
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(output_dir / "split_audit.json", split_audit)
        write_run_config(
            output_dir=output_dir,
            args=args,
            model=model,
            tokenizer=tokenizer,
            dataset_path=dataset_path,
            split_audit=split_audit,
            seed_info=seed_info,
            world_size=world_size,
            local_rank=local_rank,
            device=device,
        )
        print(
            f"Qwen2.5-Instruct eval runtime: world_size={world_size} rank={rank} local_rank={local_rank} "
            f"device={device} total_examples={len(records)} "
            f"shard_examples={(len(records) + world_size - 1 - rank) // world_size} benchmark_runs={int(args.benchmark_runs)}\n"
            f"  model_path={args.model_path}\n"
            f"  checkpoint_path_used_for_evaluation={args.model_path}\n"
            f"  tokenizer_path={args.model_path}\n"
            f"  model_class={model.__class__.__name__} model_type={getattr(model.config, 'model_type', None)}\n"
            f"  parameter_count={parameter_count(model):,}\n"
            f"  git_commit={git_commit_hash()}\n"
            f"  seed={args.seed} deterministic={seed_info['deterministic_algorithms_enabled']}\n"
            f"  max_length={int(args.max_length)}\n"
            f"  output_dir={output_dir}"
        )
        print(
            "Generation settings: do_sample=False num_beams=1 temperature=None top_p=None top_k=None "
            f"max_new_tokens={int(args.max_new_tokens)} eos_token_id={tokenizer.eos_token_id} "
            f"pad_token_id={tokenizer.pad_token_id}"
        )
    if is_dist():
        dist.barrier()

    indexed_examples = [(index, example) for index, example in enumerate(examples) if index % world_size == rank]
    batch_size = max(1, int(args.batch_size))
    benchmark_runs = max(1, int(args.benchmark_runs))
    run_summaries: list[dict[str, Any]] = []

    for run_index in range(benchmark_runs):
        run_start = time.perf_counter()
        run_output_dir = output_dir if benchmark_runs == 1 else output_dir / f"run_{run_index + 1:02d}"
        if rank == 0:
            print(
                f"[run {run_index + 1}/{benchmark_runs}] Fresh Qwen chat-template generation; writing: "
                f"{run_output_dir / 'qwen25_instruct_predictions.jsonl'}"
        )
        local_results: list[dict[str, Any]] = []
        total_loss_sum = 0.0
        total_tokens = 0
        progress = tqdm(
            range(0, len(indexed_examples), batch_size),
            desc=f"qwen25-instruct-eval-run{run_index + 1:02d}-rank{rank}",
            disable=(rank != 0),
        )
        for start in progress:
            batch_pairs = indexed_examples[start : start + batch_size]
            batch_examples = [example for _, example in batch_pairs]
            scores = score_batch(
                model=model,
                tokenizer=tokenizer,
                examples=batch_examples,
                device=device,
                max_length=int(args.max_length),
            )
            prompts = [example["prompt_text"] for _, example in batch_pairs]
            try:
                generations = generate_batch(
                    model=model,
                    tokenizer=tokenizer,
                    prompts=prompts,
                    device=device,
                    max_new_tokens=int(args.max_new_tokens),
                )
                generation_errors = ["" for _ in batch_pairs]
            except Exception as exc:
                generations = [("", 0, False) for _ in batch_pairs]
                generation_errors = [repr(exc) for _ in batch_pairs]
            for offset, (index, example) in enumerate(batch_pairs):
                score = scores[offset]
                token_count = int(score["token_count"])
                loss_sum = float(score["loss_sum"])
                total_tokens += token_count
                total_loss_sum += loss_sum
                raw_prediction, generated_count, reached_max = generations[offset]
                raw_label = str(example["response_text"])
                normalized_prediction = normalize_qwen_text(raw_prediction, tokenizer=tokenizer)
                normalized_label = normalize_qwen_text(raw_label, tokenizer=tokenizer)
                command_prediction = canonicalize_command_response(normalized_prediction)
                command_label = canonicalize_command_response(normalized_label)
                is_exact = normalized_prediction == normalized_label
                is_command_exact = command_prediction == command_label
                label_count = len(tokenizer(raw_label, add_special_tokens=False)["input_ids"])
                local_results.append(
                    {
                        "benchmark_run": run_index + 1,
                        "id": example["id"],
                        "index": index,
                        "record": example["record"],
                        "prompt": example["prompt_text"],
                        "instruction": example["instruction"],
                        "raw_prediction": raw_prediction,
                        "normalized_prediction": normalized_prediction,
                        "raw_label": raw_label,
                        "normalized_label": normalized_label,
                        "command_prediction": command_prediction,
                        "command_label": command_label,
                        "exact_match": is_exact,
                        "command_exact_match": is_command_exact,
                        "generated_token_count": int(generated_count),
                        "tokenized_prompt_length": int(example["prompt_token_count"]),
                        "label_token_count": int(label_count),
                        "response_loss": score["loss"],
                        "response_perplexity": math.exp(float(score["loss"])) if token_count > 0 else math.nan,
                        "response_token_count": token_count,
                        "generated_length": int(generated_count),
                        "label_length": int(label_count),
                        "reached_max_new_tokens": bool(reached_max),
                        "empty_prediction": not bool(normalized_prediction),
                        "generation_error": generation_errors[offset],
                        "mismatch_reason": mismatch_reason(
                            normalized_prediction,
                            normalized_label,
                            command_prediction,
                            command_label,
                        ),
                    }
                )

        if is_dist():
            local_payload = {
                "results": local_results,
                "total_loss_sum": total_loss_sum,
                "total_tokens": total_tokens,
            }
            gathered: list[dict[str, Any] | None] | None = [None for _ in range(world_size)] if rank == 0 else None
            dist.gather_object(local_payload, gathered, dst=0)
        else:
            gathered = [
                {
                    "results": local_results,
                    "total_loss_sum": total_loss_sum,
                    "total_tokens": total_tokens,
                }
            ]
        if rank != 0:
            continue

        payloads = [payload for payload in gathered or [] if payload is not None]
        total_loss_sum = sum(float(payload["total_loss_sum"]) for payload in payloads)
        total_tokens = sum(int(payload["total_tokens"]) for payload in payloads)
        results = sorted(
            [row for payload in payloads for row in payload["results"]],
            key=lambda row: int(row["index"]),
        )
        exact_correct = sum(1 for result in results if bool(result["exact_match"]))
        command_correct = sum(1 for result in results if bool(result["command_exact_match"]))
        empty_predictions = sum(1 for result in results if bool(result["empty_prediction"]))
        generation_errors = sum(1 for result in results if result.get("generation_error"))
        reached_max = sum(1 for result in results if bool(result["reached_max_new_tokens"]))
        avg_generated_tokens = sum(int(result["generated_token_count"]) for result in results) / max(1, len(results))
        avg_label_tokens = sum(int(result["label_token_count"]) for result in results) / max(1, len(results))
        mean_loss = total_loss_sum / total_tokens if total_tokens else math.nan
        generated_texts = [str(result.get("raw_prediction", "")).strip() for result in results]
        nonempty_generated = [text for text in generated_texts if text]
        prompt_copies = sum(
            1
            for result in results
            if str(result.get("raw_prediction", "")).strip()
            and str(result.get("raw_prediction", "")).strip().startswith(str(result.get("prompt", "")).strip())
        )
        if empty_predictions == len(results):
            raise RuntimeError("Generated Qwen predictions are all empty; inspect tokenizer/EOS/max_new_tokens before trusting the benchmark.")
        if len(nonempty_generated) > 1 and len(set(nonempty_generated)) == 1:
            raise RuntimeError("Generated Qwen predictions are all identical; inspect generation/checkpoint loading before trusting the benchmark.")
        if prompt_copies / max(1, len(results)) > 0.5:
            raise RuntimeError("Generated Qwen predictions are mostly prompt copies; response extraction or chat formatting is likely wrong.")
        warnings: list[str] = []
        if avg_generated_tokens >= 0.9 * int(args.max_new_tokens):
            warnings.append("Average generated length is close to max_new_tokens; the model may not be stopping cleanly.")
        if exact_correct == 0 and command_correct > 0:
            warnings.append("Exact match is 0 but command-mode match is nonzero; inspect normalization/string formatting.")
        summary = {
            "benchmark_run": run_index + 1,
            "benchmark_runs": benchmark_runs,
            "checkpoint": args.model_path,
            "dataset_file": str(dataset_path),
            "train_file": args.train_file,
            "total_examples": len(results),
            "correct_examples": exact_correct,
            "incorrect_examples": len(results) - exact_correct,
            "response_tokens": total_tokens,
            "total_loss_sum": total_loss_sum,
            "mean_response_loss": mean_loss,
            "response_perplexity": math.exp(mean_loss) if total_tokens else math.nan,
            "exact_match_accuracy": exact_correct / max(1, len(results)),
            "exact_match_correct": exact_correct,
            "command_exact_match_accuracy": command_correct / max(1, len(results)),
            "command_exact_match_correct": command_correct,
            "empty_predictions": empty_predictions,
            "generation_errors": generation_errors,
            "reached_max_new_tokens": reached_max,
            "avg_generated_tokens": avg_generated_tokens,
            "avg_label_tokens": avg_label_tokens,
            "max_new_tokens": int(args.max_new_tokens),
            "max_length": int(args.max_length),
            "batch_size": batch_size,
            "world_size": world_size,
            "uses_qwen_apply_chat_template": True,
            "comparison_slot": "Qwen2.5-0.5B-Instruct using Qwen chat-template SFT",
            "warnings": warnings,
            "prediction_file": str(run_output_dir / "qwen25_instruct_predictions.jsonl"),
            "prediction_debug_file": str(run_output_dir / "qwen25_instruct_prediction_debug.csv"),
            "failed_examples_file": str(run_output_dir / "failed_examples_20.json"),
            "eval_wall_seconds": time.perf_counter() - run_start,
        }
        run_summaries.append(summary)
        write_prediction_files(run_output_dir, results, summary)
        print(
            f"[run {run_index + 1}/{benchmark_runs}] exact_match={summary['exact_match_accuracy']:.4f} "
            f"({exact_correct}/{len(results)}) command_match={summary['command_exact_match_accuracy']:.4f} "
            f"loss={summary['mean_response_loss']:.4f} ppl={summary['response_perplexity']:.4f} "
            f"avg_generated_tokens={avg_generated_tokens:.2f}"
        )
        for warning in warnings:
            print(f"[warning] {warning}")

    if rank != 0:
        if is_dist():
            dist.barrier()
            dist.destroy_process_group()
        return

    if benchmark_runs > 1:
        benchmark_summary: dict[str, Any] = {
            "checkpoint": args.model_path,
            "dataset_file": str(dataset_path),
            "train_file": args.train_file,
            "benchmark_runs": benchmark_runs,
            "total_examples": len(records),
            "max_new_tokens": int(args.max_new_tokens),
            "max_length": int(args.max_length),
            "batch_size": batch_size,
            "world_size": world_size,
            "output_dir": str(output_dir),
            "uses_qwen_apply_chat_template": True,
            "comparison_slot": "Qwen2.5-0.5B-Instruct using Qwen chat-template SFT",
            "split_audit_file": str(output_dir / "split_audit.json"),
            "run_config_file": str(output_dir / "run_config.json"),
            "per_run_summaries": run_summaries,
        }
        for key in (
            "exact_match_accuracy",
            "command_exact_match_accuracy",
            "avg_generated_tokens",
            "avg_label_tokens",
            "mean_response_loss",
            "response_perplexity",
            "response_tokens",
            "empty_predictions",
            "generation_errors",
            "reached_max_new_tokens",
            "eval_wall_seconds",
        ):
            benchmark_summary.update(metric_mean_std(run_summaries, key))
        write_json(output_dir / "qwen25_instruct_eval_benchmark_summary.json", benchmark_summary)
        write_json(output_dir / "qwen25_instruct_eval_summary.json", benchmark_summary)
        write_json(output_dir / "metrics.json", benchmark_summary)
        last_run_dir = output_dir / f"run_{benchmark_runs:02d}"
        for filename in (
            "qwen25_instruct_predictions.jsonl",
            "qwen25_instruct_prediction_debug.csv",
            "failed_examples_20.json",
            "generation_samples.json",
            "exact_match_failure_cases.json",
        ):
            source = last_run_dir / filename
            if source.exists():
                shutil.copyfile(source, output_dir / filename)
        mirror_summary_to_root(output_root, output_dir, benchmark_summary, split_audit)
        print(
            "Qwen2.5-Instruct benchmark summary:\n"
            f"  exact_match_accuracy={benchmark_summary['exact_match_accuracy_mean_pm_std']}\n"
            f"  command_exact_match_accuracy={benchmark_summary['command_exact_match_accuracy_mean_pm_std']}\n"
            f"  loss={benchmark_summary['mean_response_loss_mean_pm_std']}\n"
            f"  ppl={benchmark_summary['response_perplexity_mean_pm_std']}\n"
            f"  avg_generated_tokens={benchmark_summary['avg_generated_tokens_mean_pm_std']}"
        )
    elif run_summaries:
        mirror_summary_to_root(output_root, output_dir, run_summaries[-1], split_audit)
        print(f"Wrote Qwen2.5-Instruct eval results to {output_dir}")

    if is_dist():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
