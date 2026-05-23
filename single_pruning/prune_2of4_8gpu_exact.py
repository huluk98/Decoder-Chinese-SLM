#!/usr/bin/env python3
"""
Single-file 8-GPU NVIDIA 2:4 pruning + exact-match benchmark for a LOCAL decoder-only HF model.

Usage with only two paths:
    python prune_2of4_8gpu_exact.py /path/to/local_model /path/to/eval.json

What it does:
  1. If multiple GPUs are available, auto-relaunches itself with torchrun using up to 8 GPUs.
  2. Loads the local model with AutoModelForCausalLM.
  3. Benchmarks dense exact-match generation on the eval file.
  4. Applies exact NVIDIA-style 2:4 pruning to every eligible nn.Linear weight.
  5. Verifies the real 2:4 pattern after pruning.
  6. Benchmarks the pruned model on the same eval file.
  7. Saves outputs next to the model path without requiring a third path.

Supported eval formats:
  - .jsonl: one dict per line
  - .json: list[dict], dict with data/examples/items/samples, or dict of id->dict
  - .csv: columns such as prompt/response, instruction/output, question/answer, input/target
  - .txt: each line as text only; will run generation but exact accuracy requires targets

Common eval row keys:
  prompt keys:   prompt, instruction, input, question, query, source, text
  target keys:   response, output, answer, target, completion, label

Notes:
  - This script uses data-parallel evaluation: each GPU loads a full copy of the model and evaluates a shard.
  - The saved checkpoint is dense-format HF weights containing 2:4 zeros. Real speedup requires a sparse runtime/kernel.
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

# =========================
# User-editable constants
# =========================
MAX_GPUS = 8
MODEL_PATH = ""  # Optional: set /path/to/local_model here, then run with no path args.
EVAL_DATASET_PATH = ""  # Optional: set /path/to/eval.json here, then run with no path args.
BATCH_SIZE = 8
MAX_INPUT_TOKENS = 2048
MAX_NEW_TOKENS = 64
TRUST_REMOTE_CODE = True
INCLUDE_LM_HEAD = True
SKIP_NON_DIVISIBLE_LINEAR = True
NORMALIZE_WHITESPACE_FOR_EXACT = True
SAVE_PRUNED_MODEL = True
SAVE_PREDICTIONS = True

# For many instruction-tuned decoder-only models, direct prompt text is what your SFT used.
# If your tokenizer has a chat template and your eval prompts require it, set USE_CHAT_TEMPLATE=True.
USE_CHAT_TEMPLATE = False

PROMPT_KEYS = ["prompt", "instruction", "input", "question", "query", "source", "text"]
TARGET_KEYS = ["response", "output", "answer", "target", "completion", "label"]


def fail(msg: str) -> None:
    print(f"\nERROR: {msg}\n", file=sys.stderr)
    sys.exit(1)


def resolve_run_paths(script_name: str) -> Tuple[Path, Path]:
    if len(sys.argv) == 3:
        model_value = sys.argv[1]
        eval_value = sys.argv[2]
    elif len(sys.argv) == 1 and MODEL_PATH.strip() and EVAL_DATASET_PATH.strip():
        model_value = MODEL_PATH
        eval_value = EVAL_DATASET_PATH
    else:
        fail(
            f"This script takes exactly two paths:\n"
            f"  python {script_name} /path/to/local_model /path/to/eval_file\n"
            "Or edit MODEL_PATH and EVAL_DATASET_PATH at the top of this script, then run it with no path args."
        )
    return Path(model_value).expanduser().resolve(), Path(eval_value).expanduser().resolve()


def maybe_relaunch_with_torchrun() -> None:
    """Allow the user to run only: python script.py MODEL_PATH EVAL_FILE."""
    if "WORLD_SIZE" in os.environ:
        return
    model_path, eval_path = resolve_run_paths("prune_2of4_8gpu_exact.py")
    if not torch.cuda.is_available():
        return
    gpu_count = torch.cuda.device_count()
    if gpu_count <= 1:
        return

    nproc = min(MAX_GPUS, gpu_count)
    env = os.environ.copy()
    env["PRUNE_2OF4_AUTOLAUNCHED"] = "1"
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={nproc}",
        str(Path(__file__).resolve()),
        str(model_path),
        str(eval_path),
    ]
    print(f"Auto-launching distributed evaluation/pruning on {nproc} GPU(s)...")
    os.execvpe(sys.executable, cmd, env)


def setup_distributed() -> Tuple[int, int, int, torch.device]:
    if "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        device = torch.device(f"cuda:{local_rank}")
        return rank, local_rank, world_size, device

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return 0, 0, 1, device


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def rank0_print(rank: int, *args: Any, **kwargs: Any) -> None:
    if rank == 0:
        print(*args, **kwargs)


def choose_dtype(device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    # H20/A100/Hopper/Ampere generally support bf16; fall back cleanly otherwise.
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def prepare_tokenizer(tokenizer: Any) -> Any:
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def load_eval_file(eval_path: Path) -> List[Dict[str, Any]]:
    suffix = eval_path.suffix.lower()
    if not eval_path.exists():
        fail(f"Eval file does not exist: {eval_path}")

    if suffix == ".jsonl":
        rows: List[Dict[str, Any]] = []
        with eval_path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if not isinstance(obj, dict):
                    obj = {"text": str(obj)}
                obj["_row_id"] = line_no - 1
                rows.append(obj)
        return rows

    if suffix == ".json":
        data = json.loads(eval_path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            rows = [x if isinstance(x, dict) else {"text": str(x)} for x in data]
        elif isinstance(data, dict):
            for key in ["data", "examples", "items", "samples", "eval", "test"]:
                if key in data and isinstance(data[key], list):
                    rows = [x if isinstance(x, dict) else {"text": str(x)} for x in data[key]]
                    break
            else:
                if all(isinstance(v, dict) for v in data.values()):
                    rows = list(data.values())
                else:
                    rows = [data]
        else:
            rows = [{"text": str(data)}]
        for i, row in enumerate(rows):
            row.setdefault("_row_id", i)
        return rows

    if suffix == ".csv":
        with eval_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            rows = [dict(row) for row in reader]
        for i, row in enumerate(rows):
            row.setdefault("_row_id", i)
        return rows

    if suffix == ".txt":
        rows = []
        with eval_path.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if line:
                    rows.append({"_row_id": i, "prompt": line})
        return rows

    fail(f"Unsupported eval file type: {eval_path.suffix}. Use json/jsonl/csv/txt.")
    return []


def first_existing(row: Dict[str, Any], keys: Iterable[str]) -> Optional[str]:
    for key in keys:
        if key in row and row[key] is not None:
            value = row[key]
            if isinstance(value, (dict, list)):
                return json.dumps(value, ensure_ascii=False, sort_keys=True)
            return str(value)
    return None


def make_prompt(row: Dict[str, Any], tokenizer: Any) -> str:
    prompt = first_existing(row, PROMPT_KEYS)
    if prompt is None:
        prompt = json.dumps({k: v for k, v in row.items() if not k.startswith("_")}, ensure_ascii=False)

    if USE_CHAT_TEMPLATE and hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": prompt}]
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            return prompt
    return prompt


def make_target(row: Dict[str, Any]) -> Optional[str]:
    return first_existing(row, TARGET_KEYS)


def normalize_for_exact(s: Optional[str]) -> str:
    if s is None:
        return ""
    text = str(s).strip()
    if NORMALIZE_WHITESPACE_FOR_EXACT:
        text = re.sub(r"\s+", "", text)
    return text


def shard_rows(rows: List[Dict[str, Any]], rank: int, world_size: int) -> List[Dict[str, Any]]:
    return [row for i, row in enumerate(rows) if i % world_size == rank]


def batch_iter(items: List[Any], batch_size: int) -> Iterable[List[Any]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def generate_batch(
    model: nn.Module,
    tokenizer: Any,
    prompts: List[str],
    device: torch.device,
) -> List[str]:
    tokenizer = prepare_tokenizer(tokenizer)

    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_INPUT_TOKENS,
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    prompt_width = int(enc["input_ids"].shape[1])

    with torch.inference_mode():
        out = model.generate(
            **enc,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            num_beams=1,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    preds: List[str] = []
    for seq in out:
        new_tokens = seq[prompt_width:]
        pred = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        preds.append(pred)
    return preds


def benchmark_exact(
    model: nn.Module,
    tokenizer: Any,
    rows: List[Dict[str, Any]],
    device: torch.device,
    rank: int,
    world_size: int,
    tag: str,
) -> Dict[str, Any]:
    local_rows = shard_rows(rows, rank, world_size)
    model.eval()

    local_predictions: List[Dict[str, Any]] = []
    correct = 0
    target_count = 0
    total_generated_tokens = 0
    start = time.time()

    for batch in batch_iter(local_rows, BATCH_SIZE):
        prompts = [make_prompt(row, tokenizer) for row in batch]
        targets = [make_target(row) for row in batch]
        preds = generate_batch(model, tokenizer, prompts, device)

        # Approximate generated token count for throughput.
        tokenized_preds = tokenizer(preds, add_special_tokens=False)["input_ids"]
        total_generated_tokens += sum(len(x) for x in tokenized_preds)

        for row, prompt, target, pred in zip(batch, prompts, targets, preds):
            is_correct: Optional[bool]
            if target is None:
                is_correct = None
            else:
                target_count += 1
                is_correct = normalize_for_exact(pred) == normalize_for_exact(target)
                correct += int(is_correct)

            local_predictions.append(
                {
                    "row_id": row.get("_row_id"),
                    "prompt": prompt,
                    "target": target,
                    "prediction": pred,
                    "correct": is_correct,
                }
            )

    elapsed = time.time() - start

    local_result = {
        "tag": tag,
        "rank": rank,
        "num_examples": len(local_rows),
        "target_count": target_count,
        "correct": correct,
        "elapsed_seconds": elapsed,
        "generated_tokens": total_generated_tokens,
        "predictions": local_predictions,
    }

    if world_size > 1:
        gathered: List[Optional[Dict[str, Any]]] = [None for _ in range(world_size)]
        dist.all_gather_object(gathered, local_result)
    else:
        gathered = [local_result]

    if rank != 0:
        return {}

    all_preds: List[Dict[str, Any]] = []
    total_examples = 0
    total_targets = 0
    total_correct = 0
    max_elapsed = 0.0
    total_tokens = 0
    for item in gathered:
        if not item:
            continue
        total_examples += int(item["num_examples"])
        total_targets += int(item["target_count"])
        total_correct += int(item["correct"])
        max_elapsed = max(max_elapsed, float(item["elapsed_seconds"]))
        total_tokens += int(item["generated_tokens"])
        all_preds.extend(item["predictions"])

    all_preds.sort(key=lambda x: int(x["row_id"]) if x.get("row_id") is not None else 0)

    return {
        "tag": tag,
        "num_examples": total_examples,
        "target_count": total_targets,
        "correct": total_correct,
        "exact_match_accuracy": (total_correct / total_targets) if total_targets else None,
        "elapsed_seconds": max_elapsed,
        "generated_tokens": total_tokens,
        "generated_tokens_per_second": total_tokens / max_elapsed if max_elapsed > 0 else None,
        "predictions": all_preds,
    }


def make_2of4_mask(weight: torch.Tensor) -> torch.Tensor:
    if weight.ndim != 2:
        raise ValueError(f"Expected 2D Linear weight, got {tuple(weight.shape)}")
    out_features, in_features = weight.shape
    if in_features % 4 != 0:
        raise ValueError(f"in_features must be divisible by 4, got {tuple(weight.shape)}")

    groups = weight.detach().abs().float().reshape(out_features, in_features // 4, 4)
    prune_idx = torch.topk(groups, k=2, dim=-1, largest=False).indices
    mask = torch.ones_like(groups, dtype=torch.bool)
    mask.scatter_(dim=-1, index=prune_idx, value=False)
    return mask.reshape_as(weight)


def is_lm_head(name: str) -> bool:
    leaf = name.lower().split(".")[-1]
    return leaf in {"lm_head", "output_head"} or "lm_head" in name.lower()


def whole_model_sparsity(model: nn.Module) -> Dict[str, Any]:
    total = 0
    zeros = 0
    for p in model.parameters():
        total += p.numel()
        zeros += int((p.detach() == 0).sum().item())
    return {
        "total_parameters": int(total),
        "zero_parameters": int(zeros),
        "nonzero_parameters": int(total - zeros),
        "sparsity": zeros / float(total or 1),
    }


def validate_2of4_for_weight(weight: torch.Tensor) -> Tuple[int, int]:
    """Return invalid group count and total group count for actual zero pattern."""
    if weight.ndim != 2 or weight.shape[1] % 4 != 0:
        return 0, 0
    groups = weight.detach().reshape(weight.shape[0], weight.shape[1] // 4, 4)
    zeros_per_group = (groups == 0).sum(dim=-1)
    invalid = int((zeros_per_group < 2).sum().item())
    total = int(zeros_per_group.numel())
    return invalid, total


def prune_model_2of4(model: nn.Module, rank: int) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    before = whole_model_sparsity(model)
    module_rows: List[Dict[str, Any]] = []

    selected_params = 0
    pruned_params = 0
    skipped: List[Dict[str, Any]] = []
    invalid_actual_groups = 0
    total_groups = 0
    masked_nonzero_violations = 0

    with torch.no_grad():
        for name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if not INCLUDE_LM_HEAD and is_lm_head(name):
                skipped.append({"module": name, "shape": list(module.weight.shape), "reason": "lm_head_skipped"})
                continue
            if module.weight.ndim != 2 or module.weight.shape[1] % 4 != 0:
                reason = "not_2d_or_in_features_not_divisible_by_4"
                if SKIP_NON_DIVISIBLE_LINEAR:
                    skipped.append({"module": name, "shape": list(module.weight.shape), "reason": reason})
                    continue
                raise ValueError(f"Cannot 2:4 prune {name} with shape {tuple(module.weight.shape)}")

            mask = make_2of4_mask(module.weight)
            before_nonzero_pruned_positions = int(torch.count_nonzero(module.weight.detach().masked_select(~mask.to(module.weight.device))).item())
            module.weight.mul_(mask.to(device=module.weight.device, dtype=module.weight.dtype))
            after_nonzero_pruned_positions = int(torch.count_nonzero(module.weight.detach().masked_select(~mask.to(module.weight.device))).item())

            invalid, groups = validate_2of4_for_weight(module.weight)
            invalid_actual_groups += invalid
            total_groups += groups
            masked_nonzero_violations += after_nonzero_pruned_positions

            weight_params = int(module.weight.numel())
            mask_pruned = int((~mask).sum().item())
            selected_params += weight_params
            pruned_params += mask_pruned

            module_rows.append(
                {
                    "module": name,
                    "shape": list(module.weight.shape),
                    "weight_parameters": weight_params,
                    "mask_pruned_parameters": mask_pruned,
                    "mask_sparsity": mask_pruned / float(weight_params or 1),
                    "actual_zero_parameters_after_prune": int((module.weight.detach() == 0).sum().item()),
                    "invalid_actual_2of4_groups": invalid,
                    "total_2of4_groups": groups,
                    "nonzero_values_removed_by_mask": before_nonzero_pruned_positions,
                    "masked_weight_nonzero_violations_after_prune": after_nonzero_pruned_positions,
                }
            )

    after = whole_model_sparsity(model)
    report = {
        "method": "nvidia_2of4_exact_decoder_only_auto_8gpu",
        "include_lm_head": INCLUDE_LM_HEAD,
        "selected_linear_modules": len(module_rows),
        "skipped_linear_modules": skipped,
        "selected_linear_weight_parameters": selected_params,
        "selected_linear_pruned_by_mask": pruned_params,
        "selected_linear_mask_sparsity": pruned_params / float(selected_params or 1),
        "whole_model_before": before,
        "whole_model_after": after,
        "valid_exact_2of4_actual_weights": invalid_actual_groups == 0,
        "invalid_actual_2of4_groups": invalid_actual_groups,
        "total_2of4_groups_checked": total_groups,
        "masked_weight_nonzero_violations": masked_nonzero_violations,
        "valid_pruned_weights": invalid_actual_groups == 0 and masked_nonzero_violations == 0,
    }
    return report, module_rows


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")


def save_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: List[str] = []
    for row in rows:
        for k in row:
            if k not in fieldnames:
                fieldnames.append(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def derived_output_dir(model_path: Path) -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    if model_path.exists() and model_path.is_dir():
        return model_path.parent / f"{model_path.name}_nvidia2of4_8gpu_exact_{stamp}"
    return Path.cwd() / f"nvidia2of4_8gpu_exact_{stamp}"


def format_metric(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return str(value)


def main() -> None:
    maybe_relaunch_with_torchrun()

    model_path, eval_path = resolve_run_paths("prune_2of4_8gpu_exact.py")

    if not model_path.exists():
        fail(f"Local model path does not exist: {model_path}")

    rank, local_rank, world_size, device = setup_distributed()
    dtype = choose_dtype(device)
    out_dir = derived_output_dir(model_path)

    try:
        rank0_print(rank, f"Model path: {model_path}")
        rank0_print(rank, f"Eval file:  {eval_path}")
        rank0_print(rank, f"Output dir: {out_dir}")
        rank0_print(
            rank,
            f"World size: {world_size}; batch_size_per_gpu: {BATCH_SIZE}; "
            f"effective_batch_size: {BATCH_SIZE * world_size}; dtype: {dtype}; device per rank: {device}",
        )

        rows = load_eval_file(eval_path)
        if not rows:
            fail("Eval file loaded zero rows.")

        rank0_print(rank, f"Loaded {len(rows)} eval rows.")

        tokenizer = prepare_tokenizer(AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=TRUST_REMOTE_CODE))

        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            torch_dtype=dtype,
            trust_remote_code=TRUST_REMOTE_CODE,
            low_cpu_mem_usage=True,
        ).to(device)
        model.eval()

        if world_size > 1:
            dist.barrier()
        rank0_print(rank, "\n[1/4] Benchmarking dense model...")
        dense_metrics = benchmark_exact(model, tokenizer, rows, device, rank, world_size, tag="dense")

        if rank == 0:
            dense_public = {k: v for k, v in dense_metrics.items() if k != "predictions"}
            rank0_print(rank, json.dumps(dense_public, indent=2, ensure_ascii=False))
            if SAVE_PREDICTIONS:
                save_jsonl(out_dir / "predictions_dense.jsonl", dense_metrics["predictions"])

        if world_size > 1:
            dist.barrier()
        rank0_print(rank, "\n[2/4] Applying exact NVIDIA 2:4 pruning on each rank...")
        prune_report, module_rows = prune_model_2of4(model, rank)

        pruned_dir = out_dir / "pruned_model"
        if rank == 0 and SAVE_PRUNED_MODEL:
            rank0_print(rank, f"\n[3/4] Saving pruned model before evaluation to {pruned_dir} ...")
            model.save_pretrained(pruned_dir, safe_serialization=True)
            tokenizer.save_pretrained(pruned_dir)
            save_json(out_dir / "nvidia_2of4_pruning_report.json", prune_report)
            save_csv(out_dir / "sparsity_by_module.csv", module_rows)

        if world_size > 1:
            dist.barrier()

        if SAVE_PRUNED_MODEL:
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
            rank0_print(rank, f"\n[4/4] Reloading saved pruned checkpoint for evaluation: {pruned_dir}")
            tokenizer = prepare_tokenizer(AutoTokenizer.from_pretrained(str(pruned_dir), trust_remote_code=TRUST_REMOTE_CODE))
            model = AutoModelForCausalLM.from_pretrained(
                str(pruned_dir),
                torch_dtype=dtype,
                trust_remote_code=TRUST_REMOTE_CODE,
                low_cpu_mem_usage=True,
            ).to(device)
            model.eval()

        rank0_print(rank, "\n[4/4] Benchmarking reloaded pruned model...")
        pruned_metrics = benchmark_exact(model, tokenizer, rows, device, rank, world_size, tag="nvidia_2of4")

        if rank == 0:
            pruned_public = {k: v for k, v in pruned_metrics.items() if k != "predictions"}
            pruned_public["checkpoint_evaluated"] = str(pruned_dir)
            rank0_print(rank, json.dumps(pruned_public, indent=2, ensure_ascii=False))

            summary = {
                "model_path": str(model_path),
                "eval_file": str(eval_path),
                "output_dir": str(out_dir),
                "world_size": world_size,
                "batch_size_per_gpu": BATCH_SIZE,
                "max_input_tokens": MAX_INPUT_TOKENS,
                "max_new_tokens": MAX_NEW_TOKENS,
                "dense": dense_public,
                "nvidia_2of4": pruned_public,
                "pruning": prune_report,
                "checkpoint_evaluated": str(pruned_dir),
            }

            save_json(out_dir / "benchmark_summary.json", summary)
            if not SAVE_PRUNED_MODEL:
                save_json(out_dir / "nvidia_2of4_pruning_report.json", prune_report)
                save_csv(out_dir / "sparsity_by_module.csv", module_rows)
            if SAVE_PREDICTIONS:
                save_jsonl(out_dir / "predictions_pruned.jsonl", pruned_metrics["predictions"])

            rows_for_csv = [
                {
                    "tag": "dense",
                    "num_examples": dense_public.get("num_examples"),
                    "target_count": dense_public.get("target_count"),
                    "correct": dense_public.get("correct"),
                    "exact_match_accuracy": dense_public.get("exact_match_accuracy"),
                    "elapsed_seconds": dense_public.get("elapsed_seconds"),
                    "generated_tokens_per_second": dense_public.get("generated_tokens_per_second"),
                    "whole_model_sparsity": prune_report["whole_model_before"]["sparsity"],
                    "selected_linear_sparsity": 0.0,
                },
                {
                    "tag": "nvidia_2of4",
                    "num_examples": pruned_public.get("num_examples"),
                    "target_count": pruned_public.get("target_count"),
                    "correct": pruned_public.get("correct"),
                    "exact_match_accuracy": pruned_public.get("exact_match_accuracy"),
                    "elapsed_seconds": pruned_public.get("elapsed_seconds"),
                    "generated_tokens_per_second": pruned_public.get("generated_tokens_per_second"),
                    "whole_model_sparsity": prune_report["whole_model_after"]["sparsity"],
                    "selected_linear_sparsity": prune_report["selected_linear_mask_sparsity"],
                },
            ]
            save_csv(out_dir / "benchmark_summary.csv", rows_for_csv)

            rank0_print(rank, "\nNVIDIA 2:4 pruning + exact benchmark complete")
            rank0_print(rank, f"Output directory: {out_dir}")
            rank0_print(rank, f"Pruned model:      {pruned_dir}")
            rank0_print(rank, f"Dense exact:       {format_metric(dense_public.get('exact_match_accuracy'))}")
            rank0_print(rank, f"Pruned exact:      {format_metric(pruned_public.get('exact_match_accuracy'))}")
            rank0_print(rank, f"Real sparsity:     {format_metric(prune_report['whole_model_after']['sparsity'])}")
            rank0_print(rank, f"Linear sparsity:   {format_metric(prune_report['selected_linear_mask_sparsity'])}")
            rank0_print(rank, f"Valid 2:4:         {prune_report['valid_pruned_weights']}")
            rank0_print(rank, f"Main summary:      {out_dir / 'benchmark_summary.json'}")

    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
