#!/usr/bin/env python3
"""
Standalone 8-GPU magnitude-pruning + exact benchmark script for decoder-only HF models.

User command, only two paths:
    python magnitude_prune_8gpu_exact.py /path/to/local_model /path/to/eval.json

What it does:
    1. Auto-relaunches itself with torchrun using up to 8 GPUs.
    2. Loads a local decoder-only model with AutoModelForCausalLM.
    3. Benchmarks dense model on the eval set.
    4. Applies 50% per-layer magnitude pruning to selected nn.Linear weights.
    5. Benchmarks the pruned model on the same eval set.
    6. Saves the pruned checkpoint and reports next to the model directory.

Supported eval formats:
    .jsonl, .json, .csv, .txt

Prompt/response field names accepted:
    prompt/instruction/input/question/query/command/source
    response/output/answer/target/label/completion

Text-only field names accepted for LM perplexity:
    text/content
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
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# -----------------------------
# Fixed experiment defaults
# -----------------------------
SPARSITY = 0.50
MODEL_PATH = ""  # Optional: set /path/to/local_model here, then run with no path args.
EVAL_DATASET_PATH = ""  # Optional: set /path/to/eval.json here, then run with no path args.
MAX_SEQ_LEN = 2048
MAX_NEW_TOKENS = 64
BATCH_SIZE = 4
DTYPE = "bf16"  # bf16, fp16, fp32, auto
TRUST_REMOTE_CODE = True
COMPARISON_MODE = "whitespace"  # exact, whitespace, lower_whitespace
SEED = 42
MAX_GPUS = 8

PROMPT_KEYS = ("prompt", "instruction", "input", "question", "query", "command", "source")
RESPONSE_KEYS = ("response", "output", "answer", "target", "label", "completion")
TEXT_KEYS = ("text", "content")


@dataclass
class EvalExample:
    idx: int
    prompt: Optional[str]
    response: Optional[str]
    text: Optional[str]


def is_worker_process() -> bool:
    return "LOCAL_RANK" in os.environ or "RANK" in os.environ


def resolve_run_paths(script_name: str) -> Tuple[Path, Path]:
    if len(sys.argv) == 3:
        model_value = sys.argv[1]
        eval_value = sys.argv[2]
    elif len(sys.argv) == 1 and MODEL_PATH.strip() and EVAL_DATASET_PATH.strip():
        model_value = MODEL_PATH
        eval_value = EVAL_DATASET_PATH
    else:
        print(
            f"Usage: python {script_name} /path/to/local_model /path/to/eval_file\n"
            "Or edit MODEL_PATH and EVAL_DATASET_PATH at the top of this script, then run it with no path args.",
            file=sys.stderr,
        )
        sys.exit(2)
    return Path(model_value).expanduser().resolve(), Path(eval_value).expanduser().resolve()


def auto_launch_if_needed() -> None:
    """Relaunch with torchrun while keeping the user command to only two paths."""
    if is_worker_process():
        return

    model_path, eval_path = resolve_run_paths("magnitude_prune_8gpu_exact.py")

    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    nproc = min(MAX_GPUS, gpu_count) if gpu_count > 0 else 1

    if nproc <= 1:
        return

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
    print(f"[launcher] Starting distributed run with {nproc} GPU processes")
    print("[launcher] " + " ".join(cmd))
    raise SystemExit(subprocess.call(cmd))


def setup_distributed() -> Tuple[int, int, int, torch.device]:
    if is_worker_process():
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    else:
        rank = 0
        world_size = 1
        local_rank = 0

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    return rank, world_size, local_rank, device


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def is_rank0(rank: int) -> bool:
    return rank == 0


def log(rank: int, msg: str) -> None:
    if is_rank0(rank):
        print(msg, flush=True)


def parse_dtype(dtype_name: str):
    if dtype_name == "auto":
        return "auto"
    if dtype_name == "bf16":
        return torch.bfloat16
    if dtype_name == "fp16":
        return torch.float16
    if dtype_name == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def make_output_dir(model_path: Path) -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return model_path.parent / f"{model_path.name}_magnitude50_8gpu_exact_{stamp}"


def normalize_text(x: str, mode: str = COMPARISON_MODE) -> str:
    x = "" if x is None else str(x)
    if mode == "exact":
        return x
    if mode == "whitespace":
        return re.sub(r"\s+", "", x)
    if mode == "lower_whitespace":
        return re.sub(r"\s+", "", x).lower()
    raise ValueError(f"Unsupported comparison mode: {mode}")


def find_first(row: Dict[str, Any], keys: Iterable[str]) -> Optional[str]:
    for key in keys:
        if key in row and row[key] is not None:
            value = row[key]
            if isinstance(value, (dict, list)):
                return json.dumps(value, ensure_ascii=False)
            return str(value)
    return None


def load_eval_file(path: Path) -> List[EvalExample]:
    suffix = path.suffix.lower()
    raw_rows: List[Any]

    if suffix == ".jsonl":
        raw_rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    raw_rows.append(json.loads(line))
    elif suffix == ".json":
        obj = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(obj, list):
            raw_rows = obj
        elif isinstance(obj, dict):
            for key in ("data", "examples", "eval", "test", "items"):
                if key in obj and isinstance(obj[key], list):
                    raw_rows = obj[key]
                    break
            else:
                raw_rows = [obj]
        else:
            raise ValueError(f"Unsupported JSON root type: {type(obj)}")
    elif suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as f:
            raw_rows = list(csv.DictReader(f))
    elif suffix == ".txt":
        raw_rows = [{"text": line.strip()} for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        raise ValueError(f"Unsupported eval file extension: {suffix}")

    examples: List[EvalExample] = []
    for i, row in enumerate(raw_rows):
        if not isinstance(row, dict):
            row = {"text": str(row)}
        prompt = find_first(row, PROMPT_KEYS)
        response = find_first(row, RESPONSE_KEYS)
        text = find_first(row, TEXT_KEYS)
        if prompt is not None and response is not None:
            examples.append(EvalExample(idx=i, prompt=prompt, response=response, text=None))
        elif text is not None:
            examples.append(EvalExample(idx=i, prompt=None, response=None, text=text))
        elif prompt is not None:
            examples.append(EvalExample(idx=i, prompt=None, response=None, text=prompt))

    if not examples:
        raise ValueError("No usable eval examples found. Need prompt/response pairs or text rows.")
    return examples


class EvalDataset(Dataset):
    def __init__(self, examples: List[EvalExample]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> EvalExample:
        return self.examples[idx]


def collate_examples(batch: List[EvalExample]) -> List[EvalExample]:
    return batch


def shard_examples(examples: List[EvalExample], rank: int, world_size: int) -> List[EvalExample]:
    return [ex for j, ex in enumerate(examples) if j % world_size == rank]


def prepare_tokenizer(tokenizer):
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    tokenizer.padding_side = "left"
    return tokenizer


def build_lm_batch(tokenizer, batch: List[EvalExample], device: torch.device) -> Dict[str, torch.Tensor]:
    texts: List[str] = []
    prompt_lengths: List[int] = []

    for ex in batch:
        if ex.prompt is not None and ex.response is not None:
            prompt = str(ex.prompt)
            response = str(ex.response)
            text = prompt + response
            prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
            prompt_lengths.append(len(prompt_ids))
            texts.append(text)
        else:
            texts.append(str(ex.text))
            prompt_lengths.append(0)

    enc = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_SEQ_LEN,
        add_special_tokens=True,
    )

    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100

    # Mask prompt tokens for prompt-response examples.
    # With left padding, content begins after pad_count.
    for i, ex in enumerate(batch):
        if ex.prompt is not None and ex.response is not None:
            pad_count = int((attention_mask[i] == 0).sum().item())
            end = min(labels.shape[1], pad_count + prompt_lengths[i])
            labels[i, :end] = -100

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def all_gather_list(obj: Any, world_size: int) -> List[Any]:
    if world_size == 1:
        return [obj]
    gathered = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, obj)
    return gathered


def benchmark_model(
    model,
    tokenizer,
    examples: List[EvalExample],
    rank: int,
    world_size: int,
    device: torch.device,
    tag: str,
    output_dir: Path,
) -> Dict[str, Any]:
    model.eval()
    local_examples = shard_examples(examples, rank, world_size)
    loader = DataLoader(EvalDataset(local_examples), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_examples)

    total_loss_sum = 0.0
    total_label_tokens = 0
    gen_records: List[Dict[str, Any]] = []
    correct = 0
    exact_total = 0

    start = time.time()
    with torch.inference_mode():
        for batch in loader:
            lm_batch = build_lm_batch(tokenizer, batch, device)
            outputs = model(**lm_batch)
            labels = lm_batch["labels"]
            label_tokens = int((labels != -100).sum().item())
            if label_tokens > 0 and outputs.loss is not None:
                total_loss_sum += float(outputs.loss.item()) * label_tokens
                total_label_tokens += label_tokens

            prompt_response_batch = [ex for ex in batch if ex.prompt is not None and ex.response is not None]
            if prompt_response_batch:
                prompts = [str(ex.prompt) for ex in prompt_response_batch]
                enc = tokenizer(
                    prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=MAX_SEQ_LEN,
                    add_special_tokens=True,
                ).to(device)
                generated = model.generate(
                    **enc,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    num_beams=1,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
                input_len = enc["input_ids"].shape[1]
                new_tokens = generated[:, input_len:]
                preds = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
                for ex, pred in zip(prompt_response_batch, preds):
                    gold = str(ex.response)
                    is_correct = normalize_text(pred) == normalize_text(gold)
                    correct += int(is_correct)
                    exact_total += 1
                    gen_records.append(
                        {
                            "idx": ex.idx,
                            "prompt": ex.prompt,
                            "gold": gold,
                            "prediction": pred,
                            "correct": bool(is_correct),
                        }
                    )

    elapsed = time.time() - start
    local_payload = {
        "loss_sum": total_loss_sum,
        "label_tokens": total_label_tokens,
        "correct": correct,
        "exact_total": exact_total,
        "elapsed": elapsed,
        "records": gen_records,
    }
    gathered = all_gather_list(local_payload, world_size)

    if not is_rank0(rank):
        return {}

    loss_sum = sum(float(x["loss_sum"]) for x in gathered)
    label_tokens = sum(int(x["label_tokens"]) for x in gathered)
    correct_sum = sum(int(x["correct"]) for x in gathered)
    exact_total_sum = sum(int(x["exact_total"]) for x in gathered)
    max_elapsed = max(float(x["elapsed"]) for x in gathered) if gathered else elapsed
    records: List[Dict[str, Any]] = []
    for x in gathered:
        records.extend(x["records"])
    records.sort(key=lambda r: r["idx"])

    avg_loss = loss_sum / float(label_tokens or 1)
    ppl = math.exp(avg_loss) if avg_loss < 80 else float("inf")
    exact_acc = correct_sum / float(exact_total_sum or 1) if exact_total_sum else None

    pred_path = output_dir / f"predictions_{tag}.jsonl"
    with pred_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return {
        "tag": tag,
        "num_examples": len(examples),
        "exact_match_examples": exact_total_sum,
        "exact_match_correct": correct_sum,
        "exact_match_accuracy": exact_acc,
        "avg_loss": avg_loss,
        "perplexity": ppl,
        "label_tokens": label_tokens,
        "elapsed_seconds": max_elapsed,
        "examples_per_second": len(examples) / float(max_elapsed or 1),
        "tokens_per_second": label_tokens / float(max_elapsed or 1),
        "prediction_file": str(pred_path),
    }


def module_sparsity_row(name: str, m: nn.Linear, mask: torch.Tensor) -> Dict[str, Any]:
    w = m.weight.detach()
    total = int(w.numel())
    zeros = int((w == 0).sum().item())
    pruned_by_mask = int((~mask).sum().item())
    return {
        "module": name,
        "shape": list(w.shape),
        "parameters": total,
        "zero_parameters": zeros,
        "actual_sparsity": zeros / float(total or 1),
        "mask_pruned_parameters": pruned_by_mask,
        "mask_sparsity": pruned_by_mask / float(total or 1),
    }


def parameter_zero_stats(model: nn.Module) -> Dict[str, Any]:
    total = 0
    zeros = 0
    for p in model.parameters():
        d = p.detach()
        total += int(d.numel())
        zeros += int((d == 0).sum().item())
    return {
        "total_parameters": total,
        "zero_parameters": zeros,
        "nonzero_parameters": total - zeros,
        "whole_model_sparsity": zeros / float(total or 1),
    }


def magnitude_prune_per_layer(model: nn.Module, sparsity: float = SPARSITY) -> Tuple[Dict[str, torch.Tensor], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Same guideline as the uploaded magnitude script:
      - iterate over nn.Linear
      - score = abs(W)
      - keep top (1 - sparsity) weights in each Linear layer
      - zero the remaining weights
    """
    masks: Dict[str, torch.Tensor] = {}
    rows: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []

    for name, m in model.named_modules():
        if not isinstance(m, nn.Linear):
            continue
        if m.weight is None or m.weight.ndim != 2:
            skipped.append({"module": name, "reason": "not_2d_linear_weight"})
            continue

        with torch.no_grad():
            W = m.weight.data
            scores = W.abs().reshape(-1)
            keep = int(scores.numel() * (1.0 - sparsity))
            if keep <= 0:
                skipped.append({"module": name, "reason": "keep_count_zero"})
                continue
            if keep >= scores.numel():
                mask = torch.ones_like(W, dtype=torch.bool)
            else:
                threshold = torch.topk(scores, keep, largest=True).values.min()
                mask = W.abs() >= threshold
            W.mul_(mask.to(dtype=W.dtype, device=W.device))

        mask_cpu = mask.detach().cpu().bool()
        masks[name] = mask_cpu
        rows.append(module_sparsity_row(name, m, mask_cpu.to(device=m.weight.device)))

    if not rows:
        raise RuntimeError("No nn.Linear layers were pruned.")
    return masks, rows, skipped


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    fields: List[str] = []
    for row in rows:
        for k in row:
            if k not in fields:
                fields.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_json(path: Path, obj: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")


def save_summary_csv(path: Path, dense: Dict[str, Any], pruned: Dict[str, Any]) -> None:
    rows = [dense, pruned]
    fields = [
        "tag",
        "num_examples",
        "exact_match_examples",
        "exact_match_correct",
        "exact_match_accuracy",
        "avg_loss",
        "perplexity",
        "label_tokens",
        "elapsed_seconds",
        "examples_per_second",
        "tokens_per_second",
        "prediction_file",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})


def format_metric(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return str(value)


def load_model_and_tokenizer(model_path: Path, device: torch.device):
    dtype = parse_dtype(DTYPE)
    kwargs: Dict[str, Any] = {"trust_remote_code": TRUST_REMOTE_CODE}
    kwargs["torch_dtype"] = dtype

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=TRUST_REMOTE_CODE)
    tokenizer = prepare_tokenizer(tokenizer)
    model = AutoModelForCausalLM.from_pretrained(str(model_path), **kwargs)
    if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    return model, tokenizer


def main_worker() -> None:
    torch.manual_seed(SEED)
    rank, world_size, local_rank, device = setup_distributed()

    model_path, eval_path = resolve_run_paths("magnitude_prune_8gpu_exact.py")

    if not model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")
    if not eval_path.exists():
        raise FileNotFoundError(f"Eval file does not exist: {eval_path}")

    output_dir = make_output_dir(model_path)
    if is_rank0(rank):
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    if world_size > 1:
        dist.barrier()

    log(rank, f"[rank0] Model: {model_path}")
    log(rank, f"[rank0] Eval:  {eval_path}")
    log(rank, f"[rank0] Output: {output_dir}")
    log(rank, f"[rank0] World size: {world_size}; local rank: {local_rank}; device: {device}")

    examples = load_eval_file(eval_path)
    if is_rank0(rank):
        save_json(
            output_dir / "eval_dataset_summary.json",
            {
                "eval_file": str(eval_path),
                "num_examples": len(examples),
                "prompt_response_examples": sum(1 for ex in examples if ex.prompt is not None and ex.response is not None),
                "text_only_examples": sum(1 for ex in examples if ex.text is not None),
            },
        )

    model, tokenizer = load_model_and_tokenizer(model_path, device)

    log(rank, "[rank0] Benchmarking dense model...")
    dense_metrics = benchmark_model(model, tokenizer, examples, rank, world_size, device, "dense", output_dir)

    before_stats = parameter_zero_stats(model)
    log(rank, "[rank0] Applying 50% per-layer magnitude pruning to nn.Linear weights...")
    masks, sparsity_rows, skipped = magnitude_prune_per_layer(model, SPARSITY)
    after_stats = parameter_zero_stats(model)

    selected_params = sum(int(r["parameters"]) for r in sparsity_rows)
    selected_zeros = sum(int(r["zero_parameters"]) for r in sparsity_rows)
    selected_pruned = sum(int(r["mask_pruned_parameters"]) for r in sparsity_rows)

    pruned_model_dir = output_dir / "pruned_model"
    if is_rank0(rank):
        log(rank, f"[rank0] Saving pruned model before evaluation: {pruned_model_dir}")
        model.save_pretrained(pruned_model_dir, safe_serialization=True)
        tokenizer.save_pretrained(pruned_model_dir)
        torch.save(masks, output_dir / "magnitude_pruning_masks.pt")
        write_csv(output_dir / "sparsity_by_module.csv", sparsity_rows)
    if world_size > 1:
        dist.barrier()

    log(rank, "[rank0] Benchmarking magnitude-pruned model...")
    pruned_metrics = benchmark_model(model, tokenizer, examples, rank, world_size, device, "pruned", output_dir)

    if is_rank0(rank):
        report = {
            "method": "magnitude_per_layer_decoder_only",
            "model_path": str(model_path),
            "eval_file": str(eval_path),
            "output_dir": str(output_dir),
            "sparsity_target": SPARSITY,
            "scope": "selected torch.nn.Linear weights in AutoModelForCausalLM",
            "rule": "per Linear layer, score=abs(weight), keep top 50%, zero remaining 50%",
            "selected_linear_modules": len(sparsity_rows),
            "skipped_linear_modules": skipped,
            "selected_linear_parameters": selected_params,
            "selected_linear_zero_parameters_after_prune": selected_zeros,
            "selected_linear_actual_sparsity": selected_zeros / float(selected_params or 1),
            "selected_linear_mask_pruned_parameters": selected_pruned,
            "selected_linear_mask_sparsity": selected_pruned / float(selected_params or 1),
            "before": before_stats,
            "after": after_stats,
            "notes": [
                "This follows the uploaded magnitude guideline but uses AutoModelForCausalLM instead of T5ForConditionalGeneration.",
                "Sparsity is exactly targeted per selected Linear layer; whole-model sparsity can be lower because embeddings/norms are not pruned.",
                "This creates dense tensors with zeros; runtime speedup requires sparse-aware kernels or export/runtime support.",
            ],
        }

        save_json(output_dir / "magnitude_pruning_report.json", report)
        save_json(output_dir / "benchmark_summary.json", {"dense": dense_metrics, "pruned": pruned_metrics})
        save_summary_csv(output_dir / "benchmark_summary.csv", dense_metrics, pruned_metrics)

        print("\nMagnitude pruning + exact benchmark complete")
        print(f"Output directory: {output_dir}")
        print(f"Pruned model:      {pruned_model_dir}")
        print(f"Dense exact:       {format_metric(dense_metrics.get('exact_match_accuracy'))}")
        print(f"Pruned exact:      {format_metric(pruned_metrics.get('exact_match_accuracy'))}")
        print(f"Real sparsity:     {format_metric(report['after']['whole_model_sparsity'])}")
        print(f"Linear sparsity:   {format_metric(report['selected_linear_mask_sparsity'])}")

    cleanup_distributed()


if __name__ == "__main__":
    auto_launch_if_needed()
    main_worker()
