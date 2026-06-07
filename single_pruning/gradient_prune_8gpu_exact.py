#!/usr/bin/env python3
"""
Standalone 8-GPU gradient/Taylor pruning + exact benchmark script for a local decoder-only
Hugging Face causal-LM checkpoint.

Run with exactly two paths:

    python gradient_prune_8gpu_exact.py /path/to/local_model /path/to/eval.json

The script auto-launches torchrun with up to 8 visible GPUs. It then:
  1. benchmarks the dense model on the eval file;
  2. computes first-order Taylor saliency |W * grad| on the eval/calibration set;
  3. prunes 50% of each eligible nn.Linear weight matrix by lowest saliency;
  4. benchmarks the pruned model on the same eval file;
  5. saves the pruned model and reports next to the model directory.

Expected eval formats:
  JSON/JSONL/CSV with any of these prompt fields: prompt, instruction, input, question, query
  and any of these response fields: response, output, answer, target, completion.

Plain text files are also accepted, but exact-match generation requires prompt/response fields.
"""

from __future__ import annotations

import csv
import datetime as _dt
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from chatlm_decoder.tokenizer import move_batch_to_device, prepare_decoder_tokenizer

# =========================
# Fixed experiment settings
# =========================
SPARSITY = 0.50
MODEL_PATH = ""  # Optional: set /path/to/local_model here, then run with no path args.
EVAL_DATASET_PATH = ""  # Optional: set /path/to/eval.json here, then run with no path args.
MAX_GPUS = 8
BATCH_SIZE = 4
MAX_PROMPT_LEN = 512
MAX_RESPONSE_LEN = 128
MAX_NEW_TOKENS = 128
DTYPE = "bf16"          # auto | bf16 | fp16 | fp32
TRUST_REMOTE_CODE = True
OVERWRITE_OUTPUT = True
INCLUDE_LM_HEAD = False
PROMPT_FORMAT = "plain" # plain | repo | chat
COMPARISON_MODE = "whitespace"  # exact | whitespace
NUM_WORKERS = 0
SEED = 42

PROMPT_KEYS = ("prompt", "instruction", "input", "question", "query")
RESPONSE_KEYS = ("response", "output", "answer", "target", "completion")
TEXT_KEYS = ("text", "content")
LEADING_MARKERS = (
    "<|assistant|>",
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
STOP_MARKERS = (
    "<|user|>",
    "<|system|>",
    "<|eos|>",
    "<|im_start|>user",
    "<|im_start|>system",
    "<|im_end|>",
    "\nUser:",
    "\n用户:",
)


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
    """Relaunch this script with torchrun when multiple GPUs are visible."""
    if os.environ.get("LOCAL_RANK") is not None:
        return

    model_path, eval_path = resolve_run_paths("gradient_prune_8gpu_exact.py")

    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    nproc = min(MAX_GPUS, gpu_count)

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
    print(f"Auto-launching distributed run on {nproc} GPUs:")
    print(" ".join(cmd))
    raise SystemExit(subprocess.call(cmd))


def setup_distributed() -> Tuple[bool, int, int, int, torch.device]:
    if os.environ.get("LOCAL_RANK") is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return False, 0, 0, 1, device

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return True, rank, local_rank, world_size, torch.device(f"cuda:{local_rank}")


def prepare_tokenizer(tokenizer: Any) -> Any:
    prepare_decoder_tokenizer(tokenizer)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token is not None else tokenizer.unk_token
    tokenizer.padding_side = "left"
    return tokenizer


def cleanup_distributed(is_dist: bool) -> None:
    if is_dist and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def is_main(rank: int) -> bool:
    return rank == 0


def log(rank: int, msg: str) -> None:
    if is_main(rank):
        print(msg, flush=True)


def dtype_from_string(dtype: str):
    if dtype == "auto":
        return "auto"
    if dtype == "bf16":
        return torch.bfloat16
    if dtype == "fp16":
        return torch.float16
    if dtype == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype}")


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def strip_wrapping_quotes(text: str) -> str:
    text = text.strip()
    changed = True
    while changed and len(text) >= 2:
        changed = False
        for left, right in (("`", "`"), ('"', '"'), ("'", "'")):
            if text.startswith(left) and text.endswith(right):
                text = text[1:-1].strip()
                changed = True
                break
    return text


def clean_prediction_text(text: str) -> str:
    text = "" if text is None else str(text)
    for marker in STOP_MARKERS:
        if marker in text:
            text = text.split(marker, 1)[0]
    changed = True
    while changed:
        changed = False
        stripped = text.strip()
        for marker in LEADING_MARKERS:
            if stripped.startswith(marker):
                text = stripped[len(marker):]
                changed = True
                break
    return strip_wrapping_quotes(text)


def normalize_text(text: str, mode: str) -> str:
    text = clean_prediction_text("" if text is None else str(text))
    if mode == "exact":
        return text.strip()
    if mode == "whitespace":
        return "".join(text.split())
    raise ValueError(f"Unsupported comparison mode: {mode}")


def first_present(d: Dict[str, Any], keys: Iterable[str]) -> Optional[Any]:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def load_records(path: Path) -> List[Dict[str, Any]]:
    suffix = path.suffix.lower()
    records: List[Dict[str, Any]] = []

    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    obj = json.loads(line)
                    records.append(obj if isinstance(obj, dict) else {"text": str(obj)})
    elif suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, list):
            records = [x if isinstance(x, dict) else {"text": str(x)} for x in obj]
        elif isinstance(obj, dict):
            for key in ("data", "examples", "items", "records"):
                if key in obj and isinstance(obj[key], list):
                    records = [x if isinstance(x, dict) else {"text": str(x)} for x in obj[key]]
                    break
            if not records:
                records = [obj]
        else:
            raise ValueError(f"Unsupported JSON root type: {type(obj)}")
    elif suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as f:
            records = list(csv.DictReader(f))
    elif suffix in {".txt", ".text"}:
        with path.open("r", encoding="utf-8") as f:
            records = [{"text": line.strip()} for line in f if line.strip()]
    else:
        raise ValueError(f"Unsupported eval file suffix: {suffix}. Use json/jsonl/csv/txt.")

    cleaned: List[Dict[str, Any]] = []
    for i, r in enumerate(records):
        prompt = first_present(r, PROMPT_KEYS)
        response = first_present(r, RESPONSE_KEYS)
        text = first_present(r, TEXT_KEYS)

        if prompt is not None and response is not None:
            cleaned.append({"id": r.get("id", i), "prompt": str(prompt), "response": str(response)})
        elif text is not None:
            cleaned.append({"id": r.get("id", i), "text": str(text)})
        else:
            raise ValueError(
                f"Record {i} lacks prompt/response or text fields. Keys found: {list(r.keys())}"
            )

    if not cleaned:
        raise ValueError(f"No usable records found in {path}")
    return cleaned


def format_prompt(prompt: str, tokenizer: Any) -> str:
    if PROMPT_FORMAT == "plain":
        return prompt
    if PROMPT_FORMAT == "repo":
        return f"Instruction: {prompt}\nResponse:"
    if PROMPT_FORMAT == "chat":
        messages = [{"role": "user", "content": prompt}]
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return f"User: {prompt}\nAssistant:"
    raise ValueError(f"Unsupported PROMPT_FORMAT: {PROMPT_FORMAT}")


class CausalEvalDataset(Dataset):
    def __init__(self, records: List[Dict[str, Any]], tokenizer: Any):
        self.records = records
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r = self.records[idx]
        item: Dict[str, Any] = {"id": r.get("id", idx)}
        if "prompt" in r and "response" in r:
            prompt_text = format_prompt(r["prompt"], self.tokenizer)
            response_text = r["response"]
            item.update({"prompt": prompt_text, "raw_prompt": r["prompt"], "response": response_text})
        else:
            item.update({"text": r["text"]})
        return item


def collate_generation(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "ids": [x["id"] for x in batch],
        "prompts": [x.get("prompt") for x in batch],
        "raw_prompts": [x.get("raw_prompt") for x in batch],
        "responses": [x.get("response") for x in batch],
        "texts": [x.get("text") for x in batch],
    }


def make_lm_batch(batch: Dict[str, Any], tokenizer: Any, device: torch.device) -> Dict[str, torch.Tensor]:
    input_ids_list: List[torch.Tensor] = []
    labels_list: List[torch.Tensor] = []

    eos = tokenizer.eos_token or ""

    for prompt, response, text in zip(batch["prompts"], batch["responses"], batch["texts"]):
        if prompt is not None and response is not None:
            prompt_ids = tokenizer(
                prompt,
                add_special_tokens=False,
                truncation=True,
                max_length=MAX_PROMPT_LEN,
                return_tensors="pt",
            )["input_ids"][0]
            response_ids = tokenizer(
                response + eos,
                add_special_tokens=False,
                truncation=True,
                max_length=MAX_RESPONSE_LEN,
                return_tensors="pt",
            )["input_ids"][0]
            ids = torch.cat([prompt_ids, response_ids], dim=0)
            labels = ids.clone()
            labels[: prompt_ids.numel()] = -100
        else:
            ids = tokenizer(
                text,
                add_special_tokens=True,
                truncation=True,
                max_length=MAX_PROMPT_LEN + MAX_RESPONSE_LEN,
                return_tensors="pt",
            )["input_ids"][0]
            labels = ids.clone()

        input_ids_list.append(ids)
        labels_list.append(labels)

    max_len = max(x.numel() for x in input_ids_list)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    input_ids = torch.full((len(input_ids_list), max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(input_ids_list), max_len), dtype=torch.long)
    labels = torch.full((len(input_ids_list), max_len), -100, dtype=torch.long)

    for i, (ids, labs) in enumerate(zip(input_ids_list, labels_list)):
        n = ids.numel()
        input_ids[i, :n] = ids
        attention_mask[i, :n] = 1
        labels[i, :n] = labs

    return {
        "input_ids": input_ids.to(device),
        "attention_mask": attention_mask.to(device),
        "labels": labels.to(device),
    }


def all_gather_objects(obj: Any, is_dist: bool, world_size: int) -> List[Any]:
    if not is_dist:
        return [obj]
    gathered = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, obj)
    return gathered


def flatten(list_of_lists: Iterable[Any]) -> List[Any]:
    out: List[Any] = []
    for x in list_of_lists:
        if isinstance(x, list):
            out.extend(x)
        else:
            out.append(x)
    return out


def exact_match_diagnostics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    predictions = [str(row.get("prediction", "")) for row in records]
    normalized_predictions = [str(row.get("normalized_prediction", "")) for row in records]
    prompts = [str(row.get("prompt", "")) for row in records]
    nonempty_predictions = [text for text in predictions if text.strip()]
    prompt_copy_count = sum(
        1
        for prompt, prediction in zip(prompts, predictions)
        if prompt.strip()
        and prediction.strip()
        and (prediction.strip() in prompt.strip() or prediction.strip().startswith(prompt.strip()))
    )
    return {
        "empty_predictions": len(predictions) - len(nonempty_predictions),
        "unique_normalized_predictions": len(set(normalized_predictions)),
        "prompt_copy_predictions": prompt_copy_count,
        "total_generation_records": len(records),
    }


def generation_warnings(tag: str, diagnostics: Dict[str, Any]) -> List[str]:
    total = int(diagnostics.get("total_generation_records", 0))
    if total <= 0:
        return []
    warnings: List[str] = []
    if int(diagnostics.get("empty_predictions", 0)) == total:
        warnings.append(f"{tag}: all generated predictions are empty; check tokenizer/EOS/max_new_tokens.")
    if int(diagnostics.get("unique_normalized_predictions", 0)) <= 1 and total > 1:
        warnings.append(f"{tag}: all generated predictions are identical; check checkpoint/generation settings.")
    if int(diagnostics.get("prompt_copy_predictions", 0)) / float(total) > 0.8:
        warnings.append(f"{tag}: most generations look like prompt copies; check response slicing/prompt format.")
    return warnings


def evaluate(
    model: nn.Module,
    tokenizer: Any,
    dataset: CausalEvalDataset,
    rank: int,
    world_size: int,
    is_dist: bool,
    device: torch.device,
    tag: str,
) -> Tuple[Optional[Dict[str, Any]], Optional[List[Dict[str, Any]]]]:
    local_records = dataset.records[rank::world_size] if is_dist else dataset.records
    local_dataset = CausalEvalDataset(local_records, tokenizer)
    loader = DataLoader(
        local_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_generation,
    )

    model.eval()
    total_loss_sum = 0.0
    total_label_tokens = 0
    total_examples = 0
    exact_correct = 0
    exact_total = 0
    total_gen_tokens = 0
    predictions: List[Dict[str, Any]] = []
    start = time.perf_counter()

    has_prompt_response = any("prompt" in r and "response" in r for r in dataset.records)

    with torch.no_grad():
        for batch in loader:
            lm_batch = make_lm_batch(batch, tokenizer, device)
            outputs = model(**lm_batch)
            valid_tokens = int((lm_batch["labels"] != -100).sum().item())
            loss_value = float(outputs.loss.detach().float().item()) if outputs.loss is not None else float("nan")
            total_loss_sum += loss_value * valid_tokens
            total_label_tokens += valid_tokens
            total_examples += int(lm_batch["input_ids"].shape[0])

            prompt_indices = [
                i
                for i, (prompt, response) in enumerate(zip(batch["prompts"], batch["responses"]))
                if prompt is not None and response is not None
            ]
            if has_prompt_response and prompt_indices:
                prompts = [batch["prompts"][i] for i in prompt_indices]
                enc = move_batch_to_device(tokenizer(
                    prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=MAX_PROMPT_LEN,
                ), device)
                gen = model.generate(
                    **enc,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    num_beams=1,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

                prompt_width = int(enc["input_ids"].shape[1])
                for gen_i, batch_i in enumerate(prompt_indices):
                    generated_ids = gen[gen_i, prompt_width:]
                    pred = tokenizer.decode(generated_ids, skip_special_tokens=True)
                    gold = batch["responses"][batch_i]
                    normalized_prediction = normalize_text(pred, COMPARISON_MODE)
                    normalized_gold = normalize_text(gold, COMPARISON_MODE)
                    ok = normalized_prediction == normalized_gold
                    exact_correct += int(ok)
                    exact_total += 1
                    total_gen_tokens += int(generated_ids.numel())
                    predictions.append(
                        {
                            "tag": tag,
                            "id": batch["ids"][batch_i],
                            "prompt": batch["raw_prompts"][batch_i],
                            "target": gold,
                            "prediction": pred,
                            "normalized_prediction": normalized_prediction,
                            "normalized_gold": normalized_gold,
                            "exact_match": bool(ok),
                        }
                    )

    elapsed = time.perf_counter() - start
    local = {
        "loss_sum": total_loss_sum,
        "label_tokens": total_label_tokens,
        "examples": total_examples,
        "exact_correct": exact_correct,
        "exact_total": exact_total,
        "gen_tokens": total_gen_tokens,
        "elapsed_sec": elapsed,
    }

    gathered_stats = all_gather_objects(local, is_dist, world_size)
    gathered_predictions = all_gather_objects(predictions, is_dist, world_size)

    if not is_main(rank):
        return None, None

    merged_preds = flatten(gathered_predictions)
    merged_preds.sort(key=lambda row: str(row.get("id", "")))
    diagnostics = exact_match_diagnostics(merged_preds)
    warnings = generation_warnings(tag, diagnostics)
    loss_sum = sum(x["loss_sum"] for x in gathered_stats)
    label_tokens = sum(x["label_tokens"] for x in gathered_stats)
    examples = sum(x["examples"] for x in gathered_stats)
    exact_correct_sum = sum(x["exact_correct"] for x in gathered_stats)
    exact_total_sum = sum(x["exact_total"] for x in gathered_stats)
    gen_tokens = sum(x["gen_tokens"] for x in gathered_stats)
    # Use max elapsed across ranks as wall time approximation.
    wall_time = max(x["elapsed_sec"] for x in gathered_stats)

    avg_loss = loss_sum / max(label_tokens, 1)
    metrics = {
        "tag": tag,
        "examples": int(examples),
        "label_tokens": int(label_tokens),
        "loss": avg_loss,
        "perplexity": math.exp(avg_loss) if avg_loss < 50 else float("inf"),
        "exact_match_accuracy": exact_correct_sum / exact_total_sum if exact_total_sum else None,
        "exact_correct": int(exact_correct_sum),
        "exact_total": int(exact_total_sum),
        "generated_tokens": int(gen_tokens),
        "wall_time_sec": wall_time,
        "examples_per_sec": examples / wall_time if wall_time > 0 else None,
        "generated_tokens_per_sec": gen_tokens / wall_time if wall_time > 0 and gen_tokens else None,
        "exact_match_diagnostics": diagnostics,
        "generation_warnings": warnings,
    }
    return metrics, merged_preds


def is_lm_head_name(name: str) -> bool:
    leaf = name.lower().split(".")[-1]
    return leaf in {"lm_head", "output_head"} or "lm_head" in name.lower()


def eligible_linear_modules(model: nn.Module) -> List[Tuple[str, nn.Linear]]:
    modules: List[Tuple[str, nn.Linear]] = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if not INCLUDE_LM_HEAD and is_lm_head_name(name):
                continue
            modules.append((name, module))
    return modules


def parameter_zero_stats(model: nn.Module) -> Dict[str, Any]:
    total = 0
    zeros = 0
    for p in model.parameters():
        data = p.detach()
        total += int(data.numel())
        zeros += int((data == 0).sum().item())
    return {
        "total_parameters": total,
        "zero_parameters": zeros,
        "nonzero_parameters": total - zeros,
        "whole_model_sparsity": zeros / float(total or 1),
    }


def compute_gradient_saliency(
    model: nn.Module,
    tokenizer: Any,
    dataset: CausalEvalDataset,
    rank: int,
    world_size: int,
    is_dist: bool,
    device: torch.device,
) -> Tuple[List[Tuple[str, nn.Linear]], Dict[str, torch.Tensor]]:
    modules = eligible_linear_modules(model)
    saliency: Dict[str, torch.Tensor] = {
        name: torch.zeros_like(module.weight.detach(), dtype=torch.float32, device=device)
        for name, module in modules
    }

    local_records = dataset.records[rank::world_size] if is_dist else dataset.records
    local_dataset = CausalEvalDataset(local_records, tokenizer)
    loader = DataLoader(
        local_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_generation,
    )

    model.train()
    model.zero_grad(set_to_none=True)

    for step, batch in enumerate(loader, start=1):
        lm_batch = make_lm_batch(batch, tokenizer, device)
        outputs = model(**lm_batch)
        loss = outputs.loss
        if loss is None:
            raise RuntimeError("Model did not return loss. Check labels construction.")
        loss.backward()

        with torch.no_grad():
            for name, module in modules:
                if module.weight.grad is not None:
                    saliency[name].add_((module.weight.detach().float() * module.weight.grad.detach().float()).abs())

        model.zero_grad(set_to_none=True)

        if step % 25 == 0:
            log(rank, f"  saliency step {step}/{len(loader)}")

    if is_dist:
        for tensor in saliency.values():
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    return modules, saliency


def apply_per_layer_gradient_prune(
    modules: List[Tuple[str, nn.Linear]],
    saliency: Dict[str, torch.Tensor],
    rank: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, torch.Tensor]]:
    """Per-layer 50% Taylor pruning across each selected Linear weight."""
    rows: List[Dict[str, Any]] = []
    masks: Dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for name, module in modules:
            score = saliency[name]
            flat_score = score.reshape(-1)
            keep = int(flat_score.numel() * (1.0 - SPARSITY))
            if keep <= 0:
                continue
            if keep >= flat_score.numel():
                mask = torch.ones_like(score, dtype=torch.bool)
            else:
                keep_idx = torch.topk(flat_score, keep, largest=True).indices
                mask_flat = torch.zeros_like(flat_score, dtype=torch.bool)
                mask_flat[keep_idx] = True
                mask = mask_flat.reshape_as(score)
            module.weight.mul_(mask.to(device=module.weight.device, dtype=module.weight.dtype))
            masks[name] = mask.detach().cpu()
            numel = int(module.weight.numel())
            zeros = int((module.weight.detach() == 0).sum().item())
            pruned_by_mask = int((~mask).sum().item())
            rows.append(
                {
                    "module": name,
                    "shape": list(module.weight.shape),
                    "weight_parameters": numel,
                    "mask_pruned_parameters": pruned_by_mask,
                    "mask_sparsity": pruned_by_mask / float(numel or 1),
                    "actual_zero_parameters_after_prune": zeros,
                    "actual_zero_fraction_after_prune": zeros / float(numel or 1),
                }
            )

    return rows, masks


def save_json(path: Path, obj: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")


def save_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    fields: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_output_dir(model_path: Path) -> Path:
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return model_path.parent / f"{model_path.name}_gradient50_8gpu_exact_{stamp}"


def format_metric(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return str(value)


def main_worker() -> None:
    model_path, eval_path = resolve_run_paths("gradient_prune_8gpu_exact.py")

    if not model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")
    if not eval_path.exists():
        raise FileNotFoundError(f"Eval file does not exist: {eval_path}")

    is_dist, rank, local_rank, world_size, device = setup_distributed()
    set_seed(SEED + rank)

    out_dir = make_output_dir(model_path)
    if is_main(rank):
        if out_dir.exists() and OVERWRITE_OUTPUT:
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
    if is_dist:
        dist.barrier()

    log(rank, f"Model path: {model_path}")
    log(rank, f"Eval file:  {eval_path}")
    log(rank, f"Output dir: {out_dir}")
    log(rank, f"World size: {world_size}; batch_size_per_gpu: {BATCH_SIZE}; effective_batch_size: {BATCH_SIZE * world_size}")

    tokenizer = prepare_tokenizer(AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=TRUST_REMOTE_CODE))

    load_kwargs: Dict[str, Any] = {"trust_remote_code": TRUST_REMOTE_CODE, "torch_dtype": dtype_from_string(DTYPE)}
    model = AutoModelForCausalLM.from_pretrained(str(model_path), **load_kwargs).to(device)
    model.config.use_cache = False

    records = load_records(eval_path)
    dataset = CausalEvalDataset(records, tokenizer)

    if is_main(rank):
        save_json(
            out_dir / "eval_dataset_summary.json",
            {
                "eval_file": str(eval_path),
                "num_records": len(records),
                "has_exact_match_fields": any("prompt" in r and "response" in r for r in records),
                "settings": {
                    "sparsity": SPARSITY,
                    "batch_size": BATCH_SIZE,
                    "max_prompt_len": MAX_PROMPT_LEN,
                    "max_response_len": MAX_RESPONSE_LEN,
                    "max_new_tokens": MAX_NEW_TOKENS,
                    "dtype": DTYPE,
                    "include_lm_head": INCLUDE_LM_HEAD,
                    "prompt_format": PROMPT_FORMAT,
                    "comparison_mode": COMPARISON_MODE,
                },
            },
        )

    log(rank, "Benchmarking dense model...")
    dense_metrics, dense_preds = evaluate(model, tokenizer, dataset, rank, world_size, is_dist, device, "dense")
    if is_main(rank):
        save_jsonl(out_dir / "predictions_dense.jsonl", dense_preds or [])
        log(rank, f"Dense metrics: {json.dumps(dense_metrics, ensure_ascii=False)}")

    if is_dist:
        dist.barrier()

    before_stats = parameter_zero_stats(model)
    log(rank, "Computing gradient/Taylor saliency |W * grad|...")
    modules, saliency = compute_gradient_saliency(model, tokenizer, dataset, rank, world_size, is_dist, device)

    log(rank, f"Applying per-layer {SPARSITY:.0%} gradient pruning to selected Linear weights...")
    prune_rows, masks = apply_per_layer_gradient_prune(modules, saliency, rank)
    after_stats = parameter_zero_stats(model)

    pruned_model_dir = out_dir / "pruned_model"
    if is_main(rank):
        log(rank, f"Saving pruned model before evaluation: {pruned_model_dir}")
        model.save_pretrained(pruned_model_dir, safe_serialization=True)
        tokenizer.save_pretrained(pruned_model_dir)
        torch.save(masks, out_dir / "gradient_pruning_masks.pt")
        save_csv(out_dir / "sparsity_by_module.csv", prune_rows)
    if is_dist:
        dist.barrier()

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    log(rank, f"Reloading saved pruned checkpoint for evaluation: {pruned_model_dir}")
    tokenizer = prepare_tokenizer(AutoTokenizer.from_pretrained(str(pruned_model_dir), trust_remote_code=TRUST_REMOTE_CODE))
    dataset.tokenizer = tokenizer
    model = AutoModelForCausalLM.from_pretrained(str(pruned_model_dir), **load_kwargs).to(device)
    model.config.use_cache = False

    log(rank, "Benchmarking reloaded pruned model...")
    pruned_metrics, pruned_preds = evaluate(model, tokenizer, dataset, rank, world_size, is_dist, device, "gradient_pruned_50")

    if is_main(rank):
        if pruned_metrics is not None:
            pruned_metrics["checkpoint_evaluated"] = str(pruned_model_dir)
        save_jsonl(out_dir / "predictions_pruned.jsonl", pruned_preds or [])

        selected_params = sum(int(r["weight_parameters"]) for r in prune_rows)
        selected_mask_pruned = sum(int(r["mask_pruned_parameters"]) for r in prune_rows)
        selected_actual_zeros = sum(int(r["actual_zero_parameters_after_prune"]) for r in prune_rows)
        report = {
            "method": "per_layer_gradient_taylor_pruning_decoder_only",
            "score": "abs(weight * gradient)",
            "sparsity_target": SPARSITY,
            "model_path": str(model_path),
            "checkpoint_evaluated": str(pruned_model_dir),
            "eval_file": str(eval_path),
            "output_dir": str(out_dir),
            "world_size": world_size,
            "selected_linear_modules": len(prune_rows),
            "include_lm_head": INCLUDE_LM_HEAD,
            "selected_linear_weight_parameters": selected_params,
            "selected_linear_mask_pruned_parameters": selected_mask_pruned,
            "selected_linear_mask_sparsity": selected_mask_pruned / float(selected_params or 1),
            "selected_linear_actual_zero_fraction": selected_actual_zeros / float(selected_params or 1),
            "before": before_stats,
            "after": after_stats,
            "note": "This is unstructured 50% per-layer gradient/Taylor pruning, not NVIDIA 2:4 pruning.",
        }
        save_json(out_dir / "gradient_pruning_report.json", report)

        summary_rows = [dense_metrics, pruned_metrics]
        summary_rows = [x for x in summary_rows if x is not None]
        save_json(out_dir / "benchmark_summary.json", summary_rows)
        save_csv(out_dir / "benchmark_summary.csv", summary_rows)

        print("\nGradient/Taylor pruning + exact benchmark complete")
        print(f"Output directory: {out_dir}")
        print(f"Pruned model:      {pruned_model_dir}")
        print(f"Dense exact:       {format_metric(dense_metrics.get('exact_match_accuracy'))}")
        print(f"Pruned exact:      {format_metric(pruned_metrics.get('exact_match_accuracy'))}")
        print(f"Real sparsity:     {format_metric(report['after']['whole_model_sparsity'])}")
        print(f"Linear sparsity:   {format_metric(report['selected_linear_mask_sparsity'])}")
        print(f"Summary CSV:       {out_dir / 'benchmark_summary.csv'}")

    cleanup_distributed(is_dist)


def main() -> None:
    auto_launch_if_needed()
    main_worker()


if __name__ == "__main__":
    main()
