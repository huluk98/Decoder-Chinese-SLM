#!/usr/bin/env python
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from tqdm.auto import trange
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from chatlm_decoder.command_eval import canonicalize_command_response
from chatlm_decoder.pruning import apply_masks, mask_sparsity
from chatlm_decoder.qwen25_instruct_data import (
    DEFAULT_SYSTEM_PROMPT,
    assert_no_legacy_tokens,
    build_qwen25_instruct_dataloader,
    format_qwen_sft_example,
    read_records,
)


def load_yaml(path: str | Path) -> dict[str, Any]:
    import yaml

    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def setup_distributed() -> tuple[torch.device, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed Qwen SFT requires CUDA.")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        return torch.device("cuda", local_rank), rank, local_rank, world_size
    if torch.cuda.is_available():
        return torch.device("cuda"), rank, local_rank, world_size
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps"), rank, local_rank, world_size
    return torch.device("cpu"), rank, local_rank, world_size


def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def maybe_print(rank: int, message: str) -> None:
    if rank == 0:
        print(message, flush=True)


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    while hasattr(model, "module") or hasattr(model, "_orig_mod"):
        model = getattr(model, "module", getattr(model, "_orig_mod", model))
    return model


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


def set_all_seeds(seed: int, rank: int) -> dict[str, Any]:
    effective_seed = int(seed) + int(rank)
    random.seed(effective_seed)
    if np is not None:
        np.random.seed(effective_seed)
    torch.manual_seed(effective_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(effective_seed)
    return {
        "python_random_seed": effective_seed,
        "numpy_seed": effective_seed if np is not None else None,
        "torch_seed": effective_seed,
        "torch_cuda_seed": effective_seed if torch.cuda.is_available() else None,
    }


def configure_cuda(tf32: bool) -> None:
    if not torch.cuda.is_available():
        return
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = bool(tf32)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high" if tf32 else "highest")


def autocast_for(device: torch.device, precision: str):
    if device.type == "cuda" and precision in {"bf16", "fp16"}:
        dtype = torch.bfloat16 if precision == "bf16" else torch.float16
        return torch.autocast("cuda", dtype=dtype)
    return nullcontext()


def dtype_kwargs(device: torch.device, precision: str) -> dict[str, Any]:
    if device.type != "cuda":
        return {}
    if precision == "bf16":
        return {"torch_dtype": torch.bfloat16}
    if precision == "fp16":
        return {"torch_dtype": torch.float16}
    return {}


def load_model(checkpoint: str, device: torch.device, config: dict[str, Any], rank: int) -> torch.nn.Module:
    precision = str(config.get("precision", "bf16")).lower()
    kwargs = dtype_kwargs(device, precision)
    if bool(config.get("flash_attention", True)):
        try:
            maybe_print(rank, "Attention implementation: trying flash_attention_2")
            return AutoModelForCausalLM.from_pretrained(
                checkpoint,
                attn_implementation="flash_attention_2",
                trust_remote_code=False,
                **kwargs,
            )
        except Exception as exc:
            maybe_print(rank, f"[warning] FlashAttention 2 unavailable ({exc}); falling back to SDPA.")
    try:
        return AutoModelForCausalLM.from_pretrained(
            checkpoint,
            attn_implementation="sdpa",
            trust_remote_code=False,
            **kwargs,
        )
    except Exception as exc:
        maybe_print(rank, f"[warning] SDPA unavailable ({exc}); loading checkpoint defaults.")
        return AutoModelForCausalLM.from_pretrained(checkpoint, trust_remote_code=False, **kwargs)


def configure_tokenizer(tokenizer: Any) -> None:
    if not hasattr(tokenizer, "apply_chat_template"):
        raise AttributeError("Qwen2.5-Instruct tokenizer must provide apply_chat_template.")
    if tokenizer.eos_token is None:
        raise ValueError("Qwen2.5-Instruct tokenizer eos_token must not be None.")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"


def count_parameters(model: torch.nn.Module) -> int:
    return sum(int(parameter.numel()) for parameter in unwrap_model(model).parameters())


def load_pruning_masks(path: str | Path) -> dict[str, torch.Tensor]:
    mask_path = Path(path).expanduser()
    masks = torch.load(mask_path, map_location="cpu")
    if not isinstance(masks, dict) or not masks:
        raise ValueError(f"Pruning mask file must contain a non-empty dict: {mask_path}")
    clean_masks: dict[str, torch.Tensor] = {}
    for name, mask in masks.items():
        if not isinstance(name, str) or not torch.is_tensor(mask):
            raise ValueError(f"Invalid pruning mask entry in {mask_path}: {name!r}")
        clean_masks[name] = mask.bool()
    return clean_masks


def mask_parameter_stats(masks: dict[str, torch.Tensor]) -> dict[str, int | float]:
    mask_parameter_count = sum(int(mask.numel()) for mask in masks.values())
    active_mask_parameters = sum(int(mask.bool().sum().item()) for mask in masks.values())
    pruned_mask_parameters = mask_parameter_count - active_mask_parameters
    return {
        "mask_parameter_count": mask_parameter_count,
        "active_mask_parameters": active_mask_parameters,
        "pruned_mask_parameters": pruned_mask_parameters,
        "active_mask_fraction": active_mask_parameters / float(mask_parameter_count or 1),
        "mask_sparsity": pruned_mask_parameters / float(mask_parameter_count or 1),
    }


def model_parameter_stats(model: torch.nn.Module) -> dict[str, int | float]:
    total_parameters = 0
    nonzero_parameters = 0
    for parameter in unwrap_model(model).parameters():
        data = parameter.detach()
        total_parameters += int(data.numel())
        nonzero_parameters += int(torch.count_nonzero(data).item())
    zero_parameters = total_parameters - nonzero_parameters
    return {
        "total_parameters": total_parameters,
        "nonzero_parameters": nonzero_parameters,
        "zero_parameters": zero_parameters,
        "nonzero_fraction": nonzero_parameters / float(total_parameters or 1),
        "model_zero_fraction": zero_parameters / float(total_parameters or 1),
    }


def write_pruning_report(
    checkpoint_dir: Path,
    model: torch.nn.Module,
    pruning_masks: dict[str, torch.Tensor],
    step: int,
    pruning_mask_source: str | None = None,
) -> None:
    metadata = {
        "method": "qwen25_instruct_sft_retune_with_fixed_pruning_masks",
        "phase": "retuned_qwen25_instruct_sft",
        "step": int(step),
        "sparsity": mask_sparsity(pruning_masks),
        "pruning_mask_source": pruning_mask_source or "",
        "mask_preserved_during_sft": True,
        "uses_qwen_apply_chat_template": True,
        **mask_parameter_stats(pruning_masks),
        **model_parameter_stats(model),
        "note": "Qwen2.5-Instruct SFT checkpoint saved with pruning masks reapplied after every optimizer step.",
    }
    with (checkpoint_dir / "pruning_report.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def lr_for_step(step: int, max_steps: int, learning_rate: float, warmup_steps: int, scheduler: str) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return learning_rate * float(step + 1) / float(warmup_steps)
    if scheduler == "constant":
        return learning_rate
    decay_steps = max(1, int(max_steps) - int(warmup_steps))
    progress = min(1.0, max(0.0, float(step - warmup_steps) / float(decay_steps)))
    return 0.5 * (1.0 + math.cos(math.pi * progress)) * learning_rate


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def remove_old_final(output_dir: Path) -> None:
    final_dir = output_dir / "final"
    if final_dir.exists() or final_dir.is_symlink():
        if final_dir.is_dir() and not final_dir.is_symlink():
            shutil.rmtree(final_dir)
        else:
            final_dir.unlink()
    final_manifest = output_dir / "final_checkpoint.json"
    if final_manifest.exists():
        final_manifest.unlink()


def save_final_checkpoint(
    model: torch.nn.Module,
    tokenizer: Any,
    output_dir: Path,
    step: int,
    config: dict[str, Any],
    pruning_masks: dict[str, torch.Tensor] | None = None,
    pruning_mask_source: str | None = None,
) -> Path:
    checkpoint_dir = output_dir / "final"
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    unwrap_model(model).save_pretrained(checkpoint_dir, safe_serialization=True)
    tokenizer.save_pretrained(checkpoint_dir)
    state = {"step": int(step), "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "config": config}
    torch.save(state, checkpoint_dir / "trainer_state.pt")
    if pruning_masks is not None:
        torch.save({name: mask.cpu() for name, mask in pruning_masks.items()}, checkpoint_dir / "pruning_masks.pt")
        write_pruning_report(checkpoint_dir, model, pruning_masks, step, pruning_mask_source=pruning_mask_source)
    manifest = {
        "checkpoint_path": str(checkpoint_dir),
        "resolved_checkpoint_path": str(checkpoint_dir.resolve()),
        "step": int(step),
        "model_class": unwrap_model(model).__class__.__name__,
        "model_type": getattr(unwrap_model(model).config, "model_type", None),
        "parameter_count": count_parameters(model),
        "tokenizer_vocab_size": len(tokenizer),
        "uses_qwen_apply_chat_template": True,
        "contains_model_safetensors": any(checkpoint_dir.glob("*.safetensors")),
        "contains_config": (checkpoint_dir / "config.json").exists(),
        "contains_tokenizer": (checkpoint_dir / "tokenizer_config.json").exists()
        and any((checkpoint_dir / name).exists() for name in ("tokenizer.json", "tokenizer.model", "vocab.json")),
        "contains_pruning_masks": (checkpoint_dir / "pruning_masks.pt").exists(),
        "contains_pruning_report": (checkpoint_dir / "pruning_report.json").exists(),
    }
    with (checkpoint_dir / "checkpoint_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    with (output_dir / "final_checkpoint.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return checkpoint_dir


def print_first_prompt(tokenizer: Any, data_path: str, system_prompt: str, rank: int) -> None:
    if rank != 0:
        return
    try:
        record = next(read_records(data_path))
    except StopIteration:
        return
    example = format_qwen_sft_example(tokenizer, record, system_prompt)
    print("First fully formatted Qwen chat prompt:\n" + example["prompt_text"])


def print_first_batch_debug(tokenizer: Any, batch: dict[str, torch.Tensor], rank: int) -> None:
    if rank != 0:
        return
    labels = batch["labels"][0].detach().cpu()
    input_ids = batch["input_ids"][0].detach().cpu()
    attention_mask = batch["attention_mask"][0].detach().cpu()
    true_length = int(attention_mask.sum().item())
    labels = labels[:true_length]
    input_ids = input_ids[:true_length]
    valid_mask = labels.ne(-100)
    supervised_ids = labels[valid_mask]
    decoded_full = tokenizer.decode(input_ids.tolist(), skip_special_tokens=False)
    decoded_response = tokenizer.decode(supervised_ids.tolist(), skip_special_tokens=False)
    assert_no_legacy_tokens(decoded_full, "first decoded Qwen batch")
    print(
        "First Qwen tokenized sample:\n"
        f"  decoded full input:\n{decoded_full}\n"
        f"  decoded supervised response region:\n{decoded_response}\n"
        f"  non_-100_label_tokens={int(valid_mask.sum().item())}"
    )


def normalize_eval_text(text: str) -> str:
    import re
    import unicodedata

    text = unicodedata.normalize("NFKC", str(text)).strip()
    text = re.sub(r"^\s*```(?:json|text|txt)?\s*|\s*```\s*$", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"[\s。．.；;，,]+$", "", text).strip()
    return " ".join(text.split()).strip()


@torch.no_grad()
def overfit_sanity_check(
    model: torch.nn.Module,
    tokenizer: Any,
    data_path: str,
    system_prompt: str,
    device: torch.device,
    max_new_tokens: int,
    max_samples: int,
    rank: int,
) -> None:
    if rank != 0:
        return
    records = list(read_records(data_path))[: int(max_samples)]
    eval_model = unwrap_model(model)
    was_training = eval_model.training
    eval_model.eval()
    failures: list[dict[str, str]] = []
    correct = 0
    for index, record in enumerate(records):
        example = format_qwen_sft_example(tokenizer, record, system_prompt)
        encoded = tokenizer(example["prompt_text"], return_tensors="pt", add_special_tokens=False).to(device)
        generated = eval_model.generate(
            **encoded,
            do_sample=False,
            num_beams=1,
            max_new_tokens=int(max_new_tokens),
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        continuation = generated[0, encoded["input_ids"].shape[1] :]
        raw_prediction = tokenizer.decode(continuation.detach().cpu().tolist(), skip_special_tokens=True)
        prediction = normalize_eval_text(raw_prediction)
        target = normalize_eval_text(example["response"])
        is_match = prediction == target
        correct += int(is_match)
        if not is_match:
            failures.append(
                {
                    "index": str(index),
                    "prompt": example["prompt_text"],
                    "raw_prediction": raw_prediction,
                    "normalized_prediction": prediction,
                    "normalized_label": target,
                    "command_prediction": canonicalize_command_response(prediction),
                    "command_label": canonicalize_command_response(target),
                }
            )
    accuracy = correct / max(1, len(records))
    print(f"Qwen debug overfit sanity exact_match={accuracy:.4f} ({correct}/{len(records)})")
    if failures:
        print("First failed Qwen debug generations:")
        print(json.dumps(failures[:10], ensure_ascii=False, indent=2))
    if accuracy < 0.95:
        raise RuntimeError("Qwen2.5-Instruct debug overfit sanity failed; the chat-template SFT pipeline is still suspect.")
    if was_training:
        eval_model.train()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen2.5-0.5B-Instruct-only SFT with official chat template formatting.")
    parser.add_argument("--config", default="configs/sft_qwen25_0p5b_instruct.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--epochs", type=float, default=None)
    parser.add_argument("--max-seq-length", "--max_seq_length", type=int, default=None)
    parser.add_argument("--per-device-train-batch-size", "--per_device_train_batch_size", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", "--gradient_accumulation_steps", type=int, default=None)
    parser.add_argument("--learning-rate", "--learning_rate", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--data-seed", "--data_seed", type=int, default=None)
    parser.add_argument("--max-steps", "--max_steps", type=int, default=None)
    parser.add_argument("--pruning-mask", "--pruning_mask", default=None)
    parser.add_argument("--debug-overfit-samples", "--debug_overfit_samples", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    checkpoint = str(args.checkpoint or config.get("model_name_or_path") or "Qwen/Qwen2.5-0.5B-Instruct")
    data_path = str(args.data_path or config.get("train_file") or "")
    if not data_path:
        raise ValueError("Set train_file in config or pass --data-path.")
    output_dir = Path(args.output_dir or config.get("output_dir") or "outputs/qwen25_0p5b_instruct_sft").expanduser()
    system_prompt = str(config.get("system_prompt") or DEFAULT_SYSTEM_PROMPT)

    max_seq_length = int(args.max_seq_length or config.get("max_seq_length", 256))
    batch_size = int(args.per_device_train_batch_size or config.get("per_device_train_batch_size", 16))
    grad_accum_steps = int(args.gradient_accumulation_steps or config.get("gradient_accumulation_steps", 1))
    learning_rate = float(args.learning_rate or config.get("learning_rate", 2.0e-5))
    epochs = float(args.epochs if args.epochs is not None else config.get("num_train_epochs", 3))
    seed = int(args.seed if args.seed is not None else config.get("seed", 42))
    data_seed = int(args.data_seed if args.data_seed is not None else config.get("data_seed", seed))
    pruning_mask_path = args.pruning_mask or config.get("pruning_mask_path")
    precision = "bf16" if bool(config.get("bf16", True)) else "fp16" if bool(config.get("fp16", False)) else "fp32"

    device, rank, local_rank, world_size = setup_distributed()
    configure_cuda(bool(config.get("tf32", True)))
    seed_info = set_all_seeds(seed, rank=rank)

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=False)
    configure_tokenizer(tokenizer)
    model = load_model(checkpoint, device, {**config, "precision": precision}, rank=rank)
    if bool(config.get("gradient_checkpointing", False)):
        model.gradient_checkpointing_enable()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
    model.to(device)
    if bool(config.get("torch_compile", False)) and hasattr(torch, "compile"):
        maybe_print(rank, "torch.compile: enabled")
        model = torch.compile(model)
    if world_size > 1:
        model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

    pruning_masks: dict[str, torch.Tensor] | None = None
    if pruning_mask_path:
        pruning_masks = load_pruning_masks(pruning_mask_path)
        apply_masks(unwrap_model(model), pruning_masks)
        maybe_print(
            rank,
            f"Pruning masks: loaded {len(pruning_masks)} tensors from {pruning_mask_path} "
            f"(sparsity={mask_sparsity(pruning_masks):.4f}); masks will be reapplied after every optimizer step.",
        )

    print_first_prompt(tokenizer, data_path, system_prompt, rank)
    dataloader = build_qwen25_instruct_dataloader(
        path=data_path,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        batch_size=batch_size,
        system_prompt=system_prompt,
        max_samples=args.debug_overfit_samples,
        group_by_length=bool(config.get("group_by_length", True)),
        shuffle=True,
        num_workers=int(config.get("dataloader_num_workers", 0)),
        pin_memory=bool(config.get("dataloader_pin_memory", False)),
        persistent_workers=bool(config.get("persistent_workers", False)),
        rank=rank,
        world_size=world_size,
        seed=data_seed,
    )
    if len(dataloader) <= 0:
        raise ValueError(f"Qwen SFT dataset produced no batches: {data_path}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=float(config.get("weight_decay", 0.01)),
        betas=(0.9, 0.95),
        eps=1e-8,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and precision == "fp16"))
    optimizer_steps_per_epoch = max(1, math.ceil(len(dataloader) / grad_accum_steps))
    max_steps = max(
        1,
        int(args.max_steps if args.max_steps is not None else math.ceil(float(epochs) * optimizer_steps_per_epoch)),
    )
    warmup_steps = int(float(config.get("warmup_ratio", 0.03)) * max_steps)
    max_new_tokens = int(config.get("max_new_tokens", 64))

    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        remove_old_final(output_dir)
        run_config = {
            "script": "scripts/sft_qwen25_instruct.py",
            "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "git_commit": git_commit_hash(),
            "model_name_or_path": checkpoint,
            "train_file": data_path,
            "eval_file": config.get("eval_file"),
            "output_dir": str(output_dir),
            "system_prompt": system_prompt,
            "uses_qwen_apply_chat_template": True,
            "world_size": world_size,
            "local_rank": local_rank,
            "parameter_count": count_parameters(model),
            "max_seq_length": max_seq_length,
            "max_new_tokens": max_new_tokens,
            "per_device_train_batch_size": batch_size,
            "gradient_accumulation_steps": grad_accum_steps,
            "effective_batch_size": batch_size * world_size * grad_accum_steps,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "max_steps": max_steps,
            "pruning_mask_path": str(pruning_mask_path) if pruning_mask_path else None,
            "pruning_mask_sparsity": mask_sparsity(pruning_masks) if pruning_masks is not None else None,
            "seed_info": {**seed_info, "base_seed": seed, "data_seed": data_seed},
            "config": config,
        }
        with (output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
            json.dump(run_config, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
    if is_dist():
        dist.barrier()

    maybe_print(
        rank,
        "Qwen2.5-Instruct SFT runtime:\n"
        f"  checkpoint={checkpoint}\n"
        f"  output_dir={output_dir}\n"
        f"  world_size={world_size} local_rank={local_rank}\n"
        f"  per_device_train_batch_size={batch_size} grad_accum={grad_accum_steps}\n"
        f"  effective_batch_size={batch_size * world_size * grad_accum_steps}\n"
        f"  max_seq_length={max_seq_length} max_steps={max_steps}\n"
        f"  parameters={count_parameters(model):,}",
    )

    model.train()
    optimizer.zero_grad(set_to_none=True)
    current_epoch = 0
    if hasattr(dataloader.sampler, "set_epoch"):
        dataloader.sampler.set_epoch(current_epoch)
    data_iter = iter(dataloader)
    first_batch_printed = False
    progress = trange(1, max_steps + 1, disable=(rank != 0), desc="qwen25-instruct-sft")
    start_time = time.perf_counter()
    for step in progress:
        lr = lr_for_step(step - 1, max_steps, learning_rate, warmup_steps, str(config.get("lr_scheduler_type", "cosine")))
        set_lr(optimizer, lr)
        raw_loss_sum = 0.0
        for micro_step in range(grad_accum_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                current_epoch += 1
                if hasattr(dataloader.sampler, "set_epoch"):
                    dataloader.sampler.set_epoch(current_epoch)
                data_iter = iter(dataloader)
                batch = next(data_iter)
            batch = {key: value.to(device, non_blocking=(device.type == "cuda")) for key, value in batch.items()}
            if not first_batch_printed:
                print_first_batch_debug(tokenizer, batch, rank)
                first_batch_printed = True
            sync_context = (
                model.no_sync()
                if world_size > 1 and hasattr(model, "no_sync") and micro_step < grad_accum_steps - 1
                else nullcontext()
            )
            with sync_context:
                with autocast_for(device, precision):
                    outputs = model(**batch, use_cache=False)
                    raw_loss = outputs.loss
                    loss = raw_loss / grad_accum_steps
                raw_loss_sum += float(raw_loss.detach().cpu())
                scaler.scale(loss).backward()
        if float(config.get("max_grad_norm", 1.0)) > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.get("max_grad_norm", 1.0)))
        scaler.step(optimizer)
        scaler.update()
        if pruning_masks is not None:
            apply_masks(unwrap_model(model), pruning_masks)
        optimizer.zero_grad(set_to_none=True)
        logged = torch.tensor(raw_loss_sum / grad_accum_steps, device=device)
        if is_dist():
            dist.all_reduce(logged, op=dist.ReduceOp.AVG)
        if rank == 0 and (step == 1 or step % int(config.get("logging_steps", 10)) == 0):
            elapsed = max(1e-6, time.perf_counter() - start_time)
            progress.set_postfix(loss=f"{float(logged.detach().cpu()):.4f}", lr=f"{lr:.2e}", step_s=f"{elapsed / step:.2f}")

    if is_dist():
        dist.barrier()
    if rank == 0:
        checkpoint_dir = save_final_checkpoint(
            model,
            tokenizer,
            output_dir,
            max_steps,
            config,
            pruning_masks=pruning_masks,
            pruning_mask_source=str(pruning_mask_path) if pruning_mask_path else None,
        )
        maybe_print(rank, f"Saved final Qwen2.5-Instruct SFT checkpoint: {checkpoint_dir}")
    if is_dist():
        dist.barrier()
    if rank == 0 and args.debug_overfit_samples:
        overfit_sanity_check(
            model=model,
            tokenizer=tokenizer,
            data_path=data_path,
            system_prompt=system_prompt,
            device=device,
            max_new_tokens=max_new_tokens,
            max_samples=int(args.debug_overfit_samples),
            rank=rank,
        )
    if is_dist():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
