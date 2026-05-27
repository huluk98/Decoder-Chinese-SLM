#!/usr/bin/env python
from __future__ import annotations

import argparse
import datetime as dt
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
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from tqdm.auto import trange
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from chatlm_decoder.config import load_config
from chatlm_decoder.command_eval import canonicalize_command_response
from chatlm_decoder.pruning import (
    apply_masks,
    mask_parameter_stats,
    masked_weight_stats,
    mask_sparsity,
    layerwise_zero_fraction,
    module_filter_report,
    sparsity_accounting,
    write_csv,
    write_json,
)
from chatlm_decoder.sft_data import EOS_TOKEN, build_sft_dataloader, normalize_sft_record, read_records

try:
    import numpy as np
except Exception:  # pragma: no cover - numpy is optional for this script.
    np = None


def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def setup_distributed() -> tuple[torch.device, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed SFT requires CUDA.")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        return torch.device("cuda", local_rank), rank, local_rank, world_size
    if torch.cuda.is_available():
        return torch.device("cuda"), rank, local_rank, world_size
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps"), rank, local_rank, world_size
    return torch.device("cpu"), rank, local_rank, world_size


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    while hasattr(model, "module") or hasattr(model, "_orig_mod"):
        model = getattr(model, "module", getattr(model, "_orig_mod", model))
    return model


def maybe_print(rank: int, message: str) -> None:
    if rank == 0:
        print(message)


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


def set_all_seeds(seed: int, rank: int, deterministic: bool) -> dict[str, Any]:
    effective_seed = int(seed) + int(rank)
    random.seed(effective_seed)
    if np is not None:
        np.random.seed(effective_seed)
    torch.manual_seed(effective_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(effective_seed)
    if deterministic:
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)
    return {
        "python_random_seed": effective_seed,
        "numpy_seed": effective_seed if np is not None else None,
        "torch_seed": effective_seed,
        "torch_cuda_seed": effective_seed if torch.cuda.is_available() else None,
        "deterministic_algorithms_enabled": bool(torch.are_deterministic_algorithms_enabled()),
    }


def autocast_for(device: torch.device, precision: str):
    if device.type == "cuda" and precision in {"fp16", "bf16"}:
        dtype = torch.float16 if precision == "fp16" else torch.bfloat16
        return torch.autocast("cuda", dtype=dtype)
    return nullcontext()


def configure_cuda(train_config: dict[str, Any], rank: int) -> None:
    if not torch.cuda.is_available():
        return
    tf32 = bool(train_config.get("tf32", True))
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = tf32
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = tf32
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high" if tf32 else "highest")
    if hasattr(torch.backends, "cuda"):
        if hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(bool(train_config.get("sdp_flash", True)))
        if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
            torch.backends.cuda.enable_mem_efficient_sdp(bool(train_config.get("sdp_mem_efficient", True)))
        if hasattr(torch.backends.cuda, "enable_math_sdp"):
            torch.backends.cuda.enable_math_sdp(bool(train_config.get("sdp_math", True)))
    maybe_print(
        rank,
        "Torch backends: "
        f"tf32={tf32} | matmul_precision={'high' if tf32 else 'highest'} | "
        f"sdp_flash={bool(train_config.get('sdp_flash', True))}",
    )


def lr_for_step(step: int, train_config: dict[str, Any]) -> float:
    max_lr = float(train_config["learning_rate"])
    min_lr = float(train_config.get("min_learning_rate", 0.0))
    warmup_steps = int(train_config.get("warmup_steps", 0))
    max_steps = int(train_config["max_steps"])
    scheduler = str(train_config.get("lr_scheduler_type", "cosine")).lower()
    if warmup_steps > 0 and step < warmup_steps:
        return max_lr * float(step + 1) / float(warmup_steps)
    if scheduler == "constant":
        return max_lr
    decay_steps = max(1, max_steps - warmup_steps)
    progress = min(1.0, max(0.0, float(step - warmup_steps) / float(decay_steps)))
    return min_lr + 0.5 * (1.0 + math.cos(math.pi * progress)) * (max_lr - min_lr)


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def save_checkpoint(
    model: torch.nn.Module,
    tokenizer: Any,
    output_dir: Path,
    step: int,
    pruning_masks: dict[str, torch.Tensor] | None = None,
    pruning_mask_source: str | None = None,
    checkpoint_name: str = "final",
) -> None:
    checkpoint_dir = output_dir / checkpoint_name
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    unwrap_model(model).save_pretrained(checkpoint_dir, safe_serialization=True)
    tokenizer.save_pretrained(checkpoint_dir)
    state = {
        "step": step,
        "checkpoint_name": checkpoint_name,
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    torch.save(state, checkpoint_dir / "trainer_state.pt")
    if pruning_masks is not None:
        torch.save({name: mask.cpu() for name, mask in pruning_masks.items()}, checkpoint_dir / "pruning_masks.pt")
        write_pruning_report(checkpoint_dir, model, pruning_masks, step, pruning_mask_source=pruning_mask_source)
        source_report = load_pruning_source_report(pruning_mask_source)
        pruning_scope = source_report.get(
            "pruning_scope",
            source_report.get("target_resolution", {}).get("pruning_scope", "transformer_linears"),
        )
        write_json(
            checkpoint_dir / "module_filter_report.json",
            module_filter_report(
                unwrap_model(model),
                include_lm_head=bool(source_report.get("include_lm_head", False)),
                scope=pruning_scope,
            ),
        )
        write_json(
            checkpoint_dir / "mask_validation.json",
            {
                "method": "sft_retune_with_fixed_pruning_masks",
                "phase": "retuned",
                **sparsity_accounting(unwrap_model(model), pruning_masks, target=mask_sparsity(pruning_masks)),
            },
        )
        write_csv(checkpoint_dir / "sparsity_by_module.csv", layerwise_zero_fraction(unwrap_model(model), pruning_masks))
        write_csv(checkpoint_dir / "layerwise_zero_fraction.csv", layerwise_zero_fraction(unwrap_model(model), pruning_masks))
        if checkpoint_name == "final":
            reload_model = AutoModelForCausalLM.from_pretrained(checkpoint_dir)
            source_report = load_pruning_source_report(pruning_mask_source)
            reload_validation = {
                "method": "sft_retune_with_fixed_pruning_masks",
                "phase": "retuned",
                "checkpoint_reloaded": str(checkpoint_dir),
                "target_sparsity_denominator": source_report.get("target_sparsity_denominator", "prunable"),
                "target_whole_model_sparsity": source_report.get("target_whole_model_sparsity"),
                **sparsity_accounting(reload_model, pruning_masks, target=mask_sparsity(pruning_masks)),
            }
            if int(reload_validation.get("masked_weight_violation_count", 0)) != 0:
                raise RuntimeError(f"Pruned weights regrew after saving/reloading retuned checkpoint: {checkpoint_dir}")
            write_json(checkpoint_dir / "checkpoint_reload_validation.json", reload_validation)
            del reload_model
    manifest = {
        **state,
        "checkpoint_path": str(checkpoint_dir),
        "resolved_checkpoint_path": str(checkpoint_dir.resolve()),
        "model_class": unwrap_model(model).__class__.__name__,
        "model_type": getattr(unwrap_model(model).config, "model_type", None),
        "parameter_count": count_all_parameters(model),
        "tokenizer_vocab_size": len(tokenizer),
        "contains_model_safetensors": any(checkpoint_dir.glob("*.safetensors")),
        "contains_config": (checkpoint_dir / "config.json").exists(),
        "contains_tokenizer": (checkpoint_dir / "tokenizer_config.json").exists()
        and any((checkpoint_dir / name).exists() for name in ("tokenizer.json", "tokenizer.model", "vocab.json")),
    }
    with (checkpoint_dir / "checkpoint_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    with (output_dir / "final_checkpoint.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(f"[checkpoint] saved SFT checkpoint: {checkpoint_dir}")


def mean_pool_last_hidden(model: torch.nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
    )
    hidden = outputs.hidden_states[-1]
    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
    return F.normalize(pooled.float(), p=2, dim=-1)


def contrastive_alignment_loss(model: torch.nn.Module, batch: dict[str, torch.Tensor], margin: float) -> torch.Tensor:
    anchor = mean_pool_last_hidden(model, batch["anchor_input_ids"], batch["anchor_attention_mask"])
    positive = mean_pool_last_hidden(model, batch["positive_input_ids"], batch["positive_attention_mask"])
    negative = mean_pool_last_hidden(model, batch["negative_input_ids"], batch["negative_attention_mask"])
    positive_distance = 1.0 - F.cosine_similarity(anchor, positive, dim=-1)
    negative_distance = 1.0 - F.cosine_similarity(anchor, negative, dim=-1)
    return F.relu(float(margin) + positive_distance - negative_distance).mean()


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


def load_pruning_source_report(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    mask_path = Path(path).expanduser()
    candidates = []
    if mask_path.name == "pruning_masks.pt":
        candidates.append(mask_path.with_name("pruning_report.json"))
    candidates.append(mask_path.parent / "pruning_report.json")
    for candidate in candidates:
        if not candidate.exists():
            continue
        with candidate.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return payload if isinstance(payload, dict) else {}
    return {}


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=(device.type == "cuda")) for key, value in batch.items()}


def trainable_parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in unwrap_model(model).parameters() if parameter.requires_grad)


def count_all_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in unwrap_model(model).parameters())


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
    mask_stats = mask_parameter_stats(pruning_masks)
    model_stats = model_parameter_stats(model)
    accounting = sparsity_accounting(unwrap_model(model), pruning_masks, target=mask_sparsity(pruning_masks))
    source_report = load_pruning_source_report(pruning_mask_source)
    target_denominator = source_report.get("target_sparsity_denominator", "prunable")
    target_whole_model_sparsity = source_report.get("target_whole_model_sparsity")
    requested_sparsity = source_report.get("requested_sparsity", source_report.get("target_sparsity", mask_sparsity(pruning_masks)))
    metadata = {
        "method": "sft_retune_with_fixed_pruning_masks",
        "phase": "retuned",
        "step": int(step),
        "sparsity": mask_sparsity(pruning_masks),
        "requested_sparsity": requested_sparsity,
        "target_sparsity": requested_sparsity,
        "pruning_scope": source_report.get(
            "pruning_scope",
            source_report.get("target_resolution", {}).get("pruning_scope", "transformer_linears"),
        ),
        "target_sparsity_denominator": target_denominator,
        "target_whole_model_sparsity": target_whole_model_sparsity,
        "source_pruning_method": source_report.get("method"),
        "source_pruning_phase": source_report.get("phase"),
        "source_target_resolution": source_report.get("target_resolution", {}),
        "pruning_mask_source": pruning_mask_source or "",
        "mask_preserved_during_sft": True,
        "target_prunable_sparsity": accounting["target_prunable_sparsity"],
        "achieved_prunable_sparsity": accounting["achieved_prunable_sparsity"],
        "achieved_whole_model_sparsity": accounting["achieved_whole_model_sparsity"],
        **mask_stats,
        **model_stats,
        **accounting,
        **masked_weight_stats(unwrap_model(model), pruning_masks),
        "note": "SFT checkpoint saved with pruning masks reapplied after every optimizer step.",
    }
    with (checkpoint_dir / "pruning_report.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def remove_stale_sft_checkpoints(output_dir: Path) -> None:
    """Remove old SFT checkpoint aliases so eval cannot pick a stale model."""
    for path in output_dir.glob("step-*"):
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
    for name in ("latest", "final"):
        path = output_dir / name
        if path.is_symlink() or path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
    final_manifest = output_dir / "final_checkpoint.json"
    if final_manifest.exists():
        final_manifest.unlink()


def _copy_flat_config(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    train_config = dict(config["train"])
    sft_config = dict(config.get("sft", {}))
    run_config = dict(config["run"])
    generation_config = dict(config.get("generation") or {})
    flat_mode = "model_name_or_path" in config or "train_file" in config or "per_device_train_batch_size" in config

    if "output_dir" in config:
        run_config["output_dir"] = config["output_dir"]
    if "model_name_or_path" in config:
        sft_config["base_model"] = config["model_name_or_path"]
    if "train_file" in config:
        sft_config["data_path"] = config["train_file"]
    if "eval_file" in config:
        sft_config["eval_path"] = config["eval_file"]
    if "max_seq_length" in config:
        sft_config["max_length"] = int(config["max_seq_length"])
    elif flat_mode:
        sft_config["max_length"] = 128
    if "max_new_tokens" in config:
        generation_config["max_new_tokens"] = int(config["max_new_tokens"])
    if "benchmark_runs" in config:
        sft_config["benchmark_runs"] = int(config["benchmark_runs"])
    if "pruning_mask_path" in config:
        sft_config["pruning_mask_path"] = config["pruning_mask_path"]

    mapping = {
        "num_train_epochs": "epochs",
        "per_device_train_batch_size": "batch_size",
        "per_device_eval_batch_size": "eval_batch_size",
        "gradient_accumulation_steps": "grad_accum_steps",
        "learning_rate": "learning_rate",
        "max_steps": "max_steps",
        "warmup_ratio": "warmup_ratio",
        "weight_decay": "weight_decay",
        "lr_scheduler_type": "lr_scheduler_type",
        "max_grad_norm": "max_grad_norm",
        "dataloader_num_workers": "num_workers",
        "dataloader_pin_memory": "pin_memory",
        "persistent_workers": "persistent_workers",
        "logging_steps": "log_every",
        "save_total_limit": "save_total_limit",
        "save_every": "save_every",
        "save_strategy": "save_strategy",
        "save_final_only": "save_final_only",
        "eval_strategy": "eval_strategy",
        "eval_steps": "eval_steps",
        "torch_compile": "compile",
        "tf32": "tf32",
        "drop_last": "drop_last",
        "remove_unused_columns": "remove_unused_columns",
        "group_by_length": "group_by_length",
        "load_in_training_dtype": "load_in_training_dtype",
    }
    for flat_key, train_key in mapping.items():
        if flat_key in config:
            train_config[train_key] = config[flat_key]

    if "gradient_checkpointing" in config:
        config["model"]["gradient_checkpointing"] = bool(config["gradient_checkpointing"])
    if "flash_attention" in config:
        train_config["flash_attention"] = bool(config["flash_attention"])
        train_config["sdp_flash"] = bool(config["flash_attention"])
    if "bf16" in config or "fp16" in config:
        if bool(config.get("bf16", False)):
            train_config["precision"] = "bf16"
        elif bool(config.get("fp16", False)):
            train_config["precision"] = "fp16"
        else:
            train_config["precision"] = "fp32"

    if flat_mode and "min_learning_rate" not in config:
        train_config["min_learning_rate"] = 0.0
    return train_config, sft_config, run_config, generation_config


def configure_tokenizer(tokenizer: Any) -> None:
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"


def model_load_kwargs(device: torch.device, train_config: dict[str, Any]) -> dict[str, Any]:
    if device.type != "cuda":
        return {}
    if not bool(train_config.get("load_in_training_dtype", False)):
        return {}
    precision = str(train_config.get("precision", "bf16")).lower()
    if precision == "bf16":
        return {"torch_dtype": torch.bfloat16}
    if precision == "fp16":
        return {"torch_dtype": torch.float16}
    return {}


def load_model(checkpoint: str, device: torch.device, train_config: dict[str, Any], rank: int) -> torch.nn.Module:
    kwargs = model_load_kwargs(device, train_config)
    if bool(train_config.get("flash_attention", False)):
        try:
            maybe_print(rank, "Attention implementation: trying flash_attention_2")
            return AutoModelForCausalLM.from_pretrained(
                checkpoint,
                attn_implementation="flash_attention_2",
                **kwargs,
            )
        except Exception as exc:
            maybe_print(rank, f"[warning] FlashAttention 2 unavailable ({exc}); falling back to SDPA.")
    try:
        return AutoModelForCausalLM.from_pretrained(checkpoint, attn_implementation="sdpa", **kwargs)
    except Exception as exc:
        maybe_print(rank, f"[warning] SDPA attention argument unavailable ({exc}); loading with checkpoint defaults.")
        return AutoModelForCausalLM.from_pretrained(checkpoint, **kwargs)


def print_startup_summary(
    rank: int,
    config_path: str,
    world_size: int,
    local_rank: int,
    device: torch.device,
    train_config: dict[str, Any],
    max_seq_length: int,
    max_new_tokens: int,
    model: torch.nn.Module,
) -> None:
    if rank != 0:
        return
    batch_size = int(train_config["batch_size"])
    grad_accum_steps = int(train_config.get("grad_accum_steps", 1))
    effective_batch = batch_size * int(world_size) * grad_accum_steps
    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else str(device)
    print(
        "Recommended 8-GPU SFT launch:\n"
        "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 TOKENIZERS_PARALLELISM=false "
        f"torchrun --standalone --nproc_per_node=8 scripts/train.py --config {config_path}"
    )
    print(
        "SFT runtime:\n"
        f"  physical CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}\n"
        f"  world_size={world_size}\n"
        f"  rank={rank} local_rank={local_rank}\n"
        f"  gpu_name={gpu_name}\n"
        f"  per_device_train_batch_size={batch_size}\n"
        f"  gradient_accumulation_steps={grad_accum_steps}\n"
        f"  effective_batch_size={effective_batch}\n"
        f"  max_seq_length={max_seq_length}\n"
        f"  max_new_tokens={max_new_tokens}\n"
        f"  trainable_parameters={trainable_parameter_count(model):,}\n"
        f"  total_parameters={count_all_parameters(model):,}"
    )


def write_sft_run_config(
    output_dir: Path,
    args: argparse.Namespace,
    config: dict[str, Any],
    train_config: dict[str, Any],
    sft_config: dict[str, Any],
    run_config: dict[str, Any],
    generation_config: dict[str, Any],
    checkpoint: str,
    data_path: str,
    eval_path: str | None,
    model: torch.nn.Module,
    tokenizer: Any,
    world_size: int,
    local_rank: int,
    seed_info: dict[str, Any],
    max_seq_length: int,
    max_new_tokens: int,
) -> None:
    payload = {
        "script": "scripts/sft.py",
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git_commit": git_commit_hash(),
        "config_path": args.config,
        "mode": str(args.mode or sft_config.get("mode", "sft")),
        "model_path": checkpoint,
        "checkpoint_path_used_for_training": checkpoint,
        "tokenizer_path": checkpoint,
        "checkpoint_files": checkpoint_file_timestamps(checkpoint),
        "train_file": str(data_path),
        "eval_file": str(eval_path) if eval_path else None,
        "output_dir": str(output_dir),
        "expected_final_checkpoint_path": str(output_dir / "final"),
        "world_size": int(world_size),
        "local_rank": int(local_rank),
        "model_class": unwrap_model(model).__class__.__name__,
        "model_type": getattr(unwrap_model(model).config, "model_type", None),
        "parameter_count": count_all_parameters(model),
        "trainable_parameter_count": trainable_parameter_count(model),
        "max_seq_length": int(max_seq_length),
        "max_new_tokens": int(max_new_tokens),
        "seed_info": seed_info,
        "train": train_config,
        "sft": sft_config,
        "run": run_config,
        "generation": generation_config,
        "tokenizer": {
            "class": tokenizer.__class__.__name__,
            "vocab_size": len(tokenizer),
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "padding_side": getattr(tokenizer, "padding_side", None),
        },
    "trainer_flags": {
            "save_strategy": train_config.get("save_strategy"),
            "eval_strategy": train_config.get("eval_strategy"),
            "save_final_only": bool(train_config.get("save_final_only", True)),
            "load_best_model_at_end": False,
            "metric_for_best_model": None,
            "greater_is_better": None,
            "remove_unused_columns": train_config.get("remove_unused_columns"),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def print_tokenized_debug_sample(tokenizer: Any, batch: dict[str, torch.Tensor], rank: int) -> None:
    if rank != 0:
        return
    input_ids = batch["input_ids"][0].detach().cpu()
    labels = batch["labels"][0].detach().cpu()
    attention_mask = batch["attention_mask"][0].detach().cpu()
    true_length = int(attention_mask.sum().item())
    input_ids = input_ids[:true_length]
    labels = labels[:true_length]
    valid_mask = labels.ne(-100)
    valid_positions = torch.nonzero(valid_mask, as_tuple=False).flatten()
    prompt_length = int(valid_positions[0].item()) if valid_positions.numel() else true_length
    response_length = int(valid_mask.sum().item())
    supervised_ids = labels[valid_mask]
    decoded_full = tokenizer.decode(input_ids.tolist(), skip_special_tokens=False)
    decoded_response = tokenizer.decode(supervised_ids.tolist(), skip_special_tokens=False) if response_length else ""
    print(
        "First tokenized SFT sample:\n"
        f"  decoded full input:\n{decoded_full}\n"
        f"  decoded supervised response region:\n{decoded_response}\n"
        f"  non_-100_label_tokens={response_length}\n"
        f"  prompt_length={prompt_length}\n"
        f"  response_length={response_length}"
    )


def warn_token_bounds(
    tokenizer: Any,
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    rank: int,
) -> None:
    if rank != 0:
        return
    vocab_size = int(unwrap_model(model).config.vocab_size)
    input_ids = batch["input_ids"]
    labels = batch["labels"]
    valid_labels = labels[labels.ne(-100)]
    print(
        "First batch token bounds:\n"
        f"  tokenizer_size={len(tokenizer)}\n"
        f"  model.config.vocab_size={vocab_size}\n"
        f"  input_ids min/max={int(input_ids.min().detach().cpu())}/{int(input_ids.max().detach().cpu())}\n"
        f"  labels min/max={int(labels.min().detach().cpu())}/{int(labels.max().detach().cpu())}\n"
        f"  labels excluding -100 min/max="
        f"{int(valid_labels.min().detach().cpu()) if valid_labels.numel() else 'none'}/"
        f"{int(valid_labels.max().detach().cpu()) if valid_labels.numel() else 'none'}"
    )
    if int(input_ids.max().detach().cpu()) >= vocab_size:
        print(f"[warning] input_ids.max() >= model.config.vocab_size ({vocab_size})")
    if valid_labels.numel() and int(valid_labels.max().detach().cpu()) >= vocab_size:
        print(f"[warning] labels excluding -100 have ids >= model.config.vocab_size ({vocab_size})")
    if bool((labels < -100).any().detach().cpu()):
        print("[warning] labels contain values less than -100")


def normalize_for_exact_match(text: str) -> str:
    text = text.replace(EOS_TOKEN, "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def mean_std(values: list[float]) -> tuple[float, float]:
    finite_values = [float(value) for value in values if math.isfinite(float(value))]
    if not finite_values:
        return math.nan, math.nan
    if len(finite_values) == 1:
        return finite_values[0], 0.0
    return statistics.mean(finite_values), statistics.stdev(finite_values)


def should_run_eval(eval_strategy: str, step: int, eval_steps: int, epoch_boundary: bool, final_step: bool) -> bool:
    strategy = str(eval_strategy).lower()
    if strategy in {"no", "none", "false", "off"}:
        return False
    if strategy == "final":
        return bool(final_step)
    if strategy == "epoch":
        return bool(epoch_boundary or final_step)
    if strategy == "steps":
        return bool(step % max(1, int(eval_steps)) == 0 or final_step)
    raise ValueError("eval_strategy must be one of: final, epoch, steps, none.")


def evaluate_exact_match(
    model: torch.nn.Module,
    tokenizer: Any,
    eval_path: str | None,
    device: torch.device,
    max_seq_length: int,
    max_new_tokens: int,
    output_dir: Path,
    step: int,
    max_samples: int | None = None,
    benchmark_runs: int = 1,
) -> dict[str, Any] | None:
    if not eval_path:
        return None
    records = list(read_records(eval_path))
    if max_samples is not None:
        records = records[: int(max_samples)]
    if not records:
        return None

    eval_model = unwrap_model(model)
    was_training = eval_model.training
    eval_model.eval()
    run_metrics: list[dict[str, Any]] = []
    benchmark_runs = max(1, int(benchmark_runs))
    for run_index in range(benchmark_runs):
        correct = 0
        generated_lengths: list[int] = []
        examples: list[dict[str, str]] = []
        with torch.no_grad():
            for index, record in enumerate(records):
                prompt, target_response = normalize_sft_record(record)
                encoded = tokenizer(
                    prompt,
                    return_tensors="pt",
                    add_special_tokens=False,
                    truncation=True,
                    max_length=int(max_seq_length),
                )
                encoded = {key: value.to(device) for key, value in encoded.items()}
                generated = eval_model.generate(
                    **encoded,
                    max_new_tokens=int(max_new_tokens),
                    do_sample=False,
                    num_beams=1,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                )
                continuation = generated[0, encoded["input_ids"].shape[1] :]
                generated_lengths.append(int(continuation.numel()))
                prediction = tokenizer.decode(continuation.detach().cpu().tolist(), skip_special_tokens=False)
                normalized_prediction = canonicalize_command_response(normalize_for_exact_match(prediction))
                normalized_target = canonicalize_command_response(normalize_for_exact_match(target_response))
                correct += int(normalized_prediction == normalized_target)
                if index < 5:
                    examples.append(
                        {
                            "prompt": prompt,
                            "prediction": normalized_prediction,
                            "target": normalized_target,
                        }
                    )
        avg_generated_tokens = sum(generated_lengths) / max(1, len(generated_lengths))
        run_metrics.append(
            {
                "benchmark_run": run_index + 1,
                "exact_match": correct / len(records),
                "correct": correct,
                "avg_generated_tokens": avg_generated_tokens,
                "examples": examples,
            }
        )
    if was_training:
        eval_model.train()

    total = len(records)
    exact_mean, exact_std = mean_std([float(run["exact_match"]) for run in run_metrics])
    length_mean, length_std = mean_std([float(run["avg_generated_tokens"]) for run in run_metrics])
    correct_mean, correct_std = mean_std([float(run["correct"]) for run in run_metrics])
    metrics = {
        "step": int(step),
        "eval_file": str(eval_path),
        "num_examples": total,
        "benchmark_runs": benchmark_runs,
        "exact_match": exact_mean,
        "exact_match_mean": exact_mean,
        "exact_match_std": exact_std,
        "correct_mean": correct_mean,
        "correct_std": correct_std,
        "avg_generated_tokens": length_mean,
        "avg_generated_tokens_mean": length_mean,
        "avg_generated_tokens_std": length_std,
        "max_new_tokens": int(max_new_tokens),
        "per_run_summaries": run_metrics,
        "examples": run_metrics[-1]["examples"] if run_metrics else [],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / f"eval_step_{step:06d}.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)
    print(
        f"Eval exact_match={metrics['exact_match_mean']:.4f} ± {metrics['exact_match_std']:.4f} "
        f"| avg_generated_tokens={metrics['avg_generated_tokens_mean']:.2f} ± "
        f"{metrics['avg_generated_tokens_std']:.2f} | runs={benchmark_runs}"
    )
    if length_mean >= 0.9 * int(max_new_tokens):
        print("[warning] Average generated length is close to max_new_tokens; the model may not be stopping cleanly.")
    return metrics


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run regular SFT, or explicitly opt into contrastive positive/negative SFT.")
    parser.add_argument("--config", default="configs/sft.yaml")
    parser.add_argument(
        "--mode",
        choices=("sft", "contrastive"),
        default=None,
        help="Default is regular SFT. Pass --mode contrastive only for positive/negative contrastive SFT.",
    )
    parser.add_argument("--checkpoint", default=None, help="Base/pretrained checkpoint to fine-tune.")
    parser.add_argument("--data-path", default=None, help="Override the SFT .jsonl or .json dataset path from the config.")
    parser.add_argument("--output-dir", default=None, help="Override the fine-tuned checkpoint output directory.")
    parser.add_argument("--epochs", type=float, default=None, help="Train for this many dataset passes.")
    parser.add_argument("--max_steps", "--max-steps", type=int, default=None, help="Override optimizer steps.")
    parser.add_argument("--debug_overfit_samples", type=int, default=None, help="Train on only this many samples.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None)
    parser.add_argument("--max_seq_length", "--max-seq-length", type=int, default=None)
    parser.add_argument("--benchmark-runs", "--benchmark_runs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None, help="Override run.seed for model/init/dropout reproducibility.")
    parser.add_argument("--data_seed", "--data-seed", type=int, default=None, help="Override dataset/DataLoader shuffle seed.")
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable deterministic PyTorch algorithms where available. This can reduce speed.",
    )
    parser.add_argument("--pruning-mask", "--pruning_mask", default=None, help="Optional pruning_masks.pt to keep zeros fixed during SFT retuning.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_config(args.config)
    train_config, sft_config, run_config, generation_config = _copy_flat_config(config)
    train_config.setdefault("save_final_only", True)

    if args.per_device_train_batch_size is not None:
        train_config["batch_size"] = int(args.per_device_train_batch_size)
    if args.gradient_accumulation_steps is not None:
        train_config["grad_accum_steps"] = int(args.gradient_accumulation_steps)
    if args.max_seq_length is not None:
        sft_config["max_length"] = int(args.max_seq_length)
    if args.benchmark_runs is not None:
        sft_config["benchmark_runs"] = int(args.benchmark_runs)
    if args.output_dir is not None:
        run_config["output_dir"] = args.output_dir
    if args.seed is not None:
        run_config["seed"] = int(args.seed)
    if args.data_seed is not None:
        config["data"]["seed"] = int(args.data_seed)
        train_config["data_seed"] = int(args.data_seed)
    if args.pruning_mask is not None:
        sft_config["pruning_mask_path"] = args.pruning_mask

    mode = str(args.mode or sft_config.get("mode", "sft"))
    contrastive = mode == "contrastive"
    device, rank, local_rank, world_size = setup_distributed()
    configure_cuda(train_config, rank)
    seed = int(run_config.get("seed", 42))
    data_seed = int(train_config.get("data_seed", config["data"].get("seed", seed)))
    seed_info = set_all_seeds(seed, rank=rank, deterministic=bool(args.deterministic))

    checkpoint = args.checkpoint or sft_config.get("base_model") or run_config.get("base_model")
    if not checkpoint:
        raise ValueError("Set model_name_or_path, sft.base_model, or pass --checkpoint.")
    checkpoint = str(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    configure_tokenizer(tokenizer)
    model = load_model(checkpoint, device, train_config, rank)

    gradient_checkpointing = bool(config["model"].get("gradient_checkpointing", False))
    if gradient_checkpointing:
        maybe_print(rank, "Gradient checkpointing: enabled")
        model.gradient_checkpointing_enable()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
    else:
        maybe_print(rank, "Gradient checkpointing: disabled")

    model.to(device)
    if bool(train_config.get("compile", False)) and hasattr(torch, "compile"):
        maybe_print(rank, "torch.compile: enabled")
        model = torch.compile(model)
    elif bool(train_config.get("compile", False)):
        maybe_print(rank, "[warning] torch.compile requested, but this PyTorch build does not support it.")

    if world_size > 1:
        model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

    pruning_masks = None
    pruning_mask_path = sft_config.get("pruning_mask_path")
    if pruning_mask_path:
        pruning_masks = load_pruning_masks(pruning_mask_path)
        apply_masks(unwrap_model(model), pruning_masks)
        maybe_print(
            rank,
            f"Pruning masks: loaded {len(pruning_masks)} tensors from {pruning_mask_path} "
            f"(sparsity={mask_sparsity(pruning_masks):.4f}); masks will be reapplied after every optimizer step.",
        )

    data_path = args.data_path or sft_config.get("data_path") or config["data"].get("sft_path")
    if not data_path:
        raise ValueError("Set train_file, sft.data_path, or pass --data-path to an SFT .jsonl or .json file.")
    max_seq_length = int(sft_config.get("max_length", 128))
    max_new_tokens = int(generation_config.get("max_new_tokens", 64))
    if max_new_tokens > 64:
        maybe_print(rank, f"[warning] max_new_tokens={max_new_tokens} is too long for this SFT task; capping to 64.")
        max_new_tokens = 64
    if max_seq_length > 512:
        maybe_print(rank, f"[warning] max_seq_length={max_seq_length} is long for short command SFT; consider 128 or 256.")

    max_samples = args.debug_overfit_samples if args.debug_overfit_samples is not None else sft_config.get("max_samples")
    dataloader = build_sft_dataloader(
        path=data_path,
        tokenizer=tokenizer,
        max_length=max_seq_length,
        batch_size=int(train_config["batch_size"]),
        num_workers=int(train_config.get("num_workers", 0)),
        shuffle=True,
        contrastive=contrastive,
        max_samples=max_samples,
        pin_memory=bool(train_config.get("pin_memory", False)),
        persistent_workers=bool(train_config.get("persistent_workers", False)),
        group_by_length=bool(train_config.get("group_by_length", False)),
        drop_last=bool(train_config.get("drop_last", False)),
        rank=rank,
        world_size=world_size,
        seed=data_seed,
    )
    if len(dataloader) <= 0:
        raise ValueError(f"SFT dataset produced no batches: {data_path}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_config["learning_rate"]),
        betas=(float(train_config.get("beta1", 0.9)), float(train_config.get("beta2", 0.95))),
        eps=float(train_config.get("adam_eps", 1e-8)),
        weight_decay=float(train_config.get("weight_decay", 0.0)),
    )
    scaler = torch.cuda.amp.GradScaler(
        enabled=(device.type == "cuda" and str(train_config["precision"]).lower() == "fp16")
    )
    grad_accum_steps = int(train_config.get("grad_accum_steps", 1))
    epochs = args.epochs if args.epochs is not None else train_config.get("epochs")
    optimizer_steps_per_epoch = max(1, int(math.ceil(len(dataloader) / grad_accum_steps)))
    if args.max_steps is not None:
        train_config["max_steps"] = int(args.max_steps)
    elif epochs is not None:
        epochs = float(epochs)
        if epochs <= 0:
            raise ValueError("--epochs must be greater than 0.")
        train_config["max_steps"] = max(1, int(math.ceil(epochs * optimizer_steps_per_epoch)))
    if "warmup_ratio" in train_config:
        train_config["warmup_steps"] = int(float(train_config["warmup_ratio"]) * int(train_config["max_steps"]))

    output_dir = Path(run_config["output_dir"]).expanduser()
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        stale_candidates = [path for path in output_dir.glob("step-*") if path.is_dir()]
        stale_candidates.extend(path for path in (output_dir / "latest", output_dir / "final") if path.exists() or path.is_symlink())
        if stale_candidates:
            maybe_print(
                rank,
                "[checkpoint] removing stale SFT checkpoint folders before training so final eval cannot reuse them: "
                + ", ".join(str(path) for path in stale_candidates),
            )
        remove_stale_sft_checkpoints(output_dir)
    if world_size > 1:
        dist.barrier()

    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else str(device)
    print(f"[rank {rank}] world_size={world_size} local_rank={local_rank} device={device} gpu_name={gpu_name}", flush=True)
    print_startup_summary(
        rank=rank,
        config_path=args.config,
        world_size=world_size,
        local_rank=local_rank,
        device=device,
        train_config=train_config,
        max_seq_length=max_seq_length,
        max_new_tokens=max_new_tokens,
        model=model,
    )
    maybe_print(rank, f"Seeds: run_seed={seed} data_seed={data_seed} deterministic={seed_info['deterministic_algorithms_enabled']}")
    if args.debug_overfit_samples is not None:
        maybe_print(rank, f"Debug overfit mode: using {args.debug_overfit_samples} samples for {train_config['max_steps']} steps.")
    maybe_print(
        rank,
        f"SFT mode={mode} | base={checkpoint} | train_file={data_path} | output={output_dir} | "
        f"micro_batches_per_epoch={len(dataloader)} | optimizer_steps_per_epoch={optimizer_steps_per_epoch}",
    )
    if contrastive:
        maybe_print(
            rank,
            "Contrastive objective: GenLoss(anchor,y) + GenLoss(positive,y) "
            "+ lambda * relu(margin + d(anchor,positive) - d(anchor,negative))",
        )
    else:
        maybe_print(rank, "Decoder-only SFT labels are unshifted; the causal LM loss shifts labels inside the model.")

    align_weight = float(sft_config.get("alignment_weight", 0.1))
    margin = float(sft_config.get("margin", 0.5))
    effective_batch_size = int(train_config["batch_size"]) * int(world_size) * grad_accum_steps
    eval_path = sft_config.get("eval_path")
    if eval_path and not Path(str(eval_path)).expanduser().exists():
        maybe_print(rank, f"[warning] eval_file does not exist and will be skipped: {eval_path}")
        eval_path = None
    eval_strategy = str(train_config.get("eval_strategy", "epoch")).lower()
    save_strategy = str(train_config.get("save_strategy", "steps")).lower()
    save_final_only = bool(train_config.get("save_final_only", True))
    save_every = int(train_config.get("save_every", 500))
    eval_steps = int(train_config.get("eval_steps", save_every))
    log_every = int(train_config.get("log_every", 10))
    maybe_print(rank, f"SFT eval_strategy={eval_strategy} | save_strategy={save_strategy} | save_final_only={save_final_only}")
    if rank == 0:
        write_sft_run_config(
            output_dir=output_dir,
            args=args,
            config=config,
            train_config=train_config,
            sft_config=sft_config,
            run_config=run_config,
            generation_config=generation_config,
            checkpoint=checkpoint,
            data_path=str(data_path),
            eval_path=str(eval_path) if eval_path else None,
            model=model,
            tokenizer=tokenizer,
            world_size=world_size,
            local_rank=local_rank,
            seed_info={**seed_info, "data_seed": data_seed, "base_seed": seed},
            max_seq_length=max_seq_length,
            max_new_tokens=max_new_tokens,
        )

    model.train()
    optimizer.zero_grad(set_to_none=True)
    current_epoch = 0
    if hasattr(dataloader.sampler, "set_epoch"):
        dataloader.sampler.set_epoch(current_epoch)
    data_iter = iter(dataloader)
    progress = trange(1, int(train_config["max_steps"]) + 1, disable=(rank != 0), desc=f"{mode}-sft")
    run_start = time.perf_counter()
    first_batch_debugged = False
    high_loss_checked = False
    retune_loss_curve: list[dict[str, Any]] = []

    for step in progress:
        total_loss_value = 0.0
        gen_loss_value = 0.0
        align_loss_value = 0.0
        lr = lr_for_step(step - 1, train_config)
        set_lr(optimizer, lr)

        for micro_step in range(grad_accum_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                current_epoch += 1
                if hasattr(dataloader.sampler, "set_epoch"):
                    dataloader.sampler.set_epoch(current_epoch)
                data_iter = iter(dataloader)
                batch = next(data_iter)
            batch = move_batch(batch, device)
            if not first_batch_debugged:
                print_tokenized_debug_sample(tokenizer, batch, rank)
                warn_token_bounds(tokenizer, model, batch, rank)
                first_batch_debugged = True

            sync_context = (
                model.no_sync()
                if world_size > 1 and hasattr(model, "no_sync") and micro_step < grad_accum_steps - 1
                else nullcontext()
            )
            with sync_context:
                with autocast_for(device, str(train_config["precision"]).lower()):
                    outputs = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"],
                        use_cache=False,
                    )
                    anchor_gen_loss = outputs.loss
                    if contrastive:
                        positive_outputs = model(
                            input_ids=batch["positive_gen_input_ids"],
                            attention_mask=batch["positive_gen_attention_mask"],
                            labels=batch["positive_gen_labels"],
                            use_cache=False,
                        )
                        positive_gen_loss = positive_outputs.loss
                        gen_loss = anchor_gen_loss + positive_gen_loss
                        align_loss = contrastive_alignment_loss(model, batch, margin=margin)
                        raw_loss = gen_loss + align_weight * align_loss
                    else:
                        gen_loss = anchor_gen_loss
                        align_loss = torch.zeros((), device=device)
                        raw_loss = gen_loss
                    loss = raw_loss / grad_accum_steps

                total_loss_value += float(raw_loss.detach().cpu())
                gen_loss_value += float(gen_loss.detach().cpu())
                align_loss_value += float(align_loss.detach().cpu())
                scaler.scale(loss).backward()

        if float(train_config.get("max_grad_norm", 0.0)) > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_config["max_grad_norm"]))
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        if pruning_masks is not None:
            apply_masks(unwrap_model(model), pruning_masks)

        logged = torch.tensor(
            [
                total_loss_value / grad_accum_steps,
                gen_loss_value / grad_accum_steps,
                align_loss_value / grad_accum_steps,
            ],
            device=device,
        )
        if is_dist():
            dist.all_reduce(logged, op=dist.ReduceOp.AVG)
        logged_loss = float(logged[0].detach().cpu())
        if rank == 0 and pruning_masks is not None:
            retune_loss_curve.append(
                {
                    "step": int(step),
                    "loss": logged_loss,
                    "gen_loss": float(logged[1].detach().cpu()),
                    "align_loss": float(logged[2].detach().cpu()),
                    "learning_rate": float(lr),
                }
            )
        if rank == 0 and not high_loss_checked:
            high_loss_checked = True
            vocab_size = int(unwrap_model(model).config.vocab_size)
            expected_random_loss = math.log(max(2, vocab_size))
            if logged_loss > 30:
                print(
                    f"[warning] First logged loss {logged_loss:.4f} is much larger than "
                    f"log(vocab_size) ~= {expected_random_loss:.2f}."
                )

        if rank == 0 and (step == 1 or step % log_every == 0):
            elapsed = max(1e-6, time.perf_counter() - run_start)
            progress.set_postfix(
                loss=f"{logged_loss:.4f}",
                gen=f"{float(logged[1]):.4f}",
                align=f"{float(logged[2]):.4f}",
                lr=f"{lr:.2e}",
                world_size=world_size,
                egb=effective_batch_size,
                step_s=f"{elapsed / step:.2f}",
            )

        epoch_boundary = step % optimizer_steps_per_epoch == 0
        final_step = step == int(train_config["max_steps"])
        should_save = (
            final_step
            if save_final_only
            else (
                (save_strategy == "epoch" and epoch_boundary)
                or (save_strategy == "steps" and step % save_every == 0)
                or final_step
            )
        )
        should_eval = bool(
            eval_path
            and should_run_eval(
                eval_strategy=eval_strategy,
                step=step,
                eval_steps=eval_steps,
                epoch_boundary=epoch_boundary,
                final_step=final_step,
            )
        )
        if should_eval or should_save:
            if is_dist():
                dist.barrier()
            if rank == 0 and should_eval:
                evaluate_exact_match(
                    model=model,
                    tokenizer=tokenizer,
                    eval_path=str(eval_path),
                    device=device,
                    max_seq_length=max_seq_length,
                    max_new_tokens=max_new_tokens,
                    output_dir=output_dir,
                    step=step,
                    max_samples=sft_config.get("eval_max_samples"),
                    benchmark_runs=int(sft_config.get("benchmark_runs", 1)),
                )
                model.train()
            if rank == 0 and should_save:
                save_checkpoint(
                    model=model,
                    tokenizer=tokenizer,
                    output_dir=output_dir,
                    step=step,
                    pruning_masks=pruning_masks,
                    pruning_mask_source=str(pruning_mask_path) if pruning_mask_path else None,
                    checkpoint_name="final" if save_final_only or final_step else f"step-{step:06d}",
                )
            if is_dist():
                dist.barrier()

    if rank == 0 and pruning_masks is not None:
        write_csv(output_dir / "retune_loss_curve.csv", retune_loss_curve)
    if is_dist():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
