#!/usr/bin/env python
from __future__ import annotations

import argparse
import math
import os
import shutil
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
from chatlm_decoder.sft_data import build_sft_dataloader


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


def autocast_for(device: torch.device, precision: str):
    if device.type == "cuda" and precision in {"fp16", "bf16"}:
        return torch.autocast("cuda", dtype=torch.float16 if precision == "fp16" else torch.bfloat16)
    return nullcontext()


def configure_cuda(train_config: dict[str, Any]) -> None:
    if not torch.cuda.is_available():
        return
    tf32 = bool(train_config.get("tf32", True))
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = tf32
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = tf32


def lr_for_step(step: int, train_config: dict[str, Any]) -> float:
    max_lr = float(train_config["learning_rate"])
    min_lr = float(train_config.get("min_learning_rate", max_lr * 0.1))
    warmup_steps = int(train_config.get("warmup_steps", 0))
    max_steps = int(train_config["max_steps"])
    if warmup_steps > 0 and step < warmup_steps:
        return max_lr * float(step + 1) / float(warmup_steps)
    decay_steps = max(1, max_steps - warmup_steps)
    progress = min(1.0, max(0.0, float(step - warmup_steps) / float(decay_steps)))
    return min_lr + 0.5 * (1.0 + math.cos(math.pi * progress)) * (max_lr - min_lr)


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def save_checkpoint(model: torch.nn.Module, tokenizer: Any, output_dir: Path, step: int, keep_last: int) -> None:
    checkpoint_dir = output_dir / f"step-{step:06d}"
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    unwrap_model(model).save_pretrained(checkpoint_dir, safe_serialization=True)
    tokenizer.save_pretrained(checkpoint_dir)
    torch.save({"step": step}, checkpoint_dir / "trainer_state.pt")
    latest = output_dir / "latest"
    if latest.is_symlink() or latest.exists():
        if latest.is_dir() and not latest.is_symlink():
            shutil.rmtree(latest)
        else:
            latest.unlink()
    try:
        latest.symlink_to(os.path.relpath(checkpoint_dir, start=output_dir), target_is_directory=True)
    except OSError:
        shutil.copytree(checkpoint_dir, latest)

    checkpoints = sorted(
        [path for path in output_dir.iterdir() if path.is_dir() and path.name.startswith("step-")],
        key=lambda path: path.name,
    )
    for stale in checkpoints[:- int(keep_last)]:
        shutil.rmtree(stale, ignore_errors=True)


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
    return (positive_distance + F.relu(float(margin) - negative_distance)).mean()


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=(device.type == "cuda")) for key, value in batch.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SFT or contrastive positive/negative SFT on a checkpoint.")
    parser.add_argument("--config", default="configs/sft.yaml")
    parser.add_argument("--mode", choices=("sft", "contrastive"), default=None)
    parser.add_argument("--checkpoint", default=None, help="Base/pretrained checkpoint to fine-tune.")
    args = parser.parse_args()

    config = load_config(args.config)
    train_config = config["train"]
    sft_config = config.get("sft", {})
    mode = args.mode or str(sft_config.get("mode", "sft"))
    contrastive = mode == "contrastive"

    device, rank, local_rank, world_size = setup_distributed()
    configure_cuda(train_config)
    torch.manual_seed(int(config["run"]["seed"]) + rank)

    checkpoint = args.checkpoint or sft_config.get("base_model") or config["run"].get("base_model")
    if not checkpoint:
        raise ValueError("Set sft.base_model or pass --checkpoint.")
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint))
    model = AutoModelForCausalLM.from_pretrained(str(checkpoint))
    if bool(config["model"].get("gradient_checkpointing", False)):
        model.gradient_checkpointing_enable()
    model.to(device)
    if world_size > 1:
        model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

    data_path = sft_config.get("data_path") or config["data"].get("sft_path")
    if not data_path:
        raise ValueError("Set sft.data_path to an SFT JSONL file.")
    dataloader = build_sft_dataloader(
        path=data_path,
        tokenizer=tokenizer,
        max_length=int(sft_config.get("max_length", config["model"].get("block_size", 2048))),
        batch_size=int(train_config["batch_size"]),
        num_workers=int(train_config.get("num_workers", 0)),
        shuffle=True,
        contrastive=contrastive,
        max_samples=sft_config.get("max_samples"),
        rank=rank,
        world_size=world_size,
    )
    data_iter = iter(dataloader)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_config["learning_rate"]),
        betas=(float(train_config.get("beta1", 0.9)), float(train_config.get("beta2", 0.95))),
        weight_decay=float(train_config.get("weight_decay", 0.0)),
    )
    scaler = torch.cuda.amp.GradScaler(
        enabled=(device.type == "cuda" and str(train_config["precision"]).lower() == "fp16")
    )
    grad_accum_steps = int(train_config.get("grad_accum_steps", 1))
    output_dir = Path(config["run"]["output_dir"]).expanduser()
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    if world_size > 1:
        dist.barrier()

    align_weight = float(sft_config.get("alignment_weight", 0.1))
    margin = float(sft_config.get("margin", 0.5))
    maybe_print(rank, f"SFT mode={mode} | base={checkpoint} | output={output_dir} | world_size={world_size}")
    maybe_print(rank, "Contrastive objective: gen_loss + lambda * (d(anchor,pos) + relu(margin - d(anchor,neg)))")

    model.train()
    optimizer.zero_grad(set_to_none=True)
    progress = trange(1, int(train_config["max_steps"]) + 1, disable=(rank != 0), desc=f"{mode}-sft")
    run_start = time.perf_counter()
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
                data_iter = iter(dataloader)
                batch = next(data_iter)
            batch = move_batch(batch, device)
            with autocast_for(device, str(train_config["precision"]).lower()):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    use_cache=False,
                )
                gen_loss = outputs.loss
                if contrastive:
                    align_loss = contrastive_alignment_loss(model, batch, margin=margin)
                    raw_loss = gen_loss + align_weight * align_loss
                else:
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
        if rank == 0 and (step == 1 or step % int(train_config.get("log_every", 10)) == 0):
            elapsed = max(1e-6, time.perf_counter() - run_start)
            progress.set_postfix(
                loss=f"{float(logged[0]):.4f}",
                gen=f"{float(logged[1]):.4f}",
                align=f"{float(logged[2]):.4f}",
                lr=f"{lr:.2e}",
                step_s=f"{elapsed / step:.2f}",
            )

        if rank == 0 and (step % int(train_config.get("save_every", 500)) == 0 or step == int(train_config["max_steps"])):
            save_checkpoint(
                model=model,
                tokenizer=tokenizer,
                output_dir=output_dir,
                step=step,
                keep_last=int(train_config.get("save_total_limit", 3)),
            )

    if is_dist():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
