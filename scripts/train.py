#!/usr/bin/env python
from __future__ import annotations

import argparse
import math
import os
import shutil
import sys
from contextlib import nullcontext
from itertools import islice
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from tqdm.auto import trange
from transformers import AutoModelForCausalLM

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from chatlm_decoder.config import load_config
from chatlm_decoder.data import build_dataloader, iter_texts
from chatlm_decoder.model import count_parameters, create_model
from chatlm_decoder.preprocess import ensure_preprocessed_data, preprocessed_data_config
from chatlm_decoder.tokenizer import load_tokenizer, train_tokenizer_from_iterator


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def setup_distributed() -> tuple[torch.device, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if world_size > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed training requires CUDA devices.")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        return torch.device("cuda", local_rank), rank, local_rank, world_size

    return select_device(), rank, local_rank, world_size


def is_main_process(rank: int) -> bool:
    return rank == 0


def maybe_print(rank: int, message: str) -> None:
    if is_main_process(rank):
        print(message)


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    unwrapped = model.module if hasattr(model, "module") else model
    return unwrapped._orig_mod if hasattr(unwrapped, "_orig_mod") else unwrapped


def autocast_for(device: torch.device, precision: str):
    if device.type == "cuda" and precision in {"fp16", "bf16"}:
        dtype = torch.float16 if precision == "fp16" else torch.bfloat16
        return torch.autocast(device_type="cuda", dtype=dtype)
    return nullcontext()


def learning_rate_for_step(step: int, train_config: dict[str, Any]) -> float:
    max_lr = float(train_config["learning_rate"])
    min_lr = float(train_config["min_learning_rate"])
    warmup_steps = int(train_config["warmup_steps"])
    max_steps = int(train_config["max_steps"])

    if warmup_steps > 0 and step < warmup_steps:
        return max_lr * float(step + 1) / float(warmup_steps)

    decay_steps = max(1, max_steps - warmup_steps)
    progress = min(1.0, max(0.0, float(step - warmup_steps) / float(decay_steps)))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + cosine * (max_lr - min_lr)


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def save_checkpoint(
    model: torch.nn.Module,
    tokenizer: Any,
    output_dir: Path,
    step: int,
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
) -> None:
    checkpoint_dir = output_dir / f"step-{step:06d}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    unwrapped = unwrap_model(model)
    unwrapped.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    torch.save({"step": step, "optimizer": optimizer.state_dict(), "config": config}, checkpoint_dir / "trainer_state.pt")

    latest_dir = output_dir / "latest"
    if latest_dir.exists():
        shutil.rmtree(latest_dir)
    shutil.copytree(checkpoint_dir, latest_dir)


def ensure_tokenizer(config: dict[str, Any], rank: int, world_size: int):
    tokenizer_path = Path(config["tokenizer"]["path"]).expanduser()
    if tokenizer_path.exists():
        return load_tokenizer(tokenizer_path)

    if not bool(config["tokenizer"].get("train_if_missing", False)):
        raise FileNotFoundError(
            f"Tokenizer not found at {tokenizer_path}. Run scripts/train_tokenizer.py first "
            "or set tokenizer.train_if_missing: true."
        )

    if rank == 0:
        data_config = ensure_preprocessed_data(config)
        texts = iter_texts(data_config)
        max_samples = config["tokenizer"].get("max_samples")
        if max_samples is not None:
            texts = islice(texts, int(max_samples))

        train_tokenizer_from_iterator(
            texts=texts,
            output_dir=tokenizer_path,
            vocab_size=int(config["tokenizer"].get("vocab_size", 29298)),
            min_frequency=int(config["tokenizer"].get("min_frequency", 2)),
            model_max_length=int(config["model"].get("block_size", 512)),
        )

    if world_size > 1:
        dist.barrier()

    return load_tokenizer(tokenizer_path)


def print_startup_launch_hint(rank: int, config_path: str) -> None:
    maybe_print(
        rank,
        "Recommended 7-GPU H20 launch when physical GPU 1 is occupied:\n"
        "CUDA_VISIBLE_DEVICES=0,2,3,4,5,6,7 \\\n"
        "HF_HUB_ENABLE_HF_TRANSFER=1 \\\n"
        "NCCL_DEBUG=WARN \\\n"
        "TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \\\n"
        f"torchrun --standalone --nproc_per_node=7 scripts/train.py --config {config_path}",
    )


def print_first_batch_debug(
    rank: int,
    local_rank: int,
    world_size: int,
    train_config: dict[str, Any],
    model: torch.nn.Module,
    tokenizer: Any,
    batch: dict[str, torch.Tensor],
    block_size: int,
    effective_global_batch: int,
) -> None:
    if not is_main_process(rank):
        return

    input_ids = batch["input_ids"]
    labels = batch["labels"]
    valid_labels = labels[labels != -100]
    model_vocab_size = int(getattr(unwrap_model(model).config, "vocab_size"))

    labels_valid_min = int(valid_labels.min().detach().cpu()) if valid_labels.numel() else "none"
    labels_valid_max = int(valid_labels.max().detach().cpu()) if valid_labels.numel() else "none"
    effective_tokens = effective_global_batch * int(block_size)

    print(
        "[debug:first_batch]\n"
        f"  CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<not set>')}\n"
        f"  world_size={world_size}\n"
        f"  rank={rank} local_rank={local_rank}\n"
        f"  per_gpu_batch_size={int(train_config['batch_size'])}\n"
        f"  grad_accum_steps={int(train_config['grad_accum_steps'])}\n"
        f"  effective_global_batch={effective_global_batch}\n"
        f"  block_size={int(block_size)}\n"
        f"  effective_tokens_per_optimizer_step={effective_tokens}\n"
        f"  tokenizer_size={len(tokenizer)}\n"
        f"  model.config.vocab_size={model_vocab_size}\n"
        f"  input_ids min/max={int(input_ids.min().detach().cpu())}/{int(input_ids.max().detach().cpu())}\n"
        f"  labels min/max={int(labels.min().detach().cpu())}/{int(labels.max().detach().cpu())}\n"
        f"  labels excluding -100 min/max={labels_valid_min}/{labels_valid_max}"
    )

    if int(input_ids.max().detach().cpu()) >= model_vocab_size:
        print(
            f"[warning] input_ids.max()={int(input_ids.max().detach().cpu())} "
            f">= model.config.vocab_size={model_vocab_size}"
        )
    if valid_labels.numel() and int(valid_labels.max().detach().cpu()) >= model_vocab_size:
        print(
            f"[warning] labels excluding -100 max={int(valid_labels.max().detach().cpu())} "
            f">= model.config.vocab_size={model_vocab_size}"
        )
    if bool((labels < -100).any().detach().cpu()):
        print("[warning] labels contain values less than -100")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a decoder-only Chinese causal language model.")
    parser.add_argument("--config", default="configs/model_0p2b.yaml", help="Path to a YAML config.")
    parser.add_argument("--resume", default=None, help="Optional checkpoint directory to resume model weights from.")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(int(config["run"]["seed"]))
    device, rank, local_rank, world_size = setup_distributed()
    print_startup_launch_hint(rank, args.config)

    tokenizer = ensure_tokenizer(config, rank=rank, world_size=world_size)
    data_config = preprocessed_data_config(config)
    if args.resume:
        model = AutoModelForCausalLM.from_pretrained(args.resume)
    else:
        model = create_model(config["model"], tokenizer)

    model.to(device)
    if bool(config["train"].get("compile", False)) and hasattr(torch, "compile"):
        model = torch.compile(model)
    if world_size > 1:
        model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

    train_config = config["train"]
    grad_accum_steps = int(train_config["grad_accum_steps"])
    per_gpu_batch_size = int(train_config["batch_size"])
    block_size = int(config["model"]["block_size"])
    effective_global_batch = world_size * per_gpu_batch_size * grad_accum_steps
    output_dir = Path(config["run"]["output_dir"]).expanduser()
    if is_main_process(rank):
        output_dir.mkdir(parents=True, exist_ok=True)
    if world_size > 1:
        dist.barrier()

    dataloader = build_dataloader(
        data_config=data_config,
        tokenizer=tokenizer,
        block_size=block_size,
        batch_size=per_gpu_batch_size,
        num_workers=int(train_config.get("num_workers", 0)),
        rank=rank,
        world_size=world_size,
    )
    data_iter = iter(dataloader)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_config["learning_rate"]),
        betas=(float(train_config["beta1"]), float(train_config["beta2"])),
        weight_decay=float(train_config["weight_decay"]),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and train_config["precision"] == "fp16"))

    maybe_print(rank, f"Device: {device} | world_size: {world_size}")
    maybe_print(rank, f"Parameters: {count_parameters(model):,}")
    maybe_print(rank, f"Tokenizer size: {len(tokenizer):,}")
    maybe_print(rank, f"Output: {output_dir}")
    maybe_print(rank, f"Effective global batch: {effective_global_batch}")
    maybe_print(rank, f"Effective tokens per optimizer step: {effective_global_batch * block_size}")

    model.train()
    optimizer.zero_grad(set_to_none=True)
    printed_first_batch_debug = False
    warned_high_first_loss = False

    progress = trange(1, int(train_config["max_steps"]) + 1, desc="training", disable=not is_main_process(rank))
    for step in progress:
        lr = learning_rate_for_step(step - 1, train_config)
        set_optimizer_lr(optimizer, lr)
        accumulated_raw_loss = 0.0

        for micro_step in range(grad_accum_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            batch = {key: value.to(device) for key, value in batch.items()}
            sync_context = (
                model.no_sync()
                if world_size > 1 and hasattr(model, "no_sync") and micro_step < grad_accum_steps - 1
                else nullcontext()
            )
            with sync_context:
                with autocast_for(device, str(train_config["precision"])):
                    outputs = model(**batch)
                    raw_loss = outputs.loss
                    loss = raw_loss / grad_accum_steps

                if not printed_first_batch_debug:
                    print_first_batch_debug(
                        rank=rank,
                        local_rank=local_rank,
                        world_size=world_size,
                        train_config=train_config,
                        model=model,
                        tokenizer=tokenizer,
                        batch=batch,
                        block_size=block_size,
                        effective_global_batch=effective_global_batch,
                    )
                    printed_first_batch_debug = True

                accumulated_raw_loss += float(raw_loss.detach().cpu())
                scaler.scale(loss).backward()

        if float(train_config["max_grad_norm"]) > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_config["max_grad_norm"]))

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        logged_loss = accumulated_raw_loss / grad_accum_steps
        if world_size > 1:
            loss_tensor = torch.tensor(logged_loss, device=device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
            logged_loss = float(loss_tensor.detach().cpu())

        if step == 1 and not warned_high_first_loss:
            if is_main_process(rank) and logged_loss > 30:
                expected_random_loss = math.log(float(getattr(unwrap_model(model).config, "vocab_size")))
                print(
                    f"[warning] first logged loss {logged_loss:.4f} is much larger than "
                    f"log(vocab_size) ~= {expected_random_loss:.2f}. Check labels/token ids."
                )
            warned_high_first_loss = True

        if step % int(train_config["log_every"]) == 0 and is_main_process(rank):
            progress.set_postfix(
                loss=f"{logged_loss:.4f}",
                lr=f"{lr:.2e}",
                world_size=world_size,
                egb=effective_global_batch,
            )

        should_save = step % int(train_config["save_every"]) == 0 or step == int(train_config["max_steps"])
        if should_save and is_main_process(rank):
            save_checkpoint(model, tokenizer, output_dir, step, optimizer, config)
        if should_save and world_size > 1:
            dist.barrier()

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
