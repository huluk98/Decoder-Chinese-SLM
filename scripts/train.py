#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import shutil
import sys
import time
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


def import_deepspeed():
    try:
        import deepspeed
    except ImportError as exc:
        raise RuntimeError(
            "DeepSpeed training was requested, but `deepspeed` is not installed. "
            "Install it with `pip install deepspeed` or recreate the conda env from environment.yml."
        ) from exc
    return deepspeed


def distributed_is_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def maybe_barrier(world_size: int) -> None:
    if world_size > 1 and distributed_is_initialized():
        dist.barrier()


def setup_distributed(use_deepspeed: bool = False, deepspeed_module: Any | None = None) -> tuple[torch.device, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if world_size > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed training requires CUDA devices.")
        torch.cuda.set_device(local_rank)
        if use_deepspeed:
            if deepspeed_module is None:
                raise RuntimeError("DeepSpeed module must be imported before distributed setup.")
            deepspeed_module.init_distributed(dist_backend="nccl")
        else:
            dist.init_process_group(backend="nccl")
        return torch.device("cuda", local_rank), rank, local_rank, world_size

    return select_device(), rank, local_rank, world_size


def is_main_process(rank: int) -> bool:
    return rank == 0


def maybe_print(rank: int, message: str) -> None:
    if is_main_process(rank):
        print(message)


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    unwrapped = model
    while True:
        if hasattr(unwrapped, "module"):
            unwrapped = unwrapped.module
            continue
        if hasattr(unwrapped, "_orig_mod"):
            unwrapped = unwrapped._orig_mod
            continue
        return unwrapped


def autocast_for(device: torch.device, precision: str):
    if device.type == "cuda" and precision in {"fp16", "bf16"}:
        dtype = torch.float16 if precision == "fp16" else torch.bfloat16
        return torch.autocast(device_type="cuda", dtype=dtype)
    return nullcontext()


def configure_torch_backends(train_config: dict[str, Any], rank: int) -> None:
    if not torch.cuda.is_available():
        return

    tf32_enabled = bool(train_config.get("tf32", False))
    matmul_precision = str(train_config.get("float32_matmul_precision", "highest"))
    cudnn_benchmark = bool(train_config.get("cudnn_benchmark", False))
    sdp_flash = bool(train_config.get("sdp_flash", True))
    sdp_mem_efficient = bool(train_config.get("sdp_mem_efficient", True))
    sdp_math = bool(train_config.get("sdp_math", True))

    fp32_backend_precision = "tf32" if tf32_enabled else "ieee"
    if hasattr(torch.backends, "fp32_precision") and hasattr(torch.backends, "cuda"):
        torch.backends.fp32_precision = fp32_backend_precision
        if hasattr(torch.backends.cuda, "matmul") and hasattr(torch.backends.cuda.matmul, "fp32_precision"):
            torch.backends.cuda.matmul.fp32_precision = fp32_backend_precision
        if hasattr(torch.backends, "cudnn") and hasattr(torch.backends.cudnn, "fp32_precision"):
            torch.backends.cudnn.fp32_precision = fp32_backend_precision
    else:
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision(matmul_precision)
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = tf32_enabled
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = tf32_enabled
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = cudnn_benchmark
    if hasattr(torch.backends, "cuda"):
        if hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(sdp_flash)
        if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
            torch.backends.cuda.enable_mem_efficient_sdp(sdp_mem_efficient)
        if hasattr(torch.backends.cuda, "enable_math_sdp"):
            torch.backends.cuda.enable_math_sdp(sdp_math)

    maybe_print(
        rank,
        f"TF32: {'enabled' if tf32_enabled else 'disabled'} | "
        f"float32_matmul_precision: {matmul_precision} | "
        f"backend_fp32_precision: {fp32_backend_precision} | "
        f"SDPA flash/mem_efficient/math: {sdp_flash}/{sdp_mem_efficient}/{sdp_math} | "
        f"cudnn_benchmark: {cudnn_benchmark}",
    )


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


class MetricsLogger:
    def __init__(self, output_dir: Path, enabled: bool = True) -> None:
        self.enabled = enabled
        self.handle = None
        self.writer = None
        if not enabled:
            return

        metrics_dir = output_dir / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        self.path = metrics_dir / "training_metrics.csv"
        is_new_file = not self.path.exists() or self.path.stat().st_size == 0
        self.handle = self.path.open("a", encoding="utf-8", newline="")
        self.writer = csv.DictWriter(
            self.handle,
            fieldnames=[
                "time_seconds",
                "step",
                "loss",
                "lr",
                "world_size",
                "effective_global_batch",
                "block_size",
                "tokens_per_step",
                "tokens_per_second",
                "seconds_per_step",
            ],
        )
        if is_new_file:
            self.writer.writeheader()
            self.handle.flush()

    def log(self, row: dict[str, Any]) -> None:
        if not self.enabled or self.writer is None or self.handle is None:
            return
        self.writer.writerow(row)
        self.handle.flush()

    def close(self) -> None:
        if self.handle is not None:
            self.handle.close()


def deepspeed_settings(train_config: dict[str, Any]) -> dict[str, Any]:
    settings = train_config.get("deepspeed", {})
    if settings is None:
        return {}
    if isinstance(settings, bool):
        return {"enabled": settings}
    if not isinstance(settings, dict):
        raise TypeError("train.deepspeed must be a mapping or boolean.")
    return settings


def resolve_project_path(path: str | Path, config: dict[str, Any]) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate

    config_dir = Path(config.get("_config_dir", PROJECT_ROOT))
    for base in (config_dir, PROJECT_ROOT):
        resolved = base / candidate
        if resolved.exists():
            return resolved
    return PROJECT_ROOT / candidate


def load_deepspeed_config(
    config: dict[str, Any],
    deepspeed_config_path: str | None,
    world_size: int,
) -> dict[str, Any]:
    train_config = config["train"]
    settings = deepspeed_settings(train_config)
    config_path = deepspeed_config_path or settings.get("config_path")
    deepspeed_config: dict[str, Any] = {}

    if config_path:
        resolved_path = resolve_project_path(str(config_path), config)
        with resolved_path.open("r", encoding="utf-8") as handle:
            deepspeed_config = json.load(handle)
    deepspeed_config = copy.deepcopy(deepspeed_config)

    precision = str(train_config["precision"]).lower()
    batch_size = int(train_config["batch_size"])
    grad_accum_steps = int(train_config["grad_accum_steps"])
    effective_global_batch = int(world_size) * batch_size * grad_accum_steps

    deepspeed_config["train_micro_batch_size_per_gpu"] = batch_size
    deepspeed_config["gradient_accumulation_steps"] = grad_accum_steps
    deepspeed_config["train_batch_size"] = effective_global_batch
    if float(train_config["max_grad_norm"]) > 0:
        deepspeed_config["gradient_clipping"] = float(train_config["max_grad_norm"])

    if precision == "bf16":
        deepspeed_config["bf16"] = {"enabled": True}
        deepspeed_config["fp16"] = {"enabled": False}
    elif precision == "fp16":
        deepspeed_config["bf16"] = {"enabled": False}
        deepspeed_config["fp16"] = {"enabled": True}
    else:
        deepspeed_config["bf16"] = {"enabled": False}
        deepspeed_config["fp16"] = {"enabled": False}

    deepspeed_config.setdefault(
        "zero_optimization",
        {
            "stage": 1,
            "contiguous_gradients": True,
            "overlap_comm": True,
        },
    )
    deepspeed_config.setdefault("wall_clock_breakdown", False)
    return deepspeed_config


def build_optimizer(
    model: torch.nn.Module,
    train_config: dict[str, Any],
    use_deepspeed: bool,
    rank: int,
) -> torch.optim.Optimizer:
    settings = deepspeed_settings(train_config)
    learning_rate = float(train_config["learning_rate"])
    betas = (float(train_config["beta1"]), float(train_config["beta2"]))
    weight_decay = float(train_config["weight_decay"])
    eps = float(train_config.get("adam_eps", 1e-8))

    if use_deepspeed and bool(settings.get("fused_adam", True)):
        try:
            from deepspeed.ops.adam import FusedAdam

            maybe_print(rank, "Optimizer: DeepSpeed FusedAdam")
            return FusedAdam(model.parameters(), lr=learning_rate, betas=betas, eps=eps, weight_decay=weight_decay)
        except Exception as exc:
            maybe_print(rank, f"[warning] DeepSpeed FusedAdam unavailable ({exc}); falling back to torch.optim.AdamW.")

    maybe_print(rank, "Optimizer: torch.optim.AdamW")
    return torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
    )


def save_checkpoint(
    model: torch.nn.Module,
    tokenizer: Any,
    output_dir: Path,
    step: int,
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    rank: int,
    world_size: int,
    use_deepspeed: bool = False,
) -> None:
    checkpoint_dir = output_dir / f"step-{step:06d}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if use_deepspeed:
        if not hasattr(model, "save_checkpoint"):
            raise RuntimeError("DeepSpeed checkpoint requested, but the model is not a DeepSpeed engine.")

        deepspeed_dir = checkpoint_dir / "deepspeed"
        model.save_checkpoint(str(deepspeed_dir), client_state={"step": step, "config": config})
        maybe_barrier(world_size)

        if is_main_process(rank):
            unwrapped = unwrap_model(model)
            unwrapped.save_pretrained(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)
            torch.save({"step": step, "config": config}, checkpoint_dir / "trainer_state.pt")

            latest_dir = output_dir / "latest"
            if latest_dir.exists():
                shutil.rmtree(latest_dir)
            shutil.copytree(checkpoint_dir, latest_dir)
        maybe_barrier(world_size)
        return

    if not is_main_process(rank):
        maybe_barrier(world_size)
        return

    unwrapped = unwrap_model(model)
    unwrapped.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    torch.save({"step": step, "optimizer": optimizer.state_dict(), "config": config}, checkpoint_dir / "trainer_state.pt")

    latest_dir = output_dir / "latest"
    if latest_dir.exists():
        shutil.rmtree(latest_dir)
    shutil.copytree(checkpoint_dir, latest_dir)
    maybe_barrier(world_size)


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

    maybe_barrier(world_size)

    return load_tokenizer(tokenizer_path)


def _h20_launch_profile(config_path: str, world_size: int) -> tuple[int, str, str]:
    if "8gpu" in config_path or world_size == 8:
        return 8, "0,1,2,3,4,5,6,7", "configs/accelerate_h20_8gpu.yaml"
    if "7gpu" in config_path or world_size == 7:
        return 7, "0,2,3,4,5,6,7", "configs/accelerate_h20_7gpu.yaml"
    return max(1, world_size), "0,1,2,3,4,5,6,7", "configs/accelerate_h20_8gpu.yaml"


def print_startup_launch_hint(rank: int, config_path: str, use_deepspeed: bool, world_size: int) -> None:
    launch_gpus, visible_devices, accelerate_config = _h20_launch_profile(config_path, world_size)
    primary_launcher = (
        f"deepspeed --num_gpus={launch_gpus} scripts/train.py --config {config_path}"
        if use_deepspeed
        else f"torchrun --standalone --nproc_per_node={launch_gpus} scripts/train.py --config {config_path}"
    )
    accelerate_launcher = (
        f"accelerate launch --config_file {accelerate_config} "
        f"scripts/train.py --config {config_path}"
    )
    occupancy_note = (
        " when physical GPU 1 is occupied"
        if launch_gpus == 7
        else " using all 8 visible H20 GPUs"
    )
    maybe_print(
        rank,
        f"Recommended {launch_gpus}-GPU H20 launch{occupancy_note}:\n"
        f"CUDA_VISIBLE_DEVICES={visible_devices} \\\n"
        "HF_HUB_ENABLE_HF_TRANSFER=1 \\\n"
        "NCCL_DEBUG=WARN \\\n"
        "TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \\\n"
        f"{primary_launcher}\n\n"
        "Accelerate launcher for the same visible GPUs:\n"
        f"CUDA_VISIBLE_DEVICES={visible_devices} \\\n"
        "HF_HUB_ENABLE_HF_TRANSFER=1 \\\n"
        "NCCL_DEBUG=WARN \\\n"
        "TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \\\n"
        f"{accelerate_launcher}",
    )


def expected_h20_world_size(config_path: str) -> int | None:
    name = Path(config_path).name.lower()
    if "h20" not in name:
        return None
    if "8gpu" in name:
        return 8
    if "7gpu" in name:
        return 7
    return None


def visible_cuda_device_count() -> int | None:
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not visible_devices:
        return None
    return len([item for item in (part.strip() for part in visible_devices.split(",")) if item])


def validate_h20_launch(rank: int, config_path: str, world_size: int) -> None:
    expected_world_size = expected_h20_world_size(config_path)
    if expected_world_size is None:
        return

    allow_mismatch = os.environ.get("ALLOW_H20_WORLD_SIZE_MISMATCH", "").lower() in {"1", "true", "yes"}
    problems: list[str] = []
    if world_size != expected_world_size:
        problems.append(f"world_size={world_size}, expected {expected_world_size}")

    visible_count = visible_cuda_device_count()
    if visible_count is not None and visible_count != expected_world_size:
        problems.append(
            f"CUDA_VISIBLE_DEVICES has {visible_count} entries, expected {expected_world_size}: "
            f"{os.environ.get('CUDA_VISIBLE_DEVICES')}"
        )

    if not problems:
        return

    message = (
        "H20 launch mismatch for this config: "
        + "; ".join(problems)
        + ". Use the matching scripts/launch_h20_* launcher, or set "
        "ALLOW_H20_WORLD_SIZE_MISMATCH=1 for a deliberate debug run."
    )
    if allow_mismatch:
        maybe_print(rank, f"[warning] {message}")
        return
    raise RuntimeError(message)


def launched_with_accelerate() -> bool:
    return any(
        key in os.environ
        for key in (
            "ACCELERATE_CONFIG_FILE",
            "ACCELERATE_DYNAMO_BACKEND",
            "ACCELERATE_MIXED_PRECISION",
            "ACCELERATE_USE_CPU",
            "ACCELERATE_USE_DEEPSPEED",
            "ACCELERATE_USE_FSDP",
        )
    )


def warn_if_accelerate_precision_differs(train_config: dict[str, Any], rank: int) -> None:
    accelerate_precision = os.environ.get("ACCELERATE_MIXED_PRECISION")
    if not accelerate_precision:
        return

    train_precision = str(train_config["precision"]).lower()
    accelerate_precision = accelerate_precision.lower()
    if accelerate_precision in {"no", "none"}:
        return
    if accelerate_precision != train_precision:
        maybe_print(
            rank,
            "[warning] Accelerate mixed precision is "
            f"{accelerate_precision}, but train.precision is {train_precision}. "
            "The training script uses train.precision for autocast/DeepSpeed config.",
        )


def print_speed_sanity_notes(
    rank: int,
    device: torch.device,
    model_config: dict[str, Any],
    train_config: dict[str, Any],
) -> None:
    if not is_main_process(rank):
        return

    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        memory_gib = props.total_memory / float(1024**3)
        print(f"CUDA device memory: {memory_gib:.1f} GiB")

        if memory_gib >= 80 and bool(model_config.get("gradient_checkpointing", False)):
            print(
                "[speed note] gradient_checkpointing is enabled on a large-memory GPU. "
                "That saves memory but adds recompute, so it can slow a 0.2B model on H20."
            )

    if int(train_config["batch_size"]) < 16:
        print(
            "[speed note] train.batch_size is below 16. On 143711 MiB H20s, a larger "
            "microbatch is usually the first throughput knob to test."
        )
    if int(train_config.get("num_workers", 0)) == 0:
        print(
            "[speed note] train.num_workers is 0. If GPU utilization dips between steps, "
            "try 4-8 workers per rank with persistent_workers=true."
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
    parser.add_argument("--deepspeed", action="store_true", help="Enable DeepSpeed using train.deepspeed config.")
    parser.add_argument("--deepspeed-config", default=None, help="Optional DeepSpeed JSON config path.")
    parser.add_argument("--local_rank", "--local-rank", type=int, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.local_rank is not None and "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    config = load_config(args.config)
    train_config = config["train"]
    use_deepspeed = bool(
        args.deepspeed
        or args.deepspeed_config
        or bool(deepspeed_settings(train_config).get("enabled", False))
    )
    deepspeed_module = import_deepspeed() if use_deepspeed else None

    set_seed(int(config["run"]["seed"]))
    device, rank, local_rank, world_size = setup_distributed(
        use_deepspeed=use_deepspeed,
        deepspeed_module=deepspeed_module,
    )
    validate_h20_launch(rank, args.config, world_size)
    configure_torch_backends(train_config, rank)
    warn_if_accelerate_precision_differs(train_config, rank)
    print_startup_launch_hint(rank, args.config, use_deepspeed=use_deepspeed, world_size=world_size)

    tokenizer = ensure_tokenizer(config, rank=rank, world_size=world_size)
    data_config = preprocessed_data_config(config)
    if args.resume:
        model = AutoModelForCausalLM.from_pretrained(args.resume)
    else:
        model = create_model(config["model"], tokenizer)

    model.to(device)
    if world_size > 1 and not use_deepspeed:
        ddp_kwargs: dict[str, Any] = {
            "device_ids": [local_rank],
            "output_device": local_rank,
            "find_unused_parameters": False,
            "static_graph": bool(train_config.get("ddp_static_graph", True)),
            "gradient_as_bucket_view": bool(train_config.get("ddp_gradient_as_bucket_view", True)),
        }
        bucket_cap_mb = train_config.get("ddp_bucket_cap_mb")
        if bucket_cap_mb is not None:
            ddp_kwargs["bucket_cap_mb"] = int(bucket_cap_mb)
        model = DistributedDataParallel(model, **ddp_kwargs)
    if bool(config["train"].get("compile", False)) and use_deepspeed:
        maybe_print(rank, "[warning] train.compile=true is ignored in DeepSpeed mode for stability.")
    elif bool(config["train"].get("compile", False)) and hasattr(torch, "compile"):
        model = torch.compile(model)

    grad_accum_steps = int(train_config["grad_accum_steps"])
    per_gpu_batch_size = int(train_config["batch_size"])
    block_size = int(config["model"]["block_size"])
    effective_global_batch = world_size * per_gpu_batch_size * grad_accum_steps
    output_dir = Path(config["run"]["output_dir"]).expanduser()
    if is_main_process(rank):
        output_dir.mkdir(parents=True, exist_ok=True)
    maybe_barrier(world_size)

    dataloader = build_dataloader(
        data_config=data_config,
        tokenizer=tokenizer,
        block_size=block_size,
        batch_size=per_gpu_batch_size,
        num_workers=int(train_config.get("num_workers", 0)),
        pin_memory=bool(train_config.get("pin_memory", False)),
        persistent_workers=bool(train_config.get("persistent_workers", False)),
        prefetch_factor=train_config.get("prefetch_factor"),
        rank=rank,
        world_size=world_size,
    )
    data_iter = iter(dataloader)

    optimizer = build_optimizer(model, train_config, use_deepspeed=use_deepspeed, rank=rank)
    if use_deepspeed:
        deepspeed_config = load_deepspeed_config(config, args.deepspeed_config, world_size=world_size)
        model, optimizer, _, _ = deepspeed_module.initialize(
            model=model,
            optimizer=optimizer,
            config=deepspeed_config,
        )
    scaler = torch.cuda.amp.GradScaler(
        enabled=(not use_deepspeed and device.type == "cuda" and train_config["precision"] == "fp16")
    )

    maybe_print(rank, f"Device: {device} | world_size: {world_size}")
    training_backend = "DeepSpeed ZeRO" if use_deepspeed else "PyTorch DDP" if world_size > 1 else "single process"
    launcher = "Accelerate" if launched_with_accelerate() else "DeepSpeed CLI" if use_deepspeed else "torchrun/Python"
    maybe_print(rank, f"Training backend: {training_backend} | launcher: {launcher}")
    maybe_print(rank, f"Parameters: {count_parameters(unwrap_model(model)):,}")
    maybe_print(rank, f"Tokenizer size: {len(tokenizer):,}")
    maybe_print(rank, f"Output: {output_dir}")
    maybe_print(rank, f"Effective global batch: {effective_global_batch}")
    maybe_print(rank, f"Effective tokens per optimizer step: {effective_global_batch * block_size}")
    maybe_print(
        rank,
        "Speed knobs: "
        f"gradient_checkpointing={bool(config['model'].get('gradient_checkpointing', False))} "
        f"compile={bool(train_config.get('compile', False))} "
        f"ddp_static_graph={bool(train_config.get('ddp_static_graph', True))} "
        f"ddp_gradient_as_bucket_view={bool(train_config.get('ddp_gradient_as_bucket_view', True))}",
    )
    maybe_print(
        rank,
        "DataLoader: "
        f"num_workers={int(train_config.get('num_workers', 0))} "
        f"pin_memory={bool(train_config.get('pin_memory', False))} "
        f"persistent_workers={bool(train_config.get('persistent_workers', False))} "
        f"prefetch_factor={train_config.get('prefetch_factor')}",
    )
    print_speed_sanity_notes(rank, device, config["model"], train_config)

    model.train()
    if use_deepspeed:
        model.zero_grad()
    else:
        optimizer.zero_grad(set_to_none=True)
    printed_first_batch_debug = False
    warned_high_first_loss = False

    progress = trange(1, int(train_config["max_steps"]) + 1, desc="training", disable=not is_main_process(rank))
    run_start_time = time.perf_counter()
    last_log_time = time.perf_counter()
    tokens_since_log = 0
    steps_since_log = 0
    metrics_logger = MetricsLogger(output_dir, enabled=is_main_process(rank))
    try:
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

                batch = {key: value.to(device, non_blocking=(device.type == "cuda")) for key, value in batch.items()}
                sync_context = (
                    model.no_sync()
                    if not use_deepspeed
                    and world_size > 1
                    and hasattr(model, "no_sync")
                    and micro_step < grad_accum_steps - 1
                    else nullcontext()
                )
                with sync_context:
                    precision_context = nullcontext() if use_deepspeed else autocast_for(device, str(train_config["precision"]))
                    with precision_context:
                        outputs = model(**batch)
                        raw_loss = outputs.loss

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
                    if use_deepspeed:
                        model.backward(raw_loss)
                        model.step()
                    else:
                        loss = raw_loss / grad_accum_steps
                        scaler.scale(loss).backward()

            if not use_deepspeed:
                if float(train_config["max_grad_norm"]) > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_config["max_grad_norm"]))

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            logged_loss = accumulated_raw_loss / grad_accum_steps
            if world_size > 1 and distributed_is_initialized():
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

            tokens_since_log += effective_global_batch * block_size
            steps_since_log += 1

            if (step == 1 or step % int(train_config["log_every"]) == 0) and is_main_process(rank):
                if device.type == "cuda" and bool(train_config.get("sync_cuda_for_timing", False)):
                    torch.cuda.synchronize(device)
                now = time.perf_counter()
                elapsed = max(1e-6, now - last_log_time)
                tokens_per_second = tokens_since_log / elapsed
                seconds_per_step = elapsed / max(1, steps_since_log)
                progress.set_postfix(
                    loss=f"{logged_loss:.4f}",
                    lr=f"{lr:.2e}",
                    world_size=world_size,
                    egb=effective_global_batch,
                    tok_s=f"{tokens_per_second / 1000.0:.1f}k",
                    step_s=f"{seconds_per_step:.2f}",
                )
                metrics_logger.log(
                    {
                        "time_seconds": f"{now - run_start_time:.3f}",
                        "step": step,
                        "loss": f"{logged_loss:.8f}",
                        "lr": f"{lr:.12g}",
                        "world_size": world_size,
                        "effective_global_batch": effective_global_batch,
                        "block_size": block_size,
                        "tokens_per_step": effective_global_batch * block_size,
                        "tokens_per_second": f"{tokens_per_second:.6f}",
                        "seconds_per_step": f"{seconds_per_step:.6f}",
                    }
                )
                last_log_time = now
                tokens_since_log = 0
                steps_since_log = 0

            should_save = step % int(train_config["save_every"]) == 0 or step == int(train_config["max_steps"])
            if should_save:
                save_checkpoint(
                    model=model,
                    tokenizer=tokenizer,
                    output_dir=output_dir,
                    step=step,
                    optimizer=optimizer,
                    config=config,
                    rank=rank,
                    world_size=world_size,
                    use_deepspeed=use_deepspeed,
                )
    finally:
        metrics_logger.close()

    if distributed_is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
