#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import torch
import yaml
from tqdm.auto import trange
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from chatlm_decoder.pruning import (
    apply_masks,
    collect_activation_scalers,
    collect_gradient_scores,
    global_magnitude_masks,
    gradient_score_masks,
    mask_sparsity,
    two_of_four_masks,
    wanda_masks,
)
from chatlm_decoder.qwen25_instruct_data import (
    DEFAULT_SYSTEM_PROMPT,
    build_qwen25_instruct_dataloader,
)


def load_yaml(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser()
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def dtype_for(name: str, device: torch.device) -> torch.dtype | None:
    if device.type != "cuda":
        return None
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    return None


def configure_tokenizer(tokenizer: Any) -> None:
    if not hasattr(tokenizer, "apply_chat_template"):
        raise AttributeError("Qwen2.5-Instruct tokenizer must provide apply_chat_template.")
    if tokenizer.eos_token is None:
        raise ValueError("Qwen2.5-Instruct tokenizer eos_token must not be None.")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"


def build_calibration_loader(tokenizer: Any, config: dict[str, Any], prune_config: dict[str, Any]):
    path = prune_config.get("calibration_data_path") or config.get("train_file")
    if not path:
        return None
    return build_qwen25_instruct_dataloader(
        path=path,
        tokenizer=tokenizer,
        max_seq_length=int(prune_config.get("max_length") or config.get("max_seq_length", 256)),
        batch_size=int(prune_config.get("batch_size", 2)),
        system_prompt=str(config.get("system_prompt") or DEFAULT_SYSTEM_PROMPT),
        max_samples=prune_config.get("max_samples"),
        group_by_length=bool(prune_config.get("group_by_length", False)),
        shuffle=False,
        num_workers=int(prune_config.get("num_workers", 0)),
        pin_memory=False,
        persistent_workers=False,
        rank=0,
        world_size=1,
        seed=int(config.get("data_seed") or config.get("seed") or 42),
    )


def make_masks(
    method: str,
    model: torch.nn.Module,
    calibration_loader: Any,
    device: torch.device,
    prune_config: dict[str, Any],
) -> dict[str, torch.Tensor]:
    sparsity = float(prune_config.get("sparsity", 0.5))
    include_lm_head = bool(prune_config.get("include_lm_head", False))
    max_batches = int(prune_config.get("calibration_batches", 128))

    if method == "2of4":
        return two_of_four_masks(model, include_lm_head=include_lm_head)
    if method == "magnitude":
        return global_magnitude_masks(model, sparsity=sparsity, include_lm_head=include_lm_head)
    if calibration_loader is None:
        raise ValueError(f"Qwen pruning method {method} requires prune.calibration_data_path or train_file.")
    if method == "wanda":
        scalers = collect_activation_scalers(
            model,
            calibration_loader,
            device=device,
            max_batches=max_batches,
            include_lm_head=include_lm_head,
        )
        return wanda_masks(
            model,
            activation_scalers=scalers,
            sparsity=sparsity,
            include_lm_head=include_lm_head,
        )
    if method == "gradient":
        scores = collect_gradient_scores(
            model,
            calibration_loader,
            device=device,
            max_batches=max_batches,
            include_lm_head=include_lm_head,
        )
        return gradient_score_masks(
            model,
            gradient_scores=scores,
            sparsity=sparsity,
            include_lm_head=include_lm_head,
        )
    raise ValueError(f"Unknown pruning method: {method}")


def recovery_tune(
    model: torch.nn.Module,
    masks: dict[str, torch.Tensor],
    calibration_loader: Any,
    device: torch.device,
    prune_config: dict[str, Any],
) -> None:
    recovery_steps = int(prune_config.get("recovery_steps", 0))
    if recovery_steps <= 0:
        return
    if calibration_loader is None:
        raise ValueError("prune.recovery_steps requires prune.calibration_data_path or train_file.")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(prune_config.get("recovery_learning_rate", 5e-6)),
        weight_decay=float(prune_config.get("recovery_weight_decay", 0.0)),
    )
    model.train()
    iterator = iter(calibration_loader)
    for _ in trange(recovery_steps, desc="qwen-sparse-recovery"):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(calibration_loader)
            batch = next(iterator)
        batch = {
            key: value.to(device)
            for key, value in batch.items()
            if key in {"input_ids", "attention_mask", "labels"}
        }
        outputs = model(**batch, use_cache=False)
        outputs.loss.backward()
        if float(prune_config.get("max_grad_norm", 1.0)) > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(prune_config.get("max_grad_norm", 1.0)))
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        apply_masks(model, masks)


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
    for parameter in model.parameters():
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


def save_pruned_model(
    model: torch.nn.Module,
    tokenizer: Any,
    masks: dict[str, torch.Tensor],
    output_dir: Path,
    method: str,
    prune_config: dict[str, Any],
    checkpoint: str,
) -> None:
    if output_dir.exists() and bool(prune_config.get("overwrite", False)):
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    mask_path = output_dir / "pruning_masks.pt"
    torch.save({name: mask.cpu() for name, mask in masks.items()}, mask_path)
    metadata = {
        "method": method,
        "phase": "one_shot_qwen25_instruct_prune",
        "base_checkpoint": checkpoint,
        "sparsity": mask_sparsity(masks),
        "target_sparsity": float(prune_config.get("sparsity", 0.5)),
        "include_lm_head": bool(prune_config.get("include_lm_head", False)),
        "recovery_steps": int(prune_config.get("recovery_steps", 0)),
        "uses_qwen_apply_chat_template": True,
        **mask_parameter_stats(masks),
        **model_parameter_stats(model),
        "note": (
            "2of4 produces an NVIDIA semi-structured 2:4 zero pattern in Qwen linear weights. "
            "Actual sparse Tensor Core speedups require a runtime that dispatches 2:4 kernels."
            if method == "2of4"
            else "Qwen2.5-Instruct checkpoint with zeros applied according to pruning masks."
        ),
    }
    with (output_dir / "pruning_report.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(f"Saved Qwen2.5-Instruct pruned checkpoint: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prune a Qwen2.5-Instruct checkpoint with Qwen chat-template calibration data."
    )
    parser.add_argument("--config", default="configs/prune_qwen25_50.yaml")
    parser.add_argument("--method", choices=("2of4", "magnitude", "wanda", "gradient"), default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--dtype", choices=("auto", "bf16", "fp16", "fp32"), default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    prune_config = dict(config.get("prune", {}) or {})
    method = args.method or str(prune_config.get("method", "magnitude"))
    checkpoint = str(args.checkpoint or prune_config.get("base_model") or config.get("model_name_or_path") or "")
    if not checkpoint:
        raise ValueError("Set prune.base_model, model_name_or_path, or pass --checkpoint.")

    output_dir = Path(args.output_dir or prune_config.get("output_dir") or f"runs/pruned-qwen25-{method}").expanduser()
    device = select_device()
    dtype_name = args.dtype or str(prune_config.get("dtype", "auto"))
    dtype = dtype_for(dtype_name, device)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=False)
    configure_tokenizer(tokenizer)
    model_kwargs = {"trust_remote_code": False}
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype
    model = AutoModelForCausalLM.from_pretrained(checkpoint, **model_kwargs).to(device)
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    calibration_loader = build_calibration_loader(tokenizer, config, prune_config)
    masks = make_masks(method, model, calibration_loader, device, prune_config)
    print(f"Applying Qwen {method} masks with actual sparsity {mask_sparsity(masks):.4f}")
    apply_masks(model, masks)
    recovery_tune(model, masks, calibration_loader, device, prune_config)
    save_pruned_model(model, tokenizer, masks, output_dir, method, prune_config, checkpoint)


if __name__ == "__main__":
    main()
