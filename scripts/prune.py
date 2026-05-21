#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import trange
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from chatlm_decoder.config import load_config
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
from chatlm_decoder.sft_data import build_sft_dataloader


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_calibration_loader(config: dict[str, Any], tokenizer: Any, prune_config: dict[str, Any]):
    path = prune_config.get("calibration_data_path") or config.get("sft", {}).get("data_path")
    if not path:
        return None
    return build_sft_dataloader(
        path=path,
        tokenizer=tokenizer,
        max_length=int(prune_config.get("max_length", config["model"].get("block_size", 2048))),
        batch_size=int(prune_config.get("batch_size", config["train"].get("batch_size", 1))),
        num_workers=int(prune_config.get("num_workers", 0)),
        shuffle=False,
        contrastive=False,
        max_samples=prune_config.get("max_samples"),
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
        raise ValueError(f"Pruning method {method} requires prune.calibration_data_path.")
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
        raise ValueError("prune.recovery_steps requires prune.calibration_data_path.")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(prune_config.get("recovery_learning_rate", 5e-6)),
        weight_decay=float(prune_config.get("recovery_weight_decay", 0.0)),
    )
    model.train()
    iterator = iter(calibration_loader)
    for _ in trange(recovery_steps, desc="sparse-recovery"):
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
        total_parameters += int(parameter.numel())
        nonzero_parameters += int(torch.count_nonzero(parameter.detach()).item())
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
) -> None:
    if output_dir.exists() and bool(prune_config.get("overwrite", False)):
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    mask_path = output_dir / "pruning_masks.pt"
    torch.save({name: mask.cpu() for name, mask in masks.items()}, mask_path)
    mask_stats = mask_parameter_stats(masks)
    model_stats = model_parameter_stats(model)
    metadata = {
        "method": method,
        "sparsity": mask_sparsity(masks),
        "target_sparsity": float(prune_config.get("sparsity", 0.5)),
        "include_lm_head": bool(prune_config.get("include_lm_head", False)),
        "recovery_steps": int(prune_config.get("recovery_steps", 0)),
        **mask_stats,
        **model_stats,
        "note": (
            "2of4 produces an NVIDIA semi-structured 2:4 zero pattern in linear weights. "
            "Actual sparse Tensor Core speedups require an inference/training runtime that uses 2:4 kernels."
            if method == "2of4"
            else "Dense checkpoint with zeros applied according to pruning masks."
        ),
    }
    with (output_dir / "pruning_report.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(f"Saved pruned checkpoint: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prune a checkpoint with 2:4, magnitude, Wanda, or gradient-score masks.")
    parser.add_argument("--config", default="configs/prune_50.yaml")
    parser.add_argument("--method", choices=("2of4", "magnitude", "wanda", "gradient"), default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    prune_config = config.get("prune", {})
    method = args.method or str(prune_config.get("method", "magnitude"))
    checkpoint = args.checkpoint or prune_config.get("base_model")
    if not checkpoint:
        raise ValueError("Set prune.base_model or pass --checkpoint.")

    output_dir = Path(args.output_dir or prune_config.get("output_dir") or f"runs/pruned-{method}").expanduser()
    device = select_device()
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint))
    model = AutoModelForCausalLM.from_pretrained(str(checkpoint)).to(device)
    model.config.use_cache = False

    calibration_loader = build_calibration_loader(config, tokenizer, prune_config)
    masks = make_masks(method, model, calibration_loader, device, prune_config)
    print(f"Applying {method} masks with actual sparsity {mask_sparsity(masks):.4f}")
    apply_masks(model, masks)
    recovery_tune(model, masks, calibration_loader, device, prune_config)
    save_pruned_model(model, tokenizer, masks, output_dir, method, prune_config)


if __name__ == "__main__":
    main()
