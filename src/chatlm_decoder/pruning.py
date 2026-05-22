from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

import torch


def named_prunable_linears(model: torch.nn.Module, include_lm_head: bool = False) -> list[tuple[str, torch.nn.Linear]]:
    modules: list[tuple[str, torch.nn.Linear]] = []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if not include_lm_head and name.endswith("lm_head"):
            continue
        modules.append((name, module))
    return modules


def apply_masks(model: torch.nn.Module, masks: dict[str, torch.Tensor]) -> None:
    module_lookup = dict(model.named_modules())
    with torch.no_grad():
        for name, mask in masks.items():
            if name not in module_lookup:
                raise KeyError(f"Pruning mask references unknown module: {name}")
            module = module_lookup[name]
            if mask.shape != module.weight.shape:
                raise ValueError(
                    f"Pruning mask shape mismatch for {name}: mask={tuple(mask.shape)} "
                    f"weight={tuple(module.weight.shape)}"
                )
            module.weight.mul_(mask.to(device=module.weight.device, dtype=module.weight.dtype))


def mask_sparsity(masks: dict[str, torch.Tensor]) -> float:
    total = sum(mask.numel() for mask in masks.values())
    zeros = sum(int((mask == 0).sum().item()) for mask in masks.values())
    return zeros / float(total or 1)


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


def mask_implied_model_stats(total_parameters: int, masks: dict[str, torch.Tensor]) -> dict[str, int | float]:
    mask_stats = mask_parameter_stats(masks)
    pruned_parameters = int(mask_stats["pruned_mask_parameters"])
    active_parameters = max(0, int(total_parameters) - pruned_parameters)
    return {
        "mask_implied_active_parameters": active_parameters,
        "mask_implied_pruned_parameters": pruned_parameters,
        "mask_implied_active_fraction": active_parameters / float(total_parameters or 1),
        "mask_implied_pruned_fraction": pruned_parameters / float(total_parameters or 1),
    }


def masked_weight_stats(model: torch.nn.Module, masks: dict[str, torch.Tensor]) -> dict[str, int | float]:
    module_lookup = dict(model.named_modules())
    masked_weight_count = 0
    masked_weight_violation_count = 0
    for name, mask in masks.items():
        if name not in module_lookup:
            raise KeyError(f"Pruning mask references unknown module: {name}")
        module = module_lookup[name]
        if mask.shape != module.weight.shape:
            raise ValueError(
                f"Pruning mask shape mismatch for {name}: mask={tuple(mask.shape)} "
                f"weight={tuple(module.weight.shape)}"
            )
        mask_on_device = mask.to(device=module.weight.device, dtype=torch.bool)
        pruned_values = module.weight.detach().masked_select(~mask_on_device)
        masked_weight_count += int(pruned_values.numel())
        masked_weight_violation_count += int(torch.count_nonzero(pruned_values).item())
    return {
        "masked_weight_count": masked_weight_count,
        "masked_weight_violation_count": masked_weight_violation_count,
        "masked_weight_violation_fraction": masked_weight_violation_count / float(masked_weight_count or 1),
    }


def exact_global_score_masks(scores: dict[str, torch.Tensor], sparsity: float) -> dict[str, torch.Tensor]:
    if not 0.0 <= float(sparsity) <= 1.0:
        raise ValueError(f"sparsity must be between 0 and 1, got {sparsity}")
    names = list(scores)
    if not names:
        raise ValueError("No prunable score tensors were provided.")
    flat_scores = torch.cat([scores[name].detach().float().flatten().cpu() for name in names])
    total = flat_scores.numel()
    keep_count = max(0, min(total, total - int(float(sparsity) * total)))
    if keep_count <= 0:
        return {name: torch.zeros_like(scores[name], dtype=torch.bool) for name in names}
    if keep_count >= total:
        return {name: torch.ones_like(scores[name], dtype=torch.bool) for name in names}

    keep_indices = torch.topk(flat_scores, k=keep_count, largest=True, sorted=False).indices
    flat_mask = torch.zeros(total, dtype=torch.bool)
    flat_mask[keep_indices] = True

    masks: dict[str, torch.Tensor] = {}
    offset = 0
    for name in names:
        numel = scores[name].numel()
        masks[name] = flat_mask[offset : offset + numel].reshape_as(scores[name]).to(scores[name].device)
        offset += numel
    return masks


def global_magnitude_masks(
    model: torch.nn.Module,
    sparsity: float = 0.5,
    include_lm_head: bool = False,
) -> dict[str, torch.Tensor]:
    layers = named_prunable_linears(model, include_lm_head=include_lm_head)
    scores = {name: module.weight.detach().abs() for name, module in layers}
    return exact_global_score_masks(scores, sparsity=sparsity)


def two_of_four_masks(model: torch.nn.Module, include_lm_head: bool = False) -> dict[str, torch.Tensor]:
    masks: dict[str, torch.Tensor] = {}
    for name, module in named_prunable_linears(model, include_lm_head=include_lm_head):
        weight = module.weight.detach()
        mask = torch.ones_like(weight, dtype=torch.bool)
        full_cols = (weight.shape[1] // 4) * 4
        if full_cols != weight.shape[1]:
            raise ValueError(
                f"2of4 pruning requires in_features divisible by 4 for exact 50% sparsity; "
                f"{name} has shape={tuple(weight.shape)}"
            )
        if full_cols == 0:
            masks[name] = mask
            continue
        view = weight[:, :full_cols].abs().reshape(weight.shape[0], full_cols // 4, 4)
        prune_idx = torch.topk(view, k=2, dim=-1, largest=False).indices
        grouped_mask = torch.ones_like(view, dtype=torch.bool)
        grouped_mask.scatter_(-1, prune_idx, False)
        mask[:, :full_cols] = grouped_mask.reshape(weight.shape[0], full_cols)
        masks[name] = mask
    return masks


def wanda_masks(
    model: torch.nn.Module,
    activation_scalers: dict[str, torch.Tensor],
    sparsity: float = 0.5,
    include_lm_head: bool = False,
) -> dict[str, torch.Tensor]:
    if not 0.0 <= float(sparsity) <= 1.0:
        raise ValueError(f"sparsity must be between 0 and 1, got {sparsity}")
    masks: dict[str, torch.Tensor] = {}
    for name, module in named_prunable_linears(model, include_lm_head=include_lm_head):
        scaler = activation_scalers.get(name)
        if scaler is None:
            score = module.weight.detach().abs()
        else:
            score = module.weight.detach().abs() * scaler.to(module.weight.device).sqrt().view(1, -1)
        prune_count = int(float(sparsity) * score.shape[1])
        if prune_count <= 0:
            masks[name] = torch.ones_like(score, dtype=torch.bool)
            continue
        prune_idx = torch.topk(score, k=min(prune_count, score.shape[1]), dim=1, largest=False).indices
        mask = torch.ones_like(score, dtype=torch.bool)
        mask.scatter_(1, prune_idx, False)
        masks[name] = mask
    return masks


def gradient_score_masks(
    model: torch.nn.Module,
    gradient_scores: dict[str, torch.Tensor],
    sparsity: float = 0.5,
    include_lm_head: bool = False,
) -> dict[str, torch.Tensor]:
    layers = named_prunable_linears(model, include_lm_head=include_lm_head)
    scores = {
        name: gradient_scores.get(name, torch.zeros_like(module.weight, device="cpu")).to(module.weight.device)
        for name, module in layers
    }
    return exact_global_score_masks(scores, sparsity=sparsity)


def collect_activation_scalers(
    model: torch.nn.Module,
    batches: Iterable[dict[str, torch.Tensor]],
    device: torch.device,
    max_batches: int,
    include_lm_head: bool = False,
) -> dict[str, torch.Tensor]:
    sums: dict[str, torch.Tensor] = {}
    counts: dict[str, int] = defaultdict(int)
    hooks = []

    def hook_for(name: str):
        def _hook(_module: torch.nn.Module, inputs: tuple[torch.Tensor, ...], _output: torch.Tensor) -> None:
            hidden = inputs[0].detach().float()
            flat = hidden.reshape(-1, hidden.shape[-1])
            sums[name] = sums.get(name, torch.zeros(flat.shape[-1], device="cpu")) + flat.pow(2).sum(dim=0).cpu()
            counts[name] += flat.shape[0]

        return _hook

    for name, module in named_prunable_linears(model, include_lm_head=include_lm_head):
        hooks.append(module.register_forward_hook(hook_for(name)))

    model.eval()
    with torch.no_grad():
        for batch_index, batch in enumerate(batches):
            if batch_index >= int(max_batches):
                break
            batch = {key: value.to(device) for key, value in batch.items() if key in {"input_ids", "attention_mask", "labels"}}
            model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False)

    for hook in hooks:
        hook.remove()

    return {
        name: sums[name] / max(1, counts[name])
        for name in sums
    }


def collect_gradient_scores(
    model: torch.nn.Module,
    batches: Iterable[dict[str, torch.Tensor]],
    device: torch.device,
    max_batches: int,
    include_lm_head: bool = False,
) -> dict[str, torch.Tensor]:
    scores = {
        name: torch.zeros_like(module.weight, device="cpu")
        for name, module in named_prunable_linears(model, include_lm_head=include_lm_head)
    }
    module_lookup = dict(model.named_modules())
    model.train()
    model.zero_grad(set_to_none=True)
    for batch_index, batch in enumerate(batches):
        if batch_index >= int(max_batches):
            break
        batch = {key: value.to(device) for key, value in batch.items() if key in {"input_ids", "attention_mask", "labels"}}
        outputs = model(**batch, use_cache=False)
        outputs.loss.backward()
        for name, score in scores.items():
            module = module_lookup[name]
            if module.weight.grad is not None:
                score.add_((module.weight.detach() * module.weight.grad.detach()).abs().cpu())
        model.zero_grad(set_to_none=True)
    return scores
