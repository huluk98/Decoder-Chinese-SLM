from __future__ import annotations

from collections import defaultdict
import csv
import json
from typing import Any, Iterable

import torch


PROTECTED_NAME_PATTERNS = (
    "embed",
    "embedding",
    "wte",
    "wpe",
    "lm_head",
    "output_head",
    "score",
    "norm",
    "layernorm",
    "ln_",
    "rotary",
    "tokenizer",
)

ATTENTION_LINEAR_NAMES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "out_proj",
    "c_attn",
    "c_proj",
    "query",
    "key",
    "value",
)

MLP_LINEAR_NAMES = (
    "gate_proj",
    "up_proj",
    "down_proj",
    "fc1",
    "fc2",
    "c_fc",
    "c_proj",
    "w1",
    "w2",
    "w3",
)


def _lower_name(name: str) -> str:
    return name.replace("-", "_").lower()


def protected_reason(name: str, module: torch.nn.Module | None = None) -> str:
    lower = _lower_name(name)
    if isinstance(module, torch.nn.Embedding) or "embed" in lower or "wte" in lower or "wpe" in lower:
        return "embedding_or_position_embedding"
    if "lm_head" in lower or "output_head" in lower:
        return "lm_head_or_output_head"
    if "norm" in lower or "layernorm" in lower or ".ln_" in lower or lower.startswith("ln_"):
        return "normalization"
    if "tokenizer" in lower:
        return "tokenizer_related"
    if module is not None and not isinstance(module, torch.nn.Linear):
        return "non_linear_module"
    return "outside_attention_or_mlp_linear_scope"


def is_attention_or_mlp_linear_name(name: str) -> bool:
    lower = _lower_name(name)
    parts = lower.split(".")
    leaf = parts[-1] if parts else lower
    in_attention = any(part in {"self_attn", "attention", "attn"} for part in parts)
    in_mlp = any(part in {"mlp", "feed_forward", "ffn"} for part in parts)
    if in_attention and leaf in ATTENTION_LINEAR_NAMES:
        return True
    if in_mlp and leaf in MLP_LINEAR_NAMES:
        return True
    return False


def is_protected_name(name: str) -> bool:
    lower = _lower_name(name)
    return any(pattern in lower for pattern in PROTECTED_NAME_PATTERNS)


def is_prunable_transformer_linear(name: str, module: torch.nn.Module) -> bool:
    if not isinstance(module, torch.nn.Linear):
        return False
    if is_protected_name(name):
        return False
    return is_attention_or_mlp_linear_name(name)


def named_prunable_linears(model: torch.nn.Module, include_lm_head: bool = False) -> list[tuple[str, torch.nn.Linear]]:
    if include_lm_head:
        raise ValueError("This pruning benchmark protects lm_head/output-head parameters.")
    modules: list[tuple[str, torch.nn.Linear]] = []
    for name, module in model.named_modules():
        if is_prunable_transformer_linear(name, module):
            modules.append((name, module))
    if not modules:
        raise ValueError("No prunable transformer Linear modules were found.")
    return modules


def module_filter_report(model: torch.nn.Module, include_lm_head: bool = False) -> dict[str, Any]:
    prunable = named_prunable_linears(model, include_lm_head=include_lm_head)
    prunable_weight_params = {f"{name}.weight" for name, _module in prunable}
    prunable_module_names = [name for name, _module in prunable]
    prunable_parameter_count = sum(int(module.weight.numel()) for _name, module in prunable)
    total_parameter_count = sum(int(parameter.numel()) for parameter in model.parameters())
    protected: list[dict[str, Any]] = []
    module_lookup = dict(model.named_modules())
    for parameter_name, parameter in model.named_parameters():
        if parameter_name in prunable_weight_params:
            continue
        module_name = parameter_name.rsplit(".", 1)[0] if "." in parameter_name else parameter_name
        module = module_lookup.get(module_name)
        protected.append(
            {
                "name": parameter_name,
                "module": module_name,
                "shape": list(parameter.shape),
                "parameter_count": int(parameter.numel()),
                "reason": protected_reason(parameter_name, module),
            }
        )
    protected_parameter_count = total_parameter_count - prunable_parameter_count
    return {
        "protocol": "decoder-only pruning scope",
        "prunable_scope": "attention and MLP torch.nn.Linear weights only",
        "prunable_module_names": prunable_module_names,
        "prunable_modules": [
            {
                "name": name,
                "shape": list(module.weight.shape),
                "parameter_count": int(module.weight.numel()),
            }
            for name, module in prunable
        ],
        "protected_parameters": protected,
        "protected_module_names": sorted({row["module"] for row in protected}),
        "protected_categories": sorted({row["reason"] for row in protected}),
        "prunable_parameter_count": prunable_parameter_count,
        "protected_parameter_count": protected_parameter_count,
        "total_parameter_count": total_parameter_count,
        "percentage_of_model_in_pruning_mask": prunable_parameter_count / float(total_parameter_count or 1),
        "excluded_from_pruning": [
            "token_embeddings",
            "positional_embeddings",
            "lm_head_or_output_head",
            "tied_output_embeddings",
            "layer_norms_or_rms_norms",
            "bias_terms",
            "tokenizer_related_parameters",
            "non_linear_modules",
        ],
    }


def normalize_sparsity_denominator(value: str | None) -> str:
    denominator = str(value or "prunable").strip().lower().replace("-", "_")
    aliases = {
        "mask": "prunable",
        "prunable_weights": "prunable",
        "prunable_linear": "prunable",
        "prunable_linears": "prunable",
        "total": "whole_model",
        "model": "whole_model",
        "whole": "whole_model",
        "whole_model": "whole_model",
    }
    denominator = aliases.get(denominator, denominator)
    if denominator not in {"prunable", "whole_model"}:
        raise ValueError(
            "prune.sparsity_denominator must be 'prunable' or 'whole_model', "
            f"got {value!r}."
        )
    return denominator


def resolve_prunable_sparsity_for_target(
    model: torch.nn.Module,
    target_sparsity: float,
    denominator: str | None = "prunable",
    include_lm_head: bool = False,
) -> dict[str, Any]:
    target = float(target_sparsity)
    if not 0.0 <= target <= 1.0:
        raise ValueError(f"sparsity must be between 0 and 1, got {target_sparsity}")

    normalized_denominator = normalize_sparsity_denominator(denominator)
    prunable = named_prunable_linears(model, include_lm_head=include_lm_head)
    prunable_parameter_count = sum(int(module.weight.numel()) for _name, module in prunable)
    model_stats = model_parameter_stats(model)
    total_parameter_count = int(model_stats["total_parameters"])
    existing_zero_parameters = int(model_stats["zero_parameters"])
    protected_parameter_count = total_parameter_count - prunable_parameter_count

    if normalized_denominator == "prunable":
        target_prunable_sparsity = target
        target_whole_model_sparsity = None
        target_pruned_prunable_parameters = int(target_prunable_sparsity * prunable_parameter_count)
    else:
        target_whole_zeros = int(round(target * total_parameter_count))
        target_pruned_prunable_parameters = max(0, target_whole_zeros - existing_zero_parameters)
        if target_pruned_prunable_parameters > prunable_parameter_count:
            raise ValueError(
                "Cannot reach requested whole-model sparsity while keeping protected parameters unchanged: "
                f"need to prune {target_pruned_prunable_parameters:,} prunable weights, but only "
                f"{prunable_parameter_count:,} are in the pruning mask."
            )
        target_prunable_sparsity = target_pruned_prunable_parameters / float(prunable_parameter_count or 1)
        target_whole_model_sparsity = target

    return {
        "requested_sparsity": target,
        "target_sparsity_denominator": normalized_denominator,
        "target_prunable_sparsity": float(target_prunable_sparsity),
        "target_whole_model_sparsity": target_whole_model_sparsity,
        "target_pruned_prunable_parameters": int(target_pruned_prunable_parameters),
        "prunable_parameter_count": int(prunable_parameter_count),
        "protected_parameter_count": int(protected_parameter_count),
        "total_parameter_count": int(total_parameter_count),
        "existing_zero_parameters_before_pruning": int(existing_zero_parameters),
        "percentage_of_model_in_pruning_mask": prunable_parameter_count / float(total_parameter_count or 1),
    }


def validate_masks_match_prunable_scope(model: torch.nn.Module, masks: dict[str, torch.Tensor]) -> None:
    prunable_names = {name for name, _module in named_prunable_linears(model)}
    missing = sorted(prunable_names - set(masks))
    if missing:
        raise ValueError(f"Pruning masks are missing prunable modules: {missing}")
    for name in masks:
        if name not in prunable_names:
            raise ValueError(f"Mask includes non-prunable module: {name}")
        if is_protected_name(name):
            raise ValueError(f"Mask includes protected module name: {name}")


def protected_parameter_snapshot(model: torch.nn.Module, masks: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    prunable_weight_names = {f"{name}.weight" for name in masks}
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if name not in prunable_weight_names
    }


def protected_parameter_validation(
    model: torch.nn.Module,
    snapshot: dict[str, torch.Tensor],
) -> dict[str, Any]:
    changed: list[dict[str, Any]] = []
    missing = sorted(set(snapshot) - {name for name, _parameter in model.named_parameters()})
    for name, parameter in model.named_parameters():
        if name not in snapshot:
            continue
        before = snapshot[name]
        after = parameter.detach().cpu()
        same_shape = tuple(before.shape) == tuple(after.shape)
        unchanged = same_shape and torch.equal(before, after)
        if not unchanged:
            diff = (after.float() - before.float()).abs() if same_shape else torch.empty(0)
            changed.append(
                {
                    "name": name,
                    "reason": protected_reason(name),
                    "shape_before": list(before.shape),
                    "shape_after": list(after.shape),
                    "max_abs_delta": float(diff.max().item()) if diff.numel() else None,
                }
            )
    return {
        "protected_parameter_count": sum(int(tensor.numel()) for tensor in snapshot.values()),
        "checked_parameter_tensors": len(snapshot),
        "changed_parameter_tensors": len(changed),
        "missing_parameter_tensors": missing,
        "changed_parameters": changed[:50],
        "valid": not changed and not missing,
    }


def assert_protected_parameters_unchanged(model: torch.nn.Module, snapshot: dict[str, torch.Tensor]) -> dict[str, Any]:
    report = protected_parameter_validation(model, snapshot)
    if not report["valid"]:
        raise RuntimeError(
            "Pruning modified protected parameters. "
            f"changed={report['changed_parameter_tensors']} missing={len(report['missing_parameter_tensors'])}"
        )
    return report


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
        "achieved_whole_model_sparsity": zero_parameters / float(total_parameters or 1),
    }


def sparsity_accounting(model: torch.nn.Module, masks: dict[str, torch.Tensor], target: float) -> dict[str, int | float]:
    mask_stats = mask_parameter_stats(masks)
    model_stats = model_parameter_stats(model)
    total_parameters = int(model_stats["total_parameters"])
    prunable_parameter_count = int(mask_stats["mask_parameter_count"])
    protected_parameter_count = total_parameters - prunable_parameter_count
    return {
        "target_prunable_sparsity": float(target),
        "achieved_prunable_sparsity": float(mask_stats["mask_sparsity"]),
        "achieved_whole_model_sparsity": float(model_stats["model_zero_fraction"]),
        "active_prunable_parameters": int(mask_stats["active_mask_parameters"]),
        "pruned_prunable_parameters": int(mask_stats["pruned_mask_parameters"]),
        "total_prunable_parameters": prunable_parameter_count,
        "prunable_parameter_count": prunable_parameter_count,
        "protected_parameters": protected_parameter_count,
        "protected_parameter_count": protected_parameter_count,
        "total_parameter_count": total_parameters,
        **mask_stats,
        **model_stats,
        **mask_implied_model_stats(total_parameters, masks),
        **masked_weight_stats(model, masks),
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


def collect_weight_norms(model: torch.nn.Module, masks: dict[str, torch.Tensor]) -> dict[str, dict[str, float | int]]:
    module_lookup = dict(model.named_modules())
    rows: dict[str, dict[str, float | int]] = {}
    for name, mask in masks.items():
        module = module_lookup[name]
        weight = module.weight.detach().float()
        mask_bool = mask.to(device=weight.device, dtype=torch.bool)
        active_values = weight.masked_select(mask_bool)
        pruned_values = weight.masked_select(~mask_bool)
        rows[name] = {
            "parameter_count": int(weight.numel()),
            "weight_l1": float(weight.abs().sum().item()),
            "weight_l2": float(torch.linalg.vector_norm(weight).item()),
            "weight_max_abs": float(weight.abs().max().item()) if weight.numel() else 0.0,
            "active_weight_l2": float(torch.linalg.vector_norm(active_values).item()) if active_values.numel() else 0.0,
            "pruned_weight_l2": float(torch.linalg.vector_norm(pruned_values).item()) if pruned_values.numel() else 0.0,
        }
    return rows


def layerwise_zero_fraction(model: torch.nn.Module, masks: dict[str, torch.Tensor]) -> list[dict[str, Any]]:
    module_lookup = dict(model.named_modules())
    rows: list[dict[str, Any]] = []
    for name, mask in masks.items():
        module = module_lookup[name]
        weight = module.weight.detach()
        zero_count = int((weight == 0).sum().item())
        total = int(weight.numel())
        mask_zeros = int((mask == 0).sum().item())
        rows.append(
            {
                "module": name,
                "weight_parameter_count": total,
                "zero_parameters": zero_count,
                "zero_fraction": zero_count / float(total or 1),
                "mask_pruned_parameters": mask_zeros,
                "mask_sparsity": mask_zeros / float(mask.numel() or 1),
            }
        )
    return rows


def weight_norms_before_after(
    before: dict[str, dict[str, float | int]],
    after: dict[str, dict[str, float | int]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name in before:
        row = {"module": name}
        for key, value in before[name].items():
            row[f"before_{key}"] = value
        for key, value in after.get(name, {}).items():
            row[f"after_{key}"] = value
        before_l2 = float(before[name].get("weight_l2", 0.0))
        after_l2 = float(after.get(name, {}).get("weight_l2", 0.0))
        row["weight_l2_delta"] = after_l2 - before_l2
        rows.append(row)
    return rows


def write_json(path: str | Any, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_csv(path: str | Any, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def tensor_stats(tensor: torch.Tensor) -> dict[str, float | int]:
    data = tensor.detach().float().cpu()
    finite = torch.isfinite(data)
    finite_values = data[finite]
    return {
        "numel": int(data.numel()),
        "finite_count": int(finite.sum().item()),
        "nan_count": int(torch.isnan(data).sum().item()),
        "inf_count": int(torch.isinf(data).sum().item()),
        "nonzero_count": int(torch.count_nonzero(data).item()),
        "min": float(finite_values.min().item()) if finite_values.numel() else 0.0,
        "max": float(finite_values.max().item()) if finite_values.numel() else 0.0,
        "mean": float(finite_values.mean().item()) if finite_values.numel() else 0.0,
        "l2": float(torch.linalg.vector_norm(finite_values).item()) if finite_values.numel() else 0.0,
    }


def gradient_saliency_report(gradient_scores: dict[str, torch.Tensor]) -> dict[str, Any]:
    modules = []
    blocking: list[str] = []
    for name, score in gradient_scores.items():
        stats = tensor_stats(score)
        modules.append({"module": name, **stats})
        if int(stats["finite_count"]) != int(stats["numel"]):
            blocking.append(f"{name}: gradient saliency contains NaN/Inf")
        if int(stats["nonzero_count"]) <= 0:
            blocking.append(f"{name}: gradient saliency is all zero")
    return {"modules": modules, "blocking_issues": blocking, "all_modules_valid": not blocking}


def wanda_activation_report(activation_scalers: dict[str, torch.Tensor], masks: dict[str, torch.Tensor]) -> dict[str, Any]:
    modules = []
    blocking: list[str] = []
    for name, mask in masks.items():
        scaler = activation_scalers.get(name)
        if scaler is None:
            blocking.append(f"{name}: missing activation scaler")
            modules.append({"module": name, "missing": True})
            continue
        stats = tensor_stats(scaler)
        expected_dim = int(mask.shape[1])
        modules.append({"module": name, "expected_in_features": expected_dim, **stats})
        if int(stats["numel"]) != expected_dim:
            blocking.append(f"{name}: activation dim {stats['numel']} != in_features {expected_dim}")
        if int(stats["finite_count"]) != int(stats["numel"]):
            blocking.append(f"{name}: activation scaler contains NaN/Inf")
        if int(stats["nonzero_count"]) <= 0:
            blocking.append(f"{name}: activation scaler is all zero")
    return {"modules": modules, "blocking_issues": blocking, "all_modules_valid": not blocking}


def validate_two_of_four_masks(masks: dict[str, torch.Tensor]) -> dict[str, Any]:
    modules = []
    total_valid_groups = 0
    total_invalid_groups = 0
    invalid_examples: list[dict[str, Any]] = []
    for name, mask in masks.items():
        if mask.shape[1] % 4 != 0:
            raise ValueError(f"2:4 validation requires in_features divisible by 4: {name} shape={tuple(mask.shape)}")
        grouped = mask.bool().cpu().reshape(mask.shape[0], mask.shape[1] // 4, 4)
        kept = grouped.sum(dim=-1)
        valid = kept.eq(2)
        invalid = ~valid
        valid_groups = int(valid.sum().item())
        invalid_groups = int(invalid.sum().item())
        total_valid_groups += valid_groups
        total_invalid_groups += invalid_groups
        if invalid_groups and len(invalid_examples) < 20:
            coords = invalid.nonzero(as_tuple=False)[: 20 - len(invalid_examples)]
            for coord in coords:
                row = int(coord[0].item())
                group = int(coord[1].item())
                invalid_examples.append(
                    {
                        "module": name,
                        "row": row,
                        "group": group,
                        "kept_count": int(kept[row, group].item()),
                    }
                )
        modules.append(
            {
                "module": name,
                "valid_2of4_groups": valid_groups,
                "invalid_2of4_groups": invalid_groups,
                "achieved_2of4_sparsity": float((~grouped).sum().item()) / float(grouped.numel() or 1),
            }
        )
    return {
        "valid": total_invalid_groups == 0,
        "total_valid_2of4_groups": total_valid_groups,
        "total_invalid_2of4_groups": total_invalid_groups,
        "invalid_group_examples": invalid_examples,
        "achieved_2of4_sparsity": 0.5 if total_valid_groups and total_invalid_groups == 0 else None,
        "modules": modules,
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


def exact_rowwise_score_masks(scores: dict[str, torch.Tensor], sparsity: float) -> dict[str, torch.Tensor]:
    if not 0.0 <= float(sparsity) <= 1.0:
        raise ValueError(f"sparsity must be between 0 and 1, got {sparsity}")
    if not scores:
        raise ValueError("No prunable score tensors were provided.")

    total = sum(int(score.numel()) for score in scores.values())
    target_pruned = max(0, min(total, int(float(sparsity) * total)))
    row_specs: list[dict[str, Any]] = []
    base_counts: dict[str, int] = {}
    base_pruned = 0
    for name, score in scores.items():
        if score.ndim != 2:
            raise ValueError(f"Row-wise pruning expects 2D score tensors, got {name} shape={tuple(score.shape)}")
        cols = int(score.shape[1])
        base_count = max(0, min(cols, int(float(sparsity) * cols)))
        base_counts[name] = base_count
        base_pruned += base_count * int(score.shape[0])
        if base_count < cols:
            next_values = torch.topk(score.detach().float().cpu(), k=base_count + 1, dim=1, largest=False).values[:, -1]
        else:
            next_values = torch.full((score.shape[0],), float("inf"))
        for row, next_value in enumerate(next_values.tolist()):
            row_specs.append({"name": name, "row": row, "base_count": base_count, "next_score": float(next_value)})

    extras_needed = max(0, min(len(row_specs), target_pruned - base_pruned))
    extra_rows = {
        (spec["name"], spec["row"])
        for spec in sorted(row_specs, key=lambda row: row["next_score"])[:extras_needed]
    }

    masks: dict[str, torch.Tensor] = {}
    for name, score in scores.items():
        mask = torch.ones_like(score, dtype=torch.bool)
        if score.shape[1] == 0:
            masks[name] = mask
            continue
        base_count = base_counts[name]
        if base_count > 0:
            prune_idx = torch.topk(score, k=min(base_count, score.shape[1]), dim=1, largest=False).indices
            mask.scatter_(1, prune_idx, False)
        rows_with_extra = [row for extra_name, row in extra_rows if extra_name == name]
        if rows_with_extra and base_count < score.shape[1]:
            next_idx = torch.topk(score, k=base_count + 1, dim=1, largest=False).indices[:, -1]
            row_idx = torch.tensor(rows_with_extra, device=score.device, dtype=torch.long)
            mask[row_idx, next_idx[row_idx]] = False
        masks[name] = mask
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
    scores: dict[str, torch.Tensor] = {}
    for name, module in named_prunable_linears(model, include_lm_head=include_lm_head):
        scaler = activation_scalers.get(name)
        if scaler is None:
            score = module.weight.detach().abs()
        else:
            score = module.weight.detach().abs() * scaler.to(module.weight.device).sqrt().view(1, -1)
        scores[name] = score
    masks.update(exact_rowwise_score_masks(scores, sparsity=sparsity))
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
