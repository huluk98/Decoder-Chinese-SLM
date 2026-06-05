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
    assert_protected_parameters_unchanged,
    collect_activation_scalers,
    collect_gradient_scores,
    collect_weight_norms,
    global_magnitude_masks,
    gradient_saliency_report,
    gradient_score_masks,
    layerwise_gradient_score_masks,
    layerwise_magnitude_masks,
    linear_masks_for_activation_report,
    masked_weight_stats,
    mask_sparsity,
    module_filter_report,
    model_parameter_stats,
    normalize_pruning_scope,
    protected_parameter_snapshot,
    resolve_prunable_sparsity_for_target,
    sparsity_accounting,
    two_of_four_masks,
    validate_masks_match_prunable_scope,
    validate_two_of_four_masks,
    wanda_activation_report,
    wanda_masks,
    layerwise_zero_fraction,
    weight_norms_before_after,
    write_csv,
    write_json,
)
from chatlm_decoder.sft_data import build_sft_dataloader


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_causal_lm(checkpoint: str, attn_implementation: str | None = None) -> torch.nn.Module:
    requested_attn = str(attn_implementation or "").strip()
    if requested_attn:
        try:
            return AutoModelForCausalLM.from_pretrained(
                str(checkpoint),
                attn_implementation=requested_attn,
            )
        except Exception as exc:
            print(
                f"[warning] requested attention implementation {requested_attn!r} unavailable ({exc}); "
                "loading checkpoint defaults."
            )
    return AutoModelForCausalLM.from_pretrained(str(checkpoint))


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
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    requested_sparsity = float(prune_config.get("sparsity", 0.5))
    scope = normalize_pruning_scope(prune_config.get("scope", "transformer_linears"))
    include_lm_head = bool(prune_config.get("include_lm_head", scope == "full_model"))
    max_batches = int(prune_config.get("calibration_batches", 128))
    granularity = str(prune_config.get("granularity", prune_config.get("pruning_granularity", "layer"))).lower()
    if granularity in {"per_layer", "layerwise", "per-module", "per_module"}:
        granularity = "layer"
    if granularity not in {"layer", "global"}:
        raise ValueError(f"prune.granularity must be 'layer' or 'global', got {granularity!r}.")
    target_resolution = resolve_prunable_sparsity_for_target(
        model,
        target_sparsity=requested_sparsity,
        denominator=str(prune_config.get("sparsity_denominator", "prunable")),
        include_lm_head=include_lm_head,
        scope=scope,
    )
    sparsity = float(target_resolution["target_prunable_sparsity"])
    prune_config["_target_resolution"] = target_resolution
    prune_config["_resolved_prunable_sparsity"] = sparsity
    prune_config["_pruning_scope"] = scope

    if method == "2of4":
        method_target_note = ""
        if scope != "full_model" and str(target_resolution["target_sparsity_denominator"]) == "whole_model" and abs(sparsity - 0.5) > 1e-12:
            requested_resolution = target_resolution
            target_resolution = resolve_prunable_sparsity_for_target(
                model,
                target_sparsity=0.5,
                denominator="prunable",
                include_lm_head=include_lm_head,
                scope=scope,
            )
            sparsity = 0.5
            prune_config["_target_resolution"] = target_resolution
            prune_config["_resolved_prunable_sparsity"] = sparsity
            method_target_note = (
                "Pure NVIDIA 2:4 is a fixed vanilla method: exactly 50% sparsity within each prunable "
                "Linear 4-weight group. It cannot also satisfy a 50% whole-model target while protected "
                "parameters remain unchanged, so this method is run and reported as 50% prunable 2:4. "
                f"The requested whole-model target would have required "
                f"{float(requested_resolution['target_prunable_sparsity']):.8f} prunable sparsity."
            )
        masks = two_of_four_masks(model, include_lm_head=include_lm_head, scope=scope, sparsity=sparsity)
        validation = validate_two_of_four_masks(masks, model=model, include_lm_head=include_lm_head, scope=scope)
        if not validation["valid"]:
            raise ValueError(
                "Invalid NVIDIA 2:4 mask: "
                f"{validation['total_invalid_2of4_groups']} invalid groups, "
                f"missing eligible modules={validation.get('missing_eligible_modules', [])}"
            )
        return masks, {
            "pruning_granularity": "per_group_of_4_input_weights",
            "score_definition": "in each group of 4 weights, keep top 2 by abs(weight)",
            "method_variant": "full_model_2of4_plus_magnitude_fallback" if scope == "full_model" else "vanilla_nvidia_2of4",
            "method_target_note": method_target_note,
            "target_resolution": target_resolution,
            "nvidia_2of4_validation": validation,
        }
    if method == "magnitude":
        masks = (
            global_magnitude_masks(model, sparsity=sparsity, include_lm_head=include_lm_head, scope=scope)
            if granularity == "global"
            else layerwise_magnitude_masks(model, sparsity=sparsity, include_lm_head=include_lm_head, scope=scope)
        )
        return masks, {
            "pruning_granularity": "global_prunable_parameter" if granularity == "global" else "per_parameter_tensor",
            "score_definition": "abs(parameter)",
            "method_variant": f"{scope}_global_magnitude" if granularity == "global" else f"{scope}_layerwise_magnitude",
            "target_resolution": target_resolution,
        }
    if calibration_loader is None:
        raise ValueError(f"Pruning method {method} requires prune.calibration_data_path.")
    if method == "wanda":
        scalers = collect_activation_scalers(
            model,
            calibration_loader,
            device=device,
            max_batches=max_batches,
            include_lm_head=include_lm_head,
            scope=scope,
        )
        masks = wanda_masks(
            model,
            activation_scalers=scalers,
            sparsity=sparsity,
            include_lm_head=include_lm_head,
            scope=scope,
            granularity=granularity,
        )
        report = wanda_activation_report(
            scalers,
            linear_masks_for_activation_report(
                model,
                masks,
                include_lm_head=include_lm_head,
                scope=scope,
            ),
        )
        if not report["all_modules_valid"]:
            raise ValueError(f"Invalid Wanda activation statistics: {report['blocking_issues']}")
        return masks, {
            "pruning_granularity": "global_prunable_parameter" if granularity == "global" else "per_output_row_linear_plus_per_parameter_tensor",
            "score_definition": "linear weights use abs(weight) * sqrt(mean(input_activation^2)); non-linear full-model tensors use abs(parameter)",
            "method_variant": f"{scope}_wanda",
            "target_resolution": target_resolution,
            "wanda_activation_report": report,
        }
    if method == "gradient":
        scores = collect_gradient_scores(
            model,
            calibration_loader,
            device=device,
            max_batches=max_batches,
            include_lm_head=include_lm_head,
            scope=scope,
        )
        report = gradient_saliency_report(scores, block_all_zero=scope != "full_model")
        if not report["all_modules_valid"]:
            raise ValueError(f"Invalid gradient saliency statistics: {report['blocking_issues']}")
        masks = (
            gradient_score_masks(
                model,
                gradient_scores=scores,
                sparsity=sparsity,
                include_lm_head=include_lm_head,
                scope=scope,
            )
            if granularity == "global"
            else layerwise_gradient_score_masks(
                model,
                gradient_scores=scores,
                sparsity=sparsity,
                include_lm_head=include_lm_head,
                scope=scope,
            )
        )
        return masks, {
            "pruning_granularity": "global_prunable_parameter" if granularity == "global" else "per_parameter_tensor",
            "score_definition": "abs(parameter * gradient)",
            "method_variant": f"{scope}_global_gradient_saliency" if granularity == "global" else f"{scope}_layerwise_gradient_saliency",
            "target_resolution": target_resolution,
            "gradient_saliency_report": report,
        }
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


def save_pruned_model(
    model: torch.nn.Module,
    tokenizer: Any,
    masks: dict[str, torch.Tensor],
    output_dir: Path,
    method: str,
    prune_config: dict[str, Any],
    checkpoint: str,
    diagnostics: dict[str, Any],
    before_norms: dict[str, dict[str, float | int]],
    device: torch.device,
) -> None:
    if output_dir.exists() and bool(prune_config.get("overwrite", False)):
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    mask_path = output_dir / "pruning_masks.pt"
    torch.save({name: mask.cpu() for name, mask in masks.items()}, mask_path)
    requested_sparsity = float(prune_config.get("sparsity", 0.5))
    scope = normalize_pruning_scope(prune_config.get("_pruning_scope", prune_config.get("scope", "transformer_linears")))
    target_resolution = prune_config.get("_target_resolution") or resolve_prunable_sparsity_for_target(
        model,
        target_sparsity=requested_sparsity,
        denominator=str(prune_config.get("sparsity_denominator", "prunable")),
        include_lm_head=bool(prune_config.get("include_lm_head", scope == "full_model")),
        scope=scope,
    )
    target_prunable_sparsity = float(target_resolution["target_prunable_sparsity"])
    accounting = sparsity_accounting(model, masks, target=target_prunable_sparsity)
    after_norms = collect_weight_norms(model, masks)
    per_layer_sparsity = layerwise_zero_fraction(model, masks)
    reload_model = load_causal_lm(
        str(output_dir),
        attn_implementation=prune_config.get("attn_implementation"),
    ).to(device)
    reload_validation = {
        "checkpoint_reloaded": str(output_dir),
        **sparsity_accounting(reload_model, masks, target=target_prunable_sparsity),
    }
    if int(reload_validation.get("masked_weight_violation_count", 0) or 0):
        raise ValueError(f"Pruned weights became nonzero after checkpoint reload: {output_dir}")
    metadata = {
        "method": method,
        "phase": "one_shot_prune",
        "base_checkpoint": checkpoint,
        "sparsity": mask_sparsity(masks),
        "requested_sparsity": requested_sparsity,
        "target_sparsity": requested_sparsity,
        "pruning_scope": scope,
        "target_sparsity_denominator": target_resolution["target_sparsity_denominator"],
        "target_resolution": target_resolution,
        "target_prunable_sparsity": target_prunable_sparsity,
        "target_whole_model_sparsity": target_resolution.get("target_whole_model_sparsity"),
        "method_variant": diagnostics.get("method_variant", ""),
        "method_target_note": diagnostics.get("method_target_note", ""),
        "achieved_prunable_sparsity": accounting["achieved_prunable_sparsity"],
        "achieved_whole_model_sparsity": accounting["achieved_whole_model_sparsity"],
        "include_lm_head": bool(prune_config.get("include_lm_head", scope == "full_model")),
        "recovery_steps": int(prune_config.get("recovery_steps", 0)),
        "checkpoint_evaluated": str(output_dir),
        "checkpoint_reload_validation": reload_validation,
        "per_layer_sparsity": per_layer_sparsity,
        **accounting,
        "note": (
            "2of4 produces an NVIDIA semi-structured 2:4 zero pattern in linear weights. "
            "Actual sparse Tensor Core speedups require an inference/training runtime that uses 2:4 kernels."
            if method == "2of4"
            else "Dense checkpoint with zeros applied according to pruning masks."
        ),
    }
    write_json(
        output_dir / "module_filter_report.json",
        module_filter_report(
            model,
            include_lm_head=bool(prune_config.get("include_lm_head", scope == "full_model")),
            scope=scope,
        ),
    )
    write_json(output_dir / "pruning_report.json", metadata)
    write_json(output_dir / "mask_validation.json", {"method": method, "phase": "one_shot_prune", **accounting})
    write_json(output_dir / "checkpoint_reload_validation.json", reload_validation)
    write_csv(output_dir / "sparsity_by_module.csv", per_layer_sparsity)
    write_csv(output_dir / "layerwise_zero_fraction.csv", per_layer_sparsity)
    write_csv(output_dir / "layerwise_weight_norms_before_after.csv", weight_norms_before_after(before_norms, after_norms))
    if diagnostics.get("gradient_saliency_report"):
        write_json(output_dir / "gradient_saliency_report.json", diagnostics["gradient_saliency_report"])
    if diagnostics.get("wanda_activation_report"):
        write_json(output_dir / "wanda_activation_report.json", diagnostics["wanda_activation_report"])
    if diagnostics.get("nvidia_2of4_validation"):
        write_json(output_dir / "nvidia_2of4_validation.json", diagnostics["nvidia_2of4_validation"])
    if diagnostics.get("protected_parameter_validation"):
        write_json(output_dir / "protected_parameter_validation.json", diagnostics["protected_parameter_validation"])
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
    model = load_causal_lm(str(checkpoint), attn_implementation=prune_config.get("attn_implementation")).to(device)
    model.config.use_cache = False

    calibration_loader = build_calibration_loader(config, tokenizer, prune_config)
    masks, diagnostics = make_masks(method, model, calibration_loader, device, prune_config)
    scope = normalize_pruning_scope(prune_config.get("_pruning_scope", prune_config.get("scope", "transformer_linears")))
    validate_masks_match_prunable_scope(
        model,
        masks,
        include_lm_head=bool(prune_config.get("include_lm_head", scope == "full_model")),
        allow_missing=method == "2of4" and scope != "full_model",
        scope=scope,
    )
    protected_snapshot = protected_parameter_snapshot(model, masks)
    before_norms = collect_weight_norms(model, masks)
    print(f"Applying {method} masks with actual sparsity {mask_sparsity(masks):.4f}")
    apply_masks(model, masks)
    recovery_tune(model, masks, calibration_loader, device, prune_config)
    protected_validation = assert_protected_parameters_unchanged(model, protected_snapshot)
    diagnostics["protected_parameter_validation"] = protected_validation
    save_pruned_model(model, tokenizer, masks, output_dir, method, prune_config, str(checkpoint), diagnostics, before_norms, device)


if __name__ == "__main__":
    main()
