#!/usr/bin/env python
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from chatlm_decoder.pruning import (  # noqa: E402
    apply_masks,
    collect_gradient_scores,
    global_magnitude_masks,
    gradient_saliency_report,
    gradient_score_masks,
    layerwise_magnitude_masks,
    layerwise_gradient_score_masks,
    mask_sparsity,
    model_parameter_stats,
    named_prunable_linears,
    sparsity_accounting,
)
from chatlm_decoder.sparsity_experiments import (  # noqa: E402
    DIFFICULTY_LEVELS,
    add_retention_metrics,
    load_benchmark_samples,
    load_prompt_response_samples,
    prediction_result_rows,
    summarize_prediction_rows,
    write_csv_rows,
)
from chatlm_decoder.tokenizer import move_batch_to_device, prepare_decoder_tokenizer  # noqa: E402

LEGACY_USER_TOKEN = "<|user|>"
LEGACY_ASSISTANT_TOKEN = "<|assistant|>"
LEGACY_EOS_TOKEN = "<|eos|>"
SUMMARY_FIELDNAMES = [
    "experiment_name",
    "model_family",
    "eval_name",
    "eval_path",
    "pruning_mode",
    "pruning_method",
    "target_sparsity",
    "targeted_linear_sparsity_actual",
    "whole_model_sparsity_actual",
    "seed",
    "em1_overall",
    "em1_overall_ci_low",
    "em1_overall_ci_high",
    "em5_overall",
    "em5_overall_ci_low",
    "em5_overall_ci_high",
    "em1_easy",
    "em5_easy",
    "count_easy",
    "em1_medium",
    "em5_medium",
    "count_medium",
    "em1_hard",
    "em5_hard",
    "count_hard",
    "em1_retention_overall",
    "em5_retention_overall",
    "em1_retention_easy",
    "em5_retention_easy",
    "em1_retention_medium",
    "em5_retention_medium",
    "em1_retention_hard",
    "em5_retention_hard",
    "decoding_config_json",
    "training_config_json",
    "pruning_config_json",
    "checkpoint_path",
    "mask_path",
]
PREDICTION_FIELDNAMES = [
    "sample_id",
    "eval_name",
    "input",
    "target",
    "difficulty",
    "top1_prediction",
    "top5_predictions",
    "em1",
    "em5",
    "model_family",
    "pruning_mode",
    "pruning_method",
    "target_sparsity",
    "targeted_linear_sparsity_actual",
    "whole_model_sparsity_actual",
    "seed",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run dense, one-shot, and progressive Linear-weight sparsity experiments for SCENIC IoT benchmarks."
    )
    parser.add_argument("--experiment_name", required=True)
    parser.add_argument("--model_family", required=True, choices=("encoder-only", "encoder_only", "decoder-only", "decoder_only", "encoder-decoder", "encoder_decoder"))
    parser.add_argument("--model_checkpoint", required=True)
    parser.add_argument("--sparsity_levels", nargs="+", type=float, default=[0.0, 0.3, 0.5])
    parser.add_argument("--pruning_modes", nargs="+", default=["dense", "oneshot", "progressive"], choices=("dense", "oneshot", "one_shot", "progressive"))
    parser.add_argument("--pruning_mode", nargs="+", default=None, choices=("dense", "oneshot", "one_shot", "progressive"))
    parser.add_argument("--prune_scope", default="linear_weights", choices=("linear_weights",))
    parser.add_argument("--prune_method", default="magnitude", choices=("magnitude", "gradient", "taylor"))
    parser.add_argument("--progressive_schedule", default="staged", choices=("staged",))
    parser.add_argument("--recovery_epochs_per_stage", type=int, default=1)
    parser.add_argument("--final_recovery_epochs", type=int, default=1)
    parser.add_argument("--gradient_calibration_batches", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--prune_output_heads", action="store_true")
    parser.add_argument("--global_pruning", action="store_true")
    parser.add_argument("--regrowth", action="store_true", help="Allow later progressive masks to regrow previously pruned weights.")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--benchmark_path", required=True)
    parser.add_argument("--benchmark_difficulty_path", default=None)
    parser.add_argument(
        "--extra_eval_path",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Additional final eval split to run after each dense/pruned checkpoint, for example training_dataset=data/train.json.",
    )
    parser.add_argument("--recovery_train_path", default=None, help="Optional prompt/response data for progressive recovery fine-tuning.")
    parser.add_argument("--validation_path", default=None, help="Optional validation data for progressive stage EM logs.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_beams", type=int, default=5)
    parser.add_argument("--num_return_sequences", type=int, default=5)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--normalization_mode", default="command", choices=("exact", "whitespace", "normalized", "punctuation", "command"))
    parser.add_argument("--prompt_format", default="auto", choices=("auto", "legacy", "raw", "qwen-instruct"))
    parser.add_argument("--system_prompt", default="")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto", choices=("auto", "fp32", "fp16", "bf16"))
    parser.add_argument(
        "--expected_world_size",
        "--expected-world-size",
        type=int,
        default=None,
        help="Fail unless the actual torch distributed WORLD_SIZE matches this value.",
    )
    parser.add_argument(
        "--expected_visible_gpu_count",
        "--expected-visible-gpu-count",
        type=int,
        default=None,
        help="Fail unless CUDA_VISIBLE_DEVICES exposes this many GPUs.",
    )
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--bootstrap_resamples", type=int, default=1000)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--length_penalty", type=float, default=1.0)
    parser.add_argument("--early_stopping", action="store_true", default=True)
    parser.add_argument("--skip_plots", action="store_true")
    return parser.parse_args()


def normalize_family(value: str) -> str:
    return value.strip().lower().replace("-", "_")


def normalize_modes(args: argparse.Namespace) -> list[str]:
    raw_modes = args.pruning_mode if args.pruning_mode is not None else args.pruning_modes
    modes = []
    for mode in raw_modes:
        normalized = str(mode).strip().lower().replace("-", "_")
        if normalized == "one_shot":
            normalized = "oneshot"
        if normalized not in {"dense", "oneshot", "progressive"}:
            raise ValueError(f"Unknown pruning mode: {mode}")
        if normalized not in modes:
            modes.append(normalized)
    return modes


def set_seeds(seed: int) -> None:
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def select_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def setup_distributed(requested: str) -> tuple[torch.device, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed sparsity experiments require CUDA.")
        torch.cuda.set_device(local_rank)
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        return torch.device("cuda", local_rank), rank, local_rank, world_size
    return select_device(requested), rank, local_rank, world_size


def visible_cuda_device_count() -> int:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible.strip() or visible.strip() == "-1":
        return int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    return len([part for part in visible.split(",") if part.strip()])


def assert_expected_distributed_run(args: argparse.Namespace, world_size: int) -> None:
    if args.expected_world_size is not None and int(world_size) != int(args.expected_world_size):
        raise RuntimeError(
            f"Linear sparsity expected world_size={int(args.expected_world_size)}, got {int(world_size)}. "
            "Check NPROC_PER_NODE and the torchrun launch command."
        )
    if args.expected_visible_gpu_count is not None:
        visible_count = visible_cuda_device_count()
        if visible_count != int(args.expected_visible_gpu_count):
            raise RuntimeError(
                f"Linear sparsity expected {int(args.expected_visible_gpu_count)} visible GPUs, got {visible_count}: "
                f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')!r}."
            )


def distributed_barrier(local_rank: int) -> None:
    if is_dist():
        if torch.cuda.is_available():
            dist.barrier(device_ids=[int(local_rank)])
        else:
            dist.barrier()


def cleanup_distributed() -> None:
    if is_dist():
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    return int(rank) == 0


def maybe_print(rank: int, message: str) -> None:
    if is_main_process(rank):
        print(message, flush=True)


def maybe_barrier(world_size: int, local_rank: int = 0) -> None:
    if int(world_size) > 1 and is_dist():
        distributed_barrier(local_rank)


def unwrap_model(model: Any) -> Any:
    while hasattr(model, "module"):
        model = model.module
    return model


def all_gather_object(obj: Any, world_size: int) -> list[Any]:
    if int(world_size) <= 1:
        return [obj]
    gathered = [None for _ in range(int(world_size))]
    dist.all_gather_object(gathered, obj)
    return gathered


def dtype_for(name: str, device: torch.device) -> torch.dtype | str:
    if name == "auto":
        return "auto"
    if name == "bf16":
        return torch.bfloat16 if device.type == "cuda" else torch.float32
    if name == "fp16":
        return torch.float16 if device.type == "cuda" else torch.float32
    return torch.float32


def resolve_output_dir(value: str | Path) -> Path:
    output_dir = Path(value).expanduser()
    if not output_dir.is_absolute():
        output_dir = (PROJECT_ROOT / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def configure_tokenizer(tokenizer: Any) -> None:
    prepare_decoder_tokenizer(tokenizer)
    if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token
    if getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})


def load_model_and_tokenizer(args: argparse.Namespace, device: torch.device) -> tuple[Any, Any]:
    from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification, AutoTokenizer

    family = normalize_family(args.model_family)
    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint, trust_remote_code=args.trust_remote_code)
    configure_tokenizer(tokenizer)
    dtype = dtype_for(args.dtype, device)
    model_kwargs: dict[str, Any] = {"trust_remote_code": args.trust_remote_code}
    if dtype != "auto":
        model_kwargs["torch_dtype"] = dtype
    if family == "decoder_only":
        model = AutoModelForCausalLM.from_pretrained(args.model_checkpoint, **model_kwargs)
    elif family == "encoder_decoder":
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_checkpoint, **model_kwargs)
    elif family == "encoder_only":
        model = AutoModelForSequenceClassification.from_pretrained(args.model_checkpoint, **model_kwargs)
    else:
        raise ValueError(f"Unsupported model family: {args.model_family}")
    model.to(device)
    model.eval()
    if hasattr(model, "get_input_embeddings") and model.get_input_embeddings() is not None:
        embedding_rows = int(model.get_input_embeddings().weight.shape[0])
        if len(tokenizer) > embedding_rows and hasattr(model, "resize_token_embeddings"):
            model.resize_token_embeddings(len(tokenizer))
    return model, tokenizer


def decoder_prompt(sample_input: str, tokenizer: Any, args: argparse.Namespace) -> str:
    requested = args.prompt_format
    if requested == "auto":
        has_chat_template = bool(getattr(tokenizer, "chat_template", None))
        requested = "qwen-instruct" if has_chat_template and "instruct" in args.model_checkpoint.lower() else "legacy"
    if requested == "raw":
        return sample_input
    if requested == "legacy":
        return f"{LEGACY_USER_TOKEN}\n{sample_input}\n{LEGACY_ASSISTANT_TOKEN}\n"
    if requested == "qwen-instruct":
        messages = []
        if args.system_prompt:
            messages.append({"role": "system", "content": args.system_prompt})
        messages.append({"role": "user", "content": sample_input})
        return str(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
    raise ValueError(f"Unsupported prompt format: {args.prompt_format}")


def clean_completion(text: str) -> str:
    value = str(text).replace(LEGACY_EOS_TOKEN, "")
    for marker in (LEGACY_USER_TOKEN, "<|im_start|>user", "<|im_start|>system"):
        if marker in value:
            value = value.split(marker, 1)[0]
    return value.strip()


@torch.no_grad()
def generate_decoder_or_seq2seq_candidates(
    model: Any,
    tokenizer: Any,
    samples: list[dict[str, Any]],
    args: argparse.Namespace,
    device: torch.device,
    rank: int = 0,
) -> list[list[str]]:
    family = normalize_family(args.model_family)
    if family == "decoder_only":
        prompts = [decoder_prompt(sample["input"], tokenizer, args) for sample in samples]
    else:
        prompts = [sample["input"] for sample in samples]
    tokenizer.padding_side = "left"
    all_candidates: list[list[str]] = []
    batch_size = max(1, int(args.batch_size))
    return_sequences = max(5, int(args.num_return_sequences))
    beams = max(int(args.num_beams), return_sequences)
    for start in tqdm(range(0, len(prompts), batch_size), desc="benchmark-generate", disable=not is_main_process(rank)):
        batch_prompts = prompts[start : start + batch_size]
        encoded = move_batch_to_device(
            tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max(1, int(args.max_length) - int(args.max_new_tokens)),
                add_special_tokens=False,
            ),
            device,
        )
        prompt_width = int(encoded["input_ids"].shape[-1])
        output_ids = model.generate(
            **encoded,
            max_new_tokens=int(args.max_new_tokens),
            do_sample=False,
            num_beams=beams,
            num_return_sequences=return_sequences,
            length_penalty=float(args.length_penalty),
            early_stopping=bool(args.early_stopping),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        for row in range(len(batch_prompts)):
            candidates = []
            for generated in output_ids[row * return_sequences : (row + 1) * return_sequences]:
                completion_ids = generated[prompt_width:] if family == "decoder_only" else generated
                text = tokenizer.decode(completion_ids, skip_special_tokens=True)
                candidates.append(clean_completion(text))
            all_candidates.append(candidates[:5])
    return all_candidates


@torch.no_grad()
def score_encoder_candidates(
    model: Any,
    tokenizer: Any,
    samples: list[dict[str, Any]],
    args: argparse.Namespace,
    device: torch.device,
    rank: int = 0,
) -> list[list[str]]:
    id2label = getattr(model.config, "id2label", None) or {}
    if not id2label:
        raise ValueError("Encoder-only EM@5 requires model.config.id2label to map class ids to canonical responses.")
    predictions: list[list[str]] = []
    batch_size = max(1, int(args.batch_size))
    for start in tqdm(range(0, len(samples), batch_size), desc="benchmark-score", disable=not is_main_process(rank)):
        batch = samples[start : start + batch_size]
        encoded = move_batch_to_device(
            tokenizer(
                [sample["input"] for sample in batch],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=int(args.max_length),
            ),
            device,
        )
        logits = model(**encoded).logits
        topk = torch.topk(logits, k=min(5, logits.shape[-1]), dim=-1).indices.detach().cpu().tolist()
        for row in topk:
            predictions.append([str(id2label.get(int(index), index)) for index in row])
    return predictions


def evaluate_model(
    model: Any,
    tokenizer: Any,
    samples: list[dict[str, Any]],
    args: argparse.Namespace,
    device: torch.device,
    rank: int = 0,
    world_size: int = 1,
) -> list[list[str]]:
    family = normalize_family(args.model_family)
    model.eval()
    if int(world_size) > 1:
        indexed_samples = [(index, sample) for index, sample in enumerate(samples) if index % int(world_size) == int(rank)]
        local_samples = [sample for _index, sample in indexed_samples]
        if family == "encoder_only":
            local_candidates = score_encoder_candidates(model, tokenizer, local_samples, args, device, rank=rank)
        else:
            local_candidates = generate_decoder_or_seq2seq_candidates(model, tokenizer, local_samples, args, device, rank=rank)
        gathered = all_gather_object(list(zip([index for index, _sample in indexed_samples], local_candidates)), world_size)
        if not is_main_process(rank):
            return []
        ordered: list[tuple[int, list[str]]] = []
        for payload in gathered:
            ordered.extend(payload or [])
        ordered.sort(key=lambda item: int(item[0]))
        return [candidates for _index, candidates in ordered]
    if family == "encoder_only":
        return score_encoder_candidates(model, tokenizer, samples, args, device, rank=rank)
    return generate_decoder_or_seq2seq_candidates(model, tokenizer, samples, args, device, rank=rank)


def targeted_linear_zero_fraction(model: torch.nn.Module, include_heads: bool) -> float:
    total = 0
    zeros = 0
    for _name, module in named_prunable_linears(model, include_lm_head=include_heads):
        data = module.weight.detach()
        total += int(data.numel())
        zeros += int((data == 0).sum().item())
    return zeros / float(total or 1)


def sparsity_stats(model: torch.nn.Module, masks: dict[str, torch.Tensor] | None, target: float, include_heads: bool) -> dict[str, float]:
    if masks:
        accounting = sparsity_accounting(model, masks, target=target)
        return {
            "targeted_linear_sparsity_actual": float(accounting["achieved_prunable_sparsity"]),
            "whole_model_sparsity_actual": float(accounting["achieved_whole_model_sparsity"]),
        }
    model_stats = model_parameter_stats(model)
    return {
        "targeted_linear_sparsity_actual": targeted_linear_zero_fraction(model, include_heads=include_heads),
        "whole_model_sparsity_actual": float(model_stats["achieved_whole_model_sparsity"]),
    }


def make_magnitude_masks(model: torch.nn.Module, sparsity: float, args: argparse.Namespace) -> dict[str, torch.Tensor]:
    if float(sparsity) <= 0.0:
        return {}
    if args.global_pruning:
        return global_magnitude_masks(model, sparsity=float(sparsity), include_lm_head=bool(args.prune_output_heads))
    return layerwise_magnitude_masks(model, sparsity=float(sparsity), include_lm_head=bool(args.prune_output_heads))


def gradient_calibration_batches(
    tokenizer: Any,
    samples: list[dict[str, Any]],
    args: argparse.Namespace,
    device: torch.device,
) -> list[dict[str, torch.Tensor]]:
    if normalize_family(args.model_family) != "decoder_only":
        raise ValueError("Gradient/Taylor progressive pruning is currently supported for decoder-only models.")
    batches: list[dict[str, torch.Tensor]] = []
    for batch in batched(samples, int(args.batch_size)):
        prompt_texts = [decoder_prompt(sample["input"], tokenizer, args) for sample in batch]
        eos = str(getattr(tokenizer, "eos_token", None) or LEGACY_EOS_TOKEN)
        target_texts = [sample["target"] + ("" if str(sample["target"]).endswith(eos) else eos) for sample in batch]
        full_texts = [prompt + target for prompt, target in zip(prompt_texts, target_texts)]
        encoded = move_batch_to_device(
            tokenizer(
                full_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=int(args.max_length),
                add_special_tokens=False,
            ),
            device,
        )
        labels = encoded["input_ids"].clone()
        prompt_encoded = tokenizer(
            prompt_texts,
            padding=True,
            truncation=True,
            max_length=int(args.max_length),
            add_special_tokens=False,
        )
        prompt_lengths = [int(sum(mask)) for mask in prompt_encoded["attention_mask"]]
        for row, prompt_length in enumerate(prompt_lengths):
            labels[row, :prompt_length] = -100
        labels[encoded["attention_mask"] == 0] = -100
        batches.append(
            {
                "input_ids": encoded["input_ids"],
                "attention_mask": encoded["attention_mask"],
                "labels": labels,
            }
        )
        if len(batches) >= int(args.gradient_calibration_batches):
            break
    return batches


def make_gradient_masks(
    model: torch.nn.Module,
    tokenizer: Any,
    sparsity: float,
    recovery_samples: list[dict[str, Any]],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    if float(sparsity) <= 0.0:
        return {}
    if not recovery_samples:
        raise ValueError("--prune_method gradient requires --recovery_train_path for calibration batches.")
    calibration_batches = gradient_calibration_batches(tokenizer, recovery_samples, args, device)
    scores = collect_gradient_scores(
        model,
        calibration_batches,
        device=device,
        max_batches=int(args.gradient_calibration_batches),
        include_lm_head=bool(args.prune_output_heads),
    )
    report = gradient_saliency_report(scores)
    if not report["all_modules_valid"]:
        raise ValueError(f"Invalid gradient saliency statistics: {report['blocking_issues']}")
    if args.global_pruning:
        return gradient_score_masks(
            model,
            gradient_scores=scores,
            sparsity=float(sparsity),
            include_lm_head=bool(args.prune_output_heads),
        )
    return layerwise_gradient_score_masks(
        model,
        gradient_scores=scores,
        sparsity=float(sparsity),
        include_lm_head=bool(args.prune_output_heads),
    )


def make_pruning_masks(
    model: torch.nn.Module,
    tokenizer: Any,
    sparsity: float,
    recovery_samples: list[dict[str, Any]],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    method = str(args.prune_method).strip().lower()
    if method == "magnitude":
        return make_magnitude_masks(model, sparsity, args)
    if method in {"gradient", "taylor"}:
        return make_gradient_masks(model, tokenizer, sparsity, recovery_samples, args, device)
    raise ValueError(f"Unsupported prune_method: {args.prune_method}")


def combine_masks(
    previous: dict[str, torch.Tensor] | None,
    current: dict[str, torch.Tensor],
    *,
    allow_regrowth: bool,
) -> dict[str, torch.Tensor]:
    if previous is None or allow_regrowth:
        return current
    combined = {}
    for name, mask in current.items():
        old = previous.get(name)
        combined[name] = mask if old is None else (old.to(mask.device).bool() & mask.bool())
    return combined


def staged_schedule(target_sparsity: float) -> list[float]:
    target = round(float(target_sparsity), 10)
    if abs(target - 0.3) < 1e-9:
        return [0.1, 0.2, 0.3]
    if abs(target - 0.5) < 1e-9:
        return [0.1, 0.2, 0.3, 0.4, 0.5]
    stages = []
    value = 0.1
    while value < target:
        stages.append(round(value, 10))
        value += 0.1
    if not stages or abs(stages[-1] - target) > 1e-9:
        stages.append(target)
    return stages


def batched(items: list[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    return [items[start : start + batch_size] for start in range(0, len(items), max(1, batch_size))]


def epoch_batches(items: list[dict[str, Any]], batch_size: int, rank: int, world_size: int) -> list[list[dict[str, Any]]]:
    if int(world_size) <= 1:
        return batched(items, batch_size)
    if not items:
        return []
    batch_size = max(1, int(batch_size))
    world_size = max(1, int(world_size))
    global_batch_size = batch_size * world_size
    steps = max(1, math.ceil(len(items) / float(global_batch_size)))
    batches: list[list[dict[str, Any]]] = []
    for step in range(steps):
        base = step * global_batch_size + int(rank) * batch_size
        local = [items[index] for index in range(base, min(base + batch_size, len(items)))]
        while len(local) < batch_size:
            local.append(items[(base + len(local)) % len(items)])
        batches.append(local)
    return batches


def train_one_epoch(
    model: Any,
    train_model: Any,
    tokenizer: Any,
    samples: list[dict[str, Any]],
    masks: dict[str, torch.Tensor],
    args: argparse.Namespace,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    rank: int = 0,
    world_size: int = 1,
) -> float | None:
    if not samples:
        return None
    family = normalize_family(args.model_family)
    if family == "encoder_only":
        raise ValueError("Progressive recovery fine-tuning for encoder-only models needs class ids and is not supported by this runner.")
    model.train()
    train_model.train()
    losses: list[float] = []
    for batch in epoch_batches(samples, int(args.batch_size), rank=rank, world_size=world_size):
        optimizer.zero_grad(set_to_none=True)
        if family == "encoder_decoder":
            encoded = move_batch_to_device(
                tokenizer(
                    [sample["input"] for sample in batch],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=int(args.max_length),
                ),
                device,
            )
            labels = tokenizer(
                [sample["target"] for sample in batch],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=int(args.max_new_tokens),
            )["input_ids"].to(device)
            labels[labels == int(tokenizer.pad_token_id)] = -100
            outputs = train_model(**encoded, labels=labels)
        else:
            prompt_texts = [decoder_prompt(sample["input"], tokenizer, args) for sample in batch]
            eos = str(getattr(tokenizer, "eos_token", None) or LEGACY_EOS_TOKEN)
            target_texts = [sample["target"] + ("" if str(sample["target"]).endswith(eos) else eos) for sample in batch]
            full_texts = [prompt + target for prompt, target in zip(prompt_texts, target_texts)]
            encoded = move_batch_to_device(
                tokenizer(
                    full_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=int(args.max_length),
                    add_special_tokens=False,
                ),
                device,
            )
            labels = encoded["input_ids"].clone()
            prompt_encoded = tokenizer(
                prompt_texts,
                padding=True,
                truncation=True,
                max_length=int(args.max_length),
                add_special_tokens=False,
            )
            prompt_lengths = [int(sum(mask)) for mask in prompt_encoded["attention_mask"]]
            for row, prompt_length in enumerate(prompt_lengths):
                labels[row, :prompt_length] = -100
            labels[encoded["attention_mask"] == 0] = -100
            outputs = train_model(**encoded, labels=labels)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(unwrap_model(train_model).parameters(), 1.0)
        optimizer.step()
        apply_masks(model, masks)
        losses.append(float(loss.detach().cpu()))
    model.eval()
    train_model.eval()
    loss_sum = sum(losses)
    loss_count = len(losses)
    if int(world_size) > 1 and is_dist():
        stats = torch.tensor([loss_sum, float(loss_count)], dtype=torch.float32, device=device)
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        loss_sum = float(stats[0].detach().cpu())
        loss_count = int(stats[1].detach().cpu())
    return loss_sum / float(loss_count or 1) if loss_count else None


def maybe_validation_metrics(
    model: Any,
    tokenizer: Any,
    validation_samples: list[dict[str, Any]],
    args: argparse.Namespace,
    device: torch.device,
    rank: int = 0,
    world_size: int = 1,
) -> tuple[float | None, float | None]:
    if not validation_samples:
        return None, None
    candidates = evaluate_model(model, tokenizer, validation_samples, args, device, rank=rank, world_size=world_size)
    if not is_main_process(rank):
        return None, None
    rows = prediction_result_rows(
        validation_samples,
        candidates,
        normalization_mode=args.normalization_mode,
        model_family=normalize_family(args.model_family),
        pruning_mode="validation",
        pruning_method=args.prune_method,
        target_sparsity=0.0,
        targeted_linear_sparsity_actual=0.0,
        whole_model_sparsity_actual=0.0,
        seed=int(args.seed),
    )
    summary = summarize_prediction_rows(rows, bootstrap_resamples=10, seed=int(args.seed))
    return summary["em1_overall"], summary["em5_overall"]


def save_run_artifacts(
    model: Any,
    tokenizer: Any,
    masks: dict[str, torch.Tensor] | None,
    output_dir: Path,
    model_family: str,
    pruning_mode: str,
    sparsity: float,
    seed: int,
) -> tuple[str, str]:
    run_slug = f"{model_family}_{pruning_mode}_{sparsity:g}_{seed}"
    checkpoint_dir = output_dir / "checkpoints" / run_slug
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint_dir, safe_serialization=True)
    tokenizer.save_pretrained(checkpoint_dir)
    mask_path = ""
    if masks is not None:
        mask_file = output_dir / "masks" / f"masks_{run_slug}.pt"
        mask_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save({name: mask.detach().cpu().bool() for name, mask in masks.items()}, mask_file)
        mask_path = str(mask_file)
    return str(checkpoint_dir), mask_path


def run_progressive(
    model: Any,
    train_model: Any,
    tokenizer: Any,
    target_sparsity: float,
    recovery_samples: list[dict[str, Any]],
    validation_samples: list[dict[str, Any]],
    args: argparse.Namespace,
    output_dir: Path,
    device: torch.device,
    rank: int = 0,
    world_size: int = 1,
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]]]:
    masks: dict[str, torch.Tensor] | None = None
    logs: list[dict[str, Any]] = []
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.learning_rate)) if recovery_samples else None
    for stage_index, stage_sparsity in enumerate(staged_schedule(target_sparsity), start=1):
        stage_masks = make_pruning_masks(model, tokenizer, stage_sparsity, recovery_samples, args, device)
        masks = combine_masks(masks, stage_masks, allow_regrowth=bool(args.regrowth))
        apply_masks(model, masks)
        epochs = max(0, int(args.recovery_epochs_per_stage))
        if epochs == 0 or optimizer is None:
            if is_main_process(rank):
                stats = sparsity_stats(model, masks, stage_sparsity, include_heads=bool(args.prune_output_heads))
                logs.append(
                    {
                        "stage": stage_index,
                        "stage_target_sparsity": stage_sparsity,
                        **stats,
                        "recovery_epoch": 0,
                        "train_loss": None,
                        "val_em1": None,
                        "val_em5": None,
                    }
                )
            continue
        for epoch in range(1, epochs + 1):
            train_loss = train_one_epoch(
                model,
                train_model,
                tokenizer,
                recovery_samples,
                masks,
                args,
                device,
                optimizer,
                rank=rank,
                world_size=world_size,
            )
            val_em1, val_em5 = maybe_validation_metrics(
                model,
                tokenizer,
                validation_samples,
                args,
                device,
                rank=rank,
                world_size=world_size,
            )
            if is_main_process(rank):
                stats = sparsity_stats(model, masks, stage_sparsity, include_heads=bool(args.prune_output_heads))
                logs.append(
                    {
                        "stage": stage_index,
                        "stage_target_sparsity": stage_sparsity,
                        **stats,
                        "recovery_epoch": epoch,
                        "train_loss": train_loss,
                        "val_em1": val_em1,
                        "val_em5": val_em5,
                    }
                )
    if optimizer is not None and int(args.final_recovery_epochs) > 0 and masks is not None:
        final_stage = len(staged_schedule(target_sparsity)) + 1
        for epoch in range(1, int(args.final_recovery_epochs) + 1):
            train_loss = train_one_epoch(
                model,
                train_model,
                tokenizer,
                recovery_samples,
                masks,
                args,
                device,
                optimizer,
                rank=rank,
                world_size=world_size,
            )
            val_em1, val_em5 = maybe_validation_metrics(
                model,
                tokenizer,
                validation_samples,
                args,
                device,
                rank=rank,
                world_size=world_size,
            )
            if is_main_process(rank):
                stats = sparsity_stats(model, masks, target_sparsity, include_heads=bool(args.prune_output_heads))
                logs.append(
                    {
                        "stage": final_stage,
                        "stage_target_sparsity": target_sparsity,
                        **stats,
                        "recovery_epoch": epoch,
                        "train_loss": train_loss,
                        "val_em1": val_em1,
                        "val_em5": val_em5,
                    }
                )
    return masks or {}, logs


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def row_config_json(args: argparse.Namespace, kind: str) -> str:
    if kind == "decoding":
        payload = {
            "num_beams": int(args.num_beams),
            "num_return_sequences": int(args.num_return_sequences),
            "max_new_tokens": int(args.max_new_tokens),
            "length_penalty": float(args.length_penalty),
            "early_stopping": bool(args.early_stopping),
            "normalization_mode": args.normalization_mode,
            "prompt_format": args.prompt_format,
        }
    elif kind == "training":
        payload = {
            "learning_rate": float(args.learning_rate),
            "batch_size": int(args.batch_size),
            "recovery_epochs_per_stage": int(args.recovery_epochs_per_stage),
            "final_recovery_epochs": int(args.final_recovery_epochs),
            "seed": int(args.seed),
            "recovery_train_path": args.recovery_train_path or "",
            "validation_path": args.validation_path or "",
            "gradient_clipping": 1.0,
        }
    else:
        payload = {
            "prune_scope": args.prune_scope,
            "prune_method": args.prune_method,
            "progressive_schedule": args.progressive_schedule,
            "prune_output_heads": bool(args.prune_output_heads),
            "global_pruning": bool(args.global_pruning),
            "regrowth": bool(args.regrowth),
        }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def write_paper_table(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    table_rows = []
    for row in rows:
        table_rows.append(
            {
                "model_family": row["model_family"],
                "pruning_mode": row["pruning_mode"],
                "target_sparsity": row["target_sparsity"],
                "overall EM@1": row.get("em1_overall"),
                "overall EM@5": row.get("em5_overall"),
                "easy EM@1": row.get("em1_easy"),
                "easy EM@5": row.get("em5_easy"),
                "medium EM@1": row.get("em1_medium"),
                "medium EM@5": row.get("em5_medium"),
                "hard EM@1": row.get("em1_hard"),
                "hard EM@5": row.get("em5_hard"),
                "targeted linear sparsity": row.get("targeted_linear_sparsity_actual"),
                "whole-model sparsity": row.get("whole_model_sparsity_actual"),
            }
        )
    write_csv_rows(output_dir / "paper_table_sparsity_difficulty.csv", table_rows)


def run_single_eval(
    model: Any,
    tokenizer: Any,
    eval_name: str,
    eval_path: str,
    samples: list[dict[str, Any]],
    args: argparse.Namespace,
    output_dir: Path,
    pruning_mode: str,
    target_sparsity: float,
    masks: dict[str, torch.Tensor] | None,
    checkpoint_path: str,
    mask_path: str,
    device: torch.device,
    rank: int = 0,
    world_size: int = 1,
) -> dict[str, Any] | None:
    stats = sparsity_stats(model, masks, target_sparsity, include_heads=bool(args.prune_output_heads))
    candidates = evaluate_model(model, tokenizer, samples, args, device, rank=rank, world_size=world_size)
    if not is_main_process(rank):
        return None
    family = normalize_family(args.model_family)
    prediction_rows = prediction_result_rows(
        samples,
        candidates,
        normalization_mode=args.normalization_mode,
        model_family=family,
        pruning_mode=pruning_mode,
        pruning_method=args.prune_method,
        target_sparsity=float(target_sparsity),
        targeted_linear_sparsity_actual=stats["targeted_linear_sparsity_actual"],
        whole_model_sparsity_actual=stats["whole_model_sparsity_actual"],
        seed=int(args.seed),
    )
    for row in prediction_rows:
        row["eval_name"] = eval_name
    prediction_path = output_dir / f"predictions_{family}_{pruning_mode}_{target_sparsity:g}_{int(args.seed)}.csv"
    if eval_name != "benchmark":
        safe_eval_name = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in eval_name)
        prediction_path = output_dir / f"predictions_{safe_eval_name}_{family}_{pruning_mode}_{target_sparsity:g}_{int(args.seed)}.csv"
    write_csv_rows(prediction_path, prediction_rows, PREDICTION_FIELDNAMES)
    summary = summarize_prediction_rows(
        prediction_rows,
        bootstrap_resamples=int(args.bootstrap_resamples),
        seed=int(args.seed),
    )
    row = {
        "experiment_name": args.experiment_name,
        "model_family": family,
        "eval_name": eval_name,
        "eval_path": eval_path,
        "pruning_mode": pruning_mode,
        "pruning_method": args.prune_method,
        "target_sparsity": float(target_sparsity),
        **stats,
        "seed": int(args.seed),
        **summary,
        "decoding_config_json": row_config_json(args, "decoding"),
        "training_config_json": row_config_json(args, "training"),
        "pruning_config_json": row_config_json(args, "pruning"),
        "checkpoint_path": checkpoint_path,
        "mask_path": mask_path,
    }
    return row


def run_plots(output_dir: Path) -> None:
    try:
        from plot_sparsity_results import plot_results

        plot_results(output_dir / "summary_metrics.csv", output_dir / "figures")
    except Exception as exc:
        print(f"[warning] Could not generate plots automatically: {exc}", flush=True)


def parse_extra_eval_paths(values: list[str]) -> list[tuple[str, str]]:
    evals: list[tuple[str, str]] = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"--extra_eval_path must be NAME=PATH, got {value!r}")
        name, path = value.split("=", 1)
        name = name.strip()
        path = path.strip()
        if not name or not path:
            raise ValueError(f"--extra_eval_path must be NAME=PATH, got {value!r}")
        evals.append((name, path))
    return evals


def main() -> None:
    args = parse_args()
    device, rank, local_rank, world_size = setup_distributed(args.device)
    assert_expected_distributed_run(args, world_size)
    set_seeds(int(args.seed) + int(rank))
    output_dir = resolve_output_dir(args.output_dir)
    benchmark_samples = load_benchmark_samples(args.benchmark_path, args.benchmark_difficulty_path)
    if args.limit is not None:
        benchmark_samples = benchmark_samples[: int(args.limit)]
    eval_sets: list[tuple[str, str, list[dict[str, Any]]]] = [("benchmark", args.benchmark_path, benchmark_samples)]
    for eval_name, eval_path in parse_extra_eval_paths(args.extra_eval_path):
        extra_samples = load_prompt_response_samples(eval_path)
        if args.limit is not None:
            extra_samples = extra_samples[: int(args.limit)]
        eval_sets.append((eval_name, eval_path, extra_samples))
    recovery_samples = load_prompt_response_samples(args.recovery_train_path) if args.recovery_train_path else []
    validation_samples = load_prompt_response_samples(args.validation_path) if args.validation_path else []
    modes = normalize_modes(args)
    sparsity_levels = sorted({float(level) for level in args.sparsity_levels})
    if "dense" in modes and 0.0 not in sparsity_levels:
        sparsity_levels.insert(0, 0.0)

    rank_device = {
        "rank": int(rank),
        "local_rank": int(local_rank),
        "device": str(device),
        "cuda_current_device": int(torch.cuda.current_device()) if torch.cuda.is_available() else None,
    }
    rank_devices = all_gather_object(rank_device, world_size)

    metadata = {
        "experiment_name": args.experiment_name,
        "created_at_utc": dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "benchmark_path": args.benchmark_path,
        "benchmark_difficulty_path": args.benchmark_difficulty_path or "",
        "extra_eval_paths": {name: path for name, path, _samples in eval_sets if name != "benchmark"},
        "difficulty_counts": {
            name: {level: sum(1 for sample in samples if sample["difficulty"] == level) for level in DIFFICULTY_LEVELS}
            for name, _path, samples in eval_sets
        },
        "args": vars(args),
    }
    if is_main_process(rank):
        metadata["distributed"] = {
            "rank": int(rank),
            "local_rank": int(local_rank),
            "world_size": int(world_size),
            "launcher": "torchrun" if int(world_size) > 1 else "python",
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "torch_cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
            "visible_cuda_device_count": visible_cuda_device_count(),
            "expected_world_size": args.expected_world_size,
            "expected_visible_gpu_count": args.expected_visible_gpu_count,
            "rank_devices": rank_devices,
        }
        write_json(output_dir / "run_config.json", metadata)
        print(
            "Linear sparsity runtime: "
            f"world_size={world_size} cuda_visible_devices={os.environ.get('CUDA_VISIBLE_DEVICES', '')} "
            f"torch_cuda_device_count={torch.cuda.device_count() if torch.cuda.is_available() else 0} "
            f"rank_devices={rank_devices}",
            flush=True,
        )
        if int(world_size) > 1:
            print("Progress bars show rank 0 only; other ranks run their distributed shards without progress bars.", flush=True)
    maybe_barrier(world_size, local_rank)

    summary_rows: list[dict[str, Any]] = []
    dense_by_family_eval: dict[tuple[str, str], dict[str, Any]] = {}
    family = normalize_family(args.model_family)

    for mode in modes:
        mode_sparsities = [0.0] if mode == "dense" else [level for level in sparsity_levels if level > 0.0]
        for target_sparsity in mode_sparsities:
            maybe_print(rank, f"\n=== {family} {mode} target_sparsity={target_sparsity:g} ===")
            model, tokenizer = load_model_and_tokenizer(args, device)
            device = next(model.parameters()).device
            train_model: Any = model
            if int(world_size) > 1:
                train_model = DistributedDataParallel(
                    model,
                    device_ids=[local_rank] if device.type == "cuda" else None,
                    output_device=local_rank if device.type == "cuda" else None,
                    find_unused_parameters=False,
                )
            masks: dict[str, torch.Tensor] | None = None
            if mode == "oneshot":
                masks = make_pruning_masks(model, tokenizer, target_sparsity, recovery_samples, args, device)
                apply_masks(model, masks)
            elif mode == "progressive":
                masks, logs = run_progressive(
                    model,
                    train_model,
                    tokenizer,
                    target_sparsity,
                    recovery_samples,
                    validation_samples,
                    args,
                    output_dir,
                    device,
                    rank=rank,
                    world_size=world_size,
                )
                log_path = output_dir / f"progressive_logs_{family}_{target_sparsity:g}_{int(args.seed)}.csv"
                if is_main_process(rank):
                    write_csv_rows(
                        log_path,
                        logs,
                        [
                            "stage",
                            "stage_target_sparsity",
                            "targeted_linear_sparsity_actual",
                            "whole_model_sparsity_actual",
                            "recovery_epoch",
                            "train_loss",
                            "val_em1",
                            "val_em5",
                        ],
                    )
            maybe_barrier(world_size, local_rank)
            checkpoint_path = ""
            mask_path = ""
            if is_main_process(rank):
                checkpoint_path, mask_path = save_run_artifacts(
                    model,
                    tokenizer,
                    masks,
                    output_dir,
                    family,
                    mode,
                    target_sparsity,
                    int(args.seed),
                )
            maybe_barrier(world_size, local_rank)
            for eval_name, eval_path, eval_samples in eval_sets:
                row = run_single_eval(
                    model,
                    tokenizer,
                    eval_name,
                    eval_path,
                    eval_samples,
                    args,
                    output_dir,
                    mode,
                    target_sparsity,
                    masks,
                    checkpoint_path,
                    mask_path,
                    device,
                    rank=rank,
                    world_size=world_size,
                )
                if not is_main_process(rank) or row is None:
                    continue
                if mode == "dense":
                    dense_by_family_eval[(family, eval_name)] = row
                row = add_retention_metrics(row, dense_by_family_eval.get((family, eval_name)))
                summary_rows.append(row)
                maybe_print(
                    rank,
                    f"{eval_name} EM@1={row['em1_overall']:.4f} EM@5={row['em5_overall']:.4f} "
                    f"targeted_linear_sparsity={row['targeted_linear_sparsity_actual']:.4f} "
                    f"whole_model_sparsity={row['whole_model_sparsity_actual']:.4f}",
                )

    if is_main_process(rank):
        write_csv_rows(output_dir / "summary_metrics.csv", summary_rows, SUMMARY_FIELDNAMES)
        write_paper_table(output_dir, summary_rows)
        if not args.skip_plots:
            run_plots(output_dir)
        print("\nSparsity experiment complete.", flush=True)
        print(f"  distributed world_size: {world_size}", flush=True)
        print(f"  predictions: {output_dir}/predictions_*.csv", flush=True)
        print(f"  summary: {output_dir / 'summary_metrics.csv'}", flush=True)
        print(f"  paper table: {output_dir / 'paper_table_sparsity_difficulty.csv'}", flush=True)
        print(f"  figures: {output_dir / 'figures'}", flush=True)
    cleanup_distributed()


if __name__ == "__main__":
    main()
