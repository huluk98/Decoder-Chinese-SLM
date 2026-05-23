#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENCHMARK_FILE = PROJECT_ROOT / "data" / "benchmarks" / "iot_instruction_benchmark_200.json"
DEFAULT_CONFIG_FILE = PROJECT_ROOT / "configs" / "iot_benchmark_eval.yaml"
LEGACY_USER_TOKEN = "<|user|>"
LEGACY_ASSISTANT_TOKEN = "<|assistant|>"
LEGACY_SYSTEM_TOKEN = "<|system|>"
LEGACY_EOS_TOKEN = "<|eos|>"
ZERO_WIDTH_PATTERN = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060\ufeff]")
SPACE_PATTERN = re.compile(r"\s+")
PUNCT_PATTERN = re.compile(r"[\s,，。．.！!？?；;：:\"'`“”‘’、]+")
LEADING_MARKERS = (
    LEGACY_ASSISTANT_TOKEN,
    "assistant:",
    "Assistant:",
    "ASSISTANT:",
    "回答:",
    "回答：",
    "答案:",
    "答案：",
    "输出:",
    "输出：",
)
STOP_MARKERS = (
    LEGACY_USER_TOKEN,
    LEGACY_SYSTEM_TOKEN,
    "\nUser:",
    "\n用户:",
    "<|im_start|>user",
    "<|im_start|>system",
)
HF_WEIGHT_FILENAMES = (
    "model.safetensors",
    "model.safetensors.index.json",
    "pytorch_model.bin",
    "pytorch_model.bin.index.json",
)


def parse_simple_yaml_scalar(value: str) -> Any:
    value = value.strip()
    if not value:
        return None
    lower = value.lower()
    if lower in {"null", "none", "~"}:
        return None
    if lower in {"true", "yes", "on"}:
        return True
    if lower in {"false", "no", "off"}:
        return False
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    if re.fullmatch(r"[-+]?\d+", value):
        return int(value)
    if re.fullmatch(r"[-+]?(?:\d+\.\d*|\.\d+)(?:[eE][-+]?\d+)?", value) or re.fullmatch(
        r"[-+]?\d+[eE][-+]?\d+", value
    ):
        return float(value)
    return value


def strip_simple_yaml_comment(value: str) -> str:
    quote: str | None = None
    escaped = False
    for index, char in enumerate(value):
        if escaped:
            escaped = False
            continue
        if char == "\\" and quote == '"':
            escaped = True
            continue
        if char in {"'", '"'}:
            if quote is None:
                quote = char
            elif quote == char:
                quote = None
            continue
        if char == "#" and quote is None and (index == 0 or value[index - 1].isspace()):
            return value[:index].rstrip()
    return value


def load_simple_yaml(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = {}
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("- "):
            raise ValueError(f"{path}:{line_number}: list syntax requires PyYAML; use flat key: value entries here.")
        if ":" not in line:
            raise ValueError(f"{path}:{line_number}: expected key: value.")
        key, value = line.split(":", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"{path}:{line_number}: empty key.")
        data[key] = parse_simple_yaml_scalar(strip_simple_yaml_comment(value))
    return data


def load_eval_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        import yaml  # type: ignore[import-not-found]

        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except ModuleNotFoundError:
        payload = load_simple_yaml(path)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a YAML mapping.")
    for section in ("iot_benchmark_eval", "iot_benchmark", "benchmark"):
        value = payload.get(section)
        if isinstance(value, dict):
            return dict(value)
    return dict(payload)


def choose_setting(args: argparse.Namespace, config: dict[str, Any], name: str, default: Any = None) -> Any:
    value = getattr(args, name)
    if value is not None:
        return value
    config_value = config.get(name, default)
    if config_value is None:
        return default
    return config_value


def resolve_project_path(value: str | Path | None, default: str | Path | None = None) -> Path:
    path_value = value if value not in (None, "") else default
    if path_value in (None, ""):
        raise ValueError("Missing required path value.")
    path = Path(str(path_value)).expanduser()
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    return path


def is_local_path_like(value: str) -> bool:
    path = Path(value).expanduser()
    return (
        path.is_absolute()
        or value.startswith(("./", "../", "~"))
        or path.exists()
        or ("/" in value and path.parent.exists())
    )


def has_hf_checkpoint_files(path: Path) -> bool:
    return (path / "config.json").exists() and any((path / filename).exists() for filename in HF_WEIGHT_FILENAMES)


def resolve_model_checkpoint_path(value: str) -> str:
    """Accept either a real checkpoint dir or a pruning run dir containing pruned_model/."""
    if not is_local_path_like(value):
        return value
    path = Path(value).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Model path does not exist: {path}")
    if path.is_file():
        raise ValueError(f"Model path must be a checkpoint directory or HF model id, got file: {path}")
    candidates = [path, path / "pruned_model", path / "final", path / "latest"]
    for candidate in candidates:
        if has_hf_checkpoint_files(candidate):
            return str(candidate.resolve())
    raise FileNotFoundError(
        f"Could not find a Hugging Face checkpoint at {path}. Expected config.json plus model weights "
        "in that directory, or in pruned_model/, final/, or latest/."
    )


def read_json_records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict):
        records = None
        for key in ("data", "records", "items", "examples", "test", "validation"):
            value = payload.get(key)
            if isinstance(value, list):
                records = value
                break
        if records is None:
            records = [payload]
    else:
        raise ValueError(f"{path} must contain a JSON list or object.")

    clean: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"{path}: record {index} must be a JSON object.")
        if not str(record.get("prompt", "")).strip() or not str(record.get("response", "")).strip():
            raise ValueError(f"{path}: record {index} must contain non-empty prompt and response fields.")
        clean.append(record)
    return clean


def select_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def dtype_for(name: str, device: torch.device) -> torch.dtype | str:
    if name == "auto":
        return "auto"
    if name == "bf16":
        return torch.bfloat16 if device.type == "cuda" else torch.float32
    if name == "fp16":
        return torch.float16 if device.type == "cuda" else torch.float32
    if name == "fp32":
        return torch.float32
    raise ValueError(f"Unknown dtype: {name}")


def configure_tokenizer(tokenizer: Any) -> None:
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    tokenizer.padding_side = "left"


def resolve_prompt_format(requested: str, model_path: str, tokenizer: Any) -> str:
    if requested != "auto":
        return requested
    lower = str(model_path).lower()
    has_chat_template = bool(getattr(tokenizer, "chat_template", None))
    if has_chat_template and "instruct" in lower:
        return "qwen-instruct"
    return "legacy"


def format_prompt(record: dict[str, Any], tokenizer: Any, prompt_format: str, system_prompt: str) -> str:
    prompt = str(record["prompt"]).strip()
    if prompt_format == "raw":
        return prompt
    if prompt_format == "legacy":
        return f"{LEGACY_USER_TOKEN}\n{prompt}\n{LEGACY_ASSISTANT_TOKEN}\n"
    if prompt_format == "qwen-instruct":
        if not hasattr(tokenizer, "apply_chat_template"):
            raise AttributeError("prompt_format=qwen-instruct requires tokenizer.apply_chat_template.")
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return str(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
    raise ValueError(f"Unknown prompt format: {prompt_format}")


def response_with_eos(response: str, tokenizer: Any, prompt_format: str) -> str:
    response = str(response).strip()
    eos = getattr(tokenizer, "eos_token", None)
    if prompt_format == "legacy":
        eos = LEGACY_EOS_TOKEN
    if eos and not response.endswith(str(eos)):
        return f"{response}{eos}"
    return response


def strip_wrapping_quotes(text: str) -> str:
    text = text.strip()
    changed = True
    while changed and len(text) >= 2:
        changed = False
        for left, right in (("`", "`"), ('"', '"'), ("'", "'")):
            if text.startswith(left) and text.endswith(right):
                text = text[1:-1].strip()
                changed = True
    return text


def clean_generated_text(text: str, tokenizer: Any) -> str:
    text = unicodedata.normalize("NFKC", str(text))
    text = ZERO_WIDTH_PATTERN.sub("", text)
    for special in (
        LEGACY_EOS_TOKEN,
        getattr(tokenizer, "bos_token", None),
        getattr(tokenizer, "eos_token", None),
        getattr(tokenizer, "pad_token", None),
        getattr(tokenizer, "unk_token", None),
        "<|im_end|>",
    ):
        if special:
            text = text.replace(str(special), "")
    for marker in STOP_MARKERS:
        if marker in text:
            text = text.split(marker, 1)[0]
    text = text.strip()
    changed = True
    while changed:
        changed = False
        for marker in LEADING_MARKERS:
            if text.startswith(marker):
                text = text[len(marker) :].strip()
                changed = True
    return strip_wrapping_quotes(text)


def normalize_text(text: str, tokenizer: Any, mode: str) -> str:
    text = clean_generated_text(text, tokenizer)
    text = SPACE_PATTERN.sub("", text)
    if mode == "exact":
        return text
    if mode == "normalized":
        return PUNCT_PATTERN.sub("", text)
    raise ValueError(f"Unknown comparison mode: {mode}")


def build_example(record: dict[str, Any], tokenizer: Any, prompt_format: str, system_prompt: str, max_length: int) -> dict[str, Any]:
    prompt_text = format_prompt(record, tokenizer=tokenizer, prompt_format=prompt_format, system_prompt=system_prompt)
    response_text = str(record["response"]).strip()
    full_text = prompt_text + response_with_eos(response_text, tokenizer=tokenizer, prompt_format=prompt_format)
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False, truncation=True, max_length=max_length)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False, truncation=True, max_length=max_length)["input_ids"]
    labels = [int(token_id) for token_id in full_ids]
    prompt_len = min(len(prompt_ids), len(labels))
    labels[:prompt_len] = [-100] * prompt_len
    return {
        "record": record,
        "prompt_text": prompt_text,
        "response_text": response_text,
        "input_ids": [int(token_id) for token_id in full_ids],
        "labels": labels,
        "prompt_token_count": int(prompt_len),
    }


def pad_examples(examples: list[dict[str, Any]], pad_token_id: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    max_len = max(len(example["input_ids"]) for example in examples)
    input_ids = torch.full((len(examples), max_len), int(pad_token_id), dtype=torch.long)
    attention_mask = torch.zeros((len(examples), max_len), dtype=torch.long)
    labels = torch.full((len(examples), max_len), -100, dtype=torch.long)
    for row, example in enumerate(examples):
        ids = torch.tensor(example["input_ids"], dtype=torch.long)
        row_labels = torch.tensor(example["labels"], dtype=torch.long)
        input_ids[row, : ids.numel()] = ids
        attention_mask[row, : ids.numel()] = 1
        labels[row, : row_labels.numel()] = row_labels
    return input_ids, attention_mask, labels


@torch.no_grad()
def score_batch(model: Any, examples: list[dict[str, Any]], pad_token_id: int, device: torch.device) -> list[dict[str, Any]]:
    input_ids, attention_mask, labels = pad_examples(examples, pad_token_id=pad_token_id)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    labels = labels.to(device)
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    rows = []
    for row in range(shift_labels.shape[0]):
        row_labels = shift_labels[row]
        token_count = int((row_labels != -100).sum().detach().cpu())
        if token_count <= 0:
            rows.append({"loss": math.nan, "loss_sum": 0.0, "token_count": 0})
            continue
        loss_sum = F.cross_entropy(
            shift_logits[row].view(-1, shift_logits.shape[-1]).float(),
            row_labels.view(-1),
            ignore_index=-100,
            reduction="sum",
        )
        rows.append(
            {
                "loss": float((loss_sum / token_count).detach().cpu()),
                "loss_sum": float(loss_sum.detach().cpu()),
                "token_count": token_count,
            }
        )
    return rows


@torch.no_grad()
def generate_batch(
    model: Any,
    tokenizer: Any,
    prompt_texts: list[str],
    device: torch.device,
    max_prompt_tokens: int,
    max_new_tokens: int,
    num_beams: int,
) -> list[tuple[str, int]]:
    inputs = tokenizer(
        prompt_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max(1, int(max_prompt_tokens)),
        add_special_tokens=False,
    ).to(device)
    prompt_width = int(inputs["input_ids"].shape[-1])
    output_ids = model.generate(
        **inputs,
        max_new_tokens=int(max_new_tokens),
        do_sample=False,
        num_beams=int(num_beams),
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    completions = []
    for row in range(int(output_ids.shape[0])):
        completion_ids = output_ids[row, prompt_width:]
        completions.append((tokenizer.decode(completion_ids, skip_special_tokens=False), int(completion_ids.numel())))
    return completions


def mean(values: list[float]) -> float | None:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return None
    return sum(finite) / len(finite)


def grouped_metrics(results: list[dict[str, Any]], field: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in results:
        key = str(row.get(field) or "unknown")
        groups.setdefault(key, []).append(row)
    return {
        key: {
            "total_examples": len(rows),
            "exact_match_correct": sum(1 for row in rows if row["exact_match"]),
            "exact_match_accuracy": sum(1 for row in rows if row["exact_match"]) / float(len(rows) or 1),
            "mean_response_loss": mean([row["response_loss"] for row in rows]),
        }
        for key, rows in sorted(groups.items())
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "id",
        "difficulty",
        "task_type",
        "exact_match",
        "response_loss",
        "generated_token_count",
        "prompt",
        "gold_response",
        "prediction",
        "normalized_prediction",
        "normalized_gold",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a model on the final 200-example IoT instruction benchmark.")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_FILE),
        help="Flat YAML config. Edit configs/iot_benchmark_eval.yaml for the one-link workflow.",
    )
    parser.add_argument("--model-path", "--checkpoint", default=None, dest="model_path")
    parser.add_argument("--benchmark-file", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--prompt-format", choices=("auto", "legacy", "raw", "qwen-instruct"), default=None)
    parser.add_argument("--system-prompt", default=None)
    parser.add_argument("--comparison-mode", choices=("exact", "normalized"), default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-beams", type=int, default=None)
    parser.add_argument("--dtype", choices=("auto", "bf16", "fp16", "fp32"), default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--trust-remote-code", action="store_true", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).expanduser() if args.config else DEFAULT_CONFIG_FILE
    if not config_path.is_absolute():
        config_path = (PROJECT_ROOT / config_path).resolve()
    config = load_eval_config(config_path)

    requested_model_path = str(choose_setting(args, config, "model_path", "") or "").strip()
    if not requested_model_path:
        raise ValueError(
            "Set model_path in configs/iot_benchmark_eval.yaml, or pass --model-path /path/or/hf-model-id."
        )
    model_path = resolve_model_checkpoint_path(requested_model_path)
    benchmark_file = resolve_project_path(choose_setting(args, config, "benchmark_file", None), DEFAULT_BENCHMARK_FILE)
    if not benchmark_file.exists():
        raise FileNotFoundError(f"Benchmark file not found: {benchmark_file}")

    output_dir_value = choose_setting(args, config, "output_dir", None)
    output_dir = Path(
        output_dir_value
        or PROJECT_ROOT
        / "runs"
        / "iot-benchmark"
        / f"{benchmark_file.stem}__{Path(model_path).name or 'model'}__{dt.datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}"
    ).expanduser()
    if not output_dir.is_absolute():
        output_dir = (PROJECT_ROOT / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt_format_requested = str(choose_setting(args, config, "prompt_format", "auto"))
    system_prompt = str(choose_setting(args, config, "system_prompt", "") or "")
    comparison_mode = str(choose_setting(args, config, "comparison_mode", "normalized"))
    max_length = int(choose_setting(args, config, "max_length", 256))
    max_new_tokens = int(choose_setting(args, config, "max_new_tokens", 64))
    batch_size = int(choose_setting(args, config, "batch_size", 16))
    num_beams = int(choose_setting(args, config, "num_beams", 1))
    dtype_name = str(choose_setting(args, config, "dtype", "bf16"))
    device_name = str(choose_setting(args, config, "device", "auto"))
    limit = choose_setting(args, config, "limit", None)
    trust_remote_code = bool(choose_setting(args, config, "trust_remote_code", False))

    device = select_device(device_name)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    configure_tokenizer(tokenizer)
    dtype = dtype_for(dtype_name, device)
    model_kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code}
    if dtype != "auto":
        model_kwargs["torch_dtype"] = dtype
    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs).to(device)
    model.eval()
    if len(tokenizer) > int(model.get_input_embeddings().weight.shape[0]):
        model.resize_token_embeddings(len(tokenizer))

    prompt_format = resolve_prompt_format(prompt_format_requested, model_path=model_path, tokenizer=tokenizer)
    records = read_json_records(benchmark_file)
    if limit is not None:
        records = records[: int(limit)]
    examples = [
        build_example(
            record,
            tokenizer=tokenizer,
            prompt_format=prompt_format,
            system_prompt=system_prompt,
            max_length=max_length,
        )
        for record in records
    ]

    results: list[dict[str, Any]] = []
    total_loss_sum = 0.0
    total_tokens = 0
    correct = 0
    for start in tqdm(range(0, len(examples), batch_size), desc="iot-benchmark"):
        batch = examples[start : start + batch_size]
        score_rows = score_batch(model, batch, pad_token_id=int(tokenizer.pad_token_id), device=device)
        generations = generate_batch(
            model,
            tokenizer,
            [example["prompt_text"] for example in batch],
            device=device,
            max_prompt_tokens=max(1, max_length - max_new_tokens),
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
        )
        for offset, example in enumerate(batch):
            record = example["record"]
            raw_prediction, generated_tokens = generations[offset]
            prediction = clean_generated_text(raw_prediction, tokenizer=tokenizer)
            normalized_prediction = normalize_text(prediction, tokenizer=tokenizer, mode=comparison_mode)
            normalized_gold = normalize_text(example["response_text"], tokenizer=tokenizer, mode=comparison_mode)
            exact = normalized_prediction == normalized_gold
            score = score_rows[offset]
            total_loss_sum += float(score["loss_sum"])
            total_tokens += int(score["token_count"])
            correct += int(exact)
            results.append(
                {
                    "id": str(record.get("id", start + offset)),
                    "difficulty": record.get("difficulty", ""),
                    "task_type": record.get("task_type", ""),
                    "source": record.get("source", ""),
                    "response_action_count": record.get("response_action_count", ""),
                    "device_term_count": record.get("device_term_count", ""),
                    "prompt": str(record["prompt"]),
                    "formatted_prompt": example["prompt_text"],
                    "prompt_token_count": example["prompt_token_count"],
                    "gold_response": example["response_text"],
                    "raw_generated_text": raw_prediction,
                    "prediction": prediction,
                    "normalized_prediction": normalized_prediction,
                    "normalized_gold": normalized_gold,
                    "exact_match": exact,
                    "response_loss": float(score["loss"]) if math.isfinite(float(score["loss"])) else math.nan,
                    "response_token_count": int(score["token_count"]),
                    "generated_token_count": generated_tokens,
                }
            )

    mean_loss = total_loss_sum / total_tokens if total_tokens else math.nan
    summary = {
        "script": "scripts/eval_iot_benchmark.py",
        "config_file": str(config_path),
        "requested_model_path": requested_model_path,
        "model_path": model_path,
        "checkpoint_path_used_for_evaluation": model_path,
        "benchmark_file": str(benchmark_file),
        "output_dir": str(output_dir),
        "prompt_format_requested": prompt_format_requested,
        "prompt_format": prompt_format,
        "comparison_mode": comparison_mode,
        "total_examples": len(results),
        "correct_examples": correct,
        "exact_match_accuracy": correct / float(len(results) or 1),
        "mean_response_loss": mean_loss,
        "response_perplexity": math.exp(mean_loss) if math.isfinite(mean_loss) else math.nan,
        "response_tokens": total_tokens,
        "avg_generated_tokens": mean([float(row["generated_token_count"]) for row in results]),
        "empty_predictions": sum(1 for row in results if not str(row["prediction"]).strip()),
        "by_difficulty": grouped_metrics(results, "difficulty"),
        "by_task_type": grouped_metrics(results, "task_type"),
        "generation_config": {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "num_beams": num_beams,
            "max_length": max_length,
        },
    }

    write_json(output_dir / "iot_benchmark_summary.json", summary)
    write_jsonl(output_dir / "iot_benchmark_predictions.jsonl", results)
    write_csv(output_dir / "iot_benchmark_predictions.csv", results)
    write_json(output_dir / "generation_samples.json", results[:50])
    write_json(output_dir / "exact_match_failure_cases.json", [row for row in results if not row["exact_match"]][:50])
    print(
        "IoT benchmark summary:\n"
        f"  requested_model={requested_model_path}\n"
        f"  evaluated_checkpoint={model_path}\n"
        f"  benchmark={benchmark_file}\n"
        f"  prompt_format={prompt_format}\n"
        f"  exact_match_accuracy={summary['exact_match_accuracy']:.4f} ({correct}/{len(results)})\n"
        f"  mean_response_loss={summary['mean_response_loss']:.4f}\n"
        f"  response_perplexity={summary['response_perplexity']:.4f}\n"
        f"  output_dir={output_dir}"
    )


if __name__ == "__main__":
    main()
