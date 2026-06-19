#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Iterable

import torch
import yaml
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import trange
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from chatlm_decoder.jepa import create_text_jepa_from_encoder_decoder  # noqa: E402


DEFAULT_CONTEXT_FIELDS = ("context", "prompt", "instruction", "input", "question")
DEFAULT_TARGET_FIELDS = ("target", "response", "output", "answer")


def read_records(path: str | Path) -> list[dict[str, Any]]:
    data_path = Path(path).expanduser()
    if data_path.suffix.lower() == ".jsonl":
        rows = []
        with data_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    with data_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        for key in ("data", "records", "examples", "items"):
            value = payload.get(key)
            if isinstance(value, list):
                return [row for row in value if isinstance(row, dict)]
    raise ValueError(f"Unsupported JEPA data shape: {data_path}")


def clean_text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def join_fields(record: dict[str, Any], fields: Iterable[str]) -> str:
    pieces = [clean_text(record.get(field)) for field in fields]
    return "\n".join(piece for piece in pieces if piece)


class TextJEPAPairDataset(Dataset):
    def __init__(self, records: list[dict[str, Any]], context_fields: list[str], target_fields: list[str]) -> None:
        self.examples: list[dict[str, str]] = []
        for record in records:
            context = join_fields(record, context_fields)
            target = join_fields(record, target_fields)
            if context and target:
                self.examples.append({"context": context, "target": target})
        if not self.examples:
            raise ValueError("No usable JEPA context/target pairs were found.")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, str]:
        return self.examples[index]


def collate_jepa_batch(
    examples: list[dict[str, str]],
    tokenizer: Any,
    max_context_length: int,
    max_target_length: int,
) -> dict[str, torch.Tensor]:
    contexts = [example["context"] for example in examples]
    targets = [example["target"] for example in examples]
    context_batch = tokenizer(
        contexts,
        max_length=max_context_length,
        truncation=True,
        padding=True,
        return_tensors="pt",
    )
    target_batch = tokenizer(
        targets,
        max_length=max_target_length,
        truncation=True,
        padding=True,
        return_tensors="pt",
    )
    return {
        "context_input_ids": context_batch["input_ids"],
        "context_attention_mask": context_batch["attention_mask"],
        "target_input_ids": target_batch["input_ids"],
        "target_attention_mask": target_batch["attention_mask"],
        "target_query_ids": torch.zeros(len(examples), dtype=torch.long),
    }


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    config.setdefault("jepa", {})
    config["jepa"].setdefault("model_name_or_path", "google/mt5-small")
    config["jepa"].setdefault("train_file", "data/scenic/SCENIC_full_training_dataset.json")
    config["jepa"].setdefault("output_dir", "runs/jepa-encoder-decoder")
    config["jepa"].setdefault("context_fields", list(DEFAULT_CONTEXT_FIELDS))
    config["jepa"].setdefault("target_fields", list(DEFAULT_TARGET_FIELDS))
    config["jepa"].setdefault("max_context_length", 128)
    config["jepa"].setdefault("max_target_length", 64)
    config["jepa"].setdefault("batch_size", 8)
    config["jepa"].setdefault("max_steps", 1000)
    config["jepa"].setdefault("learning_rate", 1.0e-4)
    config["jepa"].setdefault("weight_decay", 0.01)
    config["jepa"].setdefault("ema_decay", 0.996)
    config["jepa"].setdefault("predictor_hidden_size", None)
    config["jepa"].setdefault("num_target_queries", 1)
    config["jepa"].setdefault("dropout", 0.0)
    config["jepa"].setdefault("loss", "smooth_l1")
    config["jepa"].setdefault("normalize_targets", True)
    config["jepa"].setdefault("num_workers", 0)
    config["jepa"].setdefault("save_every", 500)
    config["jepa"].setdefault("log_every", 10)
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrain an encoder-decoder text JEPA on context/target pairs.")
    parser.add_argument("--config", default="configs/jepa_t5_encoder_decoder.yaml")
    parser.add_argument("--model-name-or-path", default=None)
    parser.add_argument("--train-file", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    jepa_config = config["jepa"]
    if args.model_name_or_path:
        jepa_config["model_name_or_path"] = args.model_name_or_path
    if args.train_file:
        jepa_config["train_file"] = args.train_file
    if args.output_dir:
        jepa_config["output_dir"] = args.output_dir
    if args.max_steps is not None:
        jepa_config["max_steps"] = args.max_steps

    torch.manual_seed(int(config.get("run", {}).get("seed", 42)))
    device = select_device()
    output_dir = Path(jepa_config["output_dir"]).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(str(jepa_config["model_name_or_path"]))
    base_model = AutoModelForSeq2SeqLM.from_pretrained(str(jepa_config["model_name_or_path"]))
    model = create_text_jepa_from_encoder_decoder(
        base_model,
        predictor_hidden_size=jepa_config.get("predictor_hidden_size"),
        num_target_queries=int(jepa_config["num_target_queries"]),
        dropout=float(jepa_config["dropout"]),
        ema_decay=float(jepa_config["ema_decay"]),
        normalize_targets=bool(jepa_config["normalize_targets"]),
        loss=str(jepa_config["loss"]),
    ).to(device)
    del base_model

    records = read_records(jepa_config["train_file"])
    dataset = TextJEPAPairDataset(
        records,
        context_fields=list(jepa_config["context_fields"]),
        target_fields=list(jepa_config["target_fields"]),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=int(jepa_config["batch_size"]),
        shuffle=True,
        num_workers=int(jepa_config["num_workers"]),
        collate_fn=lambda examples: collate_jepa_batch(
            examples,
            tokenizer=tokenizer,
            max_context_length=int(jepa_config["max_context_length"]),
            max_target_length=int(jepa_config["max_target_length"]),
        ),
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(jepa_config["learning_rate"]),
        weight_decay=float(jepa_config["weight_decay"]),
    )

    metrics_path = output_dir / "jepa_metrics.csv"
    with metrics_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["step", "loss", "prediction_norm", "target_norm"])
        writer.writeheader()

    data_iter = iter(dataloader)
    model.train()
    max_steps = int(jepa_config["max_steps"])
    progress = trange(1, max_steps + 1, desc="text-jepa")
    latest_loss = math.nan
    for step in progress:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)
        batch = {key: value.to(device) for key, value in batch.items()}
        output = model(**batch)
        output.loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        model.update_target_encoder()
        latest_loss = float(output.loss.detach().cpu())
        progress.set_postfix(loss=f"{latest_loss:.4f}")

        if step % int(jepa_config["log_every"]) == 0 or step == 1:
            with metrics_path.open("a", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["step", "loss", "prediction_norm", "target_norm"])
                writer.writerow(
                    {
                        "step": step,
                        "loss": latest_loss,
                        "prediction_norm": float(output.prediction.detach().norm(dim=-1).mean().cpu()),
                        "target_norm": float(output.target.detach().norm(dim=-1).mean().cpu()),
                    }
                )

        if step % int(jepa_config["save_every"]) == 0 or step == max_steps:
            checkpoint_dir = output_dir / "checkpoint"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), checkpoint_dir / "text_jepa.pt")
            tokenizer.save_pretrained(checkpoint_dir / "tokenizer")

    write_json(
        output_dir / "jepa_run_config.json",
        {
            "script": "scripts/pretrain_jepa.py",
            "model_name_or_path": jepa_config["model_name_or_path"],
            "train_file": jepa_config["train_file"],
            "examples": len(dataset),
            "max_steps": max_steps,
            "final_loss": latest_loss,
            "objective": "Predict target text encoder embeddings from context text encoder embeddings.",
        },
    )
    print(f"Wrote JEPA checkpoint and metrics to {output_dir}")


if __name__ == "__main__":
    main()
