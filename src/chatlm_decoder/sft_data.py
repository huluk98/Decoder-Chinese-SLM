from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Iterator

import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler

USER_TOKEN = "<|user|>"
ASSISTANT_TOKEN = "<|assistant|>"
EOS_TOKEN = "<|eos|>"
PROMPT_FIELDS = ("prompt", "instruction", "question", "input", "x")
RESPONSE_FIELDS = ("response", "responses", "output", "answer", "completion", "target", "y")


def _clean(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _first_text(record: dict[str, Any], fields: Iterable[str]) -> str:
    for field in fields:
        value = _clean(record.get(field))
        if value:
            return value
    return ""


def _field_hint(record: dict[str, Any]) -> str:
    return ", ".join(record.keys()) or "none"


def format_prompt(record: dict[str, Any]) -> str:
    instruction = _first_text(record, PROMPT_FIELDS)
    if not instruction:
        raise ValueError(
            "SFT rows must include prompt text. Expected one of "
            f"{', '.join(PROMPT_FIELDS)}. Available fields: {_field_hint(record)}."
        )
    extra_input = _clean(record.get("input")) if "instruction" in record else ""
    if extra_input and extra_input != instruction:
        instruction = f"{instruction}\n{extra_input}" if instruction else extra_input
    return f"{USER_TOKEN}\n{instruction}\n{ASSISTANT_TOKEN}\n"


def format_response(record: dict[str, Any]) -> str:
    response = _first_text(record, RESPONSE_FIELDS)
    if not response:
        raise ValueError(
            "SFT rows must include response text. Expected one of "
            f"{', '.join(RESPONSE_FIELDS)}. Available fields: {_field_hint(record)}."
        )
    return response if response.endswith(EOS_TOKEN) else f"{response}{EOS_TOKEN}"


def format_sft_text(record: dict[str, Any]) -> tuple[str, str]:
    prompt = format_prompt(record)
    return prompt, prompt + format_response(record)


def _coerce_json_records(payload: Any, path: Path) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict):
        records = None
        for key in ("data", "records", "items", "examples", "train"):
            value = payload.get(key)
            if isinstance(value, list):
                records = value
                break
        if records is None:
            records = [payload]
    else:
        raise ValueError(
            f"{path} must contain a JSON object, a JSON list, "
            "or an object with a data/records/items/examples/train list."
        )

    clean_records: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"{path}: record {index} must be a JSON object.")
        clean_records.append(record)
    return clean_records


def read_records(path: str | Path) -> Iterator[dict[str, Any]]:
    data_path = Path(path).expanduser()
    suffix = data_path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(data_path.read_text(encoding="utf-8"))
        yield from _coerce_json_records(payload, data_path)
        return

    if suffix != ".jsonl":
        raise ValueError(f"Unsupported SFT data file extension: {suffix}. Use .jsonl or .json.")

    with data_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if line:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError(f"{data_path}:{line_number} must be a JSON object.")
                yield record


class SFTDataset(Dataset):
    def __init__(self, path: str | Path, max_samples: int | None = None) -> None:
        records = list(read_records(path))
        if max_samples is not None:
            records = records[: int(max_samples)]
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        prompt, text = format_sft_text(record)
        return {"record": record, "prompt": prompt, "text": text}


class ContrastiveSFTDataset(SFTDataset):
    def __getitem__(self, index: int) -> dict[str, Any]:
        item = super().__getitem__(index)
        record = item["record"]
        positive = _first_text(record, ("positive", "pos", "x_positive", "chosen", "x_plus"))
        negative = _first_text(record, ("negative", "neg", "x_negative", "rejected", "x_minus"))
        if not positive or not negative:
            raise ValueError(
                "Contrastive SFT rows must include positive/negative fields "
                "such as positive and negative, x_positive and x_negative, or chosen and rejected."
            )
        item["positive"] = positive
        item["negative"] = negative
        return item


def _tokenize_text(
    tokenizer: Any,
    text: str,
    max_length: int,
    add_special_tokens: bool = False,
) -> list[int]:
    ids = tokenizer(text, add_special_tokens=add_special_tokens, truncation=True, max_length=max_length)["input_ids"]
    return [int(token_id) for token_id in ids]


def _pad_sequences(sequences: list[list[int]], pad_token_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(len(sequence) for sequence in sequences)
    input_ids = torch.full((len(sequences), max_len), int(pad_token_id), dtype=torch.long)
    attention_mask = torch.zeros((len(sequences), max_len), dtype=torch.long)
    for row, sequence in enumerate(sequences):
        tensor = torch.tensor(sequence, dtype=torch.long)
        input_ids[row, : tensor.numel()] = tensor
        attention_mask[row, : tensor.numel()] = 1
    return input_ids, attention_mask


def sft_collate(features: list[dict[str, Any]], tokenizer: Any, max_length: int) -> dict[str, torch.Tensor]:
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    input_sequences: list[list[int]] = []
    label_sequences: list[list[int]] = []
    for feature in features:
        prompt_ids = _tokenize_text(tokenizer, feature["prompt"], max_length)
        full_ids = _tokenize_text(tokenizer, feature["text"], max_length)
        labels = list(full_ids)
        prompt_len = min(len(prompt_ids), len(labels))
        labels[:prompt_len] = [-100] * prompt_len
        input_sequences.append(full_ids)
        label_sequences.append(labels)

    input_ids, attention_mask = _pad_sequences(input_sequences, int(pad_token_id))
    labels, _ = _pad_sequences(label_sequences, -100)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def contrastive_sft_collate(features: list[dict[str, Any]], tokenizer: Any, max_length: int) -> dict[str, torch.Tensor]:
    batch = sft_collate(features, tokenizer, max_length)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    anchor_ids = [_tokenize_text(tokenizer, feature["prompt"], max_length) for feature in features]
    positive_ids = [_tokenize_text(tokenizer, feature["positive"], max_length) for feature in features]
    negative_ids = [_tokenize_text(tokenizer, feature["negative"], max_length) for feature in features]

    anchor_input_ids, anchor_attention_mask = _pad_sequences(anchor_ids, int(pad_token_id))
    positive_input_ids, positive_attention_mask = _pad_sequences(positive_ids, int(pad_token_id))
    negative_input_ids, negative_attention_mask = _pad_sequences(negative_ids, int(pad_token_id))
    batch.update(
        {
            "anchor_input_ids": anchor_input_ids,
            "anchor_attention_mask": anchor_attention_mask,
            "positive_input_ids": positive_input_ids,
            "positive_attention_mask": positive_attention_mask,
            "negative_input_ids": negative_input_ids,
            "negative_attention_mask": negative_attention_mask,
        }
    )
    return batch


def build_sft_dataloader(
    path: str | Path,
    tokenizer: Any,
    max_length: int,
    batch_size: int,
    num_workers: int = 0,
    shuffle: bool = True,
    contrastive: bool = False,
    max_samples: int | None = None,
    rank: int = 0,
    world_size: int = 1,
) -> DataLoader:
    dataset_cls = ContrastiveSFTDataset if contrastive else SFTDataset
    dataset = dataset_cls(path, max_samples=max_samples)
    collate = contrastive_sft_collate if contrastive else sft_collate
    sampler = (
        DistributedSampler(dataset, num_replicas=int(world_size), rank=int(rank), shuffle=bool(shuffle))
        if int(world_size) > 1
        else None
    )
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle) if sampler is None else False,
        sampler=sampler,
        num_workers=int(num_workers),
        collate_fn=lambda features: collate(features, tokenizer, int(max_length)),
    )
