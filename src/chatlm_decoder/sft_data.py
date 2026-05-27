from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable, Iterator

import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler

USER_TOKEN = "<|user|>"
ASSISTANT_TOKEN = "<|assistant|>"
SYSTEM_TOKEN = "<|system|>"
EOS_TOKEN = "<|eos|>"
PROMPT_FIELDS = ("prompt", "anchor", "instruction", "question", "query", "input", "x")
INPUT_FIELDS = ("input", "context", "source", "background")
SYSTEM_FIELDS = ("system", "system_prompt")
RESPONSE_FIELDS = ("response", "responses", "output", "answer", "completion", "target", "y")
POSITIVE_FIELDS = ("positive", "pos", "x_positive", "chosen", "x_plus")
NEGATIVE_FIELDS = ("negative", "neg", "x_negative", "rejected", "x_minus")
MESSAGE_FIELDS = ("messages", "conversations")
CONTENT_FIELDS = ("content", "value", "text", "message")
ROLE_ALIASES = {
    "human": USER_TOKEN,
    "user": USER_TOKEN,
    "instruction": USER_TOKEN,
    "prompt": USER_TOKEN,
    "assistant": ASSISTANT_TOKEN,
    "gpt": ASSISTANT_TOKEN,
    "bot": ASSISTANT_TOKEN,
    "model": ASSISTANT_TOKEN,
    "system": SYSTEM_TOKEN,
}
SPECIAL_TOKEN_PATTERN = re.compile(r"<\|(?:user|assistant|system|eos)\|>")
BLANK_LINE_PATTERN = re.compile(r"\n{3,}")


def _clean(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        text = "\n".join(_clean(item) for item in value)
    elif isinstance(value, dict):
        for field in CONTENT_FIELDS + RESPONSE_FIELDS + PROMPT_FIELDS:
            if field in value:
                text = _clean(value.get(field))
                break
        else:
            text = json.dumps(value, ensure_ascii=False)
    else:
        text = str(value)
    text = text.replace("\ufeff", "").replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.strip() for line in text.splitlines())
    text = BLANK_LINE_PATTERN.sub("\n\n", text)
    return text.strip()


def _clean_content(value: Any) -> str:
    text = _clean(value)
    text = SPECIAL_TOKEN_PATTERN.sub("", text)
    text = BLANK_LINE_PATTERN.sub("\n\n", text)
    return text.strip()


def _first_text(record: dict[str, Any], fields: Iterable[str]) -> str:
    for field in fields:
        value = _clean_content(record.get(field))
        if value:
            return value
    return ""


def _field_hint(record: dict[str, Any]) -> str:
    return ", ".join(record.keys()) or "none"


def _message_role(turn: dict[str, Any]) -> str:
    raw_role = _clean(turn.get("role") or turn.get("from") or turn.get("speaker")).lower()
    return ROLE_ALIASES.get(raw_role, "")


def _message_content(turn: dict[str, Any]) -> str:
    for field in CONTENT_FIELDS:
        content = _clean_content(turn.get(field))
        if content:
            return content
    return ""


def _iter_messages(record: dict[str, Any]) -> list[dict[str, Any]]:
    for field in MESSAGE_FIELDS:
        messages = record.get(field)
        if isinstance(messages, list):
            return [turn for turn in messages if isinstance(turn, dict)]
    return []


def _format_transcript_prompt(messages: list[dict[str, Any]]) -> tuple[str, str] | None:
    assistant_indices = [
        index
        for index, turn in enumerate(messages)
        if _message_role(turn) == ASSISTANT_TOKEN and _message_content(turn)
    ]
    if not assistant_indices:
        return None

    response_index = assistant_indices[-1]
    response = _message_content(messages[response_index])
    parts: list[str] = []
    for turn in messages[:response_index]:
        role_token = _message_role(turn)
        content = _message_content(turn)
        if role_token and content:
            parts.append(f"{role_token}\n{content}")

    if not any(part.startswith(USER_TOKEN) for part in parts):
        return None
    prompt = "\n".join(parts + [ASSISTANT_TOKEN])
    return f"{prompt}\n", response


def normalize_sft_record(record: dict[str, Any]) -> tuple[str, str]:
    """Normalize one messy SFT row into the model's canonical prompt/response text."""
    messages = _iter_messages(record)
    from_messages = _format_transcript_prompt(messages) if messages else None
    if from_messages is not None:
        prompt, response = from_messages
        return prompt, response if response.endswith(EOS_TOKEN) else f"{response}{EOS_TOKEN}"

    system = _first_text(record, SYSTEM_FIELDS)
    instruction = _first_text(record, PROMPT_FIELDS)
    extra_input = ""
    if any(field in record for field in ("instruction", "prompt", "question", "anchor")):
        extra_input = _first_text(record, INPUT_FIELDS)
    if extra_input and extra_input != instruction:
        instruction = f"{instruction}\n{extra_input}" if instruction else extra_input

    if not instruction:
        raise ValueError(
            "SFT rows must include prompt text. Expected one of "
            f"{', '.join(PROMPT_FIELDS)} or a messages/conversations field. "
            f"Available fields: {_field_hint(record)}."
        )

    response = _first_text(record, RESPONSE_FIELDS)
    if not response:
        raise ValueError(
            "SFT rows must include response text. Expected one of "
            f"{', '.join(RESPONSE_FIELDS)} or a messages/conversations field "
            f"with an assistant turn. Available fields: {_field_hint(record)}."
        )

    parts: list[str] = []
    if system:
        parts.append(f"{SYSTEM_TOKEN}\n{system}")
    parts.append(f"{USER_TOKEN}\n{instruction}")
    parts.append(ASSISTANT_TOKEN)
    prompt = "\n".join(parts) + "\n"
    return prompt, response if response.endswith(EOS_TOKEN) else f"{response}{EOS_TOKEN}"


def format_prompt(record: dict[str, Any]) -> str:
    prompt, _ = normalize_sft_record(record)
    return prompt


def format_response(record: dict[str, Any]) -> str:
    _, response = normalize_sft_record(record)
    return response


def format_sft_text(record: dict[str, Any]) -> tuple[str, str]:
    prompt, response = normalize_sft_record(record)
    return prompt, prompt + response


def format_prompt_from_text(record: dict[str, Any], text: str) -> str:
    system = _first_text(record, SYSTEM_FIELDS)
    instruction = _clean_content(text)
    parts: list[str] = []
    if system:
        parts.append(f"{SYSTEM_TOKEN}\n{system}")
    parts.append(f"{USER_TOKEN}\n{instruction}")
    parts.append(ASSISTANT_TOKEN)
    return "\n".join(parts) + "\n"


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
    def __init__(
        self,
        path: str | Path,
        max_samples: int | None = None,
        group_by_length: bool = False,
    ) -> None:
        records = list(read_records(path))
        if max_samples is not None:
            records = records[: int(max_samples)]
        if group_by_length:
            records.sort(key=lambda record: len(json.dumps(record, ensure_ascii=False)))
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
        positive = _first_text(record, POSITIVE_FIELDS)
        negative = _first_text(record, NEGATIVE_FIELDS)
        if not positive or not negative:
            raise ValueError(
                "Contrastive SFT rows must include positive/negative fields "
                "such as positive and negative, x_positive and x_negative, or chosen and rejected."
            )
        response = format_response(record)
        positive_prompt = format_prompt_from_text(record, positive)
        negative_prompt = format_prompt_from_text(record, negative)
        item["positive"] = positive
        item["negative"] = negative
        item["positive_prompt"] = positive_prompt
        item["positive_text"] = positive_prompt + response
        item["negative_prompt"] = negative_prompt
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


def _collate_supervised_texts(
    features: list[dict[str, Any]],
    tokenizer: Any,
    max_length: int,
    prompt_key: str,
    text_key: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    input_sequences: list[list[int]] = []
    label_sequences: list[list[int]] = []
    for feature in features:
        prompt_ids = _tokenize_text(tokenizer, feature[prompt_key], max_length)
        full_ids = _tokenize_text(tokenizer, feature[text_key], max_length)
        labels = list(full_ids)
        prompt_len = min(len(prompt_ids), len(labels))
        labels[:prompt_len] = [-100] * prompt_len
        input_sequences.append(full_ids)
        label_sequences.append(labels)

    input_ids, attention_mask = _pad_sequences(input_sequences, int(pad_token_id))
    labels, _ = _pad_sequences(label_sequences, -100)
    return input_ids, attention_mask, labels


def sft_collate(features: list[dict[str, Any]], tokenizer: Any, max_length: int) -> dict[str, torch.Tensor]:
    input_ids, attention_mask, labels = _collate_supervised_texts(
        features,
        tokenizer,
        max_length,
        prompt_key="prompt",
        text_key="text",
    )
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def contrastive_sft_collate(features: list[dict[str, Any]], tokenizer: Any, max_length: int) -> dict[str, torch.Tensor]:
    batch = sft_collate(features, tokenizer, max_length)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    anchor_ids = [_tokenize_text(tokenizer, feature["prompt"], max_length) for feature in features]
    positive_ids = [_tokenize_text(tokenizer, feature["positive_prompt"], max_length) for feature in features]
    negative_ids = [_tokenize_text(tokenizer, feature["negative_prompt"], max_length) for feature in features]
    positive_gen_input_ids, positive_gen_attention_mask, positive_gen_labels = _collate_supervised_texts(
        features,
        tokenizer,
        max_length,
        prompt_key="positive_prompt",
        text_key="positive_text",
    )

    anchor_input_ids, anchor_attention_mask = _pad_sequences(anchor_ids, int(pad_token_id))
    positive_input_ids, positive_attention_mask = _pad_sequences(positive_ids, int(pad_token_id))
    negative_input_ids, negative_attention_mask = _pad_sequences(negative_ids, int(pad_token_id))
    batch.update(
        {
            "positive_gen_input_ids": positive_gen_input_ids,
            "positive_gen_attention_mask": positive_gen_attention_mask,
            "positive_gen_labels": positive_gen_labels,
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
    pin_memory: bool = False,
    persistent_workers: bool = False,
    group_by_length: bool = False,
    drop_last: bool = False,
    rank: int = 0,
    world_size: int = 1,
    seed: int = 42,
) -> DataLoader:
    dataset_cls = ContrastiveSFTDataset if contrastive else SFTDataset
    dataset = dataset_cls(path, max_samples=max_samples, group_by_length=group_by_length)
    collate = contrastive_sft_collate if contrastive else sft_collate
    sampler = (
        DistributedSampler(dataset, num_replicas=int(world_size), rank=int(rank), shuffle=bool(shuffle), seed=int(seed))
        if int(world_size) > 1
        else None
    )
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle) if sampler is None else False,
        sampler=sampler,
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        persistent_workers=bool(persistent_workers) and int(num_workers) > 0,
        drop_last=bool(drop_last),
        generator=generator,
        collate_fn=lambda features: collate(features, tokenizer, int(max_length)),
    )
