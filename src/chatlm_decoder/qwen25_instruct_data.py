from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Iterable, Iterator

import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler

LEGACY_TOKENS = ("<|user|>", "<|assistant|>", "<|system|>", "<|eos|>")
DEFAULT_SYSTEM_PROMPT = (
    "You are a strict smart-home command parser. "
    "Output only the normalized command response. Do not explain."
)
PROMPT_FIELDS = ("prompt", "instruction", "question", "input", "query", "x")
RESPONSE_FIELDS = ("response", "answer", "output", "completion", "target", "y")
MESSAGE_FIELDS = ("messages", "conversations")
CONTENT_FIELDS = ("content", "value", "text", "message")
ROLE_ALIASES = {
    "user": "user",
    "human": "user",
    "instruction": "user",
    "prompt": "user",
    "assistant": "assistant",
    "gpt": "assistant",
    "bot": "assistant",
    "model": "assistant",
}
JSON_LIST_KEYS = ("data", "records", "items", "examples", "train", "eval", "validation", "test")
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


def assert_no_legacy_tokens(text: str, context: str) -> None:
    found = [token for token in LEGACY_TOKENS if token in text]
    if found:
        raise ValueError(f"Qwen2.5-Instruct path must not contain legacy tokens in {context}: {found}")


def _first_text(record: dict[str, Any], fields: Iterable[str]) -> str:
    for field in fields:
        text = _clean(record.get(field))
        if text:
            return text
    return ""


def _message_role(turn: dict[str, Any]) -> str:
    role = _clean(turn.get("role") or turn.get("from") or turn.get("speaker")).lower()
    return ROLE_ALIASES.get(role, "")


def _message_content(turn: dict[str, Any]) -> str:
    for field in CONTENT_FIELDS:
        text = _clean(turn.get(field))
        if text:
            return text
    return ""


def _iter_messages(record: dict[str, Any]) -> list[dict[str, Any]]:
    for field in MESSAGE_FIELDS:
        messages = record.get(field)
        if isinstance(messages, list):
            return [turn for turn in messages if isinstance(turn, dict)]
    return []


def extract_instruction_response(record: dict[str, Any]) -> tuple[str, str]:
    messages = _iter_messages(record)
    if messages:
        assistant_indices = [
            index
            for index, turn in enumerate(messages)
            if _message_role(turn) == "assistant" and _message_content(turn)
        ]
        if assistant_indices:
            response_index = assistant_indices[-1]
            response = _message_content(messages[response_index])
            user_parts = [
                _message_content(turn)
                for turn in messages[:response_index]
                if _message_role(turn) == "user" and _message_content(turn)
            ]
            instruction = "\n".join(user_parts).strip()
            if instruction and response:
                assert_no_legacy_tokens(instruction, "message instruction")
                assert_no_legacy_tokens(response, "message response")
                return instruction, response

    instruction = _first_text(record, PROMPT_FIELDS)
    response = _first_text(record, RESPONSE_FIELDS)
    if not instruction:
        raise ValueError(
            "Qwen SFT rows must include prompt text. Expected one of "
            f"{', '.join(PROMPT_FIELDS)} or a messages/conversations transcript."
        )
    if not response:
        raise ValueError(
            "Qwen SFT rows must include response text. Expected one of "
            f"{', '.join(RESPONSE_FIELDS)} or a messages/conversations transcript with an assistant turn."
        )
    assert_no_legacy_tokens(instruction, "instruction")
    assert_no_legacy_tokens(response, "response")
    return instruction, response


def qwen_prompt_text(tokenizer: Any, instruction: str, system_prompt: str) -> str:
    if not hasattr(tokenizer, "apply_chat_template"):
        raise AttributeError("Qwen2.5-Instruct tokenizer must provide apply_chat_template.")
    if tokenizer.eos_token is None:
        raise ValueError("Qwen2.5-Instruct tokenizer eos_token must not be None.")
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": instruction},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    assert_no_legacy_tokens(prompt, "formatted Qwen prompt")
    return str(prompt)


def format_qwen_sft_example(tokenizer: Any, record: dict[str, Any], system_prompt: str) -> dict[str, str]:
    instruction, response = extract_instruction_response(record)
    prompt = qwen_prompt_text(tokenizer, instruction, system_prompt)
    eos = str(tokenizer.eos_token)
    response_with_eos = response if response.endswith(eos) else f"{response}{eos}"
    full_text = prompt + response_with_eos
    assert_no_legacy_tokens(full_text, "formatted Qwen full text")
    return {
        "instruction": instruction,
        "response": response,
        "response_with_eos": response_with_eos,
        "prompt_text": prompt,
        "full_text": full_text,
    }


def _coerce_json_records(payload: Any, path: Path) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict):
        records = None
        for key in JSON_LIST_KEYS:
            value = payload.get(key)
            if isinstance(value, list):
                records = value
                break
        if records is None:
            records = [payload]
    else:
        raise ValueError(f"{path} must contain a JSON object, JSON list, or wrapper with one of {JSON_LIST_KEYS}.")

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
        yield from _coerce_json_records(json.loads(data_path.read_text(encoding="utf-8")), data_path)
        return
    if suffix == ".jsonl":
        with data_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError(f"{data_path}:{line_number} must be a JSON object.")
                yield record
        return
    if suffix == ".csv":
        with data_path.open("r", encoding="utf-8", newline="") as handle:
            yield from csv.DictReader(handle)
        return
    raise ValueError(f"Unsupported Qwen SFT data extension: {suffix}. Use .json, .jsonl, or .csv.")


class Qwen25InstructSFTDataset(Dataset):
    def __init__(
        self,
        path: str | Path,
        tokenizer: Any,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_samples: int | None = None,
        group_by_length: bool = False,
    ) -> None:
        records = list(read_records(path))
        if max_samples is not None:
            records = records[: int(max_samples)]
        if group_by_length:
            records.sort(key=lambda record: len(json.dumps(record, ensure_ascii=False)))
        self.records = records
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        example = format_qwen_sft_example(self.tokenizer, self.records[index], self.system_prompt)
        example["record"] = self.records[index]
        return example


def qwen25_instruct_collate(features: list[dict[str, Any]], tokenizer: Any, max_seq_length: int) -> dict[str, torch.Tensor]:
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if pad_token_id is None:
        raise ValueError("Qwen tokenizer must define pad_token_id or eos_token_id.")

    input_sequences: list[list[int]] = []
    label_sequences: list[list[int]] = []
    for feature in features:
        prompt_ids = tokenizer(feature["prompt_text"], add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(
            feature["full_text"],
            add_special_tokens=False,
            truncation=True,
            max_length=int(max_seq_length),
        )["input_ids"]
        prompt_ids = [int(token_id) for token_id in prompt_ids]
        full_ids = [int(token_id) for token_id in full_ids]
        if len(prompt_ids) >= len(full_ids):
            raise ValueError(
                "Qwen SFT example lost the supervised response after truncation. "
                f"prompt_tokens={len(prompt_ids)} full_tokens={len(full_ids)} max_seq_length={max_seq_length}"
            )
        labels = list(full_ids)
        labels[: len(prompt_ids)] = [-100] * len(prompt_ids)
        input_sequences.append(full_ids)
        label_sequences.append(labels)

    max_len = max(len(sequence) for sequence in input_sequences)
    input_ids = torch.full((len(features), max_len), int(pad_token_id), dtype=torch.long)
    attention_mask = torch.zeros((len(features), max_len), dtype=torch.long)
    labels = torch.full((len(features), max_len), -100, dtype=torch.long)
    for row, sequence in enumerate(input_sequences):
        ids = torch.tensor(sequence, dtype=torch.long)
        row_labels = torch.tensor(label_sequences[row], dtype=torch.long)
        input_ids[row, : ids.numel()] = ids
        attention_mask[row, : ids.numel()] = 1
        labels[row, : row_labels.numel()] = row_labels
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def build_qwen25_instruct_dataloader(
    path: str | Path,
    tokenizer: Any,
    max_seq_length: int,
    batch_size: int,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    max_samples: int | None = None,
    group_by_length: bool = False,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    drop_last: bool = False,
    rank: int = 0,
    world_size: int = 1,
    seed: int = 42,
) -> DataLoader:
    dataset = Qwen25InstructSFTDataset(
        path=path,
        tokenizer=tokenizer,
        system_prompt=system_prompt,
        max_samples=max_samples,
        group_by_length=group_by_length,
    )
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
        collate_fn=lambda features: qwen25_instruct_collate(features, tokenizer, int(max_seq_length)),
    )
