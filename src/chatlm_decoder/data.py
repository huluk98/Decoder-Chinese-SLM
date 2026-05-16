from __future__ import annotations

import json
import os
import sys
import time
from itertools import islice
from pathlib import Path
from typing import Any, Iterable, Iterator

try:
    import torch
    from torch.utils.data import DataLoader, IterableDataset, get_worker_info
except ImportError:
    torch = None
    DataLoader = None

    class IterableDataset:  # type: ignore[no-redef]
        pass

    def get_worker_info() -> None:  # type: ignore[no-redef]
        return None

USER_TOKEN = "<|user|>"
ASSISTANT_TOKEN = "<|assistant|>"
SYSTEM_TOKEN = "<|system|>"
EOS_TOKEN = "<|eos|>"

ROLE_ALIASES = {
    "human": USER_TOKEN,
    "user": USER_TOKEN,
    "instruction": USER_TOKEN,
    "assistant": ASSISTANT_TOKEN,
    "gpt": ASSISTANT_TOKEN,
    "bot": ASSISTANT_TOKEN,
    "system": SYSTEM_TOKEN,
}


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


def _clean(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _join_fields(record: dict[str, Any], fields: Iterable[str]) -> str:
    pieces = [_clean(record.get(field)) for field in fields]
    return "\n".join(piece for piece in pieces if piece)


def _join_text_fields(record: dict[str, Any], fields: Iterable[str]) -> str | None:
    text = _join_fields(record, fields)
    return text if text else None


def _format_conversations(record: dict[str, Any]) -> str | None:
    turns = record.get("conversations") or record.get("messages")
    if not isinstance(turns, list):
        return None

    parts: list[str] = []
    for turn in turns:
        if not isinstance(turn, dict):
            continue
        raw_role = _clean(turn.get("from") or turn.get("role")).lower()
        value = _clean(turn.get("value") or turn.get("content"))
        role_token = ROLE_ALIASES.get(raw_role)
        if role_token and value:
            parts.append(f"{role_token}\n{value}")

    if not parts:
        return None
    return "\n".join(parts) + f"\n{EOS_TOKEN}"


def _format_prompt_response(record: dict[str, Any], source: dict[str, Any]) -> str | None:
    prompt_fields = _as_list(source.get("prompt_fields"))
    response_fields = _as_list(source.get("response_fields"))

    if not prompt_fields:
        prompt_fields = ["prompt", "instruction", "input", "question", "INSTRUCTION"]
    if not response_fields:
        response_fields = ["response", "output", "answer", "RESPONSE"]

    prompt = _join_fields(record, prompt_fields)
    response = _join_fields(record, response_fields)

    if not prompt or not response:
        return None
    return f"{USER_TOKEN}\n{prompt}\n{ASSISTANT_TOKEN}\n{response}{EOS_TOKEN}"


def format_record(record: dict[str, Any], source: dict[str, Any]) -> str | None:
    fmt = source.get("format", "auto")
    if fmt == "chat_conversations" or (fmt == "auto" and "conversations" in record):
        return _format_conversations(record)
    if fmt == "prompt_response":
        return _format_prompt_response(record, source)
    if fmt == "text_fields":
        fields = _as_list(source.get("text_fields"))
        if not fields:
            fields = ["title", "desc", "content", "text", "chinese"]
        text = _join_text_fields(record, fields)
        return f"{text}{EOS_TOKEN}" if text and not text.endswith(EOS_TOKEN) else text

    text_field = source.get("text_field", "text")
    text = _clean(record.get(text_field))
    if text:
        return text if text.endswith(EOS_TOKEN) else f"{text}{EOS_TOKEN}"

    if fmt == "auto":
        for fields in (
            ["title", "desc", "content"],
            ["title", "text"],
            ["content"],
            ["chinese"],
        ):
            text = _join_text_fields(record, fields)
            if text:
                return text if text.endswith(EOS_TOKEN) else f"{text}{EOS_TOKEN}"

        string_values = [_clean(value) for value in record.values() if isinstance(value, str)]
        text = "\n".join(value for value in string_values if value)
        if text:
            return text if text.endswith(EOS_TOKEN) else f"{text}{EOS_TOKEN}"

    return _format_prompt_response(record, source)


def _iter_local_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _iter_hf_dataset(
    source: dict[str, Any],
    default_streaming: bool,
    default_shuffle_buffer: int | None,
    default_seed: int,
    default_cache_dir: str | None = None,
    default_download_timeout: int | None = None,
    default_etag_timeout: int | None = None,
    default_endpoint: str | None = None,
) -> Iterator[dict[str, Any]]:
    endpoint = source.get("endpoint", default_endpoint)
    download_timeout = source.get("download_timeout", default_download_timeout)
    etag_timeout = source.get("etag_timeout", default_etag_timeout)
    if endpoint:
        os.environ.setdefault("HF_ENDPOINT", str(endpoint))
    if download_timeout is not None:
        os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", str(download_timeout))
    if etag_timeout is not None:
        os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", str(etag_timeout))

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install `datasets` to stream Hugging Face datasets.") from exc

    path = source["path"]
    split = source.get("split", "train")
    streaming = bool(source.get("streaming", default_streaming))
    name = source.get("name")
    data_files = source.get("data_files")
    cache_dir = source.get("cache_dir", default_cache_dir)
    revision = source.get("revision")
    kwargs = {"split": split, "streaming": streaming}
    if data_files:
        kwargs["data_files"] = data_files
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    if revision:
        kwargs["revision"] = revision

    dataset = load_dataset(path, name, **kwargs) if name else load_dataset(path, **kwargs)

    shuffle_buffer = source.get("shuffle_buffer", default_shuffle_buffer)
    if streaming and shuffle_buffer:
        dataset = dataset.shuffle(buffer_size=int(shuffle_buffer), seed=int(source.get("seed", default_seed)))

    max_samples = source.get("max_samples")
    if max_samples is not None:
        if streaming:
            dataset = dataset.take(int(max_samples))
        else:
            dataset = dataset.select(range(min(int(max_samples), len(dataset))))

    yield from dataset


def _iter_with_retries(
    source: dict[str, Any],
    data_config: dict[str, Any],
    default_streaming: bool,
    default_shuffle_buffer: int | None,
    default_seed: int,
    default_cache_dir: str | None,
) -> Iterator[dict[str, Any]]:
    retries = int(source.get("retries", data_config.get("hf_retries", 3)))
    sleep_seconds = float(source.get("retry_sleep_seconds", data_config.get("hf_retry_sleep_seconds", 10)))
    backoff = float(source.get("retry_backoff", data_config.get("hf_retry_backoff", 2.0)))
    attempt = 0

    while True:
        try:
            yield from _iter_hf_dataset(
                source,
                default_streaming,
                default_shuffle_buffer,
                default_seed,
                default_cache_dir,
                data_config.get("hf_download_timeout"),
                data_config.get("hf_etag_timeout"),
                data_config.get("hf_endpoint"),
            )
            return
        except Exception as exc:
            attempt += 1
            if attempt > retries:
                raise RuntimeError(
                    f"Failed to load Hugging Face source {source.get('path')} "
                    f"after {retries} retries. Last error: {exc}"
                ) from exc

            wait = sleep_seconds * (backoff ** (attempt - 1))
            print(
                f"[data] {source.get('path')} failed on attempt {attempt}/{retries}: {exc}. "
                f"Retrying in {wait:.1f}s...",
                file=sys.stderr,
            )
            time.sleep(wait)


def iter_records(
    data_config: dict[str, Any],
    rank: int = 0,
    world_size: int = 1,
) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
    default_streaming = bool(data_config.get("streaming", True))
    default_shuffle_buffer = data_config.get("shuffle_buffer")
    default_seed = int(data_config.get("seed", 42))
    default_cache_dir = data_config.get("hf_cache_dir")
    for source in data_config.get("sources", []):
        source_type = source.get("type", "hf")
        if source_type == "local_jsonl":
            records = _iter_local_jsonl(source["path"])
        elif source_type == "hf":
            records = _iter_with_retries(
                source,
                data_config,
                default_streaming,
                default_shuffle_buffer,
                default_seed,
                default_cache_dir,
            )
        else:
            raise ValueError(f"Unknown data source type: {source_type}")

        max_samples = source.get("max_samples")
        if source_type == "local_jsonl" and max_samples is not None:
            records = islice(records, int(max_samples))

        for index, record in enumerate(records):
            if world_size <= 1 or index % world_size == rank:
                yield record, source


def iter_texts(data_config: dict[str, Any], rank: int = 0, world_size: int = 1) -> Iterator[str]:
    for record, source in iter_records(data_config, rank=rank, world_size=world_size):
        text = format_record(record, source)
        if text:
            yield text


class TokenBlockDataset(IterableDataset):
    def __init__(
        self,
        data_config: dict[str, Any],
        tokenizer: Any,
        block_size: int,
        drop_last: bool = True,
        add_eos: bool = True,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        self.data_config = data_config
        self.tokenizer = tokenizer
        self.block_size = int(block_size)
        self.drop_last = bool(drop_last)
        self.add_eos = bool(add_eos)
        self.rank = int(rank)
        self.world_size = int(world_size)

    def __iter__(self) -> Iterator[dict[str, list[int]]]:
        buffer: list[int] = []
        eos_id = self.tokenizer.eos_token_id
        worker = get_worker_info()
        if worker is None:
            rank = self.rank
            world_size = self.world_size
        else:
            rank = self.rank * worker.num_workers + worker.id
            world_size = self.world_size * worker.num_workers

        for text in iter_texts(self.data_config, rank=rank, world_size=world_size):
            ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
            if self.add_eos and eos_id is not None and (not ids or ids[-1] != eos_id):
                ids.append(eos_id)
            buffer.extend(ids)

            while len(buffer) >= self.block_size:
                chunk = buffer[: self.block_size]
                del buffer[: self.block_size]
                yield {"input_ids": chunk}

        if not self.drop_last and len(buffer) > 1:
            yield {"input_ids": buffer}


def causal_lm_collate(features: list[dict[str, list[int]]], pad_token_id: int) -> dict[str, torch.Tensor]:
    if torch is None:
        raise RuntimeError("Install `torch` to build training batches.")

    max_len = max(len(feature["input_ids"]) for feature in features)
    batch_size = len(features)

    input_ids = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
    labels = torch.full((batch_size, max_len), -100, dtype=torch.long)

    for row, feature in enumerate(features):
        ids = torch.tensor(feature["input_ids"], dtype=torch.long)
        length = ids.numel()
        input_ids[row, :length] = ids
        attention_mask[row, :length] = 1
        labels[row, :length] = ids

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def build_dataloader(
    data_config: dict[str, Any],
    tokenizer: Any,
    block_size: int,
    batch_size: int,
    num_workers: int = 0,
    rank: int = 0,
    world_size: int = 1,
) -> DataLoader:
    if torch is None or DataLoader is None:
        raise RuntimeError("Install `torch` to build a training dataloader.")

    dataset = TokenBlockDataset(
        data_config=data_config,
        tokenizer=tokenizer,
        block_size=block_size,
        drop_last=bool(data_config.get("drop_last", True)),
        add_eos=bool(data_config.get("add_eos", True)),
        rank=rank,
        world_size=world_size,
    )
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        num_workers=int(num_workers),
        collate_fn=lambda batch: causal_lm_collate(batch, int(pad_token_id)),
    )
