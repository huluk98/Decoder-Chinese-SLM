from __future__ import annotations

import json
import os
import sys
import time
from itertools import islice
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

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


def _apply_hf_env(
    source: dict[str, Any],
    default_download_timeout: int | None = None,
    default_etag_timeout: int | None = None,
    default_endpoint: str | None = None,
) -> None:
    endpoint = source.get("endpoint", default_endpoint)
    download_timeout = source.get("download_timeout", default_download_timeout)
    etag_timeout = source.get("etag_timeout", default_etag_timeout)
    if endpoint:
        os.environ.setdefault("HF_ENDPOINT", str(endpoint))
    if download_timeout is not None:
        os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", str(download_timeout))
    if etag_timeout is not None:
        os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", str(etag_timeout))


def _retry_hf_call(
    label: str,
    source: dict[str, Any],
    data_config: dict[str, Any],
    call: Callable[[], Any],
) -> Any:
    retries = int(source.get("retries", data_config.get("hf_retries", 3)))
    sleep_seconds = float(source.get("retry_sleep_seconds", data_config.get("hf_retry_sleep_seconds", 10)))
    backoff = float(source.get("retry_backoff", data_config.get("hf_retry_backoff", 2.0)))
    attempt = 0

    while True:
        try:
            return call()
        except Exception as exc:
            attempt += 1
            if attempt > retries:
                raise RuntimeError(
                    f"Failed to {label} Hugging Face source {source.get('path')} "
                    f"after {retries} retries. Last error: {exc}"
                ) from exc

            wait = sleep_seconds * (backoff ** (attempt - 1))
            print(
                f"[data] {source.get('path')} failed during {label} on attempt {attempt}/{retries}: {exc}. "
                f"Retrying in {wait:.1f}s...",
                file=sys.stderr,
            )
            time.sleep(wait)


def _as_pattern_list(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        patterns: list[str] = []
        for item in value.values():
            nested = _as_pattern_list(item)
            if nested:
                patterns.extend(nested)
        return patterns
    return [str(item) for item in value]


def _resolve_local_data_files(value: Any, base_path: Path) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        if "://" in value or Path(value).is_absolute():
            return value
        return str(base_path / value)
    if isinstance(value, dict):
        return {key: _resolve_local_data_files(item, base_path) for key, item in value.items()}
    return [_resolve_local_data_files(item, base_path) for item in value]


def _data_file_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        values: list[str] = []
        for item in value.values():
            values.extend(_data_file_values(item))
        return values
    return [str(item) for item in value]


def _dataset_builder_for_data_files(data_files: Any) -> str | None:
    suffixes = {Path(path).suffix.lower() for path in _data_file_values(data_files)}
    if suffixes and suffixes <= {".json", ".jsonl"}:
        return "json"
    if suffixes == {".parquet"}:
        return "parquet"
    return None


def _find_snapshot_data_files(snapshot_path: Path) -> list[str]:
    patterns = ("*.jsonl", "*.json", "*.parquet")
    files: list[Path] = []
    for pattern in patterns:
        files.extend(snapshot_path.rglob(pattern))
    return [
        str(path)
        for path in sorted(files)
        if not any(part.startswith(".") for part in path.relative_to(snapshot_path).parts)
        and path.name not in {"dataset_infos.json"}
    ]


def download_hf_sources(
    data_config: dict[str, Any],
    force_download: bool = False,
    continue_on_error: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Download HF dataset repos to the local cache and return a cache-backed config."""
    downloaded_config = dict(data_config)
    downloaded_config["streaming"] = False
    downloaded_config.pop("shuffle_buffer", None)

    default_cache_dir = data_config.get("hf_cache_dir")
    downloaded_sources: list[dict[str, Any]] = []
    manifest_sources: list[dict[str, Any]] = []
    snapshot_download: Callable[..., Any] | None = None

    for source in data_config.get("sources", []):
        source_type = source.get("type", "hf")
        name = source.get("path", source_type)

        if source_type != "hf":
            downloaded_sources.append(dict(source))
            manifest_sources.append({"source": name, "type": source_type, "status": "local"})
            continue

        if snapshot_download is None:
            try:
                from huggingface_hub import snapshot_download as hf_snapshot_download
            except ImportError as exc:
                raise RuntimeError("Install `huggingface_hub` to download Hugging Face datasets first.") from exc
            snapshot_download = hf_snapshot_download

        _apply_hf_env(
            source,
            data_config.get("hf_download_timeout"),
            data_config.get("hf_etag_timeout"),
            data_config.get("hf_endpoint"),
        )

        cache_dir = source.get("cache_dir", default_cache_dir)
        allow_patterns = _as_pattern_list(source.get("download_allow_patterns"))
        if allow_patterns is None:
            allow_patterns = _as_pattern_list(source.get("data_files"))
        ignore_patterns = _as_pattern_list(source.get("download_ignore_patterns"))
        revision = source.get("revision")

        def _download() -> str:
            kwargs: dict[str, Any] = {
                "repo_id": source["path"],
                "repo_type": "dataset",
                "force_download": bool(force_download),
            }
            if cache_dir:
                kwargs["cache_dir"] = cache_dir
            if revision:
                kwargs["revision"] = revision
            if allow_patterns:
                kwargs["allow_patterns"] = allow_patterns
            if ignore_patterns:
                kwargs["ignore_patterns"] = ignore_patterns
            return str(snapshot_download(**kwargs))

        try:
            snapshot_path = _retry_hf_call("download", source, data_config, _download)
        except Exception as exc:
            manifest_sources.append({"source": name, "type": source_type, "status": "failed", "error": str(exc)})
            if not continue_on_error:
                raise
            print(f"[download] Skipping failed source {name}: {exc}", file=sys.stderr)
            continue

        downloaded_source = dict(source)
        downloaded_source["local_path"] = snapshot_path
        downloaded_source["streaming"] = False
        downloaded_sources.append(downloaded_source)
        manifest_sources.append(
            {
                "source": name,
                "type": source_type,
                "status": "downloaded",
                "snapshot_path": snapshot_path,
                "allow_patterns": allow_patterns,
            }
        )

    downloaded_config["sources"] = downloaded_sources
    manifest = {
        "downloaded": sum(1 for item in manifest_sources if item["status"] == "downloaded"),
        "failed": sum(1 for item in manifest_sources if item["status"] == "failed"),
        "local": sum(1 for item in manifest_sources if item["status"] == "local"),
        "force_download": bool(force_download),
        "sources": manifest_sources,
    }
    return downloaded_config, manifest


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
    _apply_hf_env(source, default_download_timeout, default_etag_timeout, default_endpoint)

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install `datasets` to stream Hugging Face datasets.") from exc

    load_from_local_snapshot = "local_path" in source
    path = source.get("local_path", source["path"])
    split = source.get("split", "train")
    streaming = bool(source.get("streaming", default_streaming))
    name = source.get("name")
    data_files = source.get("data_files")
    if data_files and load_from_local_snapshot:
        data_files = _resolve_local_data_files(data_files, Path(path))
    if load_from_local_snapshot and not data_files:
        data_files = _find_snapshot_data_files(Path(path))
    cache_dir = source.get("cache_dir", default_cache_dir)
    revision = source.get("revision")
    kwargs = {"split": split, "streaming": streaming}
    if data_files:
        kwargs["data_files"] = data_files
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    if revision and not load_from_local_snapshot:
        kwargs["revision"] = revision

    builder = _dataset_builder_for_data_files(data_files) if load_from_local_snapshot else None
    if builder:
        dataset = load_dataset(builder, **kwargs)
    else:
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
                f"[data] {source.get('path')} failed during load on attempt {attempt}/{retries}: {exc}. "
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


class PackedTokenDataset(IterableDataset):
    def __init__(
        self,
        token_ids_path: str | Path,
        token_ids_dtype: str,
        block_size: int,
        rank: int = 0,
        world_size: int = 1,
        start_block_offset: int = 0,
    ) -> None:
        self.token_ids_path = Path(token_ids_path).expanduser()
        self.token_ids_dtype = str(token_ids_dtype)
        self.block_size = int(block_size)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.start_block_offset = int(start_block_offset)

    def __iter__(self) -> Iterator[dict[str, list[int]]]:
        try:
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("Install `numpy` to train from packed token ids.") from exc

        worker = get_worker_info()
        if worker is None:
            worker_rank = self.rank
            worker_world_size = self.world_size
        else:
            worker_rank = self.rank * worker.num_workers + worker.id
            worker_world_size = self.world_size * worker.num_workers

        token_ids = np.memmap(self.token_ids_path, dtype=np.dtype(self.token_ids_dtype), mode="r")
        total_blocks = int(token_ids.shape[0]) // self.block_size
        if total_blocks <= 0:
            return

        start_offset = self.start_block_offset % total_blocks
        for local_block_index in range(worker_rank, total_blocks, worker_world_size):
            block_index = (start_offset + local_block_index) % total_blocks
            start = block_index * self.block_size
            end = start + self.block_size
            yield {"input_ids": token_ids[start:end].astype(np.int64, copy=False).tolist()}


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
    pin_memory: bool = False,
    persistent_workers: bool = False,
    prefetch_factor: int | None = None,
    rank: int = 0,
    world_size: int = 1,
) -> DataLoader:
    if torch is None or DataLoader is None:
        raise RuntimeError("Install `torch` to build a training dataloader.")

    token_ids_path = data_config.get("token_ids_path")
    if token_ids_path and Path(token_ids_path).expanduser().exists():
        dataset = PackedTokenDataset(
            token_ids_path=token_ids_path,
            token_ids_dtype=str(data_config.get("token_ids_dtype", "uint16")),
            block_size=block_size,
            rank=rank,
            world_size=world_size,
            start_block_offset=int(data_config.get("start_block_offset", 0)),
        )
        if rank == 0:
            print(f"[data] Training from packed token ids: {Path(token_ids_path).expanduser()}")
            if int(data_config.get("start_block_offset", 0)) > 0:
                print(f"[data] Packed-token resume offset: {int(data_config['start_block_offset']):,} blocks")
    else:
        if token_ids_path and rank == 0:
            print(
                f"[data] Packed token ids not found at {Path(token_ids_path).expanduser()}; "
                "falling back to on-the-fly tokenization.",
                file=sys.stderr,
            )
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
    loader_kwargs: dict[str, Any] = {
        "batch_size": int(batch_size),
        "num_workers": int(num_workers),
        "pin_memory": bool(pin_memory),
        "collate_fn": lambda batch: causal_lm_collate(batch, int(pad_token_id)),
    }
    if int(num_workers) > 0:
        loader_kwargs["persistent_workers"] = bool(persistent_workers)
        if prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = int(prefetch_factor)
    return DataLoader(dataset, **loader_kwargs)
