from __future__ import annotations

import copy
import hashlib
import json
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable, **_: Any):
        return iterable

from chatlm_decoder.data import EOS_TOKEN, format_record, iter_records


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    lines = [line.strip() for line in text.split("\n")]

    normalized_lines: list[str] = []
    blank_seen = False
    for line in lines:
        if not line:
            if not blank_seen:
                normalized_lines.append("")
            blank_seen = True
            continue
        normalized_lines.append(line)
        blank_seen = False

    text = "\n".join(normalized_lines).strip()
    return text if text.endswith(EOS_TOKEN) else f"{text}{EOS_TOKEN}"


def _source_name(source: dict[str, Any]) -> str:
    if source.get("name"):
        return f"{source.get('path')}::{source.get('name')}"
    return str(source.get("path", source.get("type", "unknown")))


def _preprocess_config(config: dict[str, Any]) -> dict[str, Any]:
    preprocess_config = dict(config.get("preprocess") or {})
    preprocess_config.setdefault("enabled", False)
    preprocess_config.setdefault("output_path", "data/processed/normalized.jsonl")
    preprocess_config.setdefault("manifest_path", f"{preprocess_config['output_path']}.manifest.json")
    preprocess_config.setdefault("overwrite", False)
    preprocess_config.setdefault("dedupe", True)
    preprocess_config.setdefault("min_chars", 8)
    preprocess_config.setdefault("max_chars", None)
    preprocess_config.setdefault("min_rows", None)
    preprocess_config.setdefault("shuffle_before_write", False)
    return preprocess_config


def preprocessed_data_config(config: dict[str, Any]) -> dict[str, Any]:
    preprocess_config = _preprocess_config(config)
    output_path = Path(preprocess_config["output_path"]).expanduser()
    if not preprocess_config["enabled"] or not output_path.exists():
        return config["data"]

    return {
        "streaming": False,
        "drop_last": config["data"].get("drop_last", True),
        "add_eos": config["data"].get("add_eos", True),
        "sources": [
            {
                "type": "local_jsonl",
                "path": str(output_path),
                "format": "text",
                "text_field": "text",
            }
        ],
    }


def preprocess_datasets(config: dict[str, Any], force: bool = False) -> dict[str, Any]:
    preprocess_config = _preprocess_config(config)
    if not preprocess_config["enabled"]:
        return {"enabled": False, "output_path": None, "written": 0}

    output_path = Path(preprocess_config["output_path"]).expanduser()
    manifest_path = Path(preprocess_config["manifest_path"]).expanduser()
    overwrite = bool(force or preprocess_config["overwrite"])

    min_rows = preprocess_config["min_rows"]
    min_rows = int(min_rows) if min_rows is not None else None

    if output_path.exists() and not overwrite:
        manifest = {
            "enabled": True,
            "output_path": str(output_path),
            "manifest_path": str(manifest_path),
            "already_exists": True,
        }
        if manifest_path.exists():
            with manifest_path.open("r", encoding="utf-8") as handle:
                manifest.update(json.load(handle))
            manifest["already_exists"] = True
        if min_rows is not None and int(manifest.get("written", 0)) < min_rows:
            raise RuntimeError(
                f"Processed dataset has {manifest.get('written', 0):,} rows, below min_rows={min_rows:,}. "
                "Run with --force after adding/fixing sources."
            )
        return manifest

    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    data_config = copy.deepcopy(config["data"])
    if not preprocess_config["shuffle_before_write"]:
        data_config.pop("shuffle_buffer", None)
        for source in data_config.get("sources", []):
            source.pop("shuffle_buffer", None)

    min_chars = int(preprocess_config["min_chars"] or 0)
    max_chars = preprocess_config["max_chars"]
    max_chars = int(max_chars) if max_chars is not None else None
    dedupe = bool(preprocess_config["dedupe"])
    seen: set[str] = set()
    counts: dict[str, dict[str, int]] = defaultdict(lambda: {"read": 0, "written": 0, "skipped": 0})
    total_read = 0
    total_written = 0
    total_skipped = 0
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")

    with tmp_path.open("w", encoding="utf-8") as handle:
        for record, source in tqdm(iter_records(data_config), desc="preprocess", unit="row"):
            total_read += 1
            name = _source_name(source)
            counts[name]["read"] += 1

            raw_text = format_record(record, source)
            if not raw_text:
                total_skipped += 1
                counts[name]["skipped"] += 1
                continue

            text = normalize_text(raw_text)
            if len(text) < min_chars or (max_chars is not None and len(text) > max_chars):
                total_skipped += 1
                counts[name]["skipped"] += 1
                continue

            digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
            if dedupe and digest in seen:
                total_skipped += 1
                counts[name]["skipped"] += 1
                continue
            seen.add(digest)

            handle.write(json.dumps({"text": text, "source": name}, ensure_ascii=False) + "\n")
            total_written += 1
            counts[name]["written"] += 1

    tmp_path.replace(output_path)

    manifest = {
        "output_path": str(output_path),
        "written": total_written,
        "read": total_read,
        "skipped": total_skipped,
        "dedupe": dedupe,
        "min_rows": min_rows,
        "sources": counts,
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    if min_rows is not None and total_written < min_rows:
        raise RuntimeError(
            f"Processed dataset has {total_written:,} rows, below min_rows={min_rows:,}. "
            "At least one configured source may be missing, empty, or using the wrong schema."
        )
    return manifest


def ensure_preprocessed_data(config: dict[str, Any], force: bool = False) -> dict[str, Any]:
    preprocess_config = _preprocess_config(config)
    if not preprocess_config["enabled"]:
        return config["data"]

    preprocess_datasets(config, force=force)
    return preprocessed_data_config(config)
