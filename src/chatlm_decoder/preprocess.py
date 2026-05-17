from __future__ import annotations

import copy
import hashlib
import json
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable, **_: Any):
        return iterable

from chatlm_decoder.data import EOS_TOKEN, download_hf_sources, format_record, iter_records


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
    preprocess_config.setdefault("strict_min_rows", True)
    preprocess_config.setdefault("continue_on_source_error", False)
    preprocess_config.setdefault("download_first", False)
    preprocess_config.setdefault("download_manifest_path", f"{preprocess_config['output_path']}.download_manifest.json")
    preprocess_config.setdefault("continue_on_download_error", None)
    preprocess_config.setdefault("shuffle_before_write", False)
    return preprocess_config


def preprocessed_data_config(config: dict[str, Any]) -> dict[str, Any]:
    preprocess_config = _preprocess_config(config)
    output_path = Path(preprocess_config["output_path"]).expanduser()
    if not preprocess_config["enabled"] or not output_path.exists():
        return config["data"]

    data_config = {
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
    for key in ("token_ids_path", "token_ids_dtype", "token_ids_manifest_path"):
        if key in config["data"]:
            data_config[key] = config["data"][key]
    return data_config


def preprocess_datasets(
    config: dict[str, Any],
    force: bool = False,
    force_download: bool = False,
) -> dict[str, Any]:
    preprocess_config = _preprocess_config(config)
    if not preprocess_config["enabled"]:
        return {"enabled": False, "output_path": None, "written": 0}

    output_path = Path(preprocess_config["output_path"]).expanduser()
    manifest_path = Path(preprocess_config["manifest_path"]).expanduser()
    overwrite = bool(force or preprocess_config["overwrite"])

    min_rows = preprocess_config["min_rows"]
    min_rows = int(min_rows) if min_rows is not None else None
    strict_min_rows = bool(preprocess_config["strict_min_rows"])
    continue_on_source_error = bool(preprocess_config["continue_on_source_error"])
    continue_on_download_error = preprocess_config["continue_on_download_error"]
    if continue_on_download_error is None:
        continue_on_download_error = continue_on_source_error
    continue_on_download_error = bool(continue_on_download_error)

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
        below_min_rows = min_rows is not None and int(manifest.get("written", 0)) < min_rows
        manifest["below_min_rows"] = bool(below_min_rows)
        if below_min_rows and strict_min_rows:
            raise RuntimeError(
                f"Processed dataset has {manifest.get('written', 0):,} rows, below min_rows={min_rows:,}. "
                "Run with --force after adding/fixing sources."
            )
        if below_min_rows:
            print(
                f"[preprocess] Existing dataset has {manifest.get('written', 0):,} rows, "
                f"below min_rows={min_rows:,}. Continuing because strict_min_rows=false.",
                file=sys.stderr,
            )
        return manifest

    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    data_config = copy.deepcopy(config["data"])
    if not preprocess_config["shuffle_before_write"]:
        data_config.pop("shuffle_buffer", None)
        for source in data_config.get("sources", []):
            source.pop("shuffle_buffer", None)

    download_manifest: dict[str, Any] | None = None
    if preprocess_config["download_first"]:
        data_config, download_manifest = download_hf_sources(
            data_config,
            force_download=force_download,
            continue_on_error=continue_on_download_error,
        )
        download_manifest_path = Path(preprocess_config["download_manifest_path"]).expanduser()
        download_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        download_manifest["manifest_path"] = str(download_manifest_path)
        with download_manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(download_manifest, handle, ensure_ascii=False, indent=2)
            handle.write("\n")

    min_chars = int(preprocess_config["min_chars"] or 0)
    max_chars = preprocess_config["max_chars"]
    max_chars = int(max_chars) if max_chars is not None else None
    dedupe = bool(preprocess_config["dedupe"])
    seen: set[str] = set()
    counts: dict[str, dict[str, int]] = defaultdict(lambda: {"read": 0, "written": 0, "skipped": 0})
    failed_sources: list[dict[str, str]] = []
    total_read = 0
    total_written = 0
    total_skipped = 0
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    sources = data_config.get("sources", [])

    with tmp_path.open("w", encoding="utf-8") as handle:
        for source_index, source in enumerate(sources, start=1):
            name = _source_name(source)
            source_data_config = copy.deepcopy(data_config)
            source_data_config["sources"] = [source]
            progress_desc = f"preprocess {source_index}/{len(sources)} {name}"

            try:
                records = iter_records(source_data_config)
                for record, record_source in tqdm(records, desc=progress_desc, unit="row"):
                    total_read += 1
                    record_name = _source_name(record_source)
                    counts[record_name]["read"] += 1

                    raw_text = format_record(record, record_source)
                    if not raw_text:
                        total_skipped += 1
                        counts[record_name]["skipped"] += 1
                        continue

                    text = normalize_text(raw_text)
                    if len(text) < min_chars or (max_chars is not None and len(text) > max_chars):
                        total_skipped += 1
                        counts[record_name]["skipped"] += 1
                        continue

                    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
                    if dedupe and digest in seen:
                        total_skipped += 1
                        counts[record_name]["skipped"] += 1
                        continue
                    seen.add(digest)

                    handle.write(json.dumps({"text": text, "source": record_name}, ensure_ascii=False) + "\n")
                    total_written += 1
                    counts[record_name]["written"] += 1
            except Exception as exc:
                failed_sources.append({"source": name, "error": str(exc)})
                counts[name]["failed"] = 1
                if not continue_on_source_error:
                    raise
                print(
                    f"[preprocess] Skipping failed source {name}: {exc}",
                    file=sys.stderr,
                )
                continue

    tmp_path.replace(output_path)
    below_min_rows = min_rows is not None and total_written < min_rows

    manifest = {
        "output_path": str(output_path),
        "manifest_path": str(manifest_path),
        "written": total_written,
        "read": total_read,
        "skipped": total_skipped,
        "dedupe": dedupe,
        "min_rows": min_rows,
        "strict_min_rows": strict_min_rows,
        "continue_on_source_error": continue_on_source_error,
        "download_first": bool(preprocess_config["download_first"]),
        "download_manifest_path": preprocess_config["download_manifest_path"],
        "download_manifest": download_manifest,
        "below_min_rows": bool(below_min_rows),
        "failed_sources": failed_sources,
        "sources": counts,
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    if below_min_rows and strict_min_rows:
        raise RuntimeError(
            f"Processed dataset has {total_written:,} rows, below min_rows={min_rows:,}. "
            "At least one configured source may be missing, empty, or using the wrong schema."
        )
    if below_min_rows:
        print(
            f"[preprocess] Wrote {total_written:,} rows, below min_rows={min_rows:,}. "
            "Continuing because strict_min_rows=false; inspect failed_sources in the manifest.",
            file=sys.stderr,
        )
    return manifest


def ensure_preprocessed_data(
    config: dict[str, Any],
    force: bool = False,
    force_download: bool = False,
) -> dict[str, Any]:
    preprocess_config = _preprocess_config(config)
    if not preprocess_config["enabled"]:
        return config["data"]

    preprocess_datasets(config, force=force, force_download=force_download)
    return preprocessed_data_config(config)
