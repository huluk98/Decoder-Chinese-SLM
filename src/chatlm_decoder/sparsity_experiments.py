from __future__ import annotations

import csv
import json
import math
import random
import re
import unicodedata
from pathlib import Path
from typing import Any

from chatlm_decoder.command_eval import canonicalize_command_response

DIFFICULTY_LEVELS = ("easy", "medium", "hard")
BENCHMARK_LIST_KEYS = ("data", "records", "items", "examples", "eval", "validation", "test")
SAMPLE_ID_FIELDS = ("id", "sample_id", "uid", "uuid", "example_id", "index")
INPUT_FIELDS = ("input", "prompt", "instruction", "question", "command", "source")
TARGET_FIELDS = ("target", "response", "answer", "output", "completion", "label")
DIFFICULTY_FIELDS = ("difficulty", "complexity", "level")
ZERO_WIDTH_PATTERN = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060\ufeff]")
SPACE_PATTERN = re.compile(r"\s+")
PUNCT_TRANSLATION = str.maketrans(
    {
        "“": '"',
        "”": '"',
        "„": '"',
        "‘": "'",
        "’": "'",
        "‚": "'",
        "（": "(",
        "）": ")",
        "，": ",",
        "。": ".",
        "．": ".",
        "！": "!",
        "？": "?",
        "：": ":",
        "；": ";",
        "［": "[",
        "］": "]",
        "｛": "{",
        "｝": "}",
        "＝": "=",
    }
)
TRAILING_PUNCT_PATTERN = re.compile(r"[\s。．.！!？?；;，,]+$")


def read_records(path: str | Path) -> list[dict[str, Any]]:
    data_path = Path(path).expanduser()
    suffix = data_path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(data_path.read_text(encoding="utf-8"))
        records = coerce_record_list(payload, data_path)
    elif suffix == ".jsonl":
        records = []
        with data_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                record = json.loads(stripped)
                if not isinstance(record, dict):
                    raise ValueError(f"{data_path}:{line_number} must be a JSON object.")
                records.append(record)
    elif suffix == ".csv":
        with data_path.open("r", encoding="utf-8", newline="") as handle:
            records = [dict(row) for row in csv.DictReader(handle)]
    else:
        raise ValueError(f"Unsupported benchmark extension {suffix!r}; use .json, .jsonl, or .csv.")
    return records


def coerce_record_list(payload: Any, path: Path) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict):
        records = None
        for key in BENCHMARK_LIST_KEYS:
            value = payload.get(key)
            if isinstance(value, list):
                records = value
                break
        if records is None:
            records = [payload]
    else:
        raise ValueError(f"{path} must contain a JSON list or an object with records.")

    clean: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"{path}: record {index} must be an object.")
        clean.append(record)
    return clean


def first_nonempty(record: dict[str, Any], fields: tuple[str, ...]) -> str:
    for field in fields:
        value = record.get(field)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def sample_id(record: dict[str, Any], fallback_index: int) -> str:
    return first_nonempty(record, SAMPLE_ID_FIELDS) or str(fallback_index)


def embedded_difficulty(record: dict[str, Any]) -> str:
    value = first_nonempty(record, DIFFICULTY_FIELDS)
    if value:
        return normalize_difficulty(value)
    metadata = record.get("metadata")
    if isinstance(metadata, dict):
        value = first_nonempty(metadata, DIFFICULTY_FIELDS)
        if value:
            return normalize_difficulty(value)
    return ""


def normalize_difficulty(value: Any) -> str:
    difficulty = str(value).strip().lower()
    aliases = {
        "simple": "easy",
        "low": "easy",
        "basic": "easy",
        "moderate": "medium",
        "mid": "medium",
        "intermediate": "medium",
        "difficult": "hard",
        "complex": "hard",
        "high": "hard",
    }
    difficulty = aliases.get(difficulty, difficulty)
    if difficulty not in DIFFICULTY_LEVELS:
        raise ValueError(
            f"Difficulty label {value!r} is not supported. Expected one of: {', '.join(DIFFICULTY_LEVELS)}."
        )
    return difficulty


def normalize_sample(record: dict[str, Any], fallback_index: int) -> dict[str, Any]:
    input_text = first_nonempty(record, INPUT_FIELDS)
    target_text = first_nonempty(record, TARGET_FIELDS)
    if not input_text or not target_text:
        raise ValueError(
            f"Benchmark record {fallback_index} must contain input/prompt/instruction and target/response/answer text."
        )
    return {
        "sample_id": sample_id(record, fallback_index),
        "input": input_text,
        "target": target_text,
        "difficulty": embedded_difficulty(record),
        "raw_record": record,
    }


def load_benchmark_samples(
    benchmark_path: str | Path,
    difficulty_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    records = read_records(benchmark_path)
    samples = [normalize_sample(record, index) for index, record in enumerate(records)]
    if all(sample["difficulty"] for sample in samples):
        return samples
    if any(sample["difficulty"] for sample in samples):
        missing = [sample["sample_id"] for sample in samples if not sample["difficulty"]]
        raise ValueError(
            "Benchmark has partial embedded difficulty labels; missing labels for sample ids: "
            + ", ".join(missing[:20])
        )
    if difficulty_path is None:
        raise ValueError(
            "Benchmark does not contain difficulty/complexity/level labels. "
            "Pass --benchmark_difficulty_path with id/sample_id/input,difficulty labels."
        )
    return attach_external_difficulty(samples, difficulty_path)


def load_prompt_response_samples(path: str | Path) -> list[dict[str, Any]]:
    records = read_records(path)
    samples = [normalize_sample(record, index) for index, record in enumerate(records)]
    for sample in samples:
        if not sample["difficulty"]:
            sample["difficulty"] = "easy"
    return samples


def attach_external_difficulty(samples: list[dict[str, Any]], difficulty_path: str | Path) -> list[dict[str, Any]]:
    difficulty_records = read_records(difficulty_path)
    by_id: dict[str, str] = {}
    by_input: dict[str, str] = {}
    for index, record in enumerate(difficulty_records):
        label = first_nonempty(record, DIFFICULTY_FIELDS)
        if not label:
            raise ValueError(f"{difficulty_path}: difficulty row {index} is missing a difficulty label.")
        difficulty = normalize_difficulty(label)
        row_id = first_nonempty(record, ("id", "sample_id"))
        row_input = first_nonempty(record, ("input", "prompt", "instruction", "question", "command"))
        if row_id:
            by_id[row_id] = difficulty
        if row_input:
            by_input[row_input] = difficulty
        if not row_id and not row_input:
            raise ValueError(f"{difficulty_path}: row {index} needs id, sample_id, or input.")

    attached: list[dict[str, Any]] = []
    missing: list[str] = []
    for sample in samples:
        difficulty = ""
        sample_key = str(sample.get("sample_id", "")).strip()
        if sample_key and sample_key in by_id:
            difficulty = by_id[sample_key]
        elif sample["input"] in by_input:
            difficulty = by_input[sample["input"]]
        else:
            missing.append(sample_key or sample["input"])
            continue
        row = dict(sample)
        row["difficulty"] = difficulty
        attached.append(row)
    if missing:
        raise ValueError(
            "Could not join difficulty labels by sample id or exact input for: "
            + ", ".join(missing[:20])
        )
    return attached


def normalize_prediction_text(text: Any, mode: str = "normalized") -> str:
    value = unicodedata.normalize("NFKC", str(text))
    value = ZERO_WIDTH_PATTERN.sub("", value)
    value = value.translate(PUNCT_TRANSLATION)
    value = TRAILING_PUNCT_PATTERN.sub("", value.strip())
    value = SPACE_PATTERN.sub(" ", value).strip()
    if mode in {"exact", "whitespace"}:
        return value
    if mode in {"normalized", "punctuation"}:
        return value
    if mode == "command":
        return canonicalize_command_response(value)
    raise ValueError(f"Unknown normalization mode: {mode}")


def exact_match_flags(target: str, predictions: list[str], normalization_mode: str = "normalized") -> tuple[bool, bool]:
    normalized_target = normalize_prediction_text(target, mode=normalization_mode)
    normalized_predictions = [
        normalize_prediction_text(prediction, mode=normalization_mode)
        for prediction in predictions[:5]
    ]
    em1 = bool(normalized_predictions) and normalized_predictions[0] == normalized_target
    em5 = normalized_target in normalized_predictions
    return em1, em5


def prediction_result_rows(
    samples: list[dict[str, Any]],
    candidate_lists: list[list[str]],
    *,
    normalization_mode: str,
    model_family: str,
    pruning_mode: str,
    pruning_method: str,
    target_sparsity: float,
    targeted_linear_sparsity_actual: float,
    whole_model_sparsity_actual: float,
    seed: int,
) -> list[dict[str, Any]]:
    if len(samples) != len(candidate_lists):
        raise ValueError(f"Expected {len(samples)} candidate lists, got {len(candidate_lists)}.")
    rows: list[dict[str, Any]] = []
    for sample, candidates in zip(samples, candidate_lists):
        clean_candidates = [str(candidate) for candidate in candidates[:5]]
        em1, em5 = exact_match_flags(sample["target"], clean_candidates, normalization_mode=normalization_mode)
        rows.append(
            {
                "sample_id": sample["sample_id"],
                "input": sample["input"],
                "target": sample["target"],
                "difficulty": sample["difficulty"],
                "top1_prediction": clean_candidates[0] if clean_candidates else "",
                "top5_predictions": json.dumps(clean_candidates, ensure_ascii=False),
                "em1": em1,
                "em5": em5,
                "model_family": model_family,
                "pruning_mode": pruning_mode,
                "pruning_method": pruning_method,
                "target_sparsity": target_sparsity,
                "targeted_linear_sparsity_actual": targeted_linear_sparsity_actual,
                "whole_model_sparsity_actual": whole_model_sparsity_actual,
                "seed": seed,
            }
        )
    return rows


def score_rate(rows: list[dict[str, Any]], field: str) -> float | None:
    if not rows:
        return None
    return sum(1 for row in rows if bool(row.get(field))) / float(len(rows))


def bootstrap_ci(
    rows: list[dict[str, Any]],
    field: str,
    *,
    resamples: int = 1000,
    seed: int = 42,
    min_n: int = 20,
) -> tuple[float | str | None, float | str | None]:
    if len(rows) < min_n:
        return "insufficient_n", "insufficient_n"
    rng = random.Random(int(seed))
    n = len(rows)
    values = [1.0 if bool(row.get(field)) else 0.0 for row in rows]
    estimates = []
    for _ in range(int(resamples)):
        total = 0.0
        for _index in range(n):
            total += values[rng.randrange(n)]
        estimates.append(total / n)
    estimates.sort()
    low_index = max(0, min(len(estimates) - 1, int(math.floor(0.025 * len(estimates)))))
    high_index = max(0, min(len(estimates) - 1, int(math.ceil(0.975 * len(estimates))) - 1))
    return estimates[low_index], estimates[high_index]


def summarize_prediction_rows(
    rows: list[dict[str, Any]],
    *,
    bootstrap_resamples: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "count_total": len(rows),
        "em1_overall": score_rate(rows, "em1"),
        "em5_overall": score_rate(rows, "em5"),
    }
    em1_low, em1_high = bootstrap_ci(rows, "em1", resamples=bootstrap_resamples, seed=seed)
    em5_low, em5_high = bootstrap_ci(rows, "em5", resamples=bootstrap_resamples, seed=seed + 1)
    summary.update(
        {
            "em1_overall_ci_low": em1_low,
            "em1_overall_ci_high": em1_high,
            "em5_overall_ci_low": em5_low,
            "em5_overall_ci_high": em5_high,
        }
    )
    for level in DIFFICULTY_LEVELS:
        group = [row for row in rows if row.get("difficulty") == level]
        summary[f"count_{level}"] = len(group)
        summary[f"em1_{level}"] = score_rate(group, "em1")
        summary[f"em5_{level}"] = score_rate(group, "em5")
        group_em1_low, group_em1_high = bootstrap_ci(group, "em1", resamples=bootstrap_resamples, seed=seed)
        group_em5_low, group_em5_high = bootstrap_ci(group, "em5", resamples=bootstrap_resamples, seed=seed + 1)
        summary[f"em1_{level}_ci_low"] = group_em1_low
        summary[f"em1_{level}_ci_high"] = group_em1_high
        summary[f"em5_{level}_ci_low"] = group_em5_low
        summary[f"em5_{level}_ci_high"] = group_em5_high
    return summary


def retention(current: float | None, dense: float | None) -> float | None:
    if current is None or dense is None or dense == 0:
        return None
    return float(current) / float(dense)


def add_retention_metrics(row: dict[str, Any], dense_row: dict[str, Any] | None) -> dict[str, Any]:
    enriched = dict(row)
    dense = dense_row or {}
    enriched["em1_retention_overall"] = retention(row.get("em1_overall"), dense.get("em1_overall"))
    enriched["em5_retention_overall"] = retention(row.get("em5_overall"), dense.get("em5_overall"))
    for level in DIFFICULTY_LEVELS:
        enriched[f"em1_retention_{level}"] = retention(row.get(f"em1_{level}"), dense.get(f"em1_{level}"))
        enriched[f"em5_retention_{level}"] = retention(row.get(f"em5_{level}"), dense.get(f"em5_{level}"))
    return enriched


def write_csv_rows(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
