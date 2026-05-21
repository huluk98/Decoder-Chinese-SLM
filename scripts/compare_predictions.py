#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def read_rows(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path).expanduser()
    if not source.exists():
        raise FileNotFoundError(source)
    if source.suffix.lower() == ".csv":
        with source.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    rows: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{source}:{line_number} must contain a JSON object.")
            rows.append(payload)
    return rows


def row_key(row: dict[str, Any], fallback: int) -> str:
    return str(row.get("id") or row.get("index") or fallback)


def prediction_text(row: dict[str, Any]) -> str:
    return str(
        row.get("normalized_prediction")
        or row.get("comparison_generated")
        or row.get("normalized_generated")
        or row.get("raw_prediction")
        or row.get("generated")
        or row.get("prediction")
        or ""
    )


def raw_prediction_text(row: dict[str, Any]) -> str:
    return str(row.get("raw_prediction") or row.get("generated") or row.get("prediction") or "")


def exact_match(row: dict[str, Any]) -> bool | None:
    value = row.get("exact_match")
    if isinstance(value, bool):
        return value
    if value is None or value == "":
        return None
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def compact_example(key: str, old: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": key,
        "prompt": old.get("prompt") or new.get("prompt") or "",
        "label": old.get("normalized_label") or old.get("normalized_target") or old.get("raw_label") or "",
        "old_exact_match": exact_match(old),
        "new_exact_match": exact_match(new),
        "old_prediction": prediction_text(old),
        "new_prediction": prediction_text(new),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two prompt-response prediction files.")
    parser.add_argument("old_predictions")
    parser.add_argument("new_predictions")
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--examples-limit", type=int, default=50)
    args = parser.parse_args()

    old_rows = {row_key(row, index): row for index, row in enumerate(read_rows(args.old_predictions))}
    new_rows = {row_key(row, index): row for index, row in enumerate(read_rows(args.new_predictions))}
    common_keys = sorted(set(old_rows) & set(new_rows))
    only_old = sorted(set(old_rows) - set(new_rows))
    only_new = sorted(set(new_rows) - set(old_rows))

    identical_normalized = 0
    identical_raw = 0
    changed = []
    corrected = []
    regressed = []
    for key in common_keys:
        old = old_rows[key]
        new = new_rows[key]
        if prediction_text(old) == prediction_text(new):
            identical_normalized += 1
        else:
            changed.append(compact_example(key, old, new))
        if raw_prediction_text(old) == raw_prediction_text(new):
            identical_raw += 1
        old_correct = exact_match(old)
        new_correct = exact_match(new)
        if old_correct is False and new_correct is True:
            corrected.append(compact_example(key, old, new))
        elif old_correct is True and new_correct is False:
            regressed.append(compact_example(key, old, new))

    report = {
        "old_predictions": str(Path(args.old_predictions).expanduser()),
        "new_predictions": str(Path(args.new_predictions).expanduser()),
        "old_rows": len(old_rows),
        "new_rows": len(new_rows),
        "common_rows": len(common_keys),
        "only_old_rows": len(only_old),
        "only_new_rows": len(only_new),
        "identical_normalized_predictions": identical_normalized,
        "identical_raw_predictions": identical_raw,
        "changed_predictions": len(changed),
        "wrong_to_correct": len(corrected),
        "correct_to_wrong": len(regressed),
        "changed_examples": changed[: int(args.examples_limit)],
        "wrong_to_correct_examples": corrected[: int(args.examples_limit)],
        "correct_to_wrong_examples": regressed[: int(args.examples_limit)],
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.output_json:
        output_path = Path(args.output_json).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
