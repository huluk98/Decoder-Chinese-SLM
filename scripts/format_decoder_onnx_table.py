#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


PRECISION_ORDER = ("fp16", "int8")
DIFFICULTY_ORDER = {
    "easy": 0,
    "medium": 1,
    "hard": 2,
    "unknown": 99,
}


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dataset_name_from_summary_path(path: Path) -> str:
    # Expected layout: <run_root>/eval/<architecture>/<precision>/<dataset>/prompt_response_eval_summary.json
    return path.parent.name


def load_rows(run_root: str | None, summary_json: str | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if run_root:
        root = Path(run_root).expanduser()
        for path in sorted((root / "eval").rglob("prompt_response_eval_summary.json")):
            payload = read_json(path)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object: {path}")
            payload = dict(payload)
            payload.setdefault("dataset_name", dataset_name_from_summary_path(path))
            payload["_summary_path"] = str(path)
            rows.append(payload)
    if summary_json:
        payload = read_json(Path(summary_json).expanduser())
        if isinstance(payload, dict):
            rows.append(payload)
        elif isinstance(payload, list):
            for item in payload:
                if not isinstance(item, dict):
                    raise ValueError(f"Expected every summary row to be an object in {summary_json}")
                rows.append(dict(item))
        else:
            raise ValueError(f"Expected JSON object or list: {summary_json}")
    return rows


def precision_for(row: dict[str, Any]) -> str:
    precision = str(row.get("precision") or "").strip().lower()
    if precision:
        return precision
    variant = str(row.get("variant") or "").lower()
    for candidate in PRECISION_ORDER:
        if candidate in variant:
            return candidate
    return ""


def filter_rows(rows: list[dict[str, Any]], dataset: str | None) -> list[dict[str, Any]]:
    if not dataset:
        return rows
    wanted = dataset.lower()
    filtered = [row for row in rows if str(row.get("dataset_name") or "").lower() == wanted]
    return filtered


def select_precision_rows(rows: list[dict[str, Any]], precision_order: tuple[str, ...]) -> dict[str, dict[str, Any] | None]:
    selected: dict[str, dict[str, Any] | None] = {}
    for precision in precision_order:
        candidates = [row for row in rows if precision_for(row) == precision]
        selected[precision] = candidates[0] if candidates else None
    return selected


def latex_escape(value: Any) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def format_ms(value: Any) -> str:
    number = float_or_none(value)
    if number is None:
        return "--"
    return f"{number:.1f} ms"


def format_seq_len(row: dict[str, Any] | None) -> str:
    if row is None:
        return "--"
    for key in ("max_seq_len", "input_length"):
        value = row.get(key)
        if value not in (None, ""):
            return str(int(float(value)))
    return "--"


def format_memory(value: Any) -> str:
    number = float_or_none(value)
    if number is None:
        return "--"
    if number >= 1024:
        return f"{number / 1024.0:.2f} GB"
    return f"{number:.0f} MB"


def format_percent(value: Any) -> str:
    number = float_or_none(value)
    if number is None:
        return "--"
    if abs(number) <= 1.0:
        number *= 100.0
    return f"{number:.1f}\\%"


def em_pair(row: dict[str, Any] | None, prefix: str | None = None) -> str:
    if row is None:
        return "--/--"
    if prefix:
        em1 = row.get(f"{prefix}_exact_match_accuracy")
        em5 = row.get(f"{prefix}_exact_match_at_5_accuracy")
        if em5 in (None, ""):
            em5 = row.get(f"{prefix}_exact_match_at_top_k_accuracy")
    else:
        em1 = row.get("exact_match_accuracy")
        em5 = row.get("exact_match_at_5_accuracy")
        if em5 in (None, ""):
            em5 = row.get("exact_match_at_top_k_accuracy")
    return f"{format_percent(em1)}/{format_percent(em5)}"


def runtime_label(row: dict[str, Any] | None, precision: str) -> str:
    runtime = str((row or {}).get("runtime") or "ONNX/TensorRT")
    precision_upper = precision.upper()
    if precision and precision not in runtime.lower():
        runtime = f"{runtime} {precision_upper}"
    return runtime


def main_table(selected: dict[str, dict[str, Any] | None], architecture: str, caption: str, label: str) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{latex_escape(caption)}}}",
        rf"\label{{{label}}}",
        r"\begin{tabular}{llccccc}",
        r"\hline",
        r"Architecture & Runtime & Seq. Len. & Latency & P95 Lat. & Memory & EM@1/EM@5 \\",
        r"\hline",
    ]
    for precision in PRECISION_ORDER:
        row = selected.get(precision)
        lines.append(
            " & ".join(
                [
                    latex_escape(architecture),
                    latex_escape(runtime_label(row, precision)),
                    format_seq_len(row),
                    format_ms((row or {}).get("mean_latency_ms") or (row or {}).get("avg_latency_ms")),
                    format_ms((row or {}).get("p95_latency_ms")),
                    format_memory((row or {}).get("peak_memory_mb")),
                    em_pair(row),
                ]
            )
            + r" \\"
        )
    lines.extend([r"\hline", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines) + "\n"


def difficulty_groups(row: dict[str, Any] | None) -> set[str]:
    if row is None:
        return set()
    groups: set[str] = set()
    prefix = "difficulty_"
    suffix = "_total_examples"
    for key in row:
        if key.startswith(prefix) and key.endswith(suffix):
            groups.add(key[len(prefix) : -len(suffix)])
    return groups


def difficulty_sort_key(value: str) -> tuple[int, str]:
    return (DIFFICULTY_ORDER.get(value.lower(), 50), value.lower())


def level_label(value: str) -> str:
    return " ".join(part.capitalize() for part in value.replace("-", "_").split("_") if part)


def accuracy_table(selected: dict[str, dict[str, Any] | None], caption: str, label: str) -> str:
    groups: set[str] = set()
    for row in selected.values():
        groups.update(difficulty_groups(row))
    levels = ["overall", *sorted(groups, key=difficulty_sort_key)]

    precision_headers = [f"{precision.upper()} EM@1/EM@5" for precision in PRECISION_ORDER]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{latex_escape(caption)}}}",
        rf"\label{{{label}}}",
        r"\begin{tabular}{l" + ("c" * len(PRECISION_ORDER)) + "}",
        r"\hline",
        "Accuracy Level & " + " & ".join(latex_escape(header) for header in precision_headers) + r" \\",
        r"\hline",
    ]
    for level in levels:
        if level == "overall":
            values = [em_pair(selected.get(precision)) for precision in PRECISION_ORDER]
            label_text = "Overall"
        else:
            prefix = f"difficulty_{level}"
            values = [em_pair(selected.get(precision), prefix=prefix) for precision in PRECISION_ORDER]
            label_text = level_label(level)
        lines.append(latex_escape(label_text) + " & " + " & ".join(values) + r" \\")
    lines.extend([r"\hline", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Format decoder ONNX/TensorRT edge eval summaries as LaTeX tables.")
    parser.add_argument("--run-root", default=None, help="Run root containing eval/*/*/*/prompt_response_eval_summary.json.")
    parser.add_argument("--summary-json", default=None, help="Optional collected summary JSON file.")
    parser.add_argument("--dataset", default="benchmark", help="Dataset folder/name to include. Pass empty string for all rows.")
    parser.add_argument("--architecture", default="Base SFT", help="Architecture label for the main table.")
    parser.add_argument("--caption", default="Decoder base SFT ONNX edge evaluation.")
    parser.add_argument("--label", default="tab:decoder-onnx-edge")
    parser.add_argument("--accuracy-caption", default="Decoder base SFT ONNX accuracy levels.")
    parser.add_argument("--accuracy-label", default="tab:decoder-onnx-accuracy-levels")
    parser.add_argument("--output-tex", default=None, help="Path for the latency/accuracy LaTeX table.")
    parser.add_argument("--output-accuracy-tex", default=None, help="Path for the accuracy-level LaTeX table.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_rows(args.run_root, args.summary_json)
    if not rows:
        raise SystemExit("No eval summary rows found.")
    dataset = args.dataset if args.dataset else None
    filtered = filter_rows(rows, dataset)
    if not filtered:
        raise SystemExit(f"No rows matched dataset={args.dataset!r}.")

    selected = select_precision_rows(filtered, PRECISION_ORDER)
    main_tex = main_table(selected, architecture=args.architecture, caption=args.caption, label=args.label)
    accuracy_tex = accuracy_table(selected, caption=args.accuracy_caption, label=args.accuracy_label)

    if args.output_tex:
        output_path = Path(args.output_tex).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(main_tex, encoding="utf-8")
        print(f"Wrote ONNX table: {output_path}")
    else:
        print(main_tex)

    if args.output_accuracy_tex:
        output_path = Path(args.output_accuracy_tex).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(accuracy_tex, encoding="utf-8")
        print(f"Wrote accuracy-level table: {output_path}")
    elif not args.output_tex:
        print()
        print(accuracy_tex)


if __name__ == "__main__":
    main()
