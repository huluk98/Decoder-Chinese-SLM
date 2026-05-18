from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

TRAINING_METRIC_FIELDNAMES = [
    "time_seconds",
    "step",
    "loss",
    "lr",
    "world_size",
    "effective_global_batch",
    "block_size",
    "tokens_per_step",
    "tokens_per_second",
    "seconds_per_step",
    "wall_hours",
    "gpu_hours",
    "estimated_total_tokens",
    "estimated_billion_tokens",
    "gpu_hours_per_billion_tokens",
]


def _as_float(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(row: dict[str, Any], key: str) -> int | None:
    value = _as_float(row, key)
    if value is None:
        return None
    return int(value)


def merged_metric_fieldnames(existing_fieldnames: list[str] | None = None) -> list[str]:
    fieldnames = list(TRAINING_METRIC_FIELDNAMES)
    for name in existing_fieldnames or []:
        if name not in fieldnames:
            fieldnames.append(name)
    return fieldnames


def read_training_metric_rows(metrics_path: str | Path) -> tuple[list[str], list[dict[str, str]]]:
    path = Path(metrics_path).expanduser()
    if not path.exists() or path.stat().st_size == 0:
        return [], []

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]
        return list(reader.fieldnames or []), rows


def cumulative_training_metric_rows(
    rows: list[dict[str, Any]],
    default_world_size: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    enriched_rows: list[dict[str, Any]] = []
    previous_time_seconds: float | None = None
    total_wall_seconds = 0.0
    total_gpu_seconds = 0.0
    total_tokens = 0.0
    segments = 0
    latest_row: dict[str, Any] | None = None

    for row in rows:
        enriched = dict(row)
        time_seconds = _as_float(enriched, "time_seconds")
        if time_seconds is None:
            enriched_rows.append(enriched)
            continue

        if previous_time_seconds is None or time_seconds < previous_time_seconds:
            delta_seconds = max(0.0, time_seconds)
            segments += 1
        else:
            delta_seconds = max(0.0, time_seconds - previous_time_seconds)

        world_size = _as_int(enriched, "world_size") or default_world_size or 1
        tokens_per_second = _as_float(enriched, "tokens_per_second")
        if tokens_per_second is not None:
            total_tokens += max(0.0, tokens_per_second * delta_seconds)

        total_wall_seconds += delta_seconds
        total_gpu_seconds += delta_seconds * world_size
        billion_tokens = total_tokens / 1_000_000_000.0
        gpu_hours = total_gpu_seconds / 3600.0

        enriched["wall_hours"] = f"{total_wall_seconds / 3600.0:.6f}"
        enriched["gpu_hours"] = f"{gpu_hours:.6f}"
        enriched["estimated_total_tokens"] = str(int(round(total_tokens)))
        enriched["estimated_billion_tokens"] = f"{billion_tokens:.6f}"
        enriched["gpu_hours_per_billion_tokens"] = (
            f"{gpu_hours / billion_tokens:.6f}" if billion_tokens > 0 else ""
        )

        enriched_rows.append(enriched)
        latest_row = enriched
        previous_time_seconds = time_seconds

    summary = {
        "rows": len(rows),
        "segments": segments,
        "wall_seconds": total_wall_seconds,
        "wall_hours": total_wall_seconds / 3600.0,
        "gpu_seconds": total_gpu_seconds,
        "gpu_hours": total_gpu_seconds / 3600.0,
        "estimated_total_tokens": int(round(total_tokens)),
        "estimated_billion_tokens": total_tokens / 1_000_000_000.0,
        "gpu_hours_per_billion_tokens": (
            (total_gpu_seconds / 3600.0) / (total_tokens / 1_000_000_000.0)
            if total_tokens > 0
            else None
        ),
        "latest_step": _as_int(latest_row or {}, "step"),
        "latest_loss": _as_float(latest_row or {}, "loss"),
        "latest_lr": _as_float(latest_row or {}, "lr"),
        "latest_world_size": _as_int(latest_row or {}, "world_size") or default_world_size,
    }
    return enriched_rows, summary


def summarize_training_metrics(
    metrics_path: str | Path,
    default_world_size: int | None = None,
) -> dict[str, Any]:
    _, rows = read_training_metric_rows(metrics_path)
    _, summary = cumulative_training_metric_rows(rows, default_world_size=default_world_size)
    return summary


def upgrade_training_metrics_file(metrics_path: str | Path, default_world_size: int | None = None) -> None:
    path = Path(metrics_path).expanduser()
    existing_fieldnames, rows = read_training_metric_rows(path)
    if not rows:
        return

    fieldnames = merged_metric_fieldnames(existing_fieldnames)
    if all(name in existing_fieldnames for name in TRAINING_METRIC_FIELDNAMES):
        return

    enriched_rows, _ = cumulative_training_metric_rows(rows, default_world_size=default_world_size)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in enriched_rows:
            writer.writerow(row)
    tmp_path.replace(path)


def format_training_metrics_summary(summary: dict[str, Any], metrics_path: str | Path) -> str:
    gpu_hours_per_billion_tokens = summary.get("gpu_hours_per_billion_tokens")
    cost_line = (
        f"{gpu_hours_per_billion_tokens:.2f}"
        if isinstance(gpu_hours_per_billion_tokens, float)
        else "n/a"
    )
    latest_loss = summary.get("latest_loss")
    latest_loss_text = f"{latest_loss:.4f}" if isinstance(latest_loss, float) else "n/a"
    latest_lr = summary.get("latest_lr")
    latest_lr_text = f"{latest_lr:.3g}" if isinstance(latest_lr, float) else "n/a"

    return "\n".join(
        [
            f"Metrics: {Path(metrics_path).expanduser()}",
            f"Rows: {summary.get('rows', 0):,}",
            f"Detected run segments: {summary.get('segments', 0):,}",
            f"Latest step: {summary.get('latest_step') or 'n/a'}",
            f"Latest loss: {latest_loss_text}",
            f"Latest learning rate: {latest_lr_text}",
            f"Latest world size: {summary.get('latest_world_size') or 'n/a'}",
            f"Estimated wall hours: {summary.get('wall_hours', 0.0):.2f}",
            f"Estimated GPU-hours: {summary.get('gpu_hours', 0.0):.2f}",
            f"Estimated tokens: {summary.get('estimated_total_tokens', 0):,}",
            f"Estimated billion tokens: {summary.get('estimated_billion_tokens', 0.0):.3f}",
            f"GPU-hours per billion tokens: {cost_line}",
        ]
    )
