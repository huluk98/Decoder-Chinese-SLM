#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from chatlm_decoder.metrics import format_training_metrics_summary, summarize_training_metrics


def resolve_metrics_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_dir():
        metrics_path = candidate / "metrics" / "training_metrics.csv"
        if metrics_path.exists():
            return metrics_path
        if candidate.name == "metrics":
            return candidate / "training_metrics.csv"
    return candidate


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize wall time, GPU-hours, tokens, and latest loss from a training_metrics.csv file."
    )
    parser.add_argument(
        "path",
        nargs="?",
        default="runs/h20-8gpu-llama-0p2b-deepspeed/metrics/training_metrics.csv",
        help="Path to training_metrics.csv, a run directory, or a metrics directory.",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=None,
        help="Fallback GPU count if older metrics rows do not include world_size.",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    args = parser.parse_args()

    metrics_path = resolve_metrics_path(args.path)
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")

    summary: dict[str, Any] = summarize_training_metrics(metrics_path, default_world_size=args.gpus)
    if args.json:
        payload = {"metrics_path": str(metrics_path), **summary}
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    print(format_training_metrics_summary(summary, metrics_path))


if __name__ == "__main__":
    main()
