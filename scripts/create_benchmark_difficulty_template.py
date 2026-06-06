#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from chatlm_decoder.sparsity_experiments import normalize_sample, read_records, write_csv_rows  # noqa: E402

GUIDANCE = """# Benchmark difficulty labeling guidance

Fill the difficulty column with one of: easy, medium, hard.

easy:
Direct, single-intent, single-device command with explicit action and target.
Example: "Turn on the bedroom light."

medium:
Paraphrased, indirect, multi-device, or slightly contextual command, but still unambiguous.
Example: "It is too dark in the bedroom."

hard:
Indirect, compositional, conditional, rare-device, multi-step, negated, or potentially ambiguous command.
Example: "If the room gets too warm, lower the AC and turn off the heater."
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a blank easy/medium/hard difficulty-label template.")
    parser.add_argument("--benchmark_path", required=True)
    parser.add_argument("--output_dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    benchmark_path = Path(args.benchmark_path).expanduser()
    output_dir = Path(args.output_dir).expanduser() if args.output_dir else benchmark_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    records = read_records(benchmark_path)
    rows = []
    for index, record in enumerate(records):
        sample = normalize_sample(record, index)
        rows.append(
            {
                "id": sample["sample_id"],
                "input": sample["input"],
                "target": sample["target"],
                "difficulty": "",
            }
        )
    csv_path = output_dir / "benchmark_difficulty_template.csv"
    readme_path = output_dir / "benchmark_difficulty_template_README.md"
    write_csv_rows(csv_path, rows, ["id", "input", "target", "difficulty"])
    readme_path.write_text(GUIDANCE, encoding="utf-8")
    print(f"Wrote {csv_path}")
    print(f"Wrote {readme_path}")


if __name__ == "__main__":
    main()
