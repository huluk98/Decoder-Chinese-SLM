#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

DIFFICULTIES = ("easy", "medium", "hard")


def as_float(value: Any) -> float | None:
    if value in (None, "", "None"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def read_summary(path: str | Path) -> list[dict[str, Any]]:
    summary_path = Path(path).expanduser()
    with summary_path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def row_label(row: dict[str, Any]) -> str:
    return f"{row.get('model_family', '')} {row.get('pruning_mode', '')} {as_float(row.get('target_sparsity')) or 0:g}"


def plot_metric_vs_sparsity(rows: list[dict[str, Any]], metric: str, title: str, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.8))
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row.get("model_family")), str(row.get("pruning_mode"))), []).append(row)
    for (family, mode), group_rows in sorted(groups.items()):
        points = sorted(
            (
                (as_float(row.get("target_sparsity")), as_float(row.get(metric)))
                for row in group_rows
            ),
            key=lambda item: item[0] if item[0] is not None else -1.0,
        )
        xs = [point[0] for point in points if point[0] is not None and point[1] is not None]
        ys = [point[1] for point in points if point[0] is not None and point[1] is not None]
        if xs and ys:
            ax.plot(xs, ys, marker="o", linewidth=2, label=f"{family} {mode}")
    ax.set_title(title)
    ax.set_xlabel("Target sparsity")
    ax.set_ylabel(metric.replace("_", " ").upper())
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_difficulty_bars(rows: list[dict[str, Any]], metric_prefix: str, title: str, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    labels = [row_label(row) for row in rows]
    if not labels:
        return
    x = list(range(len(labels)))
    width = 0.24
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.8), 5.2))
    offsets = {"easy": -width, "medium": 0.0, "hard": width}
    colors = {"easy": "#2E7D32", "medium": "#1565C0", "hard": "#C62828"}
    for difficulty in DIFFICULTIES:
        values = [as_float(row.get(f"{metric_prefix}_{difficulty}")) or 0.0 for row in rows]
        positions = [position + offsets[difficulty] for position in x]
        ax.bar(positions, values, width=width, label=difficulty, color=colors[difficulty])
    ax.set_title(title)
    ax.set_ylabel(metric_prefix.upper())
    ax.set_ylim(0.0, 1.02)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_results(summary_csv: str | Path, figures_dir: str | Path | None = None) -> None:
    rows = read_summary(summary_csv)
    if figures_dir is None:
        figures_path = Path(summary_csv).expanduser().parent / "figures"
    else:
        figures_path = Path(figures_dir).expanduser()
    figures_path.mkdir(parents=True, exist_ok=True)
    plot_metric_vs_sparsity(rows, "em1_overall", "EM@1 vs Sparsity", figures_path / "em1_vs_sparsity.png")
    plot_metric_vs_sparsity(rows, "em5_overall", "EM@5 vs Sparsity", figures_path / "em5_vs_sparsity.png")
    plot_difficulty_bars(rows, "em1", "Difficulty Breakdown EM@1", figures_path / "difficulty_em1_bar.png")
    plot_difficulty_bars(rows, "em5", "Difficulty Breakdown EM@5", figures_path / "difficulty_em5_bar.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot SCENIC linear sparsity experiment results.")
    parser.add_argument("--summary_metrics", required=True)
    parser.add_argument("--figures_dir", default=None)
    args = parser.parse_args()
    plot_results(args.summary_metrics, args.figures_dir)
    figures_dir = Path(args.figures_dir) if args.figures_dir else Path(args.summary_metrics).expanduser().parent / "figures"
    print(f"Wrote figures to {figures_dir}")


if __name__ == "__main__":
    main()
