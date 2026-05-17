#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path


def read_series(metrics_path: Path) -> tuple[list[int], list[float], list[float]]:
    steps: list[int] = []
    losses: list[float] = []
    learning_rates: list[float] = []

    with metrics_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                step = int(row["step"])
                loss = float(row["loss"])
            except (KeyError, TypeError, ValueError):
                continue

            steps.append(step)
            losses.append(loss)
            try:
                learning_rates.append(float(row.get("lr", "nan")))
            except ValueError:
                learning_rates.append(float("nan"))

    if not steps:
        raise ValueError(f"No valid loss rows found in {metrics_path}")
    return steps, losses, learning_rates


def moving_average(values: list[float], window: int) -> list[float]:
    if window <= 1:
        return values[:]

    smoothed: list[float] = []
    running_sum = 0.0
    queue: list[float] = []
    for value in values:
        queue.append(value)
        running_sum += value
        if len(queue) > window:
            running_sum -= queue.pop(0)
        smoothed.append(running_sum / len(queue))
    return smoothed


def plot_loss(
    metrics_path: Path,
    output_dir: Path,
    stage: str,
    title: str,
    smooth_window: int,
    dpi: int,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps, losses, learning_rates = read_series(metrics_path)
    smoothed_losses = moving_average(losses, smooth_window)
    output_dir.mkdir(parents=True, exist_ok=True)

    generated: list[Path] = []
    stage_loss_path = output_dir / f"{stage}_loss.png"
    generic_loss_path = output_dir / "loss.png"

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(steps, losses, color="#9aa7b1", linewidth=0.9, alpha=0.45, label="raw loss")
    ax.plot(steps, smoothed_losses, color="#2563eb", linewidth=2.0, label=f"moving average ({smooth_window})")
    ax.set_title(title)
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Causal LM cross-entropy loss")
    ax.grid(True, color="#d8dee9", linewidth=0.8, alpha=0.7)
    ax.legend()
    fig.tight_layout()
    fig.savefig(stage_loss_path, dpi=dpi)
    plt.close(fig)
    generated.append(stage_loss_path)

    if generic_loss_path != stage_loss_path:
        shutil.copyfile(stage_loss_path, generic_loss_path)
        generated.append(generic_loss_path)

    if any(rate == rate for rate in learning_rates):
        stage_loss_lr_path = output_dir / f"{stage}_loss_lr.png"
        fig, ax = plt.subplots(figsize=(10, 5.5))
        ax.plot(steps, smoothed_losses, color="#2563eb", linewidth=2.0, label="loss")
        ax.set_title(f"{title} with learning rate")
        ax.set_xlabel("Optimizer step")
        ax.set_ylabel("Causal LM cross-entropy loss", color="#2563eb")
        ax.tick_params(axis="y", labelcolor="#2563eb")
        ax.grid(True, color="#d8dee9", linewidth=0.8, alpha=0.7)

        ax_lr = ax.twinx()
        ax_lr.plot(steps, learning_rates, color="#dc2626", linewidth=1.4, alpha=0.8, label="learning rate")
        ax_lr.set_ylabel("Learning rate", color="#dc2626")
        ax_lr.tick_params(axis="y", labelcolor="#dc2626")

        lines, labels = ax.get_legend_handles_labels()
        lr_lines, lr_labels = ax_lr.get_legend_handles_labels()
        ax.legend(lines + lr_lines, labels + lr_labels, loc="upper right")
        fig.tight_layout()
        fig.savefig(stage_loss_lr_path, dpi=dpi)
        plt.close(fig)
        generated.append(stage_loss_lr_path)

    return generated


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot training loss curves from training_metrics.csv.")
    parser.add_argument("--metrics", default=None, help="Path to training_metrics.csv.")
    parser.add_argument("--run-dir", default="runs/h20-7gpu-llama-0p2b-deepspeed", help="Run directory used when --metrics is omitted.")
    parser.add_argument("--output-dir", default=None, help="Directory for generated PNG files. Defaults to the metrics directory.")
    parser.add_argument("--stage", default="pretrain", help="Stage name used for <stage>_loss.png.")
    parser.add_argument("--title", default="Decoder-Chinese-SLM 0.2B Pretraining Loss", help="Plot title.")
    parser.add_argument("--smooth-window", type=int, default=50, help="Moving-average smoothing window.")
    parser.add_argument("--dpi", type=int, default=160, help="Output image DPI.")
    args = parser.parse_args()

    metrics_path = Path(args.metrics) if args.metrics else Path(args.run_dir) / "metrics" / "training_metrics.csv"
    metrics_path = metrics_path.expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else metrics_path.parent

    generated = plot_loss(
        metrics_path=metrics_path,
        output_dir=output_dir,
        stage=str(args.stage),
        title=str(args.title),
        smooth_window=int(args.smooth_window),
        dpi=int(args.dpi),
    )
    for path in generated:
        print(path)


if __name__ == "__main__":
    main()
