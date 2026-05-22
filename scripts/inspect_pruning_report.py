#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def resolve_report_path(path: str | Path) -> Path:
    report_path = Path(path).expanduser()
    if report_path.is_dir():
        report_path = report_path / "pruning_report.json"
    if not report_path.exists():
        raise FileNotFoundError(f"Pruning report not found: {report_path}")
    return report_path


def fmt_int(value: Any) -> str:
    if value is None or value == "":
        return "missing"
    return f"{int(value):,}"


def fmt_float(value: Any) -> str:
    if value is None or value == "":
        return "missing"
    return f"{float(value):.8f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Print active/pruned parameter counts from pruning_report.json.")
    parser.add_argument("path", help="Path to pruning_report.json or a checkpoint directory containing it.")
    args = parser.parse_args()

    report_path = resolve_report_path(args.path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    mask_parameter_count = int(report.get("mask_parameter_count") or 0)
    active_mask_parameters = int(report.get("active_mask_parameters") or 0)
    pruned_mask_parameters = int(report.get("pruned_mask_parameters") or 0)
    total_parameters = int(report.get("total_parameters") or 0)
    nonzero_parameters = int(report.get("nonzero_parameters") or 0)
    masked_violations = int(report.get("masked_weight_violation_count") or 0)

    print(f"report: {report_path}")
    print(f"method: {report.get('method', 'missing')}")
    print(f"phase: {report.get('phase', 'one_shot_prune')}")
    print(f"target_sparsity: {fmt_float(report.get('target_sparsity'))}")
    print(f"mask_sparsity: {fmt_float(report.get('mask_sparsity', report.get('sparsity')))}")
    print(f"mask_parameter_count: {fmt_int(mask_parameter_count)}")
    print(f"active_mask_parameters: {fmt_int(active_mask_parameters)}")
    print(f"pruned_mask_parameters: {fmt_int(pruned_mask_parameters)}")
    print(f"active_mask_fraction: {fmt_float(report.get('active_mask_fraction'))}")
    print(f"mask_implied_active_parameters: {fmt_int(report.get('mask_implied_active_parameters'))}")
    print(f"mask_implied_pruned_parameters: {fmt_int(report.get('mask_implied_pruned_parameters'))}")
    print(f"mask_implied_active_fraction: {fmt_float(report.get('mask_implied_active_fraction'))}")
    print(f"mask_implied_pruned_fraction: {fmt_float(report.get('mask_implied_pruned_fraction'))}")
    print(f"total_parameters: {fmt_int(total_parameters)}")
    print(f"nonzero_parameters: {fmt_int(nonzero_parameters)}")
    print(f"nonzero_fraction: {fmt_float(report.get('nonzero_fraction'))}")
    print(f"masked_weight_violation_count: {fmt_int(masked_violations)}")

    if mask_parameter_count and active_mask_parameters == 0:
        print("status: ERROR active_mask_parameters is zero")
    elif mask_parameter_count and abs((active_mask_parameters / mask_parameter_count) - 0.5) > 1e-6:
        print("status: WARNING active_mask_fraction is not 0.5")
    elif masked_violations:
        print("status: ERROR pruned weights contain nonzero values")
    elif total_parameters and nonzero_parameters == 0:
        print("status: ERROR model has zero nonzero parameters")
    else:
        print("status: OK")


if __name__ == "__main__":
    main()
