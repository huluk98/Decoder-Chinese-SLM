#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from chatlm_decoder.config import load_config
from chatlm_decoder.preprocess import preprocess_datasets


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and normalize all configured datasets into one JSONL file.")
    parser.add_argument("--config", default="configs/model_0p2b.yaml", help="Path to a YAML config.")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing processed dataset.")
    args = parser.parse_args()

    config = load_config(args.config)
    manifest = preprocess_datasets(config, force=args.force)
    if manifest.get("already_exists"):
        print(f"Processed dataset already exists: {manifest['output_path']}")
        print("Use --force to rebuild it.")
        return

    print(f"Processed dataset: {manifest.get('output_path')}")
    print(f"Rows read: {manifest.get('read', 0):,}")
    print(f"Rows written: {manifest.get('written', 0):,}")
    print(f"Rows skipped: {manifest.get('skipped', 0):,}")


if __name__ == "__main__":
    main()

