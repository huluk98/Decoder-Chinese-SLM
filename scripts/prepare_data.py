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
    parser.add_argument("--force-download", action="store_true", help="Re-download Hugging Face snapshots before preprocessing.")
    args = parser.parse_args()

    config = load_config(args.config)
    manifest = preprocess_datasets(config, force=args.force, force_download=args.force_download)
    if manifest.get("already_exists"):
        print(f"Processed dataset already exists: {manifest['output_path']}")
        print("Use --force to rebuild it.")
        return

    print(f"Processed dataset: {manifest.get('output_path')}")
    print(f"Rows read: {manifest.get('read', 0):,}")
    print(f"Rows written: {manifest.get('written', 0):,}")
    print(f"Rows skipped: {manifest.get('skipped', 0):,}")
    if manifest.get("download_manifest"):
        download_manifest = manifest["download_manifest"]
        print(f"Downloaded sources: {download_manifest.get('downloaded', 0)}")
        print(f"Failed downloads: {download_manifest.get('failed', 0)}")
        print(f"Download manifest: {download_manifest.get('manifest_path')}")
    if manifest.get("below_min_rows"):
        print("Warning: processed row count is below preprocess.min_rows; inspect the manifest before a full run.")
    if manifest.get("failed_sources"):
        print(f"Skipped failed sources: {len(manifest['failed_sources'])}")
        print(f"Manifest: {manifest.get('manifest_path')}")


if __name__ == "__main__":
    main()
