#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from chatlm_decoder.config import load_config
from chatlm_decoder.data import download_hf_sources


def main() -> None:
    parser = argparse.ArgumentParser(description="Download configured Hugging Face dataset snapshots locally.")
    parser.add_argument("--config", default="configs/model_0p2b.yaml", help="Path to a YAML config.")
    parser.add_argument("--force-download", action="store_true", help="Re-download files even when cached.")
    args = parser.parse_args()

    config = load_config(args.config)
    preprocess_config = config["preprocess"]
    continue_on_error = preprocess_config["continue_on_download_error"]
    if continue_on_error is None:
        continue_on_error = preprocess_config["continue_on_source_error"]

    _, manifest = download_hf_sources(
        config["data"],
        force_download=args.force_download,
        continue_on_error=bool(continue_on_error),
    )

    manifest_path = Path(preprocess_config["download_manifest_path"]).expanduser()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest["manifest_path"] = str(manifest_path)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(f"Download manifest: {manifest_path}")
    print(f"Downloaded sources: {manifest.get('downloaded', 0)}")
    print(f"Local sources: {manifest.get('local', 0)}")
    print(f"Failed sources: {manifest.get('failed', 0)}")


if __name__ == "__main__":
    main()
