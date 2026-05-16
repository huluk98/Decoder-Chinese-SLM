#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from itertools import islice
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from chatlm_decoder.config import load_config
from chatlm_decoder.data import iter_texts
from chatlm_decoder.preprocess import ensure_preprocessed_data
from chatlm_decoder.tokenizer import train_tokenizer_from_iterator


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a BPE tokenizer for the decoder-only Chinese LM.")
    parser.add_argument("--config", default="configs/model_0p2b.yaml", help="Path to a YAML config.")
    parser.add_argument("--output-dir", default=None, help="Override tokenizer output directory.")
    parser.add_argument("--max-samples", type=int, default=None, help="Override tokenizer max sample count.")
    parser.add_argument("--skip-prepare", action="store_true", help="Train directly from configured raw sources.")
    parser.add_argument("--force-prepare", action="store_true", help="Rebuild the normalized JSONL before training.")
    parser.add_argument("--force-download", action="store_true", help="Re-download Hugging Face snapshots before preprocessing.")
    args = parser.parse_args()

    config = load_config(args.config)
    tokenizer_config = config["tokenizer"]
    data_config = (
        config["data"]
        if args.skip_prepare
        else ensure_preprocessed_data(config, force=args.force_prepare, force_download=args.force_download)
    )

    max_samples = args.max_samples if args.max_samples is not None else tokenizer_config.get("max_samples")
    texts = iter_texts(data_config)
    if max_samples is not None:
        texts = islice(texts, int(max_samples))

    output_dir = args.output_dir or tokenizer_config["path"]
    tokenizer = train_tokenizer_from_iterator(
        texts=texts,
        output_dir=output_dir,
        vocab_size=int(tokenizer_config.get("vocab_size", 29298)),
        min_frequency=int(tokenizer_config.get("min_frequency", 2)),
        model_max_length=int(config["model"].get("block_size", 512)),
    )
    print(f"Saved tokenizer with {len(tokenizer)} tokens to {output_dir}")


if __name__ == "__main__":
    main()
