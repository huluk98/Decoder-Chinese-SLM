from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    config.setdefault("run", {})
    config.setdefault("tokenizer", {})
    config.setdefault("model", {})
    config.setdefault("data", {})
    config.setdefault("train", {})
    config.setdefault("preprocess", {})

    config["run"].setdefault("seed", 42)
    config["run"].setdefault("output_dir", "runs/default")

    config["tokenizer"].setdefault("path", "artifacts/tokenizer")
    config["tokenizer"].setdefault("vocab_size", 29298)
    config["tokenizer"].setdefault("min_frequency", 2)
    config["tokenizer"].setdefault("train_if_missing", False)

    config["model"].setdefault("vocab_size", config["tokenizer"]["vocab_size"])
    config["model"].setdefault("architecture", "llama")
    config["model"].setdefault("block_size", 512)
    config["model"].setdefault("num_hidden_layers", config["model"].get("n_layer", 24))
    config["model"].setdefault("num_attention_heads", config["model"].get("n_head", 12))
    config["model"].setdefault("num_key_value_heads", config["model"].get("n_kv_head", 4))
    config["model"].setdefault("hidden_size", config["model"].get("n_embd", 768))
    config["model"].setdefault("intermediate_size", config["model"].get("n_inner", 2048))
    config["model"].setdefault("attention_dropout", config["model"].get("dropout", 0.0))
    config["model"].setdefault("tie_word_embeddings", False)
    config["model"].setdefault("gradient_checkpointing", False)

    config["data"].setdefault("sources", [])
    config["data"].setdefault("streaming", True)
    config["data"].setdefault("drop_last", True)
    config["data"].setdefault("add_eos", True)
    config["data"].setdefault("seed", config["run"]["seed"])
    config["data"].setdefault("hf_cache_dir", "data/raw/huggingface")
    config["data"].setdefault("hf_retries", 3)
    config["data"].setdefault("hf_retry_sleep_seconds", 10)
    config["data"].setdefault("hf_retry_backoff", 2.0)
    config["data"].setdefault("hf_download_timeout", 120)
    config["data"].setdefault("hf_etag_timeout", 60)
    config["data"].setdefault("hf_endpoint", None)

    config["preprocess"].setdefault("enabled", False)
    config["preprocess"].setdefault("output_path", "data/processed/normalized.jsonl")
    config["preprocess"].setdefault("manifest_path", f"{config['preprocess']['output_path']}.manifest.json")
    config["preprocess"].setdefault("overwrite", False)
    config["preprocess"].setdefault("dedupe", True)
    config["preprocess"].setdefault("min_chars", 8)
    config["preprocess"].setdefault("max_chars", None)
    config["preprocess"].setdefault("min_rows", None)
    config["preprocess"].setdefault("shuffle_before_write", False)

    config["train"].setdefault("batch_size", 1)
    config["train"].setdefault("grad_accum_steps", 1)
    config["train"].setdefault("max_steps", 1000)
    config["train"].setdefault("learning_rate", 3e-4)
    config["train"].setdefault("min_learning_rate", 3e-5)
    config["train"].setdefault("warmup_steps", 100)
    config["train"].setdefault("weight_decay", 0.1)
    config["train"].setdefault("beta1", 0.9)
    config["train"].setdefault("beta2", 0.95)
    config["train"].setdefault("max_grad_norm", 1.0)
    config["train"].setdefault("precision", "bf16")
    config["train"].setdefault("num_workers", 0)
    config["train"].setdefault("log_every", 10)
    config["train"].setdefault("save_every", 1000)
    config["train"].setdefault("compile", False)

    config["_config_path"] = str(config_path)
    config["_config_dir"] = str(config_path.parent)
    return config


def project_path(path: str | Path) -> Path:
    return Path(path).expanduser()
