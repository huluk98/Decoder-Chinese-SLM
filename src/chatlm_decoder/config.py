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
    config.setdefault("sft", {})
    config.setdefault("prune", {})

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
    config["data"].setdefault("token_ids_path", None)
    config["data"].setdefault("token_ids_dtype", "uint16")
    config["data"].setdefault("token_ids_manifest_path", None)
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
    config["preprocess"].setdefault("strict_min_rows", True)
    config["preprocess"].setdefault("continue_on_source_error", False)
    config["preprocess"].setdefault("download_first", False)
    config["preprocess"].setdefault("download_manifest_path", f"{config['preprocess']['output_path']}.download_manifest.json")
    config["preprocess"].setdefault("continue_on_download_error", None)
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
    config["train"].setdefault("tf32", False)
    config["train"].setdefault("float32_matmul_precision", "highest")
    config["train"].setdefault("sdp_flash", True)
    config["train"].setdefault("sdp_mem_efficient", True)
    config["train"].setdefault("sdp_math", True)
    config["train"].setdefault("cudnn_benchmark", False)
    config["train"].setdefault("num_workers", 0)
    config["train"].setdefault("pin_memory", False)
    config["train"].setdefault("persistent_workers", False)
    config["train"].setdefault("prefetch_factor", None)
    config["train"].setdefault("sync_cuda_for_timing", False)
    config["train"].setdefault("ddp_static_graph", True)
    config["train"].setdefault("ddp_gradient_as_bucket_view", True)
    config["train"].setdefault("ddp_bucket_cap_mb", 100)
    config["train"].setdefault("log_every", 10)
    config["train"].setdefault("save_every", 1000)
    config["train"].setdefault("compile", False)
    config["train"].setdefault("save_total_limit", 3)
    config["train"].setdefault("deepspeed", {})
    if config["train"]["deepspeed"] is None:
        config["train"]["deepspeed"] = {}
    if isinstance(config["train"]["deepspeed"], dict):
        config["train"]["deepspeed"].setdefault("enabled", False)
        config["train"]["deepspeed"].setdefault("config_path", None)
        config["train"]["deepspeed"].setdefault("fused_adam", True)

    config["sft"].setdefault("mode", "sft")
    config["sft"].setdefault("base_model", None)
    config["sft"].setdefault("data_path", None)
    config["sft"].setdefault("max_length", config["model"]["block_size"])
    config["sft"].setdefault("alignment_weight", 0.1)
    config["sft"].setdefault("margin", 0.5)
    config["sft"].setdefault("max_samples", None)

    config["prune"].setdefault("base_model", None)
    config["prune"].setdefault("output_dir", "runs/pruned-0p2b")
    config["prune"].setdefault("method", "magnitude")
    config["prune"].setdefault("sparsity", 0.5)
    config["prune"].setdefault("scope", "full_model")
    config["prune"].setdefault("sparsity_denominator", "whole_model")
    config["prune"].setdefault("granularity", "global")
    config["prune"].setdefault("include_lm_head", True)
    config["prune"].setdefault("calibration_data_path", None)
    config["prune"].setdefault("calibration_batches", 128)
    config["prune"].setdefault("recovery_steps", 0)
    config["prune"].setdefault("recovery_learning_rate", 5e-6)
    config["prune"].setdefault("overwrite", False)

    config["_config_path"] = str(config_path)
    config["_config_dir"] = str(config_path.parent)
    return config


def project_path(path: str | Path) -> Path:
    return Path(path).expanduser()
