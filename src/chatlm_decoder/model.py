from __future__ import annotations

from typing import Any

from transformers import GPT2Config, GPT2LMHeadModel, LlamaConfig, LlamaForCausalLM


def _token_id(tokenizer: Any | None, name: str) -> int | None:
    return getattr(tokenizer, name, None) if tokenizer is not None else None


def _create_llama_model(model_config: dict[str, Any], tokenizer: Any | None = None) -> LlamaForCausalLM:
    vocab_size = int(model_config.get("vocab_size") or (len(tokenizer) if tokenizer is not None else 29298))
    block_size = int(model_config.get("block_size", 512))
    rope_scaling = model_config.get("rope_scaling")

    config = LlamaConfig(
        vocab_size=vocab_size,
        max_position_embeddings=block_size,
        hidden_size=int(model_config.get("hidden_size", model_config.get("n_embd", 768))),
        intermediate_size=int(model_config.get("intermediate_size", model_config.get("n_inner", 2048))),
        num_hidden_layers=int(model_config.get("num_hidden_layers", model_config.get("n_layer", 24))),
        num_attention_heads=int(model_config.get("num_attention_heads", model_config.get("n_head", 12))),
        num_key_value_heads=int(model_config.get("num_key_value_heads", model_config.get("n_kv_head", 4))),
        hidden_act=str(model_config.get("hidden_act", "silu")),
        rms_norm_eps=float(model_config.get("rms_norm_eps", 1e-6)),
        rope_theta=float(model_config.get("rope_theta", 10000.0)),
        rope_scaling=rope_scaling,
        attention_bias=bool(model_config.get("attention_bias", False)),
        mlp_bias=bool(model_config.get("mlp_bias", False)),
        attention_dropout=float(model_config.get("attention_dropout", model_config.get("dropout", 0.0))),
        tie_word_embeddings=bool(model_config.get("tie_word_embeddings", False)),
        initializer_range=float(model_config.get("initializer_range", 0.02)),
        use_cache=False,
        bos_token_id=_token_id(tokenizer, "bos_token_id"),
        eos_token_id=_token_id(tokenizer, "eos_token_id"),
        pad_token_id=_token_id(tokenizer, "pad_token_id"),
    )

    attn_implementation = model_config.get("attn_implementation")
    if attn_implementation:
        config._attn_implementation = str(attn_implementation)

    model = LlamaForCausalLM(config)
    return model


def _create_gpt2_model(model_config: dict[str, Any], tokenizer: Any | None = None) -> GPT2LMHeadModel:
    vocab_size = int(model_config.get("vocab_size") or (len(tokenizer) if tokenizer is not None else 29298))
    block_size = int(model_config.get("block_size", 512))
    dropout = float(model_config.get("dropout", 0.1))

    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=block_size,
        n_ctx=block_size,
        n_embd=int(model_config.get("n_embd", model_config.get("hidden_size", 768))),
        n_layer=int(model_config.get("n_layer", model_config.get("num_hidden_layers", 24))),
        n_head=int(model_config.get("n_head", model_config.get("num_attention_heads", 12))),
        n_inner=int(model_config.get("n_inner", model_config.get("intermediate_size", 3072))),
        activation_function=str(model_config.get("activation_function", "gelu_new")),
        resid_pdrop=dropout,
        embd_pdrop=dropout,
        attn_pdrop=dropout,
        bos_token_id=_token_id(tokenizer, "bos_token_id"),
        eos_token_id=_token_id(tokenizer, "eos_token_id"),
        pad_token_id=_token_id(tokenizer, "pad_token_id"),
    )
    return GPT2LMHeadModel(config)


def create_model(model_config: dict[str, Any], tokenizer: Any | None = None):
    architecture = str(model_config.get("architecture", "llama")).lower()
    if architecture in {"llama", "llama_like", "llama-style"}:
        model = _create_llama_model(model_config, tokenizer)
    elif architecture in {"gpt2", "gpt"}:
        model = _create_gpt2_model(model_config, tokenizer)
    else:
        raise ValueError(f"Unknown model architecture: {architecture}")

    if tokenizer is not None and len(tokenizer) != model.config.vocab_size:
        model.resize_token_embeddings(len(tokenizer))

    if bool(model_config.get("gradient_checkpointing", False)):
        model.gradient_checkpointing_enable()

    return model


def count_parameters(model) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
