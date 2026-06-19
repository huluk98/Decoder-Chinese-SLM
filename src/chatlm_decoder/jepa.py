from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class TextJEPAOutput:
    loss: torch.Tensor
    prediction: torch.Tensor
    target: torch.Tensor
    context: torch.Tensor


def infer_hidden_size(config: Any) -> int:
    for name in ("d_model", "hidden_size", "n_embd"):
        value = getattr(config, name, None)
        if value is not None:
            return int(value)
    raise ValueError("Cannot infer encoder hidden size from model config.")


def get_encoder_module(model: nn.Module) -> nn.Module:
    if hasattr(model, "get_encoder"):
        return model.get_encoder()
    encoder = getattr(model, "encoder", None)
    if encoder is None:
        raise ValueError("Expected an encoder-decoder model with get_encoder() or .encoder.")
    return encoder


def freeze_module(module: nn.Module) -> None:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)


def mean_pool_hidden(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(dtype=hidden.dtype, device=hidden.device)
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


def encoder_last_hidden(
    encoder: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    outputs = encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
    hidden = getattr(outputs, "last_hidden_state", None)
    if hidden is None:
        if isinstance(outputs, (tuple, list)) and outputs:
            hidden = outputs[0]
        else:
            raise ValueError("Encoder output did not include last_hidden_state.")
    return hidden


class TextJEPAPredictor(nn.Module):
    """Predict target embeddings from context embeddings plus target-query ids."""

    def __init__(
        self,
        hidden_size: int,
        predictor_hidden_size: int | None = None,
        num_target_queries: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        predictor_hidden = int(predictor_hidden_size or hidden_size * 2)
        self.query_embeddings = nn.Embedding(int(num_target_queries), hidden_size)
        self.net = nn.Sequential(
            nn.Linear(hidden_size * 2, predictor_hidden),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(predictor_hidden, hidden_size),
        )

    def forward(self, context_embedding: torch.Tensor, target_query_ids: torch.Tensor | None = None) -> torch.Tensor:
        if target_query_ids is None:
            target_query_ids = context_embedding.new_zeros((context_embedding.shape[0],), dtype=torch.long)
        if target_query_ids.ndim != 1:
            raise ValueError("target_query_ids must have shape [batch].")
        query_embedding = self.query_embeddings(target_query_ids.to(device=context_embedding.device))
        return self.net(torch.cat([context_embedding, query_embedding], dim=-1))


class EncoderDecoderTextJEPA(nn.Module):
    """Text JEPA wrapper for encoder-decoder backbones such as T5 or mT5.

    The context encoder is trainable. The target encoder is an EMA copy that is
    read under no-grad, making the loss a latent prediction objective rather
    than token reconstruction.
    """

    def __init__(
        self,
        context_encoder: nn.Module,
        target_encoder: nn.Module,
        predictor: TextJEPAPredictor,
        ema_decay: float = 0.996,
        normalize_targets: bool = True,
        loss: str = "smooth_l1",
    ) -> None:
        super().__init__()
        self.context_encoder = context_encoder
        self.target_encoder = target_encoder
        self.predictor = predictor
        self.ema_decay = float(ema_decay)
        self.normalize_targets = bool(normalize_targets)
        self.loss_name = str(loss)
        freeze_module(self.target_encoder)

    def train(self, mode: bool = True) -> "EncoderDecoderTextJEPA":
        super().train(mode)
        self.target_encoder.eval()
        return self

    def forward(
        self,
        context_input_ids: torch.Tensor,
        context_attention_mask: torch.Tensor,
        target_input_ids: torch.Tensor,
        target_attention_mask: torch.Tensor,
        target_query_ids: torch.Tensor | None = None,
    ) -> TextJEPAOutput:
        context_hidden = encoder_last_hidden(self.context_encoder, context_input_ids, context_attention_mask)
        context_embedding = mean_pool_hidden(context_hidden, context_attention_mask)

        with torch.no_grad():
            target_hidden = encoder_last_hidden(self.target_encoder, target_input_ids, target_attention_mask)
            target_embedding = mean_pool_hidden(target_hidden, target_attention_mask)

        prediction = self.predictor(context_embedding, target_query_ids=target_query_ids)
        target = target_embedding.detach()
        if self.normalize_targets:
            prediction = F.normalize(prediction.float(), p=2, dim=-1)
            target = F.normalize(target.float(), p=2, dim=-1)

        if self.loss_name in {"l2", "mse"}:
            loss = F.mse_loss(prediction, target)
        elif self.loss_name in {"l1", "mae"}:
            loss = F.l1_loss(prediction, target)
        elif self.loss_name in {"smooth_l1", "huber"}:
            loss = F.smooth_l1_loss(prediction, target)
        else:
            raise ValueError(f"Unknown JEPA loss: {self.loss_name}")
        return TextJEPAOutput(loss=loss, prediction=prediction, target=target, context=context_embedding)

    @torch.no_grad()
    def update_target_encoder(self) -> None:
        context_state = dict(self.context_encoder.named_parameters())
        for name, target_parameter in self.target_encoder.named_parameters():
            context_parameter = context_state.get(name)
            if context_parameter is None:
                continue
            target_parameter.data.mul_(self.ema_decay).add_(context_parameter.data, alpha=1.0 - self.ema_decay)
        context_buffers = dict(self.context_encoder.named_buffers())
        for name, target_buffer in self.target_encoder.named_buffers():
            context_buffer = context_buffers.get(name)
            if context_buffer is None:
                continue
            if target_buffer.dtype.is_floating_point:
                target_buffer.data.mul_(self.ema_decay).add_(context_buffer.data, alpha=1.0 - self.ema_decay)
            else:
                target_buffer.data.copy_(context_buffer.data)


def create_text_jepa_from_encoder_decoder(
    model: nn.Module,
    *,
    predictor_hidden_size: int | None = None,
    num_target_queries: int = 1,
    dropout: float = 0.0,
    ema_decay: float = 0.996,
    normalize_targets: bool = True,
    loss: str = "smooth_l1",
) -> EncoderDecoderTextJEPA:
    context_encoder = get_encoder_module(model)
    target_encoder = copy.deepcopy(context_encoder)
    hidden_size = infer_hidden_size(getattr(context_encoder, "config", getattr(model, "config", None)))
    predictor = TextJEPAPredictor(
        hidden_size=hidden_size,
        predictor_hidden_size=predictor_hidden_size,
        num_target_queries=num_target_queries,
        dropout=dropout,
    )
    return EncoderDecoderTextJEPA(
        context_encoder=context_encoder,
        target_encoder=target_encoder,
        predictor=predictor,
        ema_decay=ema_decay,
        normalize_targets=normalize_targets,
        loss=loss,
    )
