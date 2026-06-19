from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from chatlm_decoder.jepa import (  # noqa: E402
    EncoderDecoderTextJEPA,
    TextJEPAPredictor,
    create_text_jepa_from_encoder_decoder,
    mean_pool_hidden,
)


class EncoderOutput:
    def __init__(self, hidden: torch.Tensor) -> None:
        self.last_hidden_state = hidden


class TinyEncoder(torch.nn.Module):
    def __init__(self, hidden_size: int = 4) -> None:
        super().__init__()
        self.config = SimpleNamespace(d_model=hidden_size)
        self.embed = torch.nn.Embedding(16, hidden_size)
        self.proj = torch.nn.Linear(hidden_size, hidden_size)

    def forward(self, input_ids, attention_mask=None, return_dict=True):
        hidden = self.proj(self.embed(input_ids))
        return EncoderOutput(hidden)


class TinySeq2Seq(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = TinyEncoder()
        self.config = self.encoder.config

    def get_encoder(self):
        return self.encoder


def test_mean_pool_hidden_ignores_padding() -> None:
    hidden = torch.tensor([[[1.0, 1.0], [3.0, 3.0], [99.0, 99.0]]])
    mask = torch.tensor([[1, 1, 0]])

    pooled = mean_pool_hidden(hidden, mask)

    assert pooled.tolist() == [[2.0, 2.0]]


def test_encoder_decoder_jepa_forward_produces_latent_loss() -> None:
    base = TinySeq2Seq()
    model = create_text_jepa_from_encoder_decoder(base, num_target_queries=2, ema_decay=0.9)
    batch = {
        "context_input_ids": torch.tensor([[1, 2, 0], [3, 4, 5]]),
        "context_attention_mask": torch.tensor([[1, 1, 0], [1, 1, 1]]),
        "target_input_ids": torch.tensor([[6, 7], [8, 9]]),
        "target_attention_mask": torch.tensor([[1, 1], [1, 1]]),
        "target_query_ids": torch.tensor([0, 1]),
    }

    output = model(**batch)

    assert output.loss.ndim == 0
    assert output.prediction.shape == output.target.shape == (2, 4)


def test_target_encoder_is_frozen_and_ema_updated() -> None:
    context_encoder = TinyEncoder()
    target_encoder = TinyEncoder()
    predictor = TextJEPAPredictor(hidden_size=4)
    model = EncoderDecoderTextJEPA(context_encoder, target_encoder, predictor, ema_decay=0.5)

    assert all(not parameter.requires_grad for parameter in model.target_encoder.parameters())
    old_target = next(model.target_encoder.parameters()).detach().clone()
    with torch.no_grad():
        next(model.context_encoder.parameters()).add_(1.0)

    model.update_target_encoder()

    new_target = next(model.target_encoder.parameters()).detach()
    assert not torch.equal(old_target, new_target)
