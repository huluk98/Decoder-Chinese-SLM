from __future__ import annotations

from chatlm_decoder.tokenizer import move_batch_to_device, prepare_decoder_tokenizer, strip_unused_decoder_model_kwargs


class DummyTokenizer:
    model_input_names = ["input_ids", "token_type_ids", "attention_mask"]
    pad_token_id = None
    eos_token_id = 2
    eos_token = "<eos>"
    pad_token = None


def test_prepare_decoder_tokenizer_removes_token_type_ids() -> None:
    tokenizer = DummyTokenizer()

    returned = prepare_decoder_tokenizer(tokenizer)

    assert returned is tokenizer
    assert tokenizer.model_input_names == ["input_ids", "attention_mask"]
    assert tokenizer.pad_token == "<eos>"


def test_strip_unused_decoder_model_kwargs_removes_token_type_ids() -> None:
    batch = {"input_ids": [1, 2], "attention_mask": [1, 1], "token_type_ids": [0, 0]}

    returned = strip_unused_decoder_model_kwargs(batch)

    assert returned is batch
    assert batch == {"input_ids": [1, 2], "attention_mask": [1, 1]}


class DummyTensor:
    def __init__(self) -> None:
        self.device = None

    def to(self, device: str) -> "DummyTensor":
        self.device = device
        return self


def test_move_batch_to_device_strips_and_moves_values() -> None:
    input_ids = DummyTensor()
    attention_mask = DummyTensor()
    token_type_ids = DummyTensor()
    batch = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
        "metadata": "kept",
    }

    moved = move_batch_to_device(batch, "cuda:0")

    assert moved == {"input_ids": input_ids, "attention_mask": attention_mask, "metadata": "kept"}
    assert input_ids.device == "cuda:0"
    assert attention_mask.device == "cuda:0"
    assert token_type_ids.device is None
