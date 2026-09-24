"""MLX runtime contract tests that run without the pack."""

from __future__ import annotations

import pytest
import torch

from deciduous.bonsai import CODE_COUNT, MASK
from deciduous.mlx import MlxModel, codes_from_encoder


def test_codes_from_encoder_derives_255_distinct_single_tokens() -> None:
    counter = iter(range(10_000))
    codes, token_ids = codes_from_encoder(lambda _text: [next(counter)])
    assert codes[0] == "A" and codes[25] == "Z" and codes[26] == "AA"
    assert len(codes) == CODE_COUNT == len(token_ids) == len(set(token_ids))


def test_codes_from_encoder_rejects_ambiguous_tokenizer() -> None:
    with pytest.raises(RuntimeError):
        codes_from_encoder(lambda _text: [1, 2])


class _Template:
    def render(self, **_kwargs: object) -> str:
        return "prompt"


def _stub(logprobs: list[float]) -> MlxModel:
    counter = iter(range(10_000))
    model = MlxModel.__new__(MlxModel)
    model.codes, model.token_ids = codes_from_encoder(lambda _text: [next(counter)])
    model.template = _Template()
    model._answer_logprobs = lambda _prompt: ([5, 6, 7, 8, 9], logprobs)  # type: ignore[method-assign]
    return model


def test_call_masks_positions_beyond_the_option_count() -> None:
    row = {"state": "The light is red.", "question": {"type": "choice", "criteria": {"stop": "Stop", "go": "Go"}}}
    model = _stub([-0.2, -1.7] + [-9.0] * (CODE_COUNT - 2))
    batch = model.prepare([row])
    out = model(batch)
    assert out.shape == (1, CODE_COUNT)
    assert torch.allclose(out[0, :2], torch.tensor([-0.2, -1.7]))
    assert out[0, 2].item() == MASK
    assert batch.input_tokens == 5


def test_prepare_rejects_empty_batches() -> None:
    with pytest.raises(ValueError):
        _stub([]).prepare([])
