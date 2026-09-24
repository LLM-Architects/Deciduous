"""Decision inference for the Ternary Bonsai 2 base on Apple Silicon through MLX.

Loads the publisher's MLX companion pack (pinned in deciduous.model) through the
pack's bundled runtime; ordinary MLX loaders skip the transforms these weights
need. Implements the same contract as BonsaiModel: 255 single-token answer
codes, prepare(), and a masked 255-way logit tensor from one forward pass per
decision. Text only; route images through the GGUF runtime.
"""

from __future__ import annotations

import itertools
import os
import string
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, cast

from deciduous.bonsai import CODE_COUNT, MASK, RuntimeBatch
from deciduous.model import MLX_MODEL, MLX_REVISION, decision_messages, options
from deciduous.types import DecisionInput

if TYPE_CHECKING:
    import torch

PACK_ENV = "DECIDUOUS_MLX_PACK"


class MlxModelError(RuntimeError):
    pass


def codes_from_encoder(encode: Callable[[str], list[int]]) -> tuple[list[str], list[int]]:
    """Derive the 255 single-token answer codes from a tokenizer encoder."""
    candidates = list(string.ascii_uppercase) + [
        "".join(pair) for pair in itertools.product(string.ascii_uppercase, repeat=2)
    ]
    codes: list[str] = []
    token_ids: list[int] = []
    for code in candidates:
        ids = encode(code)
        if len(ids) == 1:
            codes.append(code)
            token_ids.append(ids[0])
            if len(codes) == CODE_COUNT:
                break
    if len(codes) != CODE_COUNT or len(set(token_ids)) != CODE_COUNT:
        raise MlxModelError("The pack tokenizer must provide 255 distinct single-token answer codes.")
    return codes, token_ids


class MlxModel:
    """Serve one-pass decisions from the pinned MLX pack, in process."""

    def __init__(self, pack: str | Path | None = None) -> None:
        try:
            import mlx.core as mx  # noqa: F401
        except ImportError as error:
            raise MlxModelError("MLX is not installed; run uv sync --extra mlx first.") from error
        from jinja2.sandbox import ImmutableSandboxedEnvironment
        from tokenizers import Tokenizer

        self.pack = Path(pack or os.getenv(PACK_ENV) or ".").resolve()
        runtime = self.pack / "runtime"
        if not (runtime / "artifact.py").is_file():
            raise MlxModelError(f"No bundled runtime under {runtime}; download the pinned MLX pack first.")
        if str(runtime) not in sys.path:
            sys.path.insert(0, str(runtime))
        from artifact import load_model

        self.base_model = MLX_MODEL
        self.revision = MLX_REVISION
        self.model, _config = load_model(self.pack)
        self.tokenizer = Tokenizer.from_file(str(self.pack / "tokenizer.json"))
        template = self.pack / "chat_template.jinja"
        if not template.is_file():
            raise MlxModelError("The pack is missing chat_template.jinja.")
        environment = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
        self.template = environment.from_string(template.read_text())
        self.codes, self.token_ids = codes_from_encoder(
            lambda text: self.tokenizer.encode(text, add_special_tokens=False).ids
        )
        self.temperature = 1.0

    def _messages(self, row: DecisionInput) -> list[dict[str, object]]:
        template = decision_messages(row, self.codes)
        for part in cast(list[dict[str, object]], template[1]["content"]):
            if part.get("type") == "image":
                raise ValueError("The MLX runtime answers text decisions only; route images through the GGUF runtime.")
        return [template[0], {"role": "user", "content": template[1]["content"]}]

    def prepare(self, rows: Sequence[DecisionInput], max_length: int = 8192) -> RuntimeBatch:
        if not rows:
            raise ValueError("A batch must contain at least one decision.")
        return RuntimeBatch(
            rows=list(rows),
            messages=[self._messages(row) for row in rows],
            counts=tuple(len(options(row["question"])[0]) for row in rows),
            input_tokens=0,
        )

    def _answer_logprobs(self, prompt: str) -> tuple[list[int], list[float]]:
        import mlx.core as mx

        ids = self.tokenizer.encode(prompt, add_special_tokens=False).ids
        tokens = mx.array([ids])
        hidden = self.model.model(tokens)[:, -1:, :]
        logits = self.model.lm_head(hidden)[:, -1, :]
        probabilities = mx.softmax(logits[0])
        selected = probabilities[mx.array(self.token_ids)]
        logprobs = mx.log(selected + 1e-30)
        mx.eval(logprobs)
        return ids, [float(value) for value in logprobs.tolist()]

    def __call__(self, batch: RuntimeBatch, max_length: int = 8192) -> torch.Tensor:
        import torch

        logits = torch.full((len(batch.rows), CODE_COUNT), MASK, dtype=torch.float32)
        for index, (messages, count) in enumerate(zip(batch.messages, batch.counts, strict=True)):
            prompt = self.template.render(
                messages=messages, add_generation_prompt=True, enable_thinking=False
            )
            ids, row = self._answer_logprobs(prompt)
            batch.input_tokens += len(ids)
            if len(ids) > max_length:
                raise ValueError(f"Question branch exceeds the {max_length}-token limit; no input was truncated.")
            for position in range(min(count, len(row))):
                logits[index, position] = row[position]
        return logits
