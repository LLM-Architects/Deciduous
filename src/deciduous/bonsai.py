"""Decision inference for the Ternary Bonsai 2 GGUF base through a PrismML llama.cpp server.

The pinned base stores rotated ternary weights that only the PrismML llama.cpp fork can
execute, so a fresh-base model cannot be loaded in torch. This module implements the
DecisionModel contract (codes, prepare, logits over answer codes, temperature) by asking
the llama.cpp server for the answer-position token distribution: one forward pass per
decision, probabilities over the supplied choices.
"""

from __future__ import annotations

import base64
import io
import itertools
import math
import os
import string
from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import httpx
import torch

from deciduous.model import BASE_MODEL, BASE_REVISION, MAX_OPTIONS, decision_messages, open_image, options
from deciduous.types import DecisionInput, ImageInput

SERVER_ENV = "DECIDUOUS_LLAMA_URL"
DEFAULT_SERVER = "http://127.0.0.1:8080"
MASK = -1e9
CODE_COUNT = MAX_OPTIONS


class LlamaServerError(RuntimeError):
    """The llama.cpp server is missing, unhealthy, or returned an unusable response."""


@dataclass
class RuntimeBatch:
    """Messages rendered for one batch; input_tokens is filled after the forward calls."""

    rows: list[DecisionInput]
    messages: list[list[dict[str, object]]]
    counts: tuple[int, ...]
    input_tokens: int


class BonsaiModel:
    """Serve one-pass decisions from the pinned GGUF base via llama.cpp."""

    def __init__(self, base_url: str | None = None, *, timeout: float = 600.0,
                 client: httpx.Client | None = None) -> None:
        self.base_url = (base_url or os.getenv(SERVER_ENV) or DEFAULT_SERVER).rstrip("/")
        self.client = client if client is not None else httpx.Client(base_url=self.base_url, timeout=timeout)
        self.base_model = BASE_MODEL
        self.revision = BASE_REVISION
        self._probe()
        self.codes, self.token_ids = self._derive_codes()
        self._id_to_index = {token_id: index for index, token_id in enumerate(self.token_ids)}
        self.temperature = 1.0

    def _probe(self) -> None:
        try:
            response = self.client.get("/health")
        except httpx.HTTPError as error:
            raise LlamaServerError(
                f"No llama.cpp server at {self.base_url}. Start the PrismML build with the pinned GGUF "
                f"(see README) or point {SERVER_ENV} at a running server."
            ) from error
        if response.status_code != 200:
            raise LlamaServerError(f"llama.cpp server at {self.base_url} is not ready: HTTP {response.status_code}.")

    def _post(self, path: str, payload: dict[str, object]) -> dict[str, object]:
        try:
            response = self.client.post(path, json=payload)
        except httpx.HTTPError as error:
            raise LlamaServerError(f"Request to {self.base_url}{path} failed: {error}") from error
        if response.status_code != 200:
            raise LlamaServerError(f"{path} returned HTTP {response.status_code}: {response.text[:500]}")
        return cast(dict[str, object], response.json())

    def _tokenize(self, content: str) -> list[int]:
        payload = self._post("/tokenize", {"content": content, "add_special_tokens": False})
        return cast(list[int], payload["tokens"])

    def _derive_codes(self) -> tuple[list[str], list[int]]:
        candidates = list(string.ascii_uppercase) + ["".join(pair) for pair in itertools.product(string.ascii_uppercase, repeat=2)]
        codes: list[str] = []
        token_ids: list[int] = []
        for code in candidates:
            tokens = self._tokenize(code)
            if len(tokens) == 1:
                codes.append(code)
                token_ids.append(tokens[0])
                if len(codes) == CODE_COUNT:
                    break
        if len(codes) != CODE_COUNT or len(set(token_ids)) != CODE_COUNT:
            raise LlamaServerError("The GGUF tokenizer must provide 255 distinct single-token answer codes.")
        return codes, token_ids

    def _messages(self, row: DecisionInput, codes: Sequence[str]) -> list[dict[str, object]]:
        template = decision_messages(row, codes)
        content: list[dict[str, object]] = []
        images = iter(row.get("images", []))
        for part in cast(list[dict[str, object]], template[1]["content"]):
            if part["type"] == "image":
                content.append({"type": "image_url", "image_url": {"url": data_uri(next(images))}})
            else:
                content.append(part)
        return [template[0], {"role": "user", "content": content}]

    def prepare(self, rows: Sequence[DecisionInput], max_length: int = 8192) -> RuntimeBatch:
        if not rows:
            raise ValueError("A batch must contain at least one decision.")
        return RuntimeBatch(
            rows=list(rows),
            messages=[self._messages(row, self.codes) for row in rows],
            counts=tuple(len(options(row["question"])[0]) for row in rows),
            input_tokens=0,
        )

    def __call__(self, batch: RuntimeBatch, max_length: int = 8192) -> torch.Tensor:
        logits = torch.full((len(batch.rows), CODE_COUNT), MASK, dtype=torch.float32)
        for index, (messages, count) in enumerate(zip(batch.messages, batch.counts, strict=True)):
            positions, prompt_tokens = self._forward(messages)
            batch.input_tokens += prompt_tokens
            if prompt_tokens > max_length:
                raise ValueError(f"Question branch exceeds the {max_length}-token limit; no input was truncated.")
            chosen = next((entry for entry in positions if entry and entry[0].get("id") in self._id_to_index
                           and self._id_to_index[cast(int, entry[0]["id"])] < count), positions[0] if positions else [])
            for entry in chosen:
                position = self._id_to_index.get(cast(int, entry.get("id")))
                logprob = entry.get("logprob")
                if position is not None and position < count and isinstance(logprob, (int, float)):
                    logits[index, position] = float(logprob)
        return logits

    def _forward(self, messages: list[dict[str, object]]) -> tuple[list[list[dict[str, object]]], int]:
        payload = {
            "messages": messages,
            "max_tokens": 16,
            "temperature": 0,
            "stop": ["\n"],
            "logprobs": True,
            "top_logprobs": CODE_COUNT,
            "n_probs": CODE_COUNT,
            "chat_template_kwargs": {"enable_thinking": False},
            "reasoning_effort": "none",
        }
        body = self._post("/v1/chat/completions", payload)
        choices = cast(list[dict[str, object]], body.get("choices") or [])
        usage = cast(dict[str, object], body.get("usage") or {})
        usage_tokens = usage.get("prompt_tokens")
        prompt_tokens = usage_tokens if isinstance(usage_tokens, int) else 0
        if not choices:
            raise LlamaServerError("The server returned no choice for a decision.")
        logprobs = cast(dict[str, object] | None, choices[0].get("logprobs"))
        content = cast(list[dict[str, object]] | None, logprobs.get("content") if logprobs else None)
        if not content:
            raise LlamaServerError("The server returned no token logprobs; start it without cached quantized logits.")
        positions = [cast(list[dict[str, object]], entry.get("top_logprobs") or []) for entry in content]
        return positions, prompt_tokens

    @torch.inference_mode()
    def predict(self, rows: Sequence[DecisionInput], batch_size: int = 8, temperature: float | None = None) -> list[list[float]]:
        scale = self.temperature if temperature is None else temperature
        if not math.isfinite(scale) or scale <= 0 or batch_size < 1:
            raise ValueError("Temperature and batch size must be positive.")
        distributions: list[list[float]] = []
        for start in range(0, len(rows), batch_size):
            batch = self.prepare(rows[start:start + batch_size])
            probabilities = (self(batch) / scale).softmax(-1).cpu().tolist()
            distributions.extend(values[:count] for values, count in zip(probabilities, batch.counts, strict=True))
        return distributions

    def count_prompt_tokens(self, row: DecisionInput) -> int:
        """Exact prompt size for one row, image tokens included."""
        messages = self._messages(row, self.codes)
        body = self._post("/v1/chat/completions/input_tokens", {
            "messages": messages,
            "chat_template_kwargs": {"enable_thinking": False},
            "reasoning_effort": "none",
        })
        value = body.get("input_tokens")
        if isinstance(value, int):
            return value
        raise LlamaServerError(f"Unexpected token-count response: {str(body)[:200]}")

    def eval(self) -> BonsaiModel:
        return self

    def train(self, mode: bool = True) -> BonsaiModel:
        return self


def data_uri(value: ImageInput) -> str:
    """Normalize an ImageInput to a data URI the llama.cpp server accepts."""
    if isinstance(value, str) and value.startswith("data:image/"):
        return value
    buffer = io.BytesIO()
    open_image(value).save(buffer, format="JPEG")
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()
