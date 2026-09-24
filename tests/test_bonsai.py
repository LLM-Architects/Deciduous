import itertools
import json
import string
from pathlib import Path
from tempfile import NamedTemporaryFile

import httpx
import pytest
from PIL import Image

from deciduous.bonsai import BonsaiModel, data_uri
from deciduous.model import DecisionModel

CANDIDATES = list(string.ascii_uppercase) + ["".join(pair) for pair in itertools.product(string.ascii_uppercase, repeat=2)]
TOKEN_IDS = {code: 1_000 + index for index, code in enumerate(CANDIDATES)}


def handler(request: httpx.Request) -> httpx.Response:
    if request.url.path == "/health":
        return httpx.Response(200, json={"status": "ok"})
    if request.url.path == "/tokenize":
        code = json.loads(request.content)["content"]
        return httpx.Response(200, json={"tokens": [TOKEN_IDS[code]] if code in TOKEN_IDS else [1, 2]})
    if request.url.path == "/v1/chat/completions/input_tokens":
        return httpx.Response(200, json={"input_tokens": 57})
    if request.url.path == "/v1/chat/completions":
        body = json.loads(request.content)
        assert body["chat_template_kwargs"] == {"enable_thinking": False}
        assert body["reasoning_effort"] == "none"
        content = [
            {"token": "The", "top_logprobs": [{"id": 7, "logprob": -0.05, "token": "The"}]},
            {"token": "B", "top_logprobs": [
                {"id": TOKEN_IDS["B"], "logprob": -0.10, "token": "B"},
                {"id": TOKEN_IDS["A"], "logprob": -1.60, "token": "A"},
                {"id": TOKEN_IDS["D"], "logprob": -2.20, "token": "D"},
            ]},
        ]
        return httpx.Response(200, json={
            "choices": [{"logprobs": {"content": content}, "message": {"content": "B"}}],
            "usage": {"prompt_tokens": 57, "completion_tokens": 1},
        })
    raise AssertionError(f"unexpected request {request.url.path}")


def make_model() -> BonsaiModel:
    return BonsaiModel("http://llama", client=httpx.Client(base_url="http://llama", transport=httpx.MockTransport(handler)))


ROW = {
    "state": "lights are red",
    "question": {"type": "choice", "criteria": {"stop": None, "go": None, "wait": None}},
}


def test_codes_are_derived_from_the_server_tokenizer() -> None:
    model = make_model()
    assert model.codes[:26] == list(string.ascii_uppercase)
    assert len(model.codes) == 255
    assert len(set(model.token_ids)) == 255


def test_decision_uses_first_code_position_distribution() -> None:
    model = make_model()
    batch = model.prepare([ROW])
    logits = model(batch)
    assert batch.input_tokens == 57
    assert logits.shape == (1, 255)
    assert logits[0, 1] == pytest.approx(-0.10)
    assert logits[0, 0] == pytest.approx(-1.60)
    assert logits[0, 3] < -1e8
    assert logits[0, 2] < -1e8


def test_predict_renormalizes_over_the_option_codes() -> None:
    model = make_model()
    values = model.predict([ROW])[0]
    assert len(values) == 3
    assert sum(values) == pytest.approx(1.0)
    assert values[1] > values[0] > 0.0


def test_data_uri_accepts_pil_images_and_paths() -> None:
    image = Image.new("RGB", (4, 4), color=(255, 0, 0))
    assert data_uri(image).startswith("data:image/jpeg;base64,")
    with NamedTemporaryFile(suffix=".png") as temporary:
        image.save(temporary, format="PNG")
        assert data_uri(Path(temporary.name)).startswith("data:image/jpeg;base64,")


def test_count_prompt_tokens_uses_the_counting_endpoint() -> None:
    model = make_model()
    assert model.count_prompt_tokens(ROW) == 57


def test_torch_loader_refuses_the_gguf_base() -> None:
    with pytest.raises(ValueError, match="PrismML"):
        DecisionModel()
