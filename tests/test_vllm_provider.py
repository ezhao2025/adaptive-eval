import asyncio
import json

import httpx
import pytest

from adaptive_eval.providers import TransientError
from adaptive_eval.vllm_provider import VLLM_CONFIG, VLLMProvider

ITEMS = {"q1": {"question": "How many cubes?", "answer": "3", "aliases": ["three"],
                "image_b64": "aGVsbG8=", "media_type": "image/png"}}


def provider(handler):
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return VLLMProvider(VLLM_CONFIG, ITEMS, "http://pod:8000/", client=client)


def reply(text, status=200):
    body = {"choices": [{"message": {"content": text}}],
            "usage": {"prompt_tokens": 700, "completion_tokens": 2}}
    return lambda req: httpx.Response(status, json=body)


def test_request_matches_the_batch_prompt_and_grades_the_reply():
    seen = {}

    def handler(req):
        seen["url"], seen["body"] = str(req.url), json.loads(req.content)
        return reply("Three")(req)
    r = asyncio.run(provider(handler).answer("llava", "q1"))
    assert seen["url"] == "http://pod:8000/v1/chat/completions"
    b = seen["body"]
    assert b["model"] == "llava" and b["temperature"] == 0.0 and b["max_tokens"] == 64
    img, txt = b["messages"][0]["content"]
    assert img["image_url"]["url"] == "data:image/png;base64,aGVsbG8="
    assert txt["text"] == "How many cubes?\nAnswer with one word, nothing else."
    assert r.correct and r.input_tokens == 700 and r.output_tokens == 2 and r.cost_usd == 0.0
    assert not asyncio.run(provider(reply("4")).answer("llava", "q1")).correct


@pytest.mark.parametrize("status", [429, 502, 503, 504])
def test_overload_and_proxy_errors_are_transient(status):
    with pytest.raises(TransientError):
        asyncio.run(provider(reply("x", status)).answer("llava", "q1"))


def test_connection_errors_are_transient():
    def handler(req):
        raise httpx.ConnectError("refused", request=req)
    with pytest.raises(TransientError):
        asyncio.run(provider(handler).answer("llava", "q1"))


def test_bad_request_fails_immediately():
    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(provider(reply("x", 400)).answer("llava", "q1"))
