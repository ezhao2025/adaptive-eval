"""9.2: real-provider error mapping and grading. No network: the client is a fake."""
import asyncio

import anthropic
import httpx
import pytest

from adaptive_eval.real_provider import (ANTHROPIC_CONFIG, AnthropicProvider, grade,
                                         item_hashes, normalize)
from adaptive_eval.providers import TransientError

ITEMS = {"i1": {"question": "Capital of France?", "answer": "Paris"},
         "i2": {"question": "2+2?", "answer": "4", "aliases": ["four"]}}


def err(cls, status):
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    return cls("boom", response=httpx.Response(status, request=request), body=None)


class FakeMessages:
    def __init__(self, outcomes):
        self.outcomes, self.seen = list(outcomes), []

    async def create(self, **kw):
        self.seen.append(kw)
        out = self.outcomes.pop(0)
        if isinstance(out, Exception):
            raise out
        return out


class FakeClient:
    def __init__(self, outcomes):
        self.messages = FakeMessages(outcomes)


class Msg:
    def __init__(self, text, tin=100, tout=3):
        self.content = [type("B", (), {"type": "text", "text": text})()]
        self.usage = type("U", (), {"input_tokens": tin, "output_tokens": tout})()


def provider(outcomes):
    return AnthropicProvider(ANTHROPIC_CONFIG, ITEMS, client=FakeClient(outcomes))


def test_grading_and_normalization():
    assert normalize("  Paris. ") == "paris"
    assert grade("Paris.", ITEMS["i1"]) and not grade("Lyon", ITEMS["i1"])
    assert grade("four", ITEMS["i2"]) and grade("4", ITEMS["i2"])


def test_item_hash_changes_with_content():
    other = {"i1": {**ITEMS["i1"], "question": "Capital of Spain?"}}
    assert item_hashes(ITEMS)["i1"] != item_hashes(other)["i1"]


@pytest.mark.parametrize("exc", [
    err(anthropic.RateLimitError, 429),
    err(anthropic.InternalServerError, 500),
    err(anthropic.APIStatusError, 529),                       # overloaded
    anthropic.APIConnectionError(request=httpx.Request("POST", "https://api.anthropic.com")),
])
def test_retryable_errors_become_transient(exc):
    p = provider([exc])
    with pytest.raises(TransientError):
        asyncio.run(p.answer("m", "i1"))


@pytest.mark.parametrize("exc", [err(anthropic.BadRequestError, 400),
                                 err(anthropic.AuthenticationError, 401),
                                 err(anthropic.NotFoundError, 404)])
def test_client_errors_are_not_retried(exc):
    p = provider([exc])
    with pytest.raises(anthropic.APIStatusError):             # NOT TransientError
        asyncio.run(p.answer("m", "i1"))


def test_cost_and_correctness_from_usage():
    p = provider([Msg("Paris", tin=120, tout=2)])
    r = asyncio.run(p.answer("claude-haiku-4-5", "i1"))
    assert r.correct and r.input_tokens == 120 and r.output_tokens == 2
    assert r.cost_usd == pytest.approx(120 / 1000 * 0.001 + 2 / 1000 * 0.005)


def test_image_items_send_an_image_block():
    items = {"img": {"question": "What colour?", "answer": "red",
                     "image_b64": "AAAA", "media_type": "image/png"}}
    p = AnthropicProvider(ANTHROPIC_CONFIG, items, client=FakeClient([Msg("red")]))
    asyncio.run(p.answer("m", "img"))
    blocks = p.client.messages.seen[0]["messages"][0]["content"]
    assert blocks[0]["type"] == "image" and blocks[0]["source"]["data"] == "AAAA"
    assert p.client.messages.seen[0]["temperature"] == 0


def test_unsupported_sampling_params_are_dropped():
    class NarrowMessages(FakeMessages):
        async def create(self, *, model, max_tokens, messages):   # no temperature
            return await super().create(model=model, max_tokens=max_tokens, messages=messages)

    class NarrowClient:
        def __init__(self):
            self.messages = NarrowMessages([Msg("Paris")])

    p = AnthropicProvider(ANTHROPIC_CONFIG, ITEMS, client=NarrowClient())
    p._supported = {"model", "max_tokens", "messages"}     # as detected from a real SDK
    r = asyncio.run(p.answer("m", "i1"))
    assert r.correct and "temperature" not in p.client.messages.seen[0]
