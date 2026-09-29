"""A model served by `vllm serve` (OpenAI-compatible HTTP), behind the Provider interface.

Same contract as AnthropicProvider and MLXProvider: `await answer(model, item_id) ->
Response`. The request matches scripts/vllm_batch.py exactly (one image as a data URL, then
the question with "Answer with one word, nothing else.", temperature 0, 64 tokens), so a
live answer can be checked against the same model's batch answer.

Transient (retried by the worker): 429, 5xx, timeouts, connection errors -- including the
502/503/504 a proxy such as RunPod's returns while the server is still loading. Any other
4xx is a bug in our request and fails at once.

Cost is 0 per call: a rented GPU is billed by the hour, not by the call.

    vllm serve llava-hf/llava-onevision-qwen2-7b-ov-hf --api-key <key> --max-model-len 4096
    python -m adaptive_eval.b.worker --id live1 --provider vllm \\
        --base-url https://<pod-id>-8000.proxy.runpod.net --items data/big_items.json
"""
from __future__ import annotations

import os

import httpx

from .providers import ProviderConfig, Response, TransientError
from .real_provider import grade

# One server, one model. Limits only need to be loose enough not to throttle.
VLLM_CONFIG = ProviderConfig(rpm=600, tpm=10_000_000, latency=(0.2, 2.0), failure_rate=0.0,
                             usd_per_1k_input=0.0, usd_per_1k_output=0.0,
                             est_tokens_per_call=1500)


def prompt_of(item: dict) -> str:
    return f"{item['question']}\nAnswer with one word, nothing else."


class VLLMProvider:
    """client is injectable so tests can run without a server."""

    def __init__(self, cfg: ProviderConfig, items: dict[str, dict], base_url: str, *,
                 api_key: str | None = None, max_tokens: int = 64, name: str = "vllm",
                 timeout_s: float = 120.0, client: httpx.AsyncClient | None = None):
        self.name, self.cfg, self.items, self.max_tokens = name, cfg, items, max_tokens
        self.url = base_url.rstrip("/") + "/v1/chat/completions"
        key = api_key if api_key is not None else os.environ.get("VLLM_API_KEY")
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        self.client = client or httpx.AsyncClient(timeout=timeout_s, headers=headers)
        self.calls = 0

    def request(self, model: str, item: dict) -> dict:
        url = f"data:{item.get('media_type', 'image/png')};base64,{item['image_b64']}"
        return {"model": model, "temperature": 0.0, "max_tokens": self.max_tokens,
                "messages": [{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": url}},
                    {"type": "text", "text": prompt_of(item)}]}]}

    async def answer(self, model: str, item_id: str) -> Response:
        item = self.items[item_id]                # a missing item is our bug: let it raise
        self.calls += 1
        try:
            r = await self.client.post(self.url, json=self.request(model, item))
        except (httpx.TimeoutException, httpx.TransportError) as e:
            raise TransientError(f"{type(e).__name__}: {e}") from e
        if r.status_code == 429 or r.status_code >= 500:
            raise TransientError(f"HTTP {r.status_code}: {r.text[:200]}")
        r.raise_for_status()                      # other 4xx: our request is wrong
        body = r.json()
        text = body["choices"][0]["message"].get("content") or ""
        usage = body.get("usage") or {}
        return Response(grade(text, item), int(usage.get("prompt_tokens", 0)),
                        int(usage.get("completion_tokens", 0)), 0.0)
