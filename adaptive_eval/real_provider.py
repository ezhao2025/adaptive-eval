"""A real Anthropic provider for Design A/B: same interface as ReplayProvider.

Only 429, 5xx, overload, timeout and connection failures become TransientError (the engine
retries those). Every other 4xx -- a bad request, a bad key, a missing model -- is a bug in
our request, so it fails immediately instead of being retried against a broken setup.

Items come from a JSON file:
  {"item-0001": {"question": "...", "answer": "paris"},
   "item-0002": {"question": "...", "answer": "4", "aliases": ["four"],
                 "image_b64": "...", "media_type": "image/png"}}
item_hashes() feeds the cache key, so editing an item's text or image invalidates its
cached answers instead of silently reusing them.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import re
import string
import sys

from .providers import ProviderConfig, Response, TransientError

# Conservative defaults; override from your Console's rate limits and pricing page.
ANTHROPIC_CONFIG = ProviderConfig(rpm=50, tpm=30_000, latency=(0.5, 3.0), failure_rate=0.0,
                                  usd_per_1k_input=0.001, usd_per_1k_output=0.005,
                                  est_tokens_per_call=600)

_PUNCT = str.maketrans("", "", string.punctuation)


def load_items(path: str) -> dict[str, dict]:
    with open(path) as f:
        return json.load(f)


def item_hashes(items: dict[str, dict]) -> dict[str, str]:
    """Content hash per item: question, answer, aliases and image all count."""
    return {iid: hashlib.sha256(json.dumps(it, sort_keys=True).encode()).hexdigest()[:16]
            for iid, it in items.items()}


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower().translate(_PUNCT)).strip()


def grade(text: str, item: dict) -> bool:
    """Exact match after normalization, against the answer or any listed alias."""
    got = normalize(text)
    return any(got == normalize(a) for a in [item["answer"], *item.get("aliases", [])])


class AnthropicProvider:
    """client is injectable so tests can exercise retries without touching the network."""

    def __init__(self, cfg: ProviderConfig, items: dict[str, dict], *, client=None,
                 max_tokens: int = 16, name: str = "anthropic",
                 sampling: dict | None = None):
        self.name, self.cfg, self.items, self.max_tokens = name, cfg, items, max_tokens
        self.sampling = {"temperature": 0} if sampling is None else dict(sampling)
        self.calls = 0
        self._supported: set[str] | None = None
        if client is not None:
            self.client = client
        else:
            import anthropic                      # reads ANTHROPIC_API_KEY from the environment
            self.client = anthropic.AsyncAnthropic()
            # Sampling knobs come and go between API versions (current ones have no
            # temperature), so send only what this SDK accepts instead of crashing.
            self._supported = set(inspect.signature(
                type(self.client.messages).create).parameters)
            dropped = [k for k in self.sampling if k not in self._supported]
            if dropped:
                print(f"[{self.name}] SDK does not accept {dropped}; not sending them",
                      file=sys.stderr)

    def _params(self) -> dict:
        return {k: v for k, v in self.sampling.items()
                if self._supported is None or k in self._supported}

    def _content(self, item: dict) -> list[dict]:
        content = []
        if item.get("image_b64"):
            content.append({"type": "image", "source": {
                "type": "base64", "media_type": item.get("media_type", "image/png"),
                "data": item["image_b64"]}})
        content.append({"type": "text",
                        "text": f"{item['question']}\nAnswer with one word, nothing else."})
        return content

    async def answer(self, model: str, item_id: str) -> Response:
        import anthropic
        item = self.items[item_id]                # missing item = our bug, so let it raise
        self.calls += 1
        try:
            msg = await self.client.messages.create(
                model=model, max_tokens=self.max_tokens, **self._params(),
                messages=[{"role": "user", "content": self._content(item)}])
        except (anthropic.RateLimitError, anthropic.APIConnectionError,
                anthropic.APITimeoutError) as e:
            raise TransientError(f"{type(e).__name__}: {e}") from e
        except anthropic.APIStatusError as e:     # 5xx and overload are transient, 4xx are not
            if e.status_code >= 500 or e.status_code == 529:
                raise TransientError(f"{type(e).__name__}: {e}") from e
            raise
        text = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
        tin, tout = msg.usage.input_tokens, msg.usage.output_tokens
        cost = tin / 1000 * self.cfg.usd_per_1k_input + tout / 1000 * self.cfg.usd_per_1k_output
        return Response(grade(text, item), tin, tout, cost)
