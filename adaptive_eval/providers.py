"""Model providers. Everything else in the system only sees the Provider interface,
so replay (free, deterministic) and real APIs are interchangeable."""
from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass
from typing import Protocol


@dataclass
class Response:
    correct: bool
    input_tokens: int
    output_tokens: int
    cost_usd: float


class TransientError(Exception):
    """Retryable failure: timeout, 429, 5xx."""


class Provider(Protocol):
    name: str

    async def answer(self, model: str, item_id: str) -> Response: ...


@dataclass
class ProviderConfig:
    rpm: int                 # requests per minute
    tpm: int                 # tokens per minute
    latency: tuple[float, float]
    failure_rate: float
    usd_per_1k_input: float
    usd_per_1k_output: float
    est_tokens_per_call: int = 900


# Simulated limits/prices -- deliberately different so the scheduler has to juggle them.
DEFAULT_PROVIDERS = {
    "sim-openai": ProviderConfig(3000, 3_000_000, (0.02, 0.08), 0.03, 0.0025, 0.01),
    "sim-anthropic": ProviderConfig(1500, 1_500_000, (0.03, 0.10), 0.03, 0.003, 0.015),
    "sim-google": ProviderConfig(600, 1_000_000, (0.02, 0.06), 0.05, 0.00125, 0.005),
}


class ReplayProvider:
    """Answers from a logged response matrix with simulated latency and failures."""

    def __init__(self, name: str, cfg: ProviderConfig, responses: dict[tuple[str, str], bool],
                 seed: int = 0):
        self.name, self.cfg, self.responses = name, cfg, responses
        self.rng = random.Random(seed)
        self.calls = 0

    async def answer(self, model: str, item_id: str) -> Response:
        self.calls += 1
        await asyncio.sleep(self.rng.uniform(*self.cfg.latency))
        if self.rng.random() < self.cfg.failure_rate:
            raise TransientError(f"{self.name}: simulated 503")
        if (model, item_id) not in self.responses:   # a bug upstream, so not retryable
            raise LookupError(f"no logged answer for {model} on {item_id}")
        correct = self.responses[(model, item_id)]
        tin, tout = 850, 20      # rough image+prompt / short answer
        cost = tin / 1000 * self.cfg.usd_per_1k_input + tout / 1000 * self.cfg.usd_per_1k_output
        return Response(bool(correct), tin, tout, cost)


class AnthropicProvider:
    """UNTESTED skeleton for a real VLM call. Fill in item loading and grading for your
    benchmark. Map rate-limit / overload errors to TransientError so retries work."""

    def __init__(self, cfg: ProviderConfig, item_store):
        import anthropic  # pip install anthropic
        self.name, self.cfg, self.items = "anthropic", cfg, item_store
        self.client = anthropic.AsyncAnthropic()

    async def answer(self, model: str, item_id: str) -> Response:
        import anthropic
        item = self.items[item_id]  # expects {"image_b64", "media_type", "question", "answer"}
        try:
            msg = await self.client.messages.create(
                model=model, max_tokens=50, temperature=0,
                messages=[{"role": "user", "content": [
                    {"type": "image", "source": {"type": "base64",
                                                 "media_type": item["media_type"],
                                                 "data": item["image_b64"]}},
                    {"type": "text", "text": item["question"] + "\nAnswer with one word."},
                ]}],
            )
        except (anthropic.RateLimitError, anthropic.InternalServerError,
                anthropic.APIConnectionError) as e:   # 4xx other than 429 are NOT retryable
            raise TransientError(str(e)) from e
        text = msg.content[0].text.strip().lower()
        correct = text == item["answer"].lower()          # replace with your grader
        tin, tout = msg.usage.input_tokens, msg.usage.output_tokens
        cost = tin / 1000 * self.cfg.usd_per_1k_input + tout / 1000 * self.cfg.usd_per_1k_output
        return Response(correct, tin, tout, cost)
