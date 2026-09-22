"""Knobs for experiments on simulated providers.

rate_scale multiplies every provider's rpm and tpm (loose vs tight rate limits).
latency_scale multiplies replay latency (real LLM calls take ~1 s, replay ~0.05 s).
Workers and the scheduler must use the same values: the scheduler's speculation peeks at
the same Redis buckets the workers fill.
"""
from __future__ import annotations

from dataclasses import replace

from ..providers import DEFAULT_PROVIDERS, ProviderConfig


def provider_configs(rate_scale: float = 1.0, latency_scale: float = 1.0) -> dict[str, ProviderConfig]:
    return {n: replace(c, rpm=c.rpm * rate_scale, tpm=c.tpm * rate_scale,
                       latency=(c.latency[0] * latency_scale, c.latency[1] * latency_scale))
            for n, c in DEFAULT_PROVIDERS.items()}


def add_args(p) -> None:
    p.add_argument("--rate-scale", type=float, default=1.0,
                   help="multiply every provider's rate limit (experiments)")
    p.add_argument("--latency-scale", type=float, default=1.0,
                   help="multiply replay call latency (experiments)")
