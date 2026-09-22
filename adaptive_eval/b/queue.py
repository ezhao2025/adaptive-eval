"""Design B job queues on Redis Streams.

Per provider p: jobs:{p}:real and jobs:{p}:spec, read by consumer group "workers".
One results stream, read by consumer group "scheduler".
Redis holds only transient state: wiping it loses in-flight work, never correctness,
because Postgres is the source of truth and every job is idempotent.
"""
from __future__ import annotations

from dataclasses import dataclass

import redis.asyncio as redis
from redis.exceptions import ResponseError

WORKERS, SCHEDULER = "workers", "scheduler"


def connect(url: str) -> redis.Redis:
    """Always decode responses: stream fields come back as str, not bytes."""
    return redis.from_url(url, decode_responses=True)


def real_job_id(session_id: str, step: int) -> str:
    return f"{session_id}:{step}"


def spec_job_id(cache_key: str) -> str:
    return f"spec:{cache_key}"


@dataclass
class Job:
    job_id: str
    provider: str
    model: str
    item_id: str
    cache_key: str
    speculative: bool

    def fields(self) -> dict[str, str]:
        return {"job_id": self.job_id, "provider": self.provider, "model": self.model,
                "item_id": self.item_id, "cache_key": self.cache_key,
                "speculative": "1" if self.speculative else "0"}

    @classmethod
    def from_fields(cls, f: dict) -> "Job":
        return cls(f["job_id"], f["provider"], f["model"], f["item_id"], f["cache_key"],
                   f["speculative"] == "1")


@dataclass
class Delivery:
    """A job as handed to one worker. `delivery` counts how many times Redis has handed
    out this stream entry: 1 for a fresh read, 2+ after a dead worker's job is reclaimed.
    Log attempts under (job_id, delivery) so a re-paid call after reclaim is not dropped."""
    stream: str
    msg_id: str
    job: Job
    delivery: int = 1


class JobQueue:
    def __init__(self, r: redis.Redis, providers: list[str], prefix: str = ""):
        self.r, self.providers, self.prefix = r, list(providers), prefix
        self.results = f"{prefix}results"

    def stream(self, provider: str, speculative: bool) -> str:
        return f"{self.prefix}jobs:{provider}:{'spec' if speculative else 'real'}"

    def job_streams(self, speculative: bool) -> list[str]:
        return [self.stream(p, speculative) for p in self.providers]

    async def ensure_groups(self) -> None:
        """Idempotent. id="0" (not "$") so entries added before a restart are still delivered."""
        targets = [(s, WORKERS) for sp in (False, True) for s in self.job_streams(sp)]
        targets.append((self.results, SCHEDULER))
        for stream, group in targets:
            try:
                await self.r.xgroup_create(stream, group, id="0", mkstream=True)
            except ResponseError as e:
                if "BUSYGROUP" not in str(e):
                    raise

    async def enqueue(self, job: Job) -> str:
        return await self.r.xadd(self.stream(job.provider, job.speculative), job.fields())

    async def _read(self, consumer: str, streams: list[str], count: int,
                    block_ms: int | None) -> list[Delivery]:
        resp = await self.r.xreadgroup(WORKERS, consumer, {s: ">" for s in streams},
                                       count=count, block=block_ms)
        return [Delivery(stream, msg_id, Job.from_fields(fields))
                for stream, entries in (resp or []) for msg_id, fields in entries]

    async def read(self, consumer: str, count: int = 1, block_ms: int = 1000) -> list[Delivery]:
        """Real lane first; spec lane only when no real work is waiting."""
        got = await self._read(consumer, self.job_streams(False), count, None)
        if got:
            return got
        got = await self._read(consumer, self.job_streams(True), count, None)
        if got:
            return got
        got = await self._read(consumer, self.job_streams(False), count, block_ms)
        return got or await self._read(consumer, self.job_streams(True), count, None)

    async def ack(self, d: Delivery) -> None:
        async with self.r.pipeline(transaction=True) as p:
            p.xack(d.stream, WORKERS, d.msg_id)
            p.xdel(d.stream, d.msg_id)
            await p.execute()

    async def reclaim(self, consumer: str, min_idle_ms: int, count: int = 100) -> list[Delivery]:
        """XAUTOCLAIM jobs idle longer than min_idle_ms (their worker is presumed dead).
        Safe even if the old worker actually finished: jobs are idempotent."""
        out = []
        for stream in self.job_streams(False) + self.job_streams(True):
            start = "0-0"
            while True:
                nxt, entries, *_ = await self.r.xautoclaim(stream, WORKERS, consumer, min_idle_ms,
                                                           start_id=start, count=count)
                for msg_id, fields in entries:
                    if not fields:           # entry was deleted after being claimed
                        continue
                    info = await self.r.xpending_range(stream, WORKERS, min=msg_id, max=msg_id,
                                                       count=1)
                    n = info[0]["times_delivered"] if info else 2
                    out.append(Delivery(stream, msg_id, Job.from_fields(fields), n))
                if nxt in ("0-0", b"0-0"):
                    break
                start = nxt
        return out

    async def publish_result(self, job: Job, **result) -> str:
        fields = {"job_id": job.job_id, "cache_key": job.cache_key,
                  "speculative": "1" if job.speculative else "0"}
        fields.update({k: str(v) for k, v in result.items()})
        return await self.r.xadd(self.results, fields)

    async def read_results(self, consumer: str, count: int = 100,
                           block_ms: int | None = 1000) -> list[tuple[str, dict]]:
        resp = await self.r.xreadgroup(SCHEDULER, consumer, {self.results: ">"},
                                       count=count, block=block_ms)
        return [(msg_id, fields) for _, entries in (resp or []) for msg_id, fields in entries]

    async def ack_result(self, msg_id: str) -> None:
        async with self.r.pipeline(transaction=True) as p:
            p.xack(self.results, SCHEDULER, msg_id)
            p.xdel(self.results, msg_id)
            await p.execute()
