"""Design B Step 7: admission controller logic (no Redis or Postgres needed)."""
import asyncio

import pytest

from adaptive_eval.b.admission import Admission, expected_se_reduction
from adaptive_eval.b.queue import Job
from adaptive_eval.providers import ProviderConfig

CFGS = {"p": ProviderConfig(600, 10**9, (0.1, 0.1), 0.0, 0.001, 0.0)}   # est cost $0.0009


class FakeQueue:
    def __init__(self):
        self.sent = []

    async def enqueue(self, job):
        self.sent.append(job.job_id)


def job(name):
    return Job(name, "p", "m", "i", "k-" + name, False)


def test_expected_se_reduction():
    assert expected_se_reduction(0.5, 0.0) == pytest.approx(0.0)
    assert expected_se_reduction(0.5, 4.0) == pytest.approx(0.5 - 1 / 8 ** 0.5)


def test_window_default_from_rate_and_latency():
    # 600 rpm = 10/s, p50 latency 0.1 s, x1.5 -> 1.5 -> ceil 2
    assert Admission(FakeQueue(), CFGS).window["p"] == 2


@pytest.mark.parametrize("mode,expected", [("priority", ["a", "c", "d", "b"]),
                                           ("nearest", ["a", "c", "d", "b"]),
                                           ("fifo", ["a", "b", "c", "d"])])
def test_window_and_ordering(mode, expected):
    async def main():
        q = FakeQueue()
        adm = Admission(q, CFGS, mode=mode, windows={"p": 1})
        for name, prio in [("a", 1.0), ("b", 0.1), ("c", 9.0), ("d", 5.0)]:
            await adm.admit(job(name), prio)
        assert q.sent == ["a"]                      # window of 1: the rest wait
        for name in expected[:-1]:
            await adm.release("p", name, 0.0, paid=False)
        await adm.release("p", "zzz", 0.0, paid=False)   # unknown/duplicate: no effect
        return q.sent
    assert asyncio.run(main()) == expected


def test_budget_counts_committed_calls():
    async def main():
        q = FakeQueue()
        adm = Admission(q, CFGS, windows={"p": 10}, budget_usd=0.0009 * 3)
        for n in "abcde":
            await adm.admit(job(n), 1.0)
        first = list(q.sent)                        # only 3 affordable, even all in flight
        for n in first:
            await adm.release("p", n, 0.0009, paid=True)
        return first, q.sent, adm.budget_exhausted(), adm.spent
    first, sent, exhausted, spent = asyncio.run(main())
    assert first == ["a", "b", "c"] and sent == first
    assert exhausted and spent == pytest.approx(0.0027)
