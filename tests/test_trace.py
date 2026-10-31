"""Unit tests for rag/trace.py, driven by a hand-cranked clock."""

from __future__ import annotations

import pytest

from rag.trace import Trace


class FakeClock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now

    def advance(self, ms):
        self.now += ms / 1000


def test_stages_are_timed_and_nested():
    clock = FakeClock()
    trace = Trace(clock=clock)
    clock.advance(1)
    with trace.stage("retrieve"):
        clock.advance(10)
        with trace.stage("search"):
            clock.advance(20)
        with trace.stage("rerank") as rr:
            clock.advance(30)
        clock.advance(5)
    with trace.stage("generate") as gen:
        gen.tokens_in, gen.tokens_out = 800, 60
        clock.advance(400)
    clock.advance(4)
    trace.finish()
    clock.advance(1000)  # after finish, nothing changes

    names = [(s.name, s.depth, s.parent) for s in trace.stages]
    assert names == [("retrieve", 0, None), ("search", 1, "retrieve"), ("rerank", 1, "retrieve"), ("generate", 0, None)]
    assert trace.ms("retrieve") == pytest.approx(65)
    assert trace.ms("search") == pytest.approx(20)
    assert rr.ms == pytest.approx(30)
    assert trace.ms("generate") == pytest.approx(400)
    assert trace.ms("missing") is None
    assert trace.total_ms == pytest.approx(470)
    top_level = sum(s.ms for s in trace.stages if s.depth == 0)
    assert trace.total_ms >= top_level
    assert (trace.tokens_in, trace.tokens_out) == (800, 60)


def test_stage_is_closed_when_the_body_raises():
    clock = FakeClock()
    trace = Trace(clock=clock)
    with pytest.raises(RuntimeError):
        with trace.stage("generate"):
            clock.advance(7)
            raise RuntimeError("model down")
    assert trace.ms("generate") == pytest.approx(7)
    with trace.stage("after"):  # the failed stage is no longer the parent
        pass
    assert trace.stages[-1].parent is None


def test_cost_estimate_and_markdown():
    trace = Trace(clock=FakeClock(), price_per_1k_in=0.5, price_per_1k_out=2.0)
    with trace.stage("generate") as s:
        s.tokens_in, s.tokens_out = 2000, 500
    trace.finish()
    assert trace.estimated_cost == pytest.approx(2.0)
    md = trace.as_markdown()
    assert "| generate |" in md and "**total**" in md and "estimate" in md
    assert "estimate" not in Trace(clock=FakeClock()).finish().as_markdown()  # local: no price line
