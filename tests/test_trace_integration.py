"""Issue 6: every answer carries a trace of time and tokens per stage."""

from __future__ import annotations

import pytest

from rag.pipeline import Pipeline

QUESTION = "What is fog computing and why is it used?"


def check_trace(cfg):
    result = Pipeline(cfg).answer(QUESTION)
    trace = result.trace
    names = [s.name for s in trace.stages]
    assert "retrieve" in names and "generate" in names
    assert all(s.ms >= 0 for s in trace.stages)
    gen = next(s for s in trace.stages if s.name == "generate")
    assert gen.tokens_in > len(QUESTION.split())
    assert gen.tokens_out > 0
    top_level = sum(s.ms for s in trace.stages if s.depth == 0)
    assert trace.total_ms >= top_level
    assert trace.estimated_cost == 0.0  # local models by default
    return result


def test_trace_offline(offline_cfg):
    check_trace(offline_cfg())


def test_trace_with_rerank_nests_search_and_rerank(offline_cfg):
    trace = check_trace(offline_cfg(retrieval_mode="hybrid", rerank=True)).trace
    nested = {(s.name, s.parent) for s in trace.stages if s.depth == 1}
    assert {("search", "retrieve"), ("rerank", "retrieve")} <= nested
    assert trace.ms("rerank") <= trace.ms("retrieve")


def test_low_relevance_refusal_has_no_generate_stage(offline_cfg):
    result = Pipeline(offline_cfg()).answer("zorblax quintessa vuvuzela?")  # shares nothing with the notes
    assert result.refusal_reason == "low_relevance"
    assert result.trace.ms("generate") is None and result.trace.tokens_in == 0


def test_price_config_gives_an_estimate(offline_cfg):
    result = Pipeline(offline_cfg(price_per_1k_tokens_in=1.0, price_per_1k_tokens_out=1.0)).answer(QUESTION)
    expected = (result.trace.tokens_in + result.trace.tokens_out) / 1000
    assert result.trace.estimated_cost == pytest.approx(expected)
    assert "estimate" in result.trace.as_markdown()


def test_answer_eval_rows_carry_latency_and_tokens(offline_cfg):
    from evals import load_golden
    from evals.answers import evaluate_answers
    from evals.judge import LexicalJudge

    summary, rows = evaluate_answers(offline_cfg(), load_golden(limit=4), LexicalJudge())
    assert all(r["total_ms"] >= (r["retrieve_ms"] or 0) for r in rows)
    assert summary["latency_ms"]["retrieve"]["p95"] >= summary["latency_ms"]["retrieve"]["p50"] >= 0
    assert summary["mean_tokens_in"] > 0


def test_answer_eval_warms_up_before_timing(offline_cfg):
    """The first scored question must not pay for loading the models."""
    from evals import load_golden
    from evals.answers import evaluate_answers
    from evals.judge import LexicalJudge

    class Counting(Pipeline):
        calls: list[str] = []

        def answer(self, question, clock=None):
            self.calls.append(question)
            return super().answer(question, clock)

    items = load_golden(limit=3)
    pipeline = Counting(offline_cfg())
    _, rows = evaluate_answers(pipeline.cfg, items, LexicalJudge(), pipeline=pipeline)
    assert pipeline.calls == [items[0].question] + [i.question for i in items]
    assert [r["id"] for r in rows] == [i.id for i in items]


@pytest.mark.realmodel
def test_trace_real(real_cfg):
    check_trace(real_cfg())
