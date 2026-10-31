"""Issue 3 end to end: citations and refusals on the real PDF and golden set."""

from __future__ import annotations

import pytest

from evals import load_golden
from evals.answers import evaluate_answers
from evals.judge import LexicalJudge
from rag.pipeline import Pipeline

FOG = "What is fog computing and why is it used?"
OFF_TOPIC = "How much does Kubernetes cost per hour on AWS?"


def _check_fog_and_off_topic(pipeline: Pipeline):
    result = pipeline.answer(FOG)
    assert not result.refused, result.raw_answer
    assert 12 in {c.page for c in result.citations}, (result.raw_answer, [c.page for c in result.citations])
    assert all(c.chunk_id in {h.chunk_id for h in result.chunks} for c in result.citations)

    result = pipeline.answer(OFF_TOPIC)
    assert result.refused
    assert result.answer == pipeline.cfg.refusal_text


def test_fog_cites_page_12_and_off_topic_refuses_offline(offline_cfg):
    _check_fog_and_off_topic(Pipeline(offline_cfg(retrieval_mode="vector")))


def test_golden_set_citation_and_refusal_rates_offline(offline_cfg):
    cfg = offline_cfg(retrieval_mode="vector")
    summary, rows = evaluate_answers(cfg, load_golden(), LexicalJudge())
    assert summary["cited_rate_answerable"] >= 0.9
    unanswerable = [r for r in rows if r["type"] == "unanswerable"]
    assert sum(r["refused"] for r in unanswerable) >= 4


@pytest.mark.realmodel
def test_fog_cites_page_12_and_off_topic_refuses_real(real_cfg):
    _check_fog_and_off_topic(Pipeline(real_cfg(retrieval_mode="vector")))


@pytest.mark.realmodel
def test_golden_set_citation_and_refusal_rates_real(real_cfg):
    """The issue 3 acceptance numbers with the real chat model (slow: about 2 minutes)."""
    cfg = real_cfg(retrieval_mode="vector")
    summary, rows = evaluate_answers(cfg, load_golden(), LexicalJudge())
    assert summary["cited_rate_answerable"] >= 0.9
    unanswerable = [r for r in rows if r["type"] == "unanswerable"]
    assert sum(r["refused"] for r in unanswerable) >= 4
