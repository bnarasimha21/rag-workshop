"""Issue 4: BM25, hybrid fusion and reranking on the real PDF index."""

from __future__ import annotations

import pytest

from rag.app import make_handler
from rag.pipeline import Pipeline
from rag.retrievers import Index, bm25_search, retrieve

QUERY = "bargaining based approach"  # the heading sits on printed page 13
MODES = [(mode, rerank) for mode in ("vector", "bm25", "hybrid") for rerank in (False, True)]


def check_hybrid_and_rerank(cfg_factory):
    hybrid = cfg_factory(retrieval_mode="hybrid", rerank=False)
    index = Index.open(hybrid)

    hits = retrieve(index, QUERY, hybrid)
    assert 13 in [h.printed_page for h in hits[:3]]
    # Fused hits carry the scores from both lists they came from.
    assert any("vector" in h.scores and "bm25" in h.scores for h in hits)

    reranked_cfg = cfg_factory(retrieval_mode="hybrid", rerank=True, k=4, candidate_k=20)
    reranked = retrieve(index, QUERY, reranked_cfg)
    assert len(reranked) == 4
    scores = [h.score for h in reranked]
    assert scores == sorted(scores, reverse=True)
    assert reranked[0].printed_page == 13
    assert all("rerank" in h.scores for h in reranked)


def test_hybrid_and_rerank_offline(offline_cfg):
    check_hybrid_and_rerank(offline_cfg)


@pytest.mark.realmodel
def test_hybrid_and_rerank_real(real_cfg):
    check_hybrid_and_rerank(real_cfg)  # real nomic embeddings and the MiniLM cross-encoder


def test_bm25_needs_shared_words(offline_cfg):
    index = Index.open(offline_cfg())
    assert bm25_search(index, "zzzz qqqq", 5) == []
    assert bm25_search(index, "the of and", 5) == []  # only stopwords
    hits = bm25_search(index, "naive approach", 3)
    assert hits and hits[0].printed_page == 13
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


@pytest.mark.parametrize("mode,rerank", MODES)
def test_every_mode_answers_end_to_end(offline_cfg, mode, rerank):
    cfg = offline_cfg(retrieval_mode=mode, rerank=rerank)
    result = Pipeline(cfg).answer("What is fog computing and why is it used?")
    assert not result.refused, result.refusal_reason
    assert 12 in {c.page for c in result.citations}
    off_topic = Pipeline(cfg).answer("How much does a Kubernetes cluster cost per hour on AWS?")
    assert off_topic.refused


def test_app_handler_switches_mode_and_rerank(offline_cfg):
    handle = make_handler(offline_cfg(retrieval_mode="vector", rerank=False))
    for mode, rerank in MODES:
        answer, sources, _ = handle("Explain the bargaining based approach.", mode, rerank)
        assert "page 13" in sources, (mode, rerank, sources)


def test_cli_runs_each_mode(subprocess_env):
    import subprocess
    import sys

    from conftest import REPO

    for mode, rerank in MODES:
        env = {**subprocess_env, "RAG_RETRIEVAL_MODE": mode, "RAG_RERANK": "on" if rerank else "off"}
        out = subprocess.run([sys.executable, "-m", "evals.retrieval", "--offline", "--limit", "3"],
                             cwd=REPO, env=env, capture_output=True, text=True, timeout=300)
        assert out.returncode == 0, out.stderr
        expected = mode + ("+rerank" if rerank else "")
        assert expected in out.stdout, out.stdout
