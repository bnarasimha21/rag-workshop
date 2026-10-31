"""Issue 1: the whole pipeline end to end on the real PDF."""

from __future__ import annotations

import warnings

import pytest

from rag.pipeline import Pipeline

QUESTION = "Where can IoT data be offloaded to?"
# The answer (edge, fog, remote server, cloud) spans the end of printed page 11
# ("Offload location", "forwarded to the fog or cloud ... edge layer") and page 12.
ANSWER_PAGES = {11, 12}


def test_offline_pipeline_end_to_end(offline_cfg):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = Pipeline(offline_cfg(retrieval_mode="vector")).answer(QUESTION)
    assert result.answer.strip()
    assert ANSWER_PAGES & {c.printed_page for c in result.chunks[:3]}
    assert all(0.0 <= c.score <= 1.0 for c in result.chunks)
    assert [c.score for c in result.chunks] == sorted((c.score for c in result.chunks), reverse=True)
    deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning) and "langchain" in str(w.filename).lower()]
    assert not deprecations, [str(w.message) for w in deprecations]


@pytest.mark.realmodel
def test_real_pipeline_end_to_end(real_cfg):
    result = Pipeline(real_cfg(retrieval_mode="vector")).answer(QUESTION)
    assert result.answer.strip()
    assert ANSWER_PAGES & {c.printed_page for c in result.chunks[:3]}
