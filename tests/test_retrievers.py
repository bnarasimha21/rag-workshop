"""Unit tests for the retrieval building blocks (no index, no models)."""

from __future__ import annotations

import pytest

from rag.retrievers import Hit, reciprocal_rank_fusion, rerank


def test_rrf_adds_one_over_k_plus_rank_from_each_list():
    fused = dict(reciprocal_rank_fusion([["a", "b", "c"], ["b", "a", "d"]], k=60))
    assert fused["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert fused["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert fused["c"] == pytest.approx(1 / 63)
    assert fused["d"] == pytest.approx(1 / 63)


def test_rrf_document_in_only_one_list_still_ranks_but_below_shared_ones():
    order = [doc for doc, _ in reciprocal_rank_fusion([["x", "shared"], ["shared", "y"]], k=60)]
    assert order[0] == "shared"
    assert set(order) == {"x", "shared", "y"}


def test_rrf_ties_keep_first_seen_order():
    # a and b tie (each rank 1 in one list, rank 2 in the other); c and d tie at rank 3.
    order = [doc for doc, _ in reciprocal_rank_fusion([["a", "b", "c"], ["b", "a", "d"]], k=60)]
    assert order == ["a", "b", "c", "d"]


def test_rrf_smaller_k_favours_the_top_of_each_list_more():
    lists = [["a", "b", "c"], ["c", "x", "y"], ["b", "z", "w"]]
    assert reciprocal_rank_fusion(lists, k=1)[0][0] == "b"
    fused = dict(reciprocal_rank_fusion(lists, k=1))
    assert fused["b"] == pytest.approx(1 / 3 + 1 / 2)


def test_rrf_empty_input():
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []


class FixedReranker:
    def __init__(self, scores):
        self.scores = scores

    def predict(self, pairs):
        return [self.scores[p] for _, p in pairs]


def test_rerank_orders_by_reranker_score_and_keeps_top_n():
    hits = [Hit(f"c{i}", 0, f"text {i}", score=1.0, scores={"vector": 1.0}) for i in range(4)]
    out = rerank("q", hits, 2, FixedReranker({"text 0": 0.1, "text 1": 0.9, "text 2": 0.5, "text 3": -1.0}))
    assert [h.chunk_id for h in out] == ["c1", "c2"]
    assert [h.score for h in out] == [0.9, 0.5]
    assert out[0].scores == {"vector": 1.0, "rerank": 0.9}
    assert rerank("q", [], 3, FixedReranker({})) == []
