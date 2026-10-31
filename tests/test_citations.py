"""Issue 3 unit tests: citation parsing, hallucinated markers and the refusal paths."""

from __future__ import annotations

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from rag.citations import extract_citations, is_refusal_text, is_bare_refusal
from rag.pipeline import Pipeline
from rag.prompts import GROUNDED_SYSTEM, format_passages
from rag.retrievers import Hit


def hits(n: int) -> list[Hit]:
    return [Hit(chunk_id=f"p{12 + i:02d}-c00", page=11 + i, text=f"chunk text {i}", score=0.9, scores={"vector": 0.9}) for i in range(n)]


def test_extract_keeps_valid_and_strips_invalid_markers():
    text, cites = extract_citations("Fog is local [2]. Cloud is remote [9]. Both [1, 7].", hits(3))
    assert text == "Fog is local [2]. Cloud is remote. Both [1]."
    assert [(c.marker, c.page, c.chunk_id) for c in cites] == [(2, 13, "p13-c00"), (1, 12, "p12-c00")]
    assert cites[0].text == "chunk text 1"


def test_extract_handles_adjacent_and_zero_markers():
    text, cites = extract_citations("A [1][3] B [0].", hits(3))
    assert text == "A [1][3] B."
    assert [c.marker for c in cites] == [1, 3]


def test_refusal_detection_ignores_case_and_punctuation():
    refusal = "I could not find the answer in the document."
    assert is_refusal_text("i could not find the answer in the document", refusal)
    assert is_refusal_text("Sorry. I could NOT find the answer in the document!", refusal)
    assert not is_refusal_text("Fog computing is decentralized.", refusal)


def test_prompt_numbers_chunks_with_printed_pages():
    passages = format_passages(hits(2))
    assert passages.startswith("[1] (page 12)\nchunk text 0")
    assert "[2] (page 13)\nchunk text 1" in passages
    assert "I could not find" in GROUNDED_SYSTEM.format(refusal="I could not find it.")


def _pipeline_with(offline_cfg, reply: str, **overrides) -> Pipeline:
    p = Pipeline(offline_cfg(retrieval_mode="vector", **overrides))
    p._llm = FakeListChatModel(responses=[reply])
    return p


def test_fake_llm_citing_a_chunk_that_was_not_retrieved(offline_cfg):
    """The model cites [9] but saw only k=4 chunks: [9] is removed, and with
    nothing valid left the answer is withheld."""
    p = _pipeline_with(offline_cfg, "Fog computing reduces latency [9].")
    result = p.answer("What is fog computing?")
    assert result.raw_answer == "Fog computing reduces latency [9]."
    assert result.citations == []
    assert result.refused and result.refusal_reason == "no_citations"

    p = _pipeline_with(offline_cfg, "Fog computing reduces latency [9].", require_citations=False)
    result = p.answer("What is fog computing?")
    assert result.answer == "Fog computing reduces latency."
    assert not result.refused and result.citations == []


def test_every_citation_points_at_a_retrieved_chunk(offline_cfg):
    p = _pipeline_with(offline_cfg, "Fog is decentralized [1]. Also [4] and [5] and [12].")
    result = p.answer("What is fog computing?")
    retrieved = {c.chunk_id for c in result.chunks}
    assert {c.chunk_id for c in result.citations} <= retrieved
    assert [c.marker for c in result.citations] == [1, 4]
    assert "[5]" not in result.answer and "[12]" not in result.answer


def test_model_refusal_is_reported(offline_cfg):
    cfg = offline_cfg()
    p = _pipeline_with(offline_cfg, cfg.refusal_text)
    result = p.answer("What is fog computing?")
    assert result.refused and result.refusal_reason == "model"
    assert result.answer == cfg.refusal_text


def test_low_relevance_skips_the_llm(offline_cfg):
    p = _pipeline_with(offline_cfg, "should never be used [1]", min_relevance=0.99)
    result = p.answer("What is fog computing?")
    assert result.refused and result.refusal_reason == "low_relevance"
    assert result.trace.ms("generate") is None
    assert result.raw_answer == ""


def test_basic_prompt_has_no_grounding_checks(offline_cfg):
    p = _pipeline_with(offline_cfg, "Anything at all.", grounded_prompt=False, min_relevance=0.99)
    result = p.answer("What is fog computing?")
    assert result.answer == "Anything at all." and not result.refused


def test_years_in_brackets_are_not_citations():
    text, cites = extract_citations("Standardized in [2014], see [1].", hits(3))
    assert text == "Standardized in [2014], see [1]."
    assert [c.marker for c in cites] == [1]


def test_empty_refusal_text_never_matches():
    assert not is_refusal_text("Fog computing is decentralized.", "")
    assert not is_bare_refusal("Fog computing is decentralized.", "")


def test_cited_refusal_is_still_a_refusal(offline_cfg):
    """Small models append [1] to everything, including the refusal sentence."""
    cfg = offline_cfg()
    p = _pipeline_with(offline_cfg, cfg.refusal_text + " [1]")
    result = p.answer("What is fog computing?")
    assert result.refused and result.refusal_reason == "model"
    assert result.citations == []

    # An answer that mentions the sentence later, with a citation, is kept.
    p = _pipeline_with(offline_cfg, f"Fog is decentralized [1]. Beyond that, {cfg.refusal_text.lower()}")
    assert not p.answer("What is fog computing?").refused


def test_eval_counts_a_basic_prompt_refusal(offline_cfg):
    """The basic prompt has no refusal logic in the pipeline, so the eval must read the text."""
    from evals.answers import is_refusal

    cfg = offline_cfg(grounded_prompt=False)
    p = _pipeline_with(offline_cfg, cfg.refusal_text, grounded_prompt=False)
    result = p.answer("What is fog computing?")
    assert not result.refused
    assert is_refusal(cfg, result)
    p = _pipeline_with(offline_cfg, "Fog is decentralized.", grounded_prompt=False)
    assert not is_refusal(cfg, p.answer("What is fog computing?"))


def test_partial_answer_opening_with_the_refusal_is_kept(offline_cfg):
    cfg = offline_cfg()
    reply = f"{cfg.refusal_text} for the cost, but fog nodes sit near the devices [1]."
    p = _pipeline_with(offline_cfg, reply)
    result = p.answer("What is fog computing?")
    assert not result.refused and [c.marker for c in result.citations] == [1]


def test_refusal_text_check_keeps_word_breaks_across_newlines():
    assert not is_refusal_text("not find the answer in the document.\nFog is local.", "the documentfog")
    assert is_refusal_text("Sorry.\nI could not find the answer in the document.", "I could not find the answer in the document.")


def test_eval_trusts_the_grounded_pipeline(offline_cfg):
    """A kept grounded answer that mentions the refusal sentence for one part is an answer."""
    from evals.answers import is_refusal

    cfg = offline_cfg()
    p = _pipeline_with(offline_cfg, f"Fog is decentralized [1]. {cfg.refusal_text}")
    result = p.answer("What is fog computing?")
    assert not result.refused
    assert not is_refusal(cfg, result)
