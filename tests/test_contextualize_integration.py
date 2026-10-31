"""Issue 5: contextual chunks, built from the first 3 pages of the real PDF."""

from __future__ import annotations

import json
import logging

import pytest

from rag.app import format_sources
from rag.contextualize import CACHE_FILE, ContextCache, cache_key, clean_context, contextualize
from rag.ingest import MANIFEST, embedding_text, ingest, load_chunks, load_pages, split_pages
from rag.pipeline import Pipeline
from rag.retrievers import Index


def check_contextual_ingest(cfg, caplog):
    caplog.set_level(logging.INFO)
    first = ingest(cfg)
    chunks = load_chunks(first.path)
    assert first.n_chunks == len(chunks) > 0
    assert {c.metadata["page"] for c in chunks} <= {0, 1, 2}
    assert first.context_calls == len(chunks) and first.context_cache_hits == 0
    for c in chunks:
        assert c.metadata["context"].strip(), c.metadata["chunk_id"]
        assert embedding_text(c) != c.page_content
        assert embedding_text(c).endswith(c.page_content)
    manifest = json.loads((first.path / MANIFEST).read_text())
    assert manifest["context_cost"]["llm_calls"] == len(chunks)

    # Rebuild the index from scratch: every context must come from the cache.
    caplog.clear()
    again = ingest(cfg, force=True)
    assert again.context_calls == 0
    assert again.context_cache_hits == len(chunks)
    assert again.context_tokens == 0
    assert again.context_full_tokens == first.context_full_tokens
    assert "0 LLM calls" in caplog.text
    assert [c.metadata["context"] for c in load_chunks(again.path)] == [c.metadata["context"] for c in chunks]
    return again


def test_contextual_ingest_offline(offline_cfg, tmp_path, caplog):
    cfg = offline_cfg(index_dir=tmp_path, contextual_chunks=True, max_pages=3)
    built = check_contextual_ingest(cfg, caplog)
    assert (tmp_path / CACHE_FILE).exists()

    # The vector store holds the enriched text, but answers cite the original chunk text.
    index = Index(cfg, built.path)
    stored = index.store.get(include=["documents", "metadatas"])
    for doc, meta in zip(stored["documents"], stored["metadatas"]):
        assert doc == f"{meta['context']}\n\n{meta['text']}"
    result = Pipeline(cfg, index=index).answer("What is unstructured data?")
    assert not result.refused, result.refusal_reason
    sources = format_sources(result)
    for cit in result.citations:
        chunk = index.by_id[cit.chunk_id]
        assert cit.text == chunk.page_content
        assert chunk.metadata["context"] not in sources


@pytest.mark.realmodel
def test_contextual_ingest_real(real_cfg, tmp_path, caplog):
    check_contextual_ingest(real_cfg(index_dir=tmp_path, contextual_chunks=True, max_pages=3), caplog)


def test_contextual_flag_changes_the_index_folder(offline_cfg, tmp_path):
    plain = ingest(offline_cfg(index_dir=tmp_path, max_pages=3))
    enriched = ingest(offline_cfg(index_dir=tmp_path, max_pages=3, contextual_chunks=True))
    assert plain.path != enriched.path
    assert all("context" not in c.metadata for c in load_chunks(plain.path))


class CountingModel:
    """Stands in for the chat model and counts calls."""

    def __init__(self, reply):
        self.reply, self.calls = reply, 0

    def invoke(self, messages):
        from langchain_core.messages import AIMessage

        self.calls += 1
        return AIMessage(self.reply, usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15})


def test_cache_is_keyed_by_model_and_provider_and_skips_empty_replies(offline_cfg, tmp_path):
    cfg = offline_cfg(index_dir=tmp_path)
    pages = load_pages(cfg.pdf_path, max_pages=1)
    chunks = split_pages(pages, cfg)

    empty = CountingModel("   ")
    contextualize(chunks, pages, cfg, llm=empty)
    assert empty.calls == len(chunks)
    assert ContextCache(tmp_path / CACHE_FILE).data == {}  # nothing cached, so the next run retries

    good = CountingModel('"Part of the section on data formats."')
    stats = contextualize(chunks, pages, cfg, llm=good)
    assert good.calls == len(chunks) and stats.tokens == 15 * len(chunks)
    assert chunks[0].metadata["context"] == "Part of the section on data formats."

    again = CountingModel("unused")
    stats = contextualize(chunks, pages, cfg, llm=again)
    assert again.calls == 0 and stats.cache_hits == len(chunks) and stats.full_tokens == 15 * len(chunks)

    other_model = CountingModel("x")
    contextualize(chunks, pages, cfg.with_overrides(context_model="another-model"), llm=other_model)
    assert other_model.calls == len(chunks)

    key_offline = cache_key("offline:gemma2:2b", "page", "chunk")
    key_real = cache_key("ollama:gemma2:2b", "page", "chunk")
    assert key_offline != key_real


def test_clean_context():
    assert clean_context('  "Line one.\n Line two."  ') == "Line one. Line two."
    long = "First sentence. " + "word " * 200
    assert clean_context(long) == "First sentence."
