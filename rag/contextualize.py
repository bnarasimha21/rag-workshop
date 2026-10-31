"""Contextual chunks: give every chunk a sentence that says where it belongs.

A chunk such as "Size: ... Energy: ..." loses the fact that it is about IoT
device design once it is cut out of its page. Before embedding, we ask the
chat model for one or two sentences that place the chunk in the document and
store them in ``metadata["context"]``. Ingest then embeds and BM25-indexes
``context + chunk`` (see ``rag.ingest.embedding_text``), while citations keep
showing the original chunk text.

This is the "contextual retrieval" idea Anthropic described in September 2024.
Each chunk costs one LLM call, so results are cached on disk: a second ingest
of the same PDF with the same model makes no calls at all.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

from rag.config import RagConfig
from rag.text import printed_page

log = logging.getLogger("rag.contextualize")

# Bump this whenever CONTEXT_PROMPT changes, so cached contexts written with the
# old prompt are not reused.
PROMPT_VERSION = 1

# The whole PDF does not fit in a small local model's context window, so the
# "document" we show the model is the page the chunk came from.
CONTEXT_PROMPT = """<document>
{page_text}
</document>
The document above is page <page>{page}</page> of lecture notes on the Internet of Things. Here is a chunk from that page:
<chunk>
{chunk}
</chunk>
Write one or two short sentences that situate this chunk within the page, naming the section and topic it belongs to, so the chunk is easier to find in a search. Answer with only those sentences."""

CACHE_FILE = "context_cache.json"
MAX_CONTEXT_CHARS = 400


def cache_path(cfg: RagConfig) -> Path:
    """One cache for all indexes under ``index_dir``, so changing an unrelated setting keeps it."""
    return Path(cfg.index_dir) / CACHE_FILE


def cache_key(model: str, page_text: str, chunk_text: str) -> str:
    blob = json.dumps([PROMPT_VERSION, model, page_text, chunk_text]).encode()
    return hashlib.sha256(blob).hexdigest()


def clean_context(text: str) -> str:
    """One tidy line: no surrounding quotes, no line breaks, not absurdly long."""
    text = re.sub(r"\s+", " ", text).strip().strip('"').strip()
    if len(text) > MAX_CONTEXT_CHARS:
        cut = text[:MAX_CONTEXT_CHARS]
        text = cut[: cut.rfind(". ") + 1] if ". " in cut else cut
    return text


class ContextCache:
    """``key -> {"context": str, "tokens": int, "seconds": float}`` in a JSON file.

    Each entry remembers what generating it cost, so a cached rebuild can still
    report the one-time cost of contextualizing the whole document.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        try:
            data = json.loads(self.path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            data = {}
        self.data: dict[str, dict] = {k: v for k, v in data.items() if isinstance(v, dict) and v.get("context")}

    def get(self, key: str) -> dict | None:
        return self.data.get(key)

    def put(self, key: str, context: str, tokens: int, seconds: float) -> None:
        self.data[key] = {"context": context, "tokens": tokens, "seconds": round(seconds, 3)}

    def save(self) -> None:
        # Merge with what is on disk (another process may have added entries), then
        # write to a temp file and rename, so a crash never leaves half a JSON file.
        merged = {**ContextCache(self.path).data, **self.data}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(merged, indent=0, sort_keys=True))
        tmp.replace(self.path)


@dataclass
class ContextStats:
    chunks: int = 0
    calls: int = 0          # LLM calls made in this run
    cache_hits: int = 0
    tokens: int = 0         # tokens spent in this run
    seconds: float = 0.0    # wall time of this run
    full_tokens: int = 0    # what contextualizing every chunk cost when it was first done
    full_seconds: float = 0.0


def contextualize(chunks: list[Document], pages: list[Document], cfg: RagConfig, llm=None,
                  cache: ContextCache | None = None) -> ContextStats:
    """Set ``metadata["context"]`` on every chunk, calling the model only for chunks not in the cache."""
    start = time.perf_counter()
    model = cfg.effective_context_model
    if llm is None:
        from rag.models import make_chat_model

        llm = make_chat_model(cfg, model=model)
    cache = cache or ContextCache(cache_path(cfg))
    page_text = {p.metadata["page"]: p.page_content for p in pages}
    stats = ContextStats(chunks=len(chunks))
    for chunk in chunks:
        page = chunk.metadata["page"]
        # The provider is part of the key so offline stand-in contexts never pass for real ones.
        key = cache_key(f"{cfg.provider}:{model}", page_text[page], chunk.page_content)
        entry = cache.get(key)
        if entry is None:
            prompt = CONTEXT_PROMPT.format(page_text=page_text[page], page=printed_page(page), chunk=chunk.page_content)
            t0 = time.perf_counter()
            reply = llm.invoke([HumanMessage(prompt)])
            elapsed = time.perf_counter() - t0
            usage = getattr(reply, "usage_metadata", None) or {}
            tokens = int(usage.get("input_tokens", 0)) + int(usage.get("output_tokens", 0))
            entry = {"context": clean_context(str(reply.content)), "tokens": tokens, "seconds": elapsed}
            stats.calls += 1
            stats.tokens += tokens
            if entry["context"]:  # never cache an empty reply; try again next time
                cache.put(key, entry["context"], tokens, elapsed)
            if stats.calls % 10 == 0:
                cache.save()  # keep progress if a long run is interrupted
        else:
            stats.cache_hits += 1
        stats.full_tokens += int(entry.get("tokens", 0))
        stats.full_seconds += float(entry.get("seconds", 0.0))
        chunk.metadata["context"] = entry["context"]
    if stats.calls:
        cache.save()
    stats.seconds = time.perf_counter() - start
    log.info("Contextualized %d chunks with %s: %d LLM calls, %d cache hits, %d tokens, %.1fs.",
             stats.chunks, model, stats.calls, stats.cache_hits, stats.tokens, stats.seconds)
    return stats
