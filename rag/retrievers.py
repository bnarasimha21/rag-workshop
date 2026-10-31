"""Step 2 of RAG: find the chunks most likely to contain the answer."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

from rag.config import RagConfig
from rag.ingest import _chroma, embedding_text, ingest, load_chunks
from rag.text import printed_page, tokenize


@dataclass
class Hit:
    """One retrieved chunk. ``text`` is always the original chunk text."""

    chunk_id: str
    page: int  # 0-indexed, as stored
    text: str
    score: float
    context: str = ""
    scores: dict = field(default_factory=dict)

    @property
    def printed_page(self) -> int:
        return printed_page(self.page)


class Index:
    """The stored chunks plus the search structures built over them."""

    def __init__(self, cfg: RagConfig, path: Path):
        self.cfg = cfg
        self.path = Path(path)
        self.chunks = load_chunks(self.path)
        self.by_id = {c.metadata["chunk_id"]: c for c in self.chunks}
        self.store = _chroma(cfg, self.path)

    @classmethod
    def open(cls, cfg: RagConfig) -> "Index":
        """Open the index for ``cfg``, building it first if needed."""
        return cls(cfg, ingest(cfg).path)

    @cached_property
    def bm25(self):
        """A BM25 index over the same chunks, rebuilt in memory on open (it takes milliseconds).

        It indexes the same text that was embedded, so contextual chunks (issue 5)
        help keyword search too.
        """
        from rank_bm25 import BM25Okapi

        return BM25Okapi([tokenize(embedding_text(c)) or ["_empty_"] for c in self.chunks])

    def hit(self, chunk_id: str, score: float, **scores) -> Hit:
        c = self.by_id[chunk_id]
        return Hit(
            chunk_id=chunk_id,
            page=int(c.metadata["page"]),
            text=c.page_content,
            context=c.metadata.get("context", ""),
            score=float(score),
            scores=dict(scores),
        )


def vector_search(index: Index, question: str, k: int) -> list[Hit]:
    """Nearest chunks by cosine similarity of their embeddings (1.0 = identical)."""
    results = index.store.similarity_search_with_relevance_scores(question, k=k)
    return [index.hit(doc.metadata["chunk_id"], score, vector=score) for doc, score in results]


def bm25_search(index: Index, question: str, k: int) -> list[Hit]:
    """Chunks that share the question's words, scored by BM25. Chunks sharing no word are left out."""
    terms = tokenize(question)
    if not terms:
        return []
    scores = index.bm25.get_scores(terms)
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return [
        index.hit(index.chunks[i].metadata["chunk_id"], scores[i], bm25=float(scores[i]))
        for i in ranked[:k]
        if scores[i] > 0
    ]


def reciprocal_rank_fusion(ranked_lists: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Merge several ranked lists of ids into one.

    Each list gives an id the score 1 / (k + rank), rank starting at 1, and an
    id's scores from all lists are added up. Only ranks matter, so BM25 scores
    and cosine similarities never have to be put on the same scale. The constant
    k (60 in the original paper) stops the top rank of one list from drowning
    out everything else. Ties keep the order in which ids were first seen.
    """
    fused: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank)
    # sorted() is stable, so equal scores stay in first-seen order.
    return sorted(fused.items(), key=lambda item: item[1], reverse=True)


def hybrid_search(index: Index, question: str, k: int, pool: int, rrf_k: int = 60) -> list[Hit]:
    """Vector and BM25 search, each returning ``pool`` chunks, fused with RRF."""
    vector = vector_search(index, question, pool)
    keyword = bm25_search(index, question, pool)
    scores: dict[str, dict] = {}
    for h in vector + keyword:
        scores.setdefault(h.chunk_id, {}).update(h.scores)
    fused = reciprocal_rank_fusion([[h.chunk_id for h in vector], [h.chunk_id for h in keyword]], k=rrf_k)
    return [index.hit(cid, score, rrf=score, **scores[cid]) for cid, score in fused[:k]]


def rerank(question: str, hits: list[Hit], top_n: int, reranker) -> list[Hit]:
    """Score each (question, chunk) pair with a cross-encoder and keep the best ``top_n``.

    A cross-encoder reads the question and the chunk together, so it judges
    relevance far better than comparing two separate embeddings, but it is too
    slow to run over every chunk. So we run it only on a wider candidate pool.
    """
    if not hits:
        return []
    passages = [f"{h.context}\n\n{h.text}" if h.context else h.text for h in hits]
    scores = reranker.predict([(question, p) for p in passages])
    order = sorted(range(len(hits)), key=lambda i: float(scores[i]), reverse=True)[:top_n]
    out = []
    for i in order:
        h = hits[i]
        out.append(Hit(h.chunk_id, h.page, h.text, float(scores[i]), h.context, {**h.scores, "rerank": float(scores[i])}))
    return out


def retrieve(index: Index, question: str, cfg: RagConfig, k: int | None = None, reranker=None, trace=None) -> list[Hit]:
    """Run the retrieval stack chosen in ``cfg``: vector, bm25 or hybrid, then optionally rerank.

    With rerank on, the first stage returns ``candidate_k`` chunks and the
    reranker keeps the best ``k``. Pass a ``rag.trace.Trace`` to time the
    search and rerank steps separately.
    """
    from contextlib import nullcontext

    def stage(name):
        return trace.stage(name) if trace is not None else nullcontext()

    k = k or cfg.k
    pool = max(cfg.candidate_k, k) if cfg.rerank else k
    if cfg.retrieval_mode not in ("vector", "bm25", "hybrid"):
        raise ValueError(f"unknown retrieval_mode {cfg.retrieval_mode!r}")
    with stage("search"):
        if cfg.retrieval_mode == "vector":
            hits = vector_search(index, question, pool)
        elif cfg.retrieval_mode == "bm25":
            hits = bm25_search(index, question, pool)
        else:
            hits = hybrid_search(index, question, pool, pool=max(cfg.candidate_k, pool), rrf_k=cfg.rrf_k)
    if cfg.rerank:
        with stage("rerank"):
            if reranker is None:
                from rag.models import make_reranker

                reranker = make_reranker(cfg)
            hits = rerank(question, hits, k, reranker)
    return hits
