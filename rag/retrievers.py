"""Step 2 of RAG: find the chunks most likely to contain the answer."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from rag.config import RagConfig
from rag.ingest import _chroma, ingest, load_chunks
from rag.text import printed_page


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


def retrieve(index: Index, question: str, cfg: RagConfig, k: int | None = None) -> list[Hit]:
    return vector_search(index, question, k or cfg.k)
