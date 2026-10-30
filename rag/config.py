"""All the knobs of the pipeline in one place.

Every field can be overridden with an environment variable named ``RAG_<FIELD>``,
for example ``RAG_RETRIEVAL_MODE=hybrid`` or ``RAG_RERANK=on``. Each workshop step
is meant to be one of these changes, so learners can compare runs side by side.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from dataclasses import dataclass, fields, replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PDF = REPO_ROOT / "3rd module notes.pdf"
DEFAULT_INDEX_DIR = REPO_ROOT / ".rag_index"

RETRIEVAL_MODES = ("vector", "bm25", "hybrid")

# Bump when ingest changes what it stores (cleaning, prefixes, metadata), so
# old indexes on disk are rebuilt instead of reused.
INDEX_FORMAT = 1

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


@dataclass(frozen=True)
class RagConfig:
    # Document and index location
    pdf_path: Path = DEFAULT_PDF
    index_dir: Path = DEFAULT_INDEX_DIR

    # Models. "ollama" is the real, local default. "offline" swaps in the
    # deterministic stand-ins from rag/offline.py so tests and CI need no models.
    provider: str = "ollama"
    chat_model: str = "gemma2:2b"
    embed_model: str = "nomic-embed-text"
    ollama_base_url: str = "http://localhost:11434"
    temperature: float = 0.0
    num_ctx: int = 4096

    # Chunking. 800 characters is roughly one or two bullet points of this PDF,
    # small enough to be specific, big enough to keep a definition together.
    chunk_size: int = 800
    chunk_overlap: int = 120

    # Retrieval
    k: int = 4                      # chunks handed to the LLM
    retrieval_mode: str = "vector"  # vector | bm25 | hybrid
    rerank: bool = False
    candidate_k: int = 20           # pool size before fusion / reranking
    rrf_k: int = 60                 # reciprocal rank fusion constant
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Contextual chunk enrichment (issue 5)
    contextual_chunks: bool = False
    context_model: str = ""         # empty means "use chat_model"

    # Grounding (issue 3)
    grounded_prompt: bool = True    # False reproduces the original one-line prompt
    require_citations: bool = True  # refuse when the answer cites nothing valid
    min_relevance: float = 0.55     # refuse before generation below this vector score
    refusal_text: str = "I could not find the answer in the document."

    # Cost estimate for a hosted model (issue 6). Local Ollama is free, so 0.
    price_per_1k_tokens_in: float = 0.0
    price_per_1k_tokens_out: float = 0.0

    # Answer-eval judge (issue 2): ollama | offline | anthropic | openai
    judge_provider: str = "ollama"
    judge_model: str = ""           # empty means "use chat_model" for ollama

    def __post_init__(self) -> None:
        if self.retrieval_mode not in RETRIEVAL_MODES:
            raise ValueError(
                f"retrieval_mode must be one of {RETRIEVAL_MODES}, got {self.retrieval_mode!r}"
            )
        if self.provider not in ("ollama", "offline"):
            raise ValueError(f"provider must be 'ollama' or 'offline', got {self.provider!r}")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if self.k < 1 or self.candidate_k < 1:
            raise ValueError("k and candidate_k must be at least 1")

    # ---- derived values -------------------------------------------------

    @property
    def effective_context_model(self) -> str:
        return self.context_model or self.chat_model

    @property
    def embed_name(self) -> str:
        return "hashing" if self.provider == "offline" else self.embed_model

    @property
    def name(self) -> str:
        """Short label used for eval result files, e.g. ``hybrid+rerank+ctx``."""
        parts = [self.retrieval_mode]
        if self.rerank:
            parts.append("rerank")
        if self.contextual_chunks:
            parts.append("ctx")
        return "+".join(parts)

    def index_fingerprint(self) -> str:
        """Everything that changes what is stored in the index.

        Two configs with the same fingerprint can share an index; anything else
        gets its own folder, so switching a flag never silently reuses stale vectors.
        """
        relevant = {
            "format": INDEX_FORMAT,
            "provider": self.provider,
            "embed": self.embed_name,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "contextual": self.contextual_chunks,
            "context_model": self.effective_context_model if self.contextual_chunks else "",
        }
        blob = json.dumps(relevant, sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:12]

    def with_overrides(self, **kwargs) -> "RagConfig":
        return replace(self, **kwargs)

    # ---- construction from the environment ------------------------------

    @classmethod
    def from_env(cls, environ: dict | None = None, defaults: dict | None = None, **overrides) -> "RagConfig":
        """Build a config: ``defaults``, then ``RAG_*`` variables, then ``overrides``."""
        env = os.environ if environ is None else environ
        values: dict = dict(defaults or {})
        for f in fields(cls):
            raw = env.get(f"RAG_{f.name.upper()}")
            if raw is not None:
                values[f.name] = _parse(f, raw)
        values.update(overrides)
        return cls(**values)


# Settings for the offline stand-in models (rag/offline.py). The hashing
# embeddings give lower similarity scores than nomic-embed-text, so the
# pre-generation relevance gate needs its own threshold, tuned on the golden set.
OFFLINE_DEFAULTS = {"provider": "offline", "min_relevance": 0.12, "judge_provider": "offline"}


def offline_config(**overrides) -> RagConfig:
    """Config used by the test suite and CI: no Ollama, no downloads."""
    return RagConfig(**{**OFFLINE_DEFAULTS, **overrides})


def cli_config(offline: bool = False, **overrides) -> RagConfig:
    """Config for command-line tools: environment variables, optionally offline."""
    return RagConfig.from_env(defaults=OFFLINE_DEFAULTS if offline else None, **overrides)


def _parse(f: dataclasses.Field, raw: str):
    kind = f.type if isinstance(f.type, str) else getattr(f.type, "__name__", str(f.type))
    if kind == "bool":
        low = raw.strip().lower()
        if low in _TRUE:
            return True
        if low in _FALSE:
            return False
        raise ValueError(f"RAG_{f.name.upper()} must be on/off, got {raw!r}")
    if kind == "int":
        return int(raw)
    if kind == "float":
        return float(raw)
    if kind == "Path":
        return Path(raw).expanduser()
    return raw
