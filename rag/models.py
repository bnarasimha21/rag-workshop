"""Builds the model objects for a config. Nothing here talks to a server until used."""

from __future__ import annotations

from functools import lru_cache

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel

from rag.config import RagConfig


def make_chat_model(cfg: RagConfig, model: str | None = None) -> BaseChatModel:
    if cfg.provider == "offline":
        from rag.offline import OfflineChatModel

        return OfflineChatModel(refusal_text=cfg.refusal_text)
    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=model or cfg.chat_model,
        base_url=cfg.ollama_base_url,
        temperature=cfg.temperature,
        num_ctx=cfg.num_ctx,
    )


# Some embedding models are trained with a task prefix and retrieve noticeably
# worse without it. nomic-embed-text expects these two (see its model card).
TASK_PREFIXES = {
    "nomic-embed-text": ("search_query: ", "search_document: "),
}


class PrefixedEmbeddings(Embeddings):
    """Adds the model's query/document prefixes before embedding."""

    def __init__(self, base: Embeddings, query_prefix: str, doc_prefix: str):
        self.base, self.query_prefix, self.doc_prefix = base, query_prefix, doc_prefix

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.base.embed_documents([self.doc_prefix + t for t in texts])

    def embed_query(self, text: str) -> list[float]:
        return self.base.embed_query(self.query_prefix + text)


def make_embeddings(cfg: RagConfig) -> Embeddings:
    if cfg.provider == "offline":
        from rag.offline import HashingEmbeddings

        return HashingEmbeddings()
    from langchain_ollama import OllamaEmbeddings

    base = OllamaEmbeddings(model=cfg.embed_model, base_url=cfg.ollama_base_url)
    prefixes = TASK_PREFIXES.get(cfg.embed_model.split(":")[0])
    return PrefixedEmbeddings(base, *prefixes) if prefixes else base


def make_reranker(cfg: RagConfig):
    """Anything with ``predict(list[(question, passage)]) -> list[float]``."""
    if cfg.provider == "offline":
        from rag.offline import OverlapReranker

        return OverlapReranker()
    return _cross_encoder(cfg.reranker_model)


@lru_cache(maxsize=2)
def _cross_encoder(name: str):
    # Imported lazily: sentence-transformers pulls in torch, which takes a few
    # seconds to import and is only needed when reranking is switched on.
    from sentence_transformers import CrossEncoder

    model = CrossEncoder(name, device="cpu")
    model.model_name = name
    return model
