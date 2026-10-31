"""Step 3 of RAG: retrieve chunks, build a prompt, ask the LLM, check the answer.

Nothing here runs at import time: no index is built, no model is called and no
server is started until you call ``answer()``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from langchain_core.messages import HumanMessage, SystemMessage

from rag.citations import Citation, extract_citations, is_refusal_text, starts_with_refusal
from rag.config import RagConfig
from rag.models import make_chat_model
from rag.prompts import BASIC_PROMPT, GROUNDED_SYSTEM, GROUNDED_USER, format_passages
from rag.retrievers import Hit, Index, retrieve, vector_search
from rag.trace import Trace


@dataclass
class RagResult:
    question: str
    answer: str
    chunks: list[Hit]
    citations: list[Citation] = field(default_factory=list)
    refused: bool = False
    # Why we refused: "low_relevance" (nothing retrieved was close enough, the
    # LLM was not called), "model" (the LLM said the document lacks the
    # answer) or "no_citations" (the LLM answered but cited nothing valid).
    refusal_reason: str | None = None
    raw_answer: str = ""  # what the LLM actually said, before any checks
    trace: Trace = field(default_factory=Trace)  # time and tokens per stage, see rag/trace.py


class Pipeline:
    def __init__(self, cfg: RagConfig | None = None, index: Index | None = None):
        self.cfg = cfg or RagConfig.from_env()
        self._index = index
        self._llm = None

    @property
    def index(self) -> Index:
        if self._index is None:
            self._index = Index.open(self.cfg)
        return self._index

    @property
    def llm(self):
        if self._llm is None:
            self._llm = make_chat_model(self.cfg)
        return self._llm

    def messages(self, question: str, chunks: list[Hit]) -> list:
        if not self.cfg.grounded_prompt:
            context = " ".join(c.text for c in chunks)
            return [HumanMessage(BASIC_PROMPT.format(context=context, question=question))]
        return [
            SystemMessage(GROUNDED_SYSTEM.format(refusal=self.cfg.refusal_text)),
            HumanMessage(GROUNDED_USER.format(passages=format_passages(chunks), question=question)),
        ]

    def answer(self, question: str, clock=None) -> RagResult:
        cfg = self.cfg
        trace = Trace(price_per_1k_in=cfg.price_per_1k_tokens_in, price_per_1k_out=cfg.price_per_1k_tokens_out,
                      **({"clock": clock} if clock else {}))
        try:
            return self._answer(question, trace)
        finally:
            trace.finish()

    def _answer(self, question: str, trace: Trace) -> RagResult:
        cfg = self.cfg
        with trace.stage("retrieve"):
            chunks = retrieve(self.index, question, cfg, trace=trace)
        result = RagResult(question=question, answer="", chunks=chunks, trace=trace)

        if cfg.grounded_prompt:
            with trace.stage("relevance_gate"):
                too_far = self._too_far(question, chunks)
            if too_far:
                return self._refuse(result, "low_relevance")

        with trace.stage("generate") as stage:
            reply = self.llm.invoke(self.messages(question, chunks))
            usage = getattr(reply, "usage_metadata", None) or {}
            stage.tokens_in = int(usage.get("input_tokens", 0))
            stage.tokens_out = int(usage.get("output_tokens", 0))
        result.raw_answer = str(reply.content).strip()

        if not cfg.grounded_prompt:
            result.answer = result.raw_answer
            return result

        text, citations = extract_citations(result.raw_answer, chunks)
        if (is_refusal_text(text, cfg.refusal_text) and not citations) or starts_with_refusal(text, cfg.refusal_text):
            return self._refuse(result, "model")
        if not citations and cfg.require_citations:
            return self._refuse(result, "no_citations")
        result.answer, result.citations = text, citations
        return result

    def _too_far(self, question: str, chunks: list[Hit]) -> bool:
        """Groundedness gate: skip the LLM when nothing in the document is semantically close.

        It compares the question's best cosine similarity over the whole index
        with ``min_relevance``. That is the same in every retrieval mode, so
        BM25 mode is gated too, and a reranker that drops the closest chunk
        cannot cause a refusal. In plain vector mode the top hit already holds
        that number; otherwise it costs one extra top-1 vector lookup.
        An empty retrieval result always counts as too far.
        """
        if not chunks:
            return True
        if self.cfg.retrieval_mode == "vector" and not self.cfg.rerank:
            best = max(c.scores.get("vector", float("-inf")) for c in chunks)
        else:
            top = vector_search(self.index, question, 1)
            best = top[0].score if top else float("-inf")
        return best < self.cfg.min_relevance

    def _refuse(self, result: RagResult, reason: str) -> RagResult:
        result.answer = self.cfg.refusal_text
        result.refused = True
        result.refusal_reason = reason
        result.citations = []
        return result


_default: Pipeline | None = None


def answer(question: str, cfg: RagConfig | None = None) -> RagResult:
    """Convenience wrapper: ``from rag.pipeline import answer; answer("...")``."""
    global _default
    if cfg is not None:
        return Pipeline(cfg).answer(question)
    if _default is None:
        _default = Pipeline()
    return _default.answer(question)
