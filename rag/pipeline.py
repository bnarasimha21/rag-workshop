"""Step 3 of RAG: retrieve chunks, build a prompt, ask the LLM.

Nothing here runs at import time: no index is built, no model is called and no
server is started until you call ``answer()``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from langchain_core.messages import HumanMessage

from rag.config import RagConfig
from rag.models import make_chat_model
from rag.retrievers import Hit, Index, retrieve

BASIC_PROMPT = "Given the context: {context}, answer the question: {question}"


@dataclass
class RagResult:
    question: str
    answer: str
    chunks: list[Hit]
    timings: dict = field(default_factory=dict)  # stage name -> milliseconds


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

    def answer(self, question: str) -> RagResult:
        timings = {}
        t0 = time.perf_counter()
        chunks = retrieve(self.index, question, self.cfg)
        timings["retrieve"] = (time.perf_counter() - t0) * 1000

        context = " ".join(c.text for c in chunks)
        t0 = time.perf_counter()
        reply = self.llm.invoke([HumanMessage(BASIC_PROMPT.format(context=context, question=question))])
        timings["generate"] = (time.perf_counter() - t0) * 1000
        return RagResult(question, str(reply.content).strip(), chunks, timings)


_default: Pipeline | None = None


def answer(question: str, cfg: RagConfig | None = None) -> RagResult:
    """Convenience wrapper: ``from rag.pipeline import answer; answer("...")``."""
    global _default
    if cfg is not None:
        return Pipeline(cfg).answer(question)
    if _default is None:
        _default = Pipeline()
    return _default.answer(question)
