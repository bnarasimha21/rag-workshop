"""Deterministic stand-ins for the real models.

The workshop runs on Ollama, but the test suite and CI must not need a model
server or any downloads. These classes implement the same LangChain interfaces
as the real ones, so the pipeline code path is identical; only the "brains" are
replaced by simple word-overlap rules. Numbers produced with them measure the
plumbing (chunking, fusion, citations, refusals), not model quality.

- ``HashingEmbeddings``: bag-of-words feature hashing, a lexical embedding.
- ``OfflineChatModel``: answers by quoting the context sentence with the most
  word overlap, cites it as ``[n]``, and refuses when nothing overlaps enough.
- ``OverlapReranker``: scores (question, passage) pairs by word overlap, with
  the same ``predict`` method as ``sentence_transformers.CrossEncoder``.
"""

from __future__ import annotations

import math
import re
import zlib
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from rag.text import tokenize


def _features(text: str) -> list[str]:
    toks = tokenize(text)
    return toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]


class HashingEmbeddings(Embeddings):
    def __init__(self, dims: int = 1024):
        self.dims = dims

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dims
        counts: dict[str, int] = {}
        for feat in _features(text):
            counts[feat] = counts.get(feat, 0) + 1
        for feat, n in counts.items():
            h = zlib.crc32(feat.encode())
            sign = 1.0 if (h >> 31) & 1 else -1.0
            vec[h % self.dims] += sign * (1.0 + math.log(n))
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def overlap_score(question: str, passage: str) -> float:
    q = set(tokenize(question))
    if not q:
        return 0.0
    p = set(tokenize(passage))
    q_bi = set(_features(question)) - q
    p_bi = set(_features(passage)) - p
    return len(q & p) / len(q) + 0.5 * len(q_bi & p_bi) / max(len(q_bi), 1)


class OverlapReranker:
    model_name = "offline-overlap"

    def predict(self, pairs: list[tuple[str, str]], **_: Any) -> list[float]:
        return [overlap_score(q, p) for q, p in pairs]


_BLOCK = re.compile(r"^\[(\d+)\] \(page (\d+)\)\n(.*?)(?=^\[\d+\] \(page \d+\)\n|\Z)", re.M | re.S)
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n\s*-\s*")


def _sentences(text: str) -> list[str]:
    out = []
    for s in _SENT_SPLIT.split(text):
        s = " ".join(s.split()).strip(" -")
        if len(s) > 20:
            out.append(s)
    return out


class OfflineChatModel(BaseChatModel):
    """An extractive "LLM" that follows the grounded prompt's rules literally."""

    refusal_text: str = "I could not find the answer in the document."
    min_overlap: int = 2
    min_fraction: float = 0.4

    @property
    def _llm_type(self) -> str:
        return "offline-extractive"

    def _generate(self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs) -> ChatResult:
        prompt = "\n".join(str(m.content) for m in messages)
        user = str(messages[-1].content)
        if "<chunk>" in user:
            text = self._contextualize(user)
        elif _BLOCK.search(user):
            text = self._answer_grounded(user)
        else:
            text = self._answer_plain(user)
        msg = AIMessage(
            content=text,
            usage_metadata={
                "input_tokens": len(prompt.split()),
                "output_tokens": len(text.split()),
                "total_tokens": len(prompt.split()) + len(text.split()),
            },
        )
        return ChatResult(generations=[ChatGeneration(message=msg)])

    # -- behaviours -------------------------------------------------------

    def _best(self, question: str, blocks: list[tuple[int, str]]) -> list[tuple[float, int, str]]:
        q = set(tokenize(question))
        scored = []
        for n, text in blocks:
            for sent in _sentences(text):
                hits = len(q & set(tokenize(sent)))
                if hits >= self.min_overlap and q and hits / len(q) >= self.min_fraction:
                    scored.append((overlap_score(question, sent), n, sent))
        scored.sort(key=lambda t: (-t[0], t[1]))
        return scored

    def _answer_grounded(self, user: str) -> str:
        question = user.rsplit("Question:", 1)[-1].strip()
        blocks = [(int(m.group(1)), m.group(3)) for m in _BLOCK.finditer(user.split("Question:")[0])]
        best = self._best(question, blocks)
        if not best:
            return self.refusal_text
        picked = [best[0]]
        for cand in best[1:]:
            if cand[0] >= 0.8 * best[0][0] and cand[2] != best[0][2]:
                picked.append(cand)
                break
        return " ".join(f"{sent} [{n}]" for _, n, sent in picked)

    def _answer_plain(self, user: str) -> str:
        m = re.search(r"Given the context: (.*), answer the question: (.*)$", user, re.S)
        if not m:
            return self.refusal_text
        best = self._best(m.group(2), [(0, m.group(1))])
        return best[0][2] if best else self.refusal_text

    def _contextualize(self, user: str) -> str:
        doc = re.search(r"<document>(.*?)</document>", user, re.S)
        page = re.search(r"<page>(\d+)</page>", user)
        heading = "this topic"
        for line in (doc.group(1) if doc else "").splitlines():
            line = line.strip()
            if 3 < len(line) < 60 and not line.startswith("-"):
                heading = line.rstrip(":")
                break
        where = f" on page {page.group(1)}" if page else ""
        return f"From the IoT module notes{where}, in the part about {heading}."
