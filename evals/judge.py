"""Judges that score an answer. Used by evals/answers.py.

We use a small hand-written judge prompt rather than Ragas. Ragas metrics ask
the judge for structured JSON in several steps (split into claims, then verify
each), which small local models often break, and the extra dependency hides
what is being measured. Here each metric is one question with a one-word
answer (YES, PARTLY or NO), which a 2B model can follow and a learner can read.

Small local judges are noisy: they miss unsupported claims and sometimes
grade a correct answer down. Treat their scores as a trend across a whole
run, and spot-check a sample with a stronger judge (``--judge anthropic`` or
``--judge openai``) before trusting a small difference.
"""

from __future__ import annotations

import math
import re

from rag.config import RagConfig
from rag.text import tokenize

FAITHFULNESS_PROMPT = """You are checking an answer against source text.

Source text:
{context}

Answer:
{answer}

Is every factual claim in the answer supported by the source text? Reply with exactly one word: YES, PARTLY or NO."""

# Calibrated against gemma2:2b on tests/test_evals_integration.py::test_judge_calibration_real.
# Leaving the question out and saying "wording may differ" matters: with the question in the
# prompt, the 2B judge marked short, fully correct answers PARTLY.
CORRECTNESS_PROMPT = """Reference answer: {reference}

Candidate answer: {answer}

Does the candidate answer contain the same information as the reference answer? Wording may differ and extra detail is fine. Reply YES, PARTLY (some of it is missing) or NO (none of it, or it contradicts the reference). One word."""

_VERDICT = re.compile(r"\b(YES|PARTLY|PARTIAL|PARTIALLY|NO)\b")
_SCORES = {"YES": 1.0, "PARTLY": 0.5, "PARTIAL": 0.5, "PARTIALLY": 0.5, "NO": 0.0}
_CITATION = re.compile(r"\s*\[\d+(?:\s*,\s*\d+)*\]")


def strip_citations(text: str) -> str:
    return _CITATION.sub("", text).strip()


def parse_verdict(text: str) -> float:
    """First YES/PARTLY/NO in the reply; NaN when the judge said none of them."""
    m = _VERDICT.search(text.upper())
    return _SCORES[m.group(1)] if m else math.nan


class LLMJudge:
    def __init__(self, chat_model, name: str):
        self.llm = chat_model
        self.name = name

    def _ask(self, prompt: str) -> float:
        from langchain_core.messages import HumanMessage

        return parse_verdict(str(self.llm.invoke([HumanMessage(prompt)]).content))

    def faithfulness(self, answer: str, contexts: list[str]) -> float:
        context = "\n\n".join(contexts)
        return self._ask(FAITHFULNESS_PROMPT.format(context=context, answer=strip_citations(answer)))

    def correctness(self, question: str, answer: str, reference: str) -> float:
        return self._ask(CORRECTNESS_PROMPT.format(question=question, reference=reference, answer=strip_citations(answer)))


class LexicalJudge:
    """Offline stand-in: word overlap instead of judgement. Deterministic, crude.

    faithfulness = share of the answer's content words found in the context
    correctness  = share of the reference answer's content words found in the answer
    """

    name = "lexical (offline heuristic)"

    def faithfulness(self, answer: str, contexts: list[str]) -> float:
        words = tokenize(strip_citations(answer))
        if not words:
            return math.nan
        context = set(tokenize(" ".join(contexts)))
        return sum(w in context for w in words) / len(words)

    def correctness(self, question: str, answer: str, reference: str) -> float:
        ref = set(tokenize(reference))
        if not ref:
            return math.nan
        return len(ref & set(tokenize(strip_citations(answer)))) / len(ref)


def make_judge(cfg: RagConfig, provider: str | None = None, model: str | None = None):
    provider = provider or cfg.judge_provider
    model = model or cfg.judge_model
    if provider == "offline":
        return LexicalJudge()
    if provider == "ollama":
        from rag.models import make_chat_model

        name = model or cfg.chat_model
        return LLMJudge(make_chat_model(cfg.with_overrides(provider="ollama"), model=name), f"ollama/{name}")
    if provider in ("anthropic", "openai"):
        if not model:
            raise ValueError(f"Set RAG_JUDGE_MODEL (or --judge-model) to the {provider} model to use as judge.")
        package = {"anthropic": "langchain-anthropic", "openai": "langchain-openai"}[provider]
        try:
            if provider == "anthropic":
                from langchain_anthropic import ChatAnthropic as Chat
            else:
                from langchain_openai import ChatOpenAI as Chat
        except ImportError as exc:
            raise RuntimeError(f"A hosted {provider} judge needs `pip install {package}` and its API key in the environment.") from exc
        return LLMJudge(Chat(model=model, temperature=0), f"{provider}/{model}")
    raise ValueError(f"Unknown judge provider {provider!r}; use ollama, offline, anthropic or openai.")
