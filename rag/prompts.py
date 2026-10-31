"""The prompts. Kept in one file so learners can read and tweak them together."""

from __future__ import annotations

# The original workshop prompt, kept so the evals can show what changed.
BASIC_PROMPT = "Given the context: {context}, answer the question: {question}"

GROUNDED_SYSTEM = """You answer questions about a document using only the numbered context passages you are given.

Rules:
1. Use only facts stated in the context passages. Do not add outside knowledge.
2. End every sentence that states a fact with the number of the passage it came from, in square brackets, for example [1] or [2].
3. If the passages do not contain the answer, reply with exactly this sentence and nothing else: {refusal}
4. Keep the answer short: one to four sentences."""

GROUNDED_USER = """Context passages:

{passages}

Question: {question}"""


def format_passages(chunks) -> str:
    """Number each chunk as ``[n] (page p)`` so the model can cite it."""
    return "\n\n".join(f"[{n}] (page {c.printed_page})\n{c.text}" for n, c in enumerate(chunks, start=1))
