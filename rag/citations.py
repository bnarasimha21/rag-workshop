"""Turn ``[n]`` markers in an answer into page references, and drop fake ones."""

from __future__ import annotations

import re
from dataclasses import dataclass

# [1], [2, 3] and [1][2] are all accepted.
_MARKER = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


@dataclass
class Citation:
    marker: int        # the n in [n]
    chunk_id: str
    page: int          # printed page number
    text: str          # the original chunk text, never the generated context


def extract_citations(answer: str, chunks) -> tuple[str, list[Citation]]:
    """Return the answer with invalid markers removed, and the valid citations.

    A marker is valid only if it points at a chunk that was actually given to
    the model. A small model will sometimes write [9] when it saw four chunks;
    that is a cheap, reliable hallucination signal, so we strip it.
    """
    cited: dict[int, Citation] = {}

    def keep_valid(match: re.Match) -> str:
        numbers = [int(n) for n in match.group(1).split(",")]
        valid = [n for n in numbers if 1 <= n <= len(chunks)]
        for n in valid:
            if n not in cited:
                c = chunks[n - 1]
                cited[n] = Citation(n, c.chunk_id, c.printed_page, c.text)
        return "[" + ", ".join(map(str, valid)) + "]" if valid else ""

    cleaned = _MARKER.sub(keep_valid, answer)
    cleaned = re.sub(r"[ \t]+([.,;:])", r"\1", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned).strip()
    return cleaned, list(cited.values())


def is_refusal_text(answer: str, refusal: str) -> bool:
    norm = lambda s: re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()
    return norm(refusal) in norm(answer)
