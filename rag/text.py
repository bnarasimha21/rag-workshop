"""Small text helpers shared by ingest, BM25 and the offline stand-ins."""

from __future__ import annotations

import re

# PyPDFLoader stores 0-indexed page numbers; people (and the golden set, and
# citations) use the printed page number. This is the one place that converts.
def printed_page(meta_page: int) -> int:
    return int(meta_page) + 1


def meta_page(printed: int) -> int:
    return int(printed) - 1


# Every page of the lecture notes starts with the same running header, for
# example "Module \n3\n \nMoModul\ne 3 \n Internet of Things[21CS735]   12".
# It carries no content and would make every chunk look alike to the retriever.
_HEADER = re.compile(
    r"^\s*Module\s*3\s*MoModul\s*e\s*3\s*Internet of Things\[21CS735\]\s*\d+\s*",
    re.MULTILINE,
)
# The PDF uses private-use glyphs: U+F0B7 for bullet points, U+F020 as a space.
_BULLET = re.compile("[\uf0b7\u2022]")
_PRIVATE_USE = re.compile("[\ue000-\uf8ff]")
_BLANK_RUNS = re.compile(r"\n\s*\n(\s*\n)+")


def clean_page_text(text: str) -> str:
    text = _HEADER.sub("", text, count=1)
    text = _BULLET.sub("-", text)
    text = _PRIVATE_USE.sub(" ", text)
    text = _BLANK_RUNS.sub("\n\n", text)
    return text.strip()


STOPWORDS = frozenset(
    """a about above after again all also an and any are as at be because been
    being but by can could did do does doing for from further had has have
    having how i if in into is it its itself just may might more most much must
    no nor not of off on once only or other our out over own same should so
    some such than that the their them then there these they this those
    through to too under until up very was we were what when where which while
    who whom why will with would you your""".split()
)

_WORD = re.compile(r"[a-z0-9]+")


def _stem(word: str) -> str:
    # Deliberately tiny: enough that "offloading", "offloaded" and "offload"
    # meet, without pulling in a stemming library.
    for suffix in ("ing", "ed", "es", "s"):
        if len(word) > len(suffix) + 3 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


def tokenize(text: str, keep_stopwords: bool = False) -> list[str]:
    words = _WORD.findall(text.lower())
    if not keep_stopwords:
        words = [w for w in words if w not in STOPWORDS]
    return [_stem(w) for w in words]
