"""Evaluation harness: measure retrieval and answers against evals/golden.jsonl."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

EVALS_DIR = Path(__file__).resolve().parent
GOLDEN = EVALS_DIR / "golden.jsonl"
RESULTS = EVALS_DIR / "results"
BASELINES = EVALS_DIR / "baselines.json"


@dataclass(frozen=True)
class GoldenItem:
    id: str
    question: str
    expected_pages: tuple[int, ...]  # printed page numbers, 1-indexed
    reference_answer: str
    type: str

    @property
    def answerable(self) -> bool:
        return self.type != "unanswerable"


def load_golden(path: Path = GOLDEN, ids: list[str] | None = None, limit: int | None = None) -> list[GoldenItem]:
    items = []
    with open(path) as fh:
        for line in fh:
            if line.strip():
                row = json.loads(line)
                row["expected_pages"] = tuple(row["expected_pages"])
                items.append(GoldenItem(**row))
    if ids:
        wanted = set(ids)
        items = [i for i in items if i.id in wanted]
    return items[:limit] if limit else items


def results_dir(offline: bool) -> Path:
    """Real-model numbers go in evals/results/, offline stand-in numbers in evals/results/offline/."""
    return RESULTS / "offline" if offline else RESULTS
