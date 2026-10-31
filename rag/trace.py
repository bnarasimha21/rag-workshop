"""Where the time and tokens go for one question.

    trace = Trace()
    with trace.stage("retrieve"):
        with trace.stage("rerank"):   # nested stages are allowed
            ...
    with trace.stage("generate") as s:
        s.tokens_in, s.tokens_out = 812, 64
    trace.finish()

The clock is injectable so tests can drive it by hand.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class Stage:
    name: str
    start: float          # seconds, from the trace's clock
    end: float | None = None
    depth: int = 0        # 0 for top-level stages, 1 for a stage inside another, ...
    parent: str | None = None
    tokens_in: int = 0
    tokens_out: int = 0

    @property
    def ms(self) -> float:
        return 0.0 if self.end is None else (self.end - self.start) * 1000


@dataclass
class Trace:
    clock: Callable[[], float] = time.perf_counter
    price_per_1k_in: float = 0.0
    price_per_1k_out: float = 0.0
    stages: list[Stage] = field(default_factory=list)
    started: float = field(init=False)
    ended: float | None = field(default=None, init=False)
    _open: list[Stage] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self):
        self.started = self.clock()

    @contextmanager
    def stage(self, name: str):
        parent = self._open[-1] if self._open else None
        s = Stage(name, self.clock(), depth=len(self._open), parent=parent.name if parent else None)
        self.stages.append(s)
        self._open.append(s)
        try:
            yield s
        finally:
            s.end = self.clock()
            self._open.pop()

    def finish(self) -> "Trace":
        if self.ended is None:
            self.ended = self.clock()
        return self

    # ---- reading it --------------------------------------------------------

    @property
    def total_ms(self) -> float:
        end = self.ended if self.ended is not None else self.clock()
        return (end - self.started) * 1000

    def ms(self, name: str) -> float | None:
        """Total time spent in stages called ``name``; None if no such stage ran."""
        found = [s.ms for s in self.stages if s.name == name]
        return sum(found) if found else None

    @property
    def tokens_in(self) -> int:
        return sum(s.tokens_in for s in self.stages)

    @property
    def tokens_out(self) -> int:
        return sum(s.tokens_out for s in self.stages)

    @property
    def estimated_cost(self) -> float:
        """What these tokens would cost at the configured hosted-model prices. An estimate, 0 for local models."""
        return self.tokens_in / 1000 * self.price_per_1k_in + self.tokens_out / 1000 * self.price_per_1k_out

    def as_markdown(self) -> str:
        lines = ["| stage | ms | tokens in | tokens out |", "|---|---|---|---|"]
        for s in self.stages:
            indent = "  " * s.depth + ("↳ " if s.depth else "")
            lines.append(f"| {indent}{s.name} | {s.ms:.0f} | {s.tokens_in or ''} | {s.tokens_out or ''} |")
        lines.append(f"| **total** | **{self.total_ms:.0f}** | **{self.tokens_in}** | **{self.tokens_out}** |")
        text = "\n".join(lines)
        if self.price_per_1k_in or self.price_per_1k_out:
            text += (f"\n\nEstimated cost at the configured hosted-model prices: ${self.estimated_cost:.5f} "
                     "(an estimate; the local model costs nothing per token).")
        return text
