"""Markdown result files, so every change can commit a before/after table."""

from __future__ import annotations

import math
import time
from pathlib import Path

from rag.config import RagConfig


def percentile(values, q: float) -> float:
    """Nearest-rank percentile (q from 0 to 100): the smallest value with at least q% of values at or below it."""
    vals = sorted(v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v)))
    if not vals:
        return math.nan
    rank = max(1, math.ceil(q / 100 * len(vals)))
    return vals[rank - 1]


def ms(value: float) -> str:
    return "n/a" if value is None or (isinstance(value, float) and math.isnan(value)) else f"{value:.0f}"


def fmt(value) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return "n/a" if math.isnan(value) else f"{value:.3f}"
    return str(value)


def markdown_table(headers: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in rows]
    return "\n".join(lines)


def describe(cfg: RagConfig, retrieval: bool = True) -> str:
    """One line describing a config. ``retrieval=False`` leaves out mode and rerank, for tables that vary them."""
    models = "offline stand-ins (hashing embeddings, extractive model)" if cfg.provider == "offline" \
        else f"embeddings `{cfg.embed_model}`, chat `{cfg.chat_model}`"
    mode = f"mode `{cfg.retrieval_mode}`; rerank {'on' if cfg.rerank else 'off'}; " if retrieval else ""
    return (f"{models}; chunks {cfg.chunk_size}/{cfg.chunk_overlap}; k={cfg.k}; {mode}"
            f"contextual chunks {'on' if cfg.contextual_chunks else 'off'}")


RETRIEVAL_HEADERS = ["config", "Recall@1", "Recall@3", "Recall@5", "MRR", "p50 ms", "p95 ms"]


def retrieval_row(label: str, report) -> list:
    m = report.metrics
    return [label, m["recall@1"], m["recall@3"], m["recall@5"], m["mrr"], ms(report.p50_ms), ms(report.p95_ms)]


def write_markdown(path: Path, title: str, sections: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y-%m-%d")
    path.write_text(f"# {title}\n\nGenerated {stamp} by `evals/`. Golden set: `evals/golden.jsonl`.\n\n" + "\n\n".join(sections) + "\n")
    return path


def write_retrieval_report(cfg: RagConfig, report, path: Path) -> Path:
    table = markdown_table(RETRIEVAL_HEADERS, [retrieval_row(cfg.name, report)])
    per_q = markdown_table(
        ["id", "expected pages", "retrieved pages (top 5)", "RR"],
        [[r["id"], ", ".join(map(str, r["expected"])), ", ".join(map(str, r["retrieved"])), r["rr"]] for r in report.per_question],
    )
    return write_markdown(path, f"Retrieval: {cfg.name}", [
        f"Config: {describe(cfg)}.", f"{report.n_questions} answerable questions.", table,
        "<details><summary>Per question</summary>\n\n" + per_q + "\n\n</details>",
    ])
