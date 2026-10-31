"""Retrieval metrics: did the right pages come back, and how high up?

    python -m evals.retrieval                    # real models (Ollama)
    python -m evals.retrieval --offline          # stand-in models, no Ollama
    python -m evals.retrieval --check-baseline   # exit 1 if Recall@5 regressed
    python -m evals.retrieval --write            # save evals/results/<config>.md

No LLM is involved, so for a fixed embedding model the numbers are deterministic.

- Recall@k: of the pages that contain the answer, the fraction that appear
  among the top k retrieved chunks. Averaged over answerable questions.
- MRR (mean reciprocal rank): 1 / rank of the first chunk from a correct
  page (1.0 if it is the top chunk, 0.5 if second, 0 if none in the top 5).
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
import sys
from dataclasses import dataclass, field

from evals import BASELINES, GoldenItem, load_golden, results_dir
from rag.config import RagConfig, cli_config
from rag.retrievers import Index, retrieve

DEPTH = 5
KS = (1, 3, 5)


def recall_at_k(retrieved_pages: list[int], expected: tuple[int, ...], k: int) -> float:
    if not expected:
        raise ValueError("recall is undefined for a question with no expected pages")
    return len(set(retrieved_pages[:k]) & set(expected)) / len(set(expected))


def reciprocal_rank(retrieved_pages: list[int], expected: tuple[int, ...]) -> float:
    for rank, page in enumerate(retrieved_pages, start=1):
        if page in expected:
            return 1.0 / rank
    return 0.0


@dataclass
class RetrievalReport:
    config_name: str
    n_questions: int
    metrics: dict
    per_question: list = field(default_factory=list)

    @property
    def median_ms(self) -> float:
        """Median retrieval time per question, in milliseconds."""
        times = [r["ms"] for r in self.per_question]
        return statistics.median(times) if times else math.nan


def evaluate_retrieval(cfg: RagConfig, items: list[GoldenItem], index: Index | None = None, retriever=None) -> RetrievalReport:
    """Run every answerable golden question through retrieval and score it.

    ``retriever(question) -> list[Hit]`` can be swapped in (the tests use this to
    prove the CI gate fails when retrieval is broken).
    """
    index = index or (Index.open(cfg) if retriever is None else None)
    retriever = retriever or (lambda q: retrieve(index, q, cfg, k=DEPTH))
    rows = []
    answerable = [item for item in items if item.answerable]
    if answerable:
        retriever(answerable[0].question)  # warm-up: load models once so the timings are per query
    for item in answerable:
        t0 = time.perf_counter()
        hits = retriever(item.question)[:DEPTH]
        elapsed_ms = (time.perf_counter() - t0) * 1000
        pages = [h.printed_page for h in hits]
        row = {"id": item.id, "question": item.question, "expected": list(item.expected_pages), "retrieved": pages}
        for k in KS:
            row[f"recall@{k}"] = recall_at_k(pages, item.expected_pages, k)
        row["rr"] = reciprocal_rank(pages, item.expected_pages)
        row["ms"] = elapsed_ms
        rows.append(row)
    n = len(rows)
    metrics = {f"recall@{k}": sum(r[f"recall@{k}"] for r in rows) / n for k in KS}
    metrics["mrr"] = sum(r["rr"] for r in rows) / n
    return RetrievalReport(cfg.name, n, metrics, rows)


# ---- the CI gate ---------------------------------------------------------


def baseline_key(cfg: RagConfig) -> str:
    return f"{cfg.embed_name}/{cfg.name}"


def load_baselines() -> dict:
    return json.loads(BASELINES.read_text()) if BASELINES.exists() else {}


def check_baseline(cfg: RagConfig, metrics: dict, tolerance: float = 0.02) -> tuple[bool, str]:
    key = baseline_key(cfg)
    base = load_baselines().get(key)
    if base is None:
        return False, f"No committed baseline for {key!r} in {BASELINES.name}; run with --update-baseline first."
    floor = base["recall@5"] - tolerance
    ok = metrics["recall@5"] >= floor
    verdict = "OK" if ok else "REGRESSION"
    return ok, f"{verdict}: Recall@5 {metrics['recall@5']:.3f} vs baseline {base['recall@5']:.3f} (floor {floor:.3f}) for {key}"


def update_baseline(cfg: RagConfig, metrics: dict) -> None:
    data = load_baselines()
    data[baseline_key(cfg)] = {k: round(v, 4) for k, v in metrics.items() if k.startswith(("recall", "mrr"))}
    BASELINES.write_text(json.dumps(dict(sorted(data.items())), indent=2) + "\n")


def format_metrics(metrics: dict) -> str:
    return "  ".join(f"{k}={metrics[k]:.3f}" for k in [f"recall@{k}" for k in KS] + ["mrr"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--offline", action="store_true", help="use the offline stand-in models")
    parser.add_argument("--limit", type=int, help="only the first N golden questions")
    parser.add_argument("--check-baseline", action="store_true", help="exit 1 if Recall@5 is below the committed baseline")
    parser.add_argument("--tolerance", type=float, default=0.02)
    parser.add_argument("--update-baseline", action="store_true", help="store these numbers as the new baseline")
    parser.add_argument("--write", action="store_true", help="write evals/results/<config>.md")
    args = parser.parse_args(argv)

    cfg = cli_config(offline=args.offline)
    report = evaluate_retrieval(cfg, load_golden(limit=args.limit))
    print(f"{cfg.name} ({cfg.embed_name}), {report.n_questions} questions: {format_metrics(report.metrics)}")
    if args.write:
        from evals.report import write_retrieval_report

        path = write_retrieval_report(cfg, report, results_dir(args.offline) / f"{cfg.name}.md")
        print(f"wrote {path}")
    if args.update_baseline:
        update_baseline(cfg, report.metrics)
        print(f"updated {BASELINES.name} for {baseline_key(cfg)}")
    if args.check_baseline:
        ok, message = check_baseline(cfg, report.metrics, args.tolerance)
        print(message)
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
