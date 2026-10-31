"""Named experiments that write the committed comparison tables.

    python -m evals.experiments baseline            # real models
    python -m evals.experiments baseline --offline  # stand-in models

Each experiment runs a few configs over the golden set and writes one file in
evals/results/ (real models) or evals/results/offline/ (stand-ins).
"""

from __future__ import annotations

import argparse
import sys

from evals import load_golden, results_dir
from evals.answers import evaluate_answers, write_csv
from evals.judge import make_judge
from evals.report import RETRIEVAL_HEADERS, describe, markdown_table, retrieval_row, write_markdown
from evals.retrieval import evaluate_retrieval
from rag.config import RagConfig, cli_config

ANSWER_HEADERS = ["config", "faithfulness", "correctness", "answered with citations",
                  "refused unanswerable", "false refusals"]


def answer_row(label: str, s: dict) -> list:
    return [label, s["faithfulness"], s["correctness"], f"{s['cited_rate_answerable']:.0%}",
            f"{s['refusal_rate_unanswerable']:.0%}", s["false_refusals"]]


def run_baseline(base: RagConfig, offline: bool, judge_provider: str | None, judge_model: str | None) -> str:
    cfg = base.with_overrides(retrieval_mode="vector", rerank=False, contextual_chunks=False, grounded_prompt=False)
    items = load_golden()
    retrieval = evaluate_retrieval(cfg, items)
    judge = make_judge(cfg, judge_provider, judge_model)
    summary, rows = evaluate_answers(cfg, items, judge)
    out = results_dir(offline)
    write_csv(rows, out / "baseline-answers.csv")
    path = write_markdown(out / "baseline.md", "Baseline (issue 1 pipeline)", [
        f"Config: {describe(cfg)}. Answers judged by {judge.name}.",
        "## Retrieval\n\n" + markdown_table(RETRIEVAL_HEADERS, [retrieval_row(cfg.name, retrieval.metrics)]),
        "## Answers\n\n" + markdown_table(ANSWER_HEADERS, [answer_row(cfg.name, summary)])
        + "\n\nPer-question answers and scores: `baseline-answers.csv`.",
    ])
    return str(path)


def run_citations(base: RagConfig, offline: bool, judge_provider: str | None, judge_model: str | None) -> str:
    """Issue 3: the basic prompt (before) against the grounded, cited prompt (after)."""
    cfg = base.with_overrides(retrieval_mode="vector", rerank=False, contextual_chunks=False)
    items = load_golden()
    judge = make_judge(cfg, judge_provider, judge_model)
    out = results_dir(offline)
    table_rows = []
    for label, grounded in (("before: basic prompt", False), ("after: grounded + citations", True)):
        run_cfg = cfg.with_overrides(grounded_prompt=grounded)
        summary, rows = evaluate_answers(run_cfg, items, judge)
        write_csv(rows, out / f"citations-{'after' if grounded else 'before'}-answers.csv")
        table_rows.append(answer_row(label, summary))
    path = write_markdown(out / "citations.md", "Grounding and citations (issue 3)", [
        f"Config: {describe(cfg)}. Answers judged by {judge.name}.",
        "Faithfulness is scored only on answered questions. \"Answered with citations\" counts answerable "
        "questions whose answer carries at least one valid `[n]` marker. The basic prompt never cites.",
        markdown_table(ANSWER_HEADERS, table_rows),
        "Per-question rows: `citations-before-answers.csv` and `citations-after-answers.csv`.",
        "Reading the correctness column: the basic prompt writes long answers that mention more of the reference "
        "facts, sometimes from the model's own knowledge, and it answers every unanswerable question anyway. "
        "The grounded answers are shorter and often list only the items in the four retrieved chunks, so the judge "
        "marks them PARTLY. Better retrieval (issue 4) is the fix for that, not a looser prompt: asking the model to "
        "name every item did not raise correctness and lowered faithfulness.",
    ])
    return str(path)


EXPERIMENTS = {"baseline": run_baseline, "citations": run_citations}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("name", choices=sorted(EXPERIMENTS))
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--judge", help="judge provider for answer metrics")
    parser.add_argument("--judge-model")
    args = parser.parse_args(argv)
    path = EXPERIMENTS[args.name](cli_config(offline=args.offline), args.offline, args.judge, args.judge_model)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
