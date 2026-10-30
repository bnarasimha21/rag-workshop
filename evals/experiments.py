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

ANSWER_HEADERS = ["config", "faithfulness", "correctness", "refused unanswerable", "false refusals"]


def answer_row(label: str, s: dict) -> list:
    return [label, s["faithfulness"], s["correctness"], f"{s['refusal_rate_unanswerable']:.0%}", s["false_refusals"]]


def run_baseline(base: RagConfig, offline: bool, judge_provider: str | None, judge_model: str | None) -> str:
    cfg = base.with_overrides(retrieval_mode="vector", rerank=False, contextual_chunks=False)
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


EXPERIMENTS = {"baseline": run_baseline}


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
