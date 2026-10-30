"""Answer metrics: is the answer supported by the retrieved text, and is it right?

    python -m evals.answers                  # real models, judged by a local Ollama model
    python -m evals.answers --offline        # stand-in models and the lexical judge
    python -m evals.answers --judge anthropic --judge-model <model>   # hosted judge

Prints the means and writes one CSV row per question to
evals/results/[offline/]<config>-answers.csv.

- Faithfulness: is every claim in the answer supported by the retrieved
  chunks? Scored only for questions the pipeline answered (a refusal makes no
  claims, so it would score a free 1.0).
- Correctness: does the answer match the reference answer? For the
  unanswerable questions, a refusal counts as correct and anything else as wrong.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from pathlib import Path

from evals import GoldenItem, load_golden, results_dir
from evals.judge import make_judge
from rag.config import RagConfig, cli_config
from rag.pipeline import Pipeline


def _mean(values) -> float:
    vals = [v for v in values if v is not None and not math.isnan(v)]
    return statistics.fmean(vals) if vals else math.nan


def is_refusal(cfg: RagConfig, result) -> bool:
    refused = getattr(result, "refused", None)
    if refused is not None:
        return refused
    return cfg.refusal_text.lower().rstrip(".") in result.answer.lower()


def evaluate_answers(cfg: RagConfig, items: list[GoldenItem], judge, pipeline: Pipeline | None = None) -> tuple[dict, list[dict]]:
    pipeline = pipeline or Pipeline(cfg)
    rows = []
    for item in items:
        result = pipeline.answer(item.question)
        refused = is_refusal(cfg, result)
        contexts = [c.text for c in result.chunks]
        if item.answerable:
            correctness = 0.0 if refused else judge.correctness(item.question, result.answer, item.reference_answer)
        else:
            correctness = 1.0 if refused else 0.0
        faithfulness = math.nan if refused else judge.faithfulness(result.answer, contexts)
        rows.append({
            "id": item.id,
            "type": item.type,
            "question": item.question,
            "answer": result.answer,
            "refused": refused,
            "retrieved_pages": " ".join(str(c.printed_page) for c in result.chunks),
            "faithfulness": faithfulness,
            "correctness": correctness,
        })
    return summarize(rows), rows


def summarize(rows: list[dict]) -> dict:
    answerable = [r for r in rows if r["type"] != "unanswerable"]
    unanswerable = [r for r in rows if r["type"] == "unanswerable"]
    summary = {
        "questions": len(rows),
        "faithfulness": _mean(r["faithfulness"] for r in rows),
        "correctness": _mean(r["correctness"] for r in answerable),
        "answered": sum(not r["refused"] for r in rows),
        "false_refusals": sum(r["refused"] for r in answerable),
        "refusal_rate_unanswerable": (sum(r["refused"] for r in unanswerable) / len(unanswerable)) if unanswerable else math.nan,
    }
    return summary


# Model output is stored with long dashes (U+2013, U+2014) replaced by a plain
# hyphen, so committed result files stay plain ASCII punctuation.
_DASHES = str.maketrans({"\u2013": "-", "\u2014": "-"})


def _cell(value):
    if isinstance(value, float):
        return f"{value:.3f}"
    return value.translate(_DASHES) if isinstance(value, str) else value


def write_csv(rows: list[dict], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _cell(v) for k, v in row.items()})
    return path


def format_summary(s: dict) -> str:
    return (f"faithfulness={s['faithfulness']:.3f}  correctness={s['correctness']:.3f}  "
            f"refused {s['refusal_rate_unanswerable']:.0%} of unanswerable, "
            f"{s['false_refusals']} false refusals")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--offline", action="store_true", help="use the offline stand-in models and lexical judge")
    parser.add_argument("--limit", type=int, help="only the first N golden questions")
    parser.add_argument("--judge", help="judge provider: ollama, offline, anthropic or openai")
    parser.add_argument("--judge-model", help="judge model name (default: the chat model)")
    args = parser.parse_args(argv)

    cfg = cli_config(offline=args.offline)
    judge = make_judge(cfg, args.judge, args.judge_model)
    summary, rows = evaluate_answers(cfg, load_golden(limit=args.limit), judge)
    path = write_csv(rows, results_dir(args.offline) / f"{cfg.name}-answers.csv")
    print(f"{cfg.name} with {cfg.chat_model if cfg.provider == 'ollama' else 'offline model'}, judge {judge.name}")
    print(format_summary(summary))
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
