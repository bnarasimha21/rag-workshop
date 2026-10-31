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
from evals.report import RETRIEVAL_HEADERS, describe, markdown_table, ms, retrieval_row, write_markdown
from evals.retrieval import evaluate_retrieval
from rag.config import RagConfig, cli_config

ANSWER_HEADERS = ["config", "faithfulness", "correctness", "answered with citations",
                  "refused unanswerable", "false refusals"]
COST_HEADERS = ["config", "retrieve ms p50 / p95", "generate ms p50 / p95", "total ms p50 / p95",
                "mean tokens in", "mean tokens out"]


def answer_row(label: str, s: dict) -> list:
    return [label, s["faithfulness"], s["correctness"], f"{s['cited_rate_answerable']:.0%}",
            f"{s['refusal_rate_unanswerable']:.0%}", s["false_refusals"]]


def cost_row(label: str, s: dict) -> list:
    lat = s["latency_ms"]
    pair = lambda stage: f"{ms(lat[stage]['p50'])} / {ms(lat[stage]['p95'])}" if stage in lat else "n/a"
    return [label, pair("retrieve"), pair("generate"), pair("total"),
            f"{s['mean_tokens_in']:.0f}", f"{s['mean_tokens_out']:.0f}"]


def cost_section(rows: list[list]) -> str:
    return ("## Latency and tokens\n\nPer question, on this machine. Generate time and tokens cover only the "
            "questions that reached the model (a low-relevance refusal skips it).\n\n"
            + markdown_table(COST_HEADERS, rows))


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
        "## Retrieval\n\n" + markdown_table(RETRIEVAL_HEADERS, [retrieval_row(cfg.name, retrieval)]),
        "## Answers\n\n" + markdown_table(ANSWER_HEADERS, [answer_row(cfg.name, summary)])
        + "\n\nPer-question answers and scores: `baseline-answers.csv`.",
        cost_section([cost_row(cfg.name, summary)]),
    ])
    return str(path)


def citations_note(before: dict, after: dict, k: int) -> str:
    """The "how to read this" paragraph, written from this run's own numbers."""
    b, a = before["correctness"], after["correctness"]
    rb, ra = before["refusal_rate_unanswerable"], after["refusal_rate_unanswerable"]
    text = (f"Reading the table: correctness on answerable questions went from {b:.3f} to {a:.3f}, and the share of "
            f"unanswerable questions refused went from {rb:.0%} to {ra:.0%}.")
    if a < b:
        text += (f" The grounded answers are limited to the {k} retrieved chunks and are usually shorter, so when the "
                 "reference lists more items than those chunks hold, the judge marks them PARTLY; the basic prompt "
                 "can fill the gaps from the model's own knowledge.")
    elif a > b:
        text += " Here the grounded prompt was also more correct."
    if rb < 1:
        text += (" A question counts as refused only when the answer contains the exact refusal sentence, so a model "
                 "that declines in its own words under the basic prompt is not counted: read the unanswerable rows "
                 "(ids starting with u) of `citations-before-answers.csv`.")
    return text


def run_citations(base: RagConfig, offline: bool, judge_provider: str | None, judge_model: str | None) -> str:
    """Issue 3: the basic prompt (before) against the grounded, cited prompt (after)."""
    cfg = base.with_overrides(retrieval_mode="vector", rerank=False, contextual_chunks=False)
    items = load_golden()
    judge = make_judge(cfg, judge_provider, judge_model)
    out = results_dir(offline)
    table_rows, cost_rows, summaries = [], [], []
    for label, grounded in (("before: basic prompt", False), ("after: grounded + citations", True)):
        run_cfg = cfg.with_overrides(grounded_prompt=grounded)
        summary, rows = evaluate_answers(run_cfg, items, judge)
        write_csv(rows, out / f"citations-{'after' if grounded else 'before'}-answers.csv")
        table_rows.append(answer_row(label, summary))
        cost_rows.append(cost_row(label, summary))
        summaries.append(summary)
    path = write_markdown(out / "citations.md", "Grounding and citations (issue 3)", [
        f"Config: {describe(cfg)}. Answers judged by {judge.name}.",
        "Faithfulness is scored only on answered questions. \"Answered with citations\" counts answerable "
        "questions whose answer carries at least one valid `[n]` marker. The basic prompt never cites.",
        markdown_table(ANSWER_HEADERS, table_rows),
        cost_section(cost_rows),
        "Per-question rows: `citations-before-answers.csv` and `citations-after-answers.csv`.",
        citations_note(summaries[0], summaries[1], cfg.k),
    ])
    return str(path)


MODE_HEADERS = ["mode", "rerank", "Recall@1", "Recall@3", "Recall@5", "MRR", "p50 ms", "p95 ms"]


def retrieval_mode_rows(base: RagConfig, contextual: bool = False) -> list[list]:
    """Recall, MRR and median latency for every retrieval mode, with and without the reranker."""
    from rag.retrievers import Index

    index = Index.open(base.with_overrides(contextual_chunks=contextual))
    rows = []
    for mode in ("vector", "bm25", "hybrid"):
        for rerank in (False, True):
            cfg = base.with_overrides(retrieval_mode=mode, rerank=rerank, contextual_chunks=contextual)
            report = evaluate_retrieval(cfg, load_golden(), index=index)
            m = report.metrics
            rows.append([mode, rerank, m["recall@1"], m["recall@3"], m["recall@5"], m["mrr"],
                         ms(report.p50_ms), ms(report.p95_ms)])
    return rows


def run_retrieval_modes(base: RagConfig, offline: bool, judge_provider: str | None, judge_model: str | None) -> str:
    """Issue 4: vector, BM25 and hybrid retrieval, each with and without the cross-encoder."""
    cfg = base.with_overrides(contextual_chunks=False)
    reranker = "offline word-overlap stand-in" if offline else f"`{cfg.reranker_model}` on CPU"
    path = write_markdown(results_dir(offline) / "retrieval-modes.md", "Retrieval modes (issue 4)", [
        f"Config: {describe(cfg, retrieval=False)}. Reranker: {reranker}, choosing {cfg.k} of {cfg.candidate_k} candidates "
        f"(5 for the Recall@5 column). Hybrid fuses the top {cfg.candidate_k} of each list with RRF, k={cfg.rrf_k}.",
        "Latency is the wall-clock time of one retrieval call on this machine (median and 95th percentile "
        "over the questions), after one warm-up query, and includes embedding the question.",
        markdown_table(MODE_HEADERS, retrieval_mode_rows(cfg)),
    ])
    return str(path)


CONTEXTUAL_HEADERS = ["contextual chunks", "mode", "Recall@1", "Recall@3", "Recall@5", "MRR", "p50 ms", "p95 ms"]


def run_contextual(base: RagConfig, offline: bool, judge_provider: str | None, judge_model: str | None) -> str:
    """Issue 5: retrieval with and without a generated context line on every chunk."""
    import json

    from rag.ingest import MANIFEST, ingest
    from rag.retrievers import Index

    rows = []
    cost = None
    for contextual in (False, True):
        cfg = base.with_overrides(contextual_chunks=contextual)
        built = ingest(cfg)
        if contextual:
            cost = json.loads((built.path / MANIFEST).read_text())["context_cost"]
        index = Index(cfg, built.path)
        for mode, rerank in (("vector", False), ("hybrid", True)):
            run_cfg = cfg.with_overrides(retrieval_mode=mode, rerank=rerank)
            report = evaluate_retrieval(run_cfg, load_golden(), index=index)
            m = report.metrics
            rows.append(["on" if contextual else "off", run_cfg.name.replace("+ctx", ""), m["recall@1"], m["recall@3"],
                         m["recall@5"], m["mrr"], ms(report.p50_ms), ms(report.p95_ms)])
    model = "the offline stand-in" if offline else f"`{base.effective_context_model}`"
    path = write_markdown(results_dir(offline) / "contextual.md", "Contextual chunks (issue 5)", [
        f"Config: {describe(base, retrieval=False, contextual=False)}. Contexts written by {model}, one call per chunk, "
        "with the page as the document.",
        markdown_table(CONTEXTUAL_HEADERS, rows),
        "## One-time ingest cost\n\n" + markdown_table(
            ["LLM calls", "tokens", "seconds"], [[cost["llm_calls"], cost["tokens"], cost["seconds"]]])
        + "\n\nPaid once per PDF and model: contexts are cached in `.rag_index/context_cache.json`, "
          "so rebuilding the index makes no LLM calls.",
    ])
    return str(path)


EXPERIMENTS = {"baseline": run_baseline, "citations": run_citations, "retrieval-modes": run_retrieval_modes,
               "contextual": run_contextual}


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
