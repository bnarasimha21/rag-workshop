"""Issue 2: the golden set, retrieval metrics, answer metrics and the CI gate."""

from __future__ import annotations

import csv
import math
import random
import subprocess
import sys

import pytest

from evals import load_golden
from evals.answers import evaluate_answers, write_csv
from evals.judge import LexicalJudge, make_judge, parse_verdict, strip_citations
from evals.retrieval import check_baseline, evaluate_retrieval, load_baselines, recall_at_k, reciprocal_rank
from rag.ingest import load_pages
from rag.retrievers import Index
from rag.text import printed_page, tokenize

from conftest import REPO

TOLERANCE = 0.05


# ---- the golden set itself ----------------------------------------------

def test_golden_set_shape():
    items = load_golden()
    assert 25 <= len(items) <= 30
    assert len({i.id for i in items}) == len(items)
    unanswerable = [i for i in items if not i.answerable]
    assert 4 <= len(unanswerable) <= 5
    for item in items:
        assert item.question.endswith("?") or item.question.endswith(".")
        if item.answerable:
            assert item.expected_pages and all(1 <= p <= 16 for p in item.expected_pages)
            assert item.reference_answer
        else:
            assert not item.expected_pages


def test_golden_answers_are_grounded_in_their_pages():
    """Most content words of each reference answer appear on its expected pages."""
    pages = {printed_page(p.metadata["page"]): set(tokenize(p.page_content)) for p in load_pages(REPO / "3rd module notes.pdf")}
    for item in load_golden():
        if not item.answerable:
            continue
        words = set(tokenize(item.reference_answer))
        on_pages = set().union(*(pages[p] for p in item.expected_pages))
        assert len(words & on_pages) / len(words) >= 0.6, (item.id, words - on_pages)


# ---- metric definitions ----------------------------------------------------

def test_recall_and_reciprocal_rank():
    assert recall_at_k([12, 3, 11], (11, 12), 1) == 0.5
    assert recall_at_k([12, 3, 11], (11, 12), 3) == 1.0
    assert recall_at_k([12, 12, 12], (11, 12), 3) == 0.5
    assert reciprocal_rank([3, 4, 12], (12,)) == pytest.approx(1 / 3)
    assert reciprocal_rank([3, 4], (12,)) == 0.0
    with pytest.raises(ValueError):
        recall_at_k([1], (), 1)


def test_judge_helpers():
    assert parse_verdict("YES.") == 1.0
    assert parse_verdict("Partly, it misses one.") == 0.5
    assert parse_verdict("no") == 0.0
    assert math.isnan(parse_verdict("maybe"))
    assert strip_citations("Fog and cloud [1] and edge [2, 3].") == "Fog and cloud and edge."
    judge = LexicalJudge()
    assert judge.faithfulness("fog computing", ["Fog computing is decentralized"]) == 1.0
    assert judge.correctness("q", "edge and fog", "edge, fog, cloud") == pytest.approx(2 / 3)


def test_hosted_judge_requires_a_model_name(offline_cfg):
    with pytest.raises(ValueError, match="RAG_JUDGE_MODEL"):
        make_judge(offline_cfg(), "anthropic", "")


# ---- end to end on the real PDF -------------------------------------------

def test_retrieval_eval_end_to_end_offline(offline_cfg):
    cfg = offline_cfg(retrieval_mode="vector")
    report = evaluate_retrieval(cfg, load_golden(limit=5))
    assert report.n_questions == 5
    for value in report.metrics.values():
        assert 0.0 <= value <= 1.0
    baseline = load_baselines()["hashing/vector"]
    assert report.metrics["recall@5"] >= baseline["recall@5"] - TOLERANCE


def test_full_offline_retrieval_matches_committed_baseline(offline_cfg):
    cfg = offline_cfg(retrieval_mode="vector")
    report = evaluate_retrieval(cfg, load_golden())
    ok, message = check_baseline(cfg, report.metrics)
    assert ok, message


def test_gate_fails_when_retrieval_is_broken(offline_cfg):
    """k=1 over a shuffled index: the CI gate must catch it."""
    cfg = offline_cfg(retrieval_mode="vector")
    index = Index.open(cfg)
    rng = random.Random(0)

    def broken(question):
        chunk = rng.choice(index.chunks)
        return [index.hit(chunk.metadata["chunk_id"], 0.0)]

    report = evaluate_retrieval(cfg, load_golden(), retriever=broken)
    ok, message = check_baseline(cfg, report.metrics)
    assert not ok and message.startswith("REGRESSION")


def test_retrieval_cli_gate_passes(subprocess_env):
    env = dict(subprocess_env, RAG_RETRIEVAL_MODE="vector")
    out = subprocess.run([sys.executable, "-m", "evals.retrieval", "--offline", "--check-baseline"],
                         cwd=REPO, env=env, capture_output=True, text=True, timeout=180)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "recall@1=" in out.stdout and "mrr=" in out.stdout and "OK: Recall@5" in out.stdout


def test_answer_eval_end_to_end_offline(offline_cfg, tmp_path):
    cfg = offline_cfg(retrieval_mode="vector")
    items = [i for i in load_golden() if i.id in {"q05", "q17", "q23", "u01", "u02"}]
    summary, rows = evaluate_answers(cfg, items, LexicalJudge())
    assert len(rows) == 5
    assert 0.0 <= summary["faithfulness"] <= 1.0 and 0.0 <= summary["correctness"] <= 1.0
    path = write_csv(rows, tmp_path / "answers.csv")
    with open(path) as fh:
        written = list(csv.DictReader(fh))
    assert [r["id"] for r in written] == [i.id for i in items]
    assert {"faithfulness", "correctness", "refused", "answer"} <= set(written[0])


@pytest.mark.realmodel
def test_retrieval_eval_real_models(real_cfg):
    cfg = real_cfg(retrieval_mode="vector", rerank=False, contextual_chunks=False)
    report = evaluate_retrieval(cfg, load_golden(limit=5))
    for value in report.metrics.values():
        assert 0.0 <= value <= 1.0
    baseline = load_baselines()["nomic-embed-text/vector"]
    assert report.metrics["recall@5"] >= baseline["recall@5"] - TOLERANCE


# Hand-labelled cases: (question, reference, candidate, expected verdict).
CALIBRATION = [
    ("What kind of database systems are structured data associated with, and which language is used to access them?",
     "Relational database management systems (RDBMS), accessed with Structured Query Language (SQL).",
     "Structured data is associated with relational database management systems (RDBMS). Established languages "
     "such as Structured Query Language (SQL) are used for accessing these data.", 1.0),
    ("What are the two large processing topologies for IoT?", "On-site processing and off-site processing.",
     "The two large processing topologies for IoT are On-site and Off-site.", 1.0),
    ("Into which three categories is IoT data divided based on the urgency of processing?",
     "Very time critical, time critical, and normal.",
     "IoT data is divided into three categories based on the urgency of processing: very time critical, time critical, and normal.", 1.0),
    ("Into which three categories is IoT data divided based on the urgency of processing?",
     "Very time critical, time critical, and normal.",
     "IoT data is divided by urgency; one category is very time critical data.", 0.5),
    ("What kind of database systems are structured data associated with, and which language is used to access them?",
     "Relational database management systems (RDBMS), accessed with Structured Query Language (SQL).",
     "Structured data lives in NoSQL document stores accessed with GraphQL.", 0.0),
    ("What are the two large processing topologies for IoT?", "On-site processing and off-site processing.",
     "Cloud processing and fog processing.", 0.0),
]


@pytest.mark.realmodel
def test_judge_calibration_real(real_cfg):
    """The default local judge must agree with hand labels on most cases, and never call a correct answer wrong."""
    cfg = real_cfg()
    judge = make_judge(cfg, "ollama", None)
    verdicts = [(judge.correctness(q, a, r), want) for q, r, a, want in CALIBRATION]
    agree = sum(got == want for got, want in verdicts)
    assert agree >= 5, verdicts
    assert all(got == 1.0 for got, want in verdicts if want == 1.0), verdicts
