# Baseline (issue 1 pipeline)

Generated 2026-10-30 by `evals/`. Golden set: `evals/golden.jsonl`.

Config: offline stand-ins (hashing embeddings, extractive model); chunks 800/120; k=4; mode `vector`; rerank off; contextual chunks off. Answers judged by lexical (offline heuristic).

## Retrieval

| config | Recall@1 | Recall@3 | Recall@5 | MRR |
|---|---|---|---|---|
| vector | 0.800 | 0.900 | 0.940 | 0.920 |

## Answers

| config | faithfulness | correctness | refused unanswerable | false refusals |
|---|---|---|---|---|
| vector | 1.000 | 0.475 | 100% | 0 |

Per-question answers and scores: `baseline-answers.csv`.
