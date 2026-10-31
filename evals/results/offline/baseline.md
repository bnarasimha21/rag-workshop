# Baseline (issue 1 pipeline)

Generated 2026-10-31 by `evals/`. Golden set: `evals/golden.jsonl`.

Config: offline stand-ins (hashing embeddings, extractive model); chunks 800/120; k=4; mode `vector`; rerank off; contextual chunks off. Answers judged by lexical (offline heuristic).

## Retrieval

| config | Recall@1 | Recall@3 | Recall@5 | MRR | p50 ms | p95 ms |
|---|---|---|---|---|---|---|
| vector | 0.800 | 0.900 | 0.940 | 0.920 | 0 | 0 |

## Answers

| config | faithfulness | correctness | answered with citations | refused unanswerable | false refusals |
|---|---|---|---|---|---|
| vector | 1.000 | 0.475 | 0% | 100% | 0 |

Per-question answers and scores: `baseline-answers.csv`.

## Latency and tokens

Per question, on this machine. Generate time and tokens cover only the questions that reached the model (a low-relevance refusal skips it).

| config | retrieve ms p50 / p95 | generate ms p50 / p95 | total ms p50 / p95 | mean tokens in | mean tokens out |
|---|---|---|---|---|---|
| vector | 0 / 0 | 0 / 0 | 1 / 1 | 392 | 18 |
