# Baseline (issue 1 pipeline)

Generated 2026-10-31 by `evals/`. Golden set: `evals/golden.jsonl`.

Config: embeddings `nomic-embed-text`, chat `gemma2:2b`; chunks 800/120; k=4; mode `vector`; rerank off; contextual chunks off. Answers judged by ollama/gemma2:2b.

## Retrieval

| config | Recall@1 | Recall@3 | Recall@5 | MRR | p50 ms | p95 ms |
|---|---|---|---|---|---|---|
| vector | 0.800 | 0.940 | 0.980 | 0.927 | 14 | 19 |

## Answers

| config | faithfulness | correctness | answered with citations | refused unanswerable | false refusals |
|---|---|---|---|---|---|
| vector | 0.917 | 0.940 | 0% | 0% | 0 |

Per-question answers and scores: `baseline-answers.csv`.

## Latency and tokens

Per question, on this machine. Generate time and tokens cover only the questions that reached the model (a low-relevance refusal skips it).

| config | retrieve ms p50 / p95 | generate ms p50 / p95 | total ms p50 / p95 | mean tokens in | mean tokens out |
|---|---|---|---|---|---|
| vector | 22 / 42 | 6576 / 24439 | 6603 / 24467 | 576 | 287 |
