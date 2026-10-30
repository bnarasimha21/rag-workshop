# Baseline (issue 1 pipeline)

Generated 2026-10-30 by `evals/`. Golden set: `evals/golden.jsonl`.

Config: embeddings `nomic-embed-text`, chat `gemma2:2b`; chunks 800/120; k=4; mode `vector`; rerank off; contextual chunks off. Answers judged by ollama/gemma2:2b.

## Retrieval

| config | Recall@1 | Recall@3 | Recall@5 | MRR |
|---|---|---|---|---|
| vector | 0.800 | 0.940 | 0.980 | 0.927 |

## Answers

| config | faithfulness | correctness | refused unanswerable | false refusals |
|---|---|---|---|---|
| vector | 0.917 | 0.780 | 0% | 0 |

Per-question answers and scores: `baseline-answers.csv`.
