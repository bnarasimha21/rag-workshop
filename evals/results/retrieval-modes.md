# Retrieval modes (issue 4)

Generated 2026-10-31 by `evals/`. Golden set: `evals/golden.jsonl`.

Config: embeddings `nomic-embed-text`, chat `gemma2:2b`; chunks 800/120; k=4; contextual chunks off. Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2` on CPU, choosing 4 of 20 candidates (5 for the Recall@5 column). Hybrid fuses the top 20 of each list with RRF, k=60.

Latency is the wall-clock time of one retrieval call on this machine (median and 95th percentile over the questions), after one warm-up query, and includes embedding the question.

| mode | rerank | Recall@1 | Recall@3 | Recall@5 | MRR | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|
| vector | no | 0.800 | 0.940 | 0.980 | 0.927 | 24 | 27 |
| vector | yes | 0.840 | 0.920 | 0.960 | 0.960 | 464 | 590 |
| bm25 | no | 0.800 | 0.900 | 0.940 | 0.913 | 0 | 0 |
| bm25 | yes | 0.840 | 0.920 | 0.980 | 0.960 | 550 | 645 |
| hybrid | no | 0.780 | 0.900 | 0.920 | 0.893 | 34 | 44 |
| hybrid | yes | 0.840 | 0.920 | 0.980 | 0.960 | 415 | 631 |
