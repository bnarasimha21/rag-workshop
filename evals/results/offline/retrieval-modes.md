# Retrieval modes (issue 4)

Generated 2026-10-31 by `evals/`. Golden set: `evals/golden.jsonl`.

Config: offline stand-ins (hashing embeddings, extractive model); chunks 800/120; k=4; contextual chunks off. Reranker: offline word-overlap stand-in, choosing 4 of 20 candidates (5 for the Recall@5 column). Hybrid fuses the top 20 of each list with RRF, k=60.

Latency is the median wall-clock time of one retrieval call on this machine, after one warm-up query, and includes embedding the question.

| mode | rerank | Recall@1 | Recall@3 | Recall@5 | MRR | median retrieval ms |
|---|---|---|---|---|---|---|
| vector | no | 0.800 | 0.900 | 0.940 | 0.920 | 0 |
| vector | yes | 0.800 | 0.880 | 0.940 | 0.920 | 2 |
| bm25 | no | 0.800 | 0.900 | 0.940 | 0.913 | 0 |
| bm25 | yes | 0.800 | 0.900 | 0.940 | 0.913 | 1 |
| hybrid | no | 0.800 | 0.900 | 0.940 | 0.920 | 1 |
| hybrid | yes | 0.800 | 0.900 | 0.940 | 0.920 | 2 |
