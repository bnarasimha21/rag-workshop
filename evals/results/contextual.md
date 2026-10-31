# Contextual chunks (issue 5)

Generated 2026-10-31 by `evals/`. Golden set: `evals/golden.jsonl`.

Config: embeddings `nomic-embed-text`, chat `gemma2:2b`; chunks 800/120; k=4. Contexts written by `gemma2:2b`, one call per chunk, with the page as the document.

| contextual chunks | mode | Recall@1 | Recall@3 | Recall@5 | MRR | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|
| off | vector | 0.800 | 0.940 | 0.980 | 0.927 | 17 | 24 |
| off | hybrid+rerank | 0.840 | 0.920 | 0.980 | 0.960 | 159 | 206 |
| on | vector | 0.840 | 0.920 | 0.960 | 0.960 | 16 | 19 |
| on | hybrid+rerank | 0.840 | 0.880 | 0.980 | 0.950 | 250 | 312 |

## One-time ingest cost

| LLM calls | tokens | seconds |
|---|---|---|
| 41 | 24939 | 57.400 |

Paid once per PDF and model: contexts are cached in `.rag_index/context_cache.json`, so rebuilding the index makes no LLM calls.
