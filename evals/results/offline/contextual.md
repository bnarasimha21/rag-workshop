# Contextual chunks (issue 5)

Generated 2026-10-31 by `evals/`. Golden set: `evals/golden.jsonl`.

Config: offline stand-ins (hashing embeddings, extractive model); chunks 800/120; k=4. Contexts written by the offline stand-in, one call per chunk, with the page as the document.

| contextual chunks | mode | Recall@1 | Recall@3 | Recall@5 | MRR | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|
| off | vector | 0.800 | 0.900 | 0.940 | 0.920 | 0 | 0 |
| off | hybrid+rerank | 0.800 | 0.900 | 0.940 | 0.920 | 2 | 2 |
| on | vector | 0.800 | 0.900 | 0.980 | 0.930 | 0 | 0 |
| on | hybrid+rerank | 0.800 | 0.900 | 0.940 | 0.920 | 2 | 2 |

## One-time ingest cost

| LLM calls | tokens | seconds |
|---|---|---|
| 41 | 16788 | 0.000 |

Paid once per PDF and model: contexts are cached in `.rag_index/context_cache.json`, so rebuilding the index makes no LLM calls.
