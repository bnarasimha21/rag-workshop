# Grounding and citations (issue 3)

Generated 2026-10-31 by `evals/`. Golden set: `evals/golden.jsonl`.

Config: embeddings `nomic-embed-text`, chat `gemma2:2b`; chunks 800/120; k=4; mode `vector`; rerank off; contextual chunks off. Answers judged by ollama/gemma2:2b.

Faithfulness is scored only on answered questions. "Answered with citations" counts answerable questions whose answer carries at least one valid `[n]` marker. The basic prompt never cites.

| config | faithfulness | correctness | answered with citations | refused unanswerable | false refusals |
|---|---|---|---|---|---|
| before: basic prompt | 0.917 | 0.940 | 0% | 0% | 0 |
| after: grounded + citations | 0.940 | 0.720 | 100% | 100% | 0 |

## Latency and tokens

Per question, on this machine. Generate time and tokens cover only the questions that reached the model (a low-relevance refusal skips it).

| config | retrieve ms p50 / p95 | generate ms p50 / p95 | total ms p50 / p95 | mean tokens in | mean tokens out |
|---|---|---|---|---|---|
| before: basic prompt | 20 / 27 | 6612 / 23630 | 6633 / 23646 | 576 | 287 |
| after: grounded + citations | 31 / 45 | 2970 / 3768 | 2970 / 3788 | 733 | 40 |

Per-question rows: `citations-before-answers.csv` and `citations-after-answers.csv`.

Reading the correctness column: the basic prompt writes long answers that mention more of the reference facts, sometimes from the model's own knowledge, and it answers every unanswerable question anyway. The grounded answers are shorter and often list only the items in the four retrieved chunks, so the judge marks them PARTLY. Better retrieval (issue 4) is the fix for that, not a looser prompt: asking the model to name every item did not raise correctness and lowered faithfulness.
