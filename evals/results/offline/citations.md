# Grounding and citations (issue 3)

Generated 2026-10-31 by `evals/`. Golden set: `evals/golden.jsonl`.

Config: offline stand-ins (hashing embeddings, extractive model); chunks 800/120; k=4; mode `vector`; rerank off; contextual chunks off. Answers judged by lexical (offline heuristic).

Faithfulness is scored only on answered questions. "Answered with citations" counts answerable questions whose answer carries at least one valid `[n]` marker. The basic prompt never cites.

| config | faithfulness | correctness | answered with citations | refused unanswerable | false refusals |
|---|---|---|---|---|---|
| before: basic prompt | 0.833 | 0.475 | 0% | 0% | 0 |
| after: grounded + citations | 1.000 | 0.523 | 100% | 100% | 0 |

Per-question rows: `citations-before-answers.csv` and `citations-after-answers.csv`.

Reading the correctness column: the basic prompt writes long answers that mention more of the reference facts, sometimes from the model's own knowledge, and it answers every unanswerable question anyway. The grounded answers are shorter and often list only the items in the four retrieved chunks, so the judge marks them PARTLY. Better retrieval (issue 4) is the fix for that, not a looser prompt: asking the model to name every item did not raise correctness and lowered faithfulness.
