# RAG Workshop

Hands-on workshop for building a Retrieval-Augmented Generation (RAG) pipeline over a PDF with LangChain, Ollama and ChromaDB. Everything runs locally and for free: the embedding model and the chat model both run in Ollama on your laptop.

The sample document is `3rd module notes.pdf`, 16 pages of lecture notes on IoT processing topologies (data types, on-site vs off-site processing, device design, processing offloading).

## Tech stack

- **Language:** Python 3.10+
- **RAG framework:** LangChain (`langchain-core`, `langchain-ollama`, `langchain-chroma`, `langchain-text-splitters`)
- **Embeddings:** `nomic-embed-text` via Ollama
- **LLM:** `gemma2:2b` via Ollama (any Ollama chat model works, see Configuration)
- **Vector store:** ChromaDB, persisted on disk
- **PDF parsing:** pypdf
- **Web UI:** Gradio
- **Tests:** pytest

## Prerequisites

1. [Ollama](https://ollama.com/) installed and running (`ollama serve`, or the desktop app).
2. The two models pulled:

   ```bash
   ollama pull nomic-embed-text   # embeddings, about 270 MB
   ollama pull gemma2:2b          # chat model, about 1.6 GB
   ```

   On a machine with 16 GB of RAM or more, a bigger model such as `gemma3:4b` gives better answers: `ollama pull gemma3:4b`, then run with `RAG_CHAT_MODEL=gemma3:4b`. The default stays small so it runs on any laptop.

## Setup

```bash
git clone https://github.com/bnarasimha21/rag-workshop.git
cd rag-workshop
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
python -m rag.ingest     # 1. build the index (once)
python -m rag.app        # 2. open the web UI and ask questions
```

`python ollama_rag.py` still works and does the same as `python -m rag.app`. The app builds the index on first use if you skipped step 1.

From Python:

```python
from rag.pipeline import answer

result = answer("Where can IoT data be offloaded to?")
print(result.answer)
for chunk in result.chunks:
    print(chunk.printed_page, round(chunk.score, 3), chunk.text[:80])
```

## How the pipeline works

The code lives in the `rag/` package. Read it in this order:

| Step | File | What it does |
|------|------|--------------|
| Settings | `rag/config.py` | One `RagConfig` dataclass with every knob. Each field can be overridden with an environment variable `RAG_<FIELD>`. |
| 1. Ingest | `rag/ingest.py` | Reads the PDF page by page, strips the running page header, splits each page into chunks of about 800 characters with 120 characters of overlap (`RecursiveCharacterTextSplitter`), embeds them and saves them in Chroma under `.rag_index/`. |
| 2. Retrieve | `rag/retrievers.py` | Finds the `k` chunks whose embeddings are closest to the question's embedding. |
| 3. Generate | `rag/pipeline.py`, `rag/prompts.py` | Numbers the chunks `[1]`, `[2]` ... in a grounded prompt and asks the chat model. `answer(question)` returns a `RagResult`. |
| 4. Cite | `rag/citations.py` | Turns the `[n]` markers in the answer into citations (chunk id, page, original text) and drops markers that point at nothing. |
| 5. UI | `rag/app.py` | A Gradio page around `Pipeline.answer`, with a Sources panel showing each cited chunk. Importing it starts nothing; the server starts in `main()`. |

Things worth noticing while you read:

- **Chunking.** The retrieval unit is a chunk, not a page. Smaller chunks are more specific; too small and a definition gets cut in half. Try `RAG_CHUNK_SIZE=400` and compare.
- **Every chunk keeps its page.** Chunks carry `source`, `page` and a stable `chunk_id` such as `p12-c01` (printed page 12, chunk 01 of that page). Pages are split one at a time, so a chunk never spans two pages. `pypdf` numbers pages from 0; `rag/text.py` holds the single conversion to the printed page number that people see.
- **The index persists.** Ingest stores a manifest with the PDF's SHA-256 and the settings that affect the index. Running it again with the same PDF and settings prints `Index is up to date ... skipping embedding.` Changing the PDF or a setting such as chunk size builds a new index in its own folder. `python -m rag.ingest --force` rebuilds anyway.
- **Task prefixes matter.** `nomic-embed-text` was trained with `search_query: ` and `search_document: ` prefixes, and retrieves worse without them. `rag/models.py` adds them.

### Grounding: answer from the notes or say so

A small model will happily answer "What port does MQTT use?" from its own memory, even though the notes never mention it. The pipeline has three guards against that:

1. **A relevance gate before the model.** If even the best retrieved chunk scores below `min_relevance` (cosine similarity, 0.60 for `nomic-embed-text`), the question is refused without calling the chat model. On the golden set, answerable questions score at least 0.64 and the off-topic ones between 0.50 and 0.63, so the gate catches the clearly unrelated questions and leaves the close calls to the next two guards.
2. **A grounded prompt.** `rag/prompts.py` tells the model to use only the numbered passages, end each factual sentence with its `[n]`, and reply with the exact refusal sentence when the passages do not contain the answer.
3. **Citations are checked.** `rag/citations.py` keeps only markers that point at a retrieved passage. With `require_citations` on (the default), an answer with no valid citation is replaced by the refusal, because an uncited answer cannot be checked.

`RagResult.refusal_reason` says which guard fired: `low_relevance`, `model` or `no_citations`. Set `RAG_GROUNDED_PROMPT=off` to see the original behaviour. `evals/results/citations.md` compares the two on the golden set. With `gemma2:2b`, grounding took refusals of unanswerable questions from 0 of 5 to 5 of 5 and put a valid citation on all 25 answerable ones, with no false refusals. The price is completeness: judged correctness fell from 0.94 to 0.72, because the shorter grounded answers often name only the items present in the four retrieved chunks, while the basic prompt fills gaps from the model's own knowledge.

## Measuring it: evals

A RAG change is only an improvement if you can measure it. `evals/golden.jsonl` holds 30 questions written from the PDF: 25 answerable ones, each with the printed page numbers that contain the answer and a reference answer, and 5 the PDF cannot answer (tagged `unanswerable`). They cover every section: data types (p1 to 2), urgency levels (p3), processing topologies (p4 to 7), device design factors (p8 to 9), offload locations (p11 to 12), decision approaches (p13 to 14) and offloading considerations (p14 to 15).

Retrieval and generation are measured separately, because they fail for different reasons:

```bash
python -m evals.retrieval            # Recall@1/3/5 and MRR, no LLM needed
python -m evals.answers              # faithfulness and correctness, judged by an LLM
python -m evals.experiments baseline # both, written to evals/results/baseline.md
python -m evals.experiments citations # basic prompt vs grounded prompt, evals/results/citations.md
```

- **Recall@k**: of the pages that hold the answer, the share found in the top k chunks.
- **MRR**: 1 / the rank of the first chunk from a correct page, averaged.
- **Faithfulness**: is every claim in the answer supported by the retrieved chunks? (Scored only for answered questions.)
- **Correctness**: does the answer state the facts in the reference answer? For unanswerable questions, a refusal is correct.

The judge is a short hand-written prompt (`evals/judge.py`) that asks for one word, YES, PARTLY or NO, rather than Ragas. Ragas asks the judge for multi-step JSON that 2B models often break, and a one-line prompt is easier for learners to read. **Small local judges are noisy**: while building this, `gemma2:2b` graded short, fully correct answers as PARTLY until the question was dropped from the correctness prompt and it said wording may differ, and `qwen2.5:1.5b` said YES to everything. `test_judge_calibration_real` in `tests/test_evals_integration.py` checks the local judge against six hand-labelled answers; run it after changing the judge model or prompt. Read the scores as a trend over the whole set, and spot-check a sample with a stronger hosted judge before trusting a small difference:

```bash
pip install langchain-anthropic   # or langchain-openai
python -m evals.answers --judge anthropic --judge-model <model name>   # API key from the environment
```

Every command also takes `--offline` to use the stand-in models (see the tests section); those numbers land in `evals/results/offline/` and measure the plumbing, not model quality.

**Continuous integration.** `.github/workflows/eval.yml` runs the offline test suite and an offline retrieval gate on every push and pull request. A second job installs Ollama on the runner, pulls only `nomic-embed-text`, ingests the PDF and fails the build if Recall@5 drops below the baseline committed in `evals/baselines.json` (`python -m evals.retrieval --check-baseline`). Answer evals need a chat model and a judge, which is slow and noisy on a CI runner, so they run only when started by hand (`workflow_dispatch`). `tests/test_evals_integration.py` proves the gate fails when retrieval is broken (k=1 over a shuffled index).

## Configuration

Every field of `RagConfig` in `rag/config.py` can be set with an environment variable:

| Variable | Default | Meaning |
|----------|---------|---------|
| `RAG_CHAT_MODEL` | `gemma2:2b` | Ollama chat model |
| `RAG_EMBED_MODEL` | `nomic-embed-text` | Ollama embedding model |
| `RAG_CHUNK_SIZE` / `RAG_CHUNK_OVERLAP` | `800` / `120` | Chunking, in characters |
| `RAG_K` | `4` | Chunks given to the LLM |
| `RAG_GROUNDED_PROMPT` | `on` | Grounded, cited prompt; `off` uses the original basic prompt |
| `RAG_REQUIRE_CITATIONS` | `on` | Refuse answers that carry no valid `[n]` citation |
| `RAG_MIN_RELEVANCE` | `0.60` | Refuse before calling the model when the best chunk scores lower |
| `RAG_INDEX_DIR` | `.rag_index` | Where indexes are stored |
| `RAG_PDF_PATH` | `3rd module notes.pdf` | The document |
| `RAG_PROVIDER` | `ollama` | `offline` swaps in the test stand-ins (no Ollama needed) |

## How to run the tests

The project uses **pytest** with tests in the top-level `tests/` folder.

```bash
pip install -r requirements-dev.txt
pytest                      # everything; real-model tests skip if Ollama is not running
pytest -m "not realmodel"   # only the offline suite (what CI runs)
pytest -m realmodel         # only the tests that call Ollama
```

There are two kinds of tests:

- **Offline tests** run the real pipeline end to end on the real PDF (real chunking, a real Chroma index on disk), but with deterministic stand-in models from `rag/offline.py`: hashing embeddings, an extractive "LLM" that quotes the best-matching sentence, and a word-overlap reranker. They need no Ollama and no downloads, so they are fast and give the same result every time.
- **`realmodel` tests** run the same flows against Ollama with the configured models. They skip with a message saying what is missing when Ollama is not reachable or a model is not pulled.

## License

MIT
