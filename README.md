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
| 3. Generate | `rag/pipeline.py` | Puts the chunks into a prompt and asks the chat model. `answer(question)` returns a `RagResult`. |
| 4. UI | `rag/app.py` | A Gradio page around `Pipeline.answer`. Importing it starts nothing; the server starts in `main()`. |

Things worth noticing while you read:

- **Chunking.** The retrieval unit is a chunk, not a page. Smaller chunks are more specific; too small and a definition gets cut in half. Try `RAG_CHUNK_SIZE=400` and compare.
- **Every chunk keeps its page.** Chunks carry `source`, `page` and a stable `chunk_id` such as `p12-c01` (printed page 12, chunk 01 of that page). Pages are split one at a time, so a chunk never spans two pages. `pypdf` numbers pages from 0; `rag/text.py` holds the single conversion to the printed page number that people see.
- **The index persists.** Ingest stores a manifest with the PDF's SHA-256 and the settings that affect the index. Running it again with the same PDF and settings prints `Index is up to date ... skipping embedding.` Changing the PDF or a setting such as chunk size builds a new index in its own folder. `python -m rag.ingest --force` rebuilds anyway.
- **Task prefixes matter.** `nomic-embed-text` was trained with `search_query: ` and `search_document: ` prefixes, and retrieves worse without them. `rag/models.py` adds them.

## Configuration

Every field of `RagConfig` in `rag/config.py` can be set with an environment variable:

| Variable | Default | Meaning |
|----------|---------|---------|
| `RAG_CHAT_MODEL` | `gemma2:2b` | Ollama chat model |
| `RAG_EMBED_MODEL` | `nomic-embed-text` | Ollama embedding model |
| `RAG_CHUNK_SIZE` / `RAG_CHUNK_OVERLAP` | `800` / `120` | Chunking, in characters |
| `RAG_K` | `4` | Chunks given to the LLM |
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
