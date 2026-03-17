# RAG Workshop

Hands-on workshop materials for building a Retrieval-Augmented Generation (RAG) pipeline using LangChain, Ollama, and ChromaDB. Load a PDF document, create vector embeddings, and query it with natural language through a Gradio interface.

## Features

- **PDF Document Loading** -- Ingest PDF files using LangChain's PyPDFLoader
- **Vector Store** -- Create and query document embeddings with ChromaDB and GPT4All embeddings
- **Local LLM** -- Run inference locally using Ollama with the Gemma model
- **Interactive Q&A Interface** -- Ask questions about your documents via a Gradio web UI
- **Similarity Search** -- Retrieve the top-k most relevant document chunks for each query

## Tech Stack

- **Language:** Python
- **RAG Framework:** LangChain, LangChain Community
- **Vector Store:** ChromaDB
- **Embeddings:** GPT4All Embeddings
- **LLM:** Ollama (Gemma model)
- **PDF Processing:** PyPDF2 / pypdf
- **Web Interface:** Gradio
- **ML:** PyTorch, Sentence Transformers

## Prerequisites

- [Ollama](https://ollama.com/) installed and running locally
- Gemma model pulled: `ollama pull gemma`

## Setup / Installation

```bash
git clone https://github.com/bnarasimha21/rag-workshop.git
cd rag-workshop
pip install -r requirements.txt
```

## Usage

```bash
python ollama_rag.py
```

This will:

1. Load the included PDF document
2. Create vector embeddings and store them in ChromaDB
3. Launch a Gradio interface where you can ask questions about the document

Open the Gradio URL in your browser and type your question to get answers grounded in the document content.

## License

MIT
