"""Step 1 of RAG: turn the PDF into searchable chunks and store them.

Run it with ``python -m rag.ingest``. It reads the PDF page by page, cleans
each page, splits it into overlapping chunks, embeds the chunks and saves them
in a Chroma database on disk. A second run with the same PDF and settings
finds the saved index and skips the (slow) embedding step.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.config import RagConfig, cli_config
from rag.text import clean_page_text, printed_page

log = logging.getLogger("rag.ingest")

COLLECTION = "chunks"
MANIFEST = "manifest.json"
CHUNKS_FILE = "chunks.jsonl"


# ---- loading and splitting ---------------------------------------------


def load_pages(pdf_path: Path, max_pages: int = 0) -> list[Document]:
    """One Document per PDF page, with 0-indexed ``page`` metadata.

    This is what LangChain's ``PyPDFLoader`` does; we call ``pypdf`` directly
    so the workshop does not need the large ``langchain-community`` package.
    """
    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    pages = []
    for i, page in enumerate(reader.pages):
        if max_pages and i >= max_pages:
            break
        text = clean_page_text(page.extract_text() or "")
        if text:
            pages.append(Document(page_content=text, metadata={"source": Path(pdf_path).name, "page": i}))
    return pages


def make_splitter(cfg: RagConfig) -> RecursiveCharacterTextSplitter:
    # Tries to split on paragraphs first, then lines, then sentences, then
    # words, so chunks end at natural boundaries whenever they can.
    return RecursiveCharacterTextSplitter(
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def split_pages(pages: list[Document], cfg: RagConfig) -> list[Document]:
    """Split each page on its own, so every chunk belongs to exactly one page."""
    splitter = make_splitter(cfg)
    chunks: list[Document] = []
    for page in pages:
        for j, piece in enumerate(splitter.split_text(page.page_content)):
            meta = dict(page.metadata)
            meta["chunk_id"] = f"p{printed_page(meta['page']):02d}-c{j:02d}"
            chunks.append(Document(page_content=piece, metadata=meta))
    return chunks


# ---- the persistent index ------------------------------------------------


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 16), b""):
            h.update(block)
    return h.hexdigest()


def _index_prefix(cfg: RagConfig) -> str:
    slug = "".join(c if c.isalnum() or c in "-." else "-" for c in cfg.embed_name)
    return f"{slug}-{cfg.index_fingerprint()}-"


def index_path(cfg: RagConfig, pdf_hash: str | None = None) -> Path:
    """One folder per (settings, PDF version), e.g. ``.rag_index/nomic-embed-text-3f2a..-9c1e..``.

    A new PDF or new settings get a new folder rather than overwriting the old
    one, because Chroma keeps open database handles per folder for the life of
    the process; rewriting a folder in place breaks a running app.
    """
    pdf_hash = pdf_hash or file_sha256(Path(cfg.pdf_path))
    return Path(cfg.index_dir) / f"{_index_prefix(cfg)}{pdf_hash[:10]}"


@dataclass
class IngestReport:
    path: Path
    skipped: bool
    n_chunks: int
    seconds: float
    context_calls: int = 0
    context_tokens: int = 0
    context_cache_hits: int = 0
    context_seconds: float = 0.0
    context_full_tokens: int = 0       # one-time cost of contextualizing every chunk,
    context_full_seconds: float = 0.0  # even when this run read them all from the cache


def _chroma(cfg: RagConfig, path: Path):
    import chromadb
    from langchain_chroma import Chroma

    from rag.models import make_embeddings

    return Chroma(
        collection_name=COLLECTION,
        embedding_function=make_embeddings(cfg),
        persist_directory=str(path),
        collection_metadata={"hnsw:space": "cosine"},
        client_settings=chromadb.config.Settings(anonymized_telemetry=False, is_persistent=True),
    )


def _is_current(path: Path, pdf_hash: str, cfg: RagConfig) -> bool:
    manifest = path / MANIFEST
    if not manifest.exists() or not (path / CHUNKS_FILE).exists():
        return False
    try:
        data = json.loads(manifest.read_text())
    except (json.JSONDecodeError, UnicodeDecodeError):
        return False  # a manifest cut short by a crash: rebuild
    return data.get("pdf_sha256") == pdf_hash and data.get("fingerprint") == cfg.index_fingerprint()


def embedding_text(chunk: Document) -> str:
    """What gets embedded and BM25-indexed: the generated context (if any) plus the chunk."""
    context = chunk.metadata.get("context", "")
    return f"{context}\n\n{chunk.page_content}" if context else chunk.page_content


def ingest(cfg: RagConfig, force: bool = False) -> IngestReport:
    start = time.perf_counter()
    pdf = Path(cfg.pdf_path)
    if not pdf.exists():
        raise FileNotFoundError(f"PDF not found: {pdf}")
    pdf_hash = file_sha256(pdf)
    path = index_path(cfg, pdf_hash)

    if not force and _is_current(path, pdf_hash, cfg):
        n = sum(1 for _ in open(path / CHUNKS_FILE))
        log.info("Index is up to date (%s, %d chunks); PDF unchanged, skipping embedding.", path, n)
        return IngestReport(path, True, n, time.perf_counter() - start)

    if path.exists():  # a half-built index from an interrupted run
        _forget_chroma_clients()
        shutil.rmtree(path)
    path.mkdir(parents=True)

    pages = load_pages(pdf, cfg.max_pages)
    chunks = split_pages(pages, cfg)
    log.info("Loaded %d pages, split into %d chunks (size %d, overlap %d).",
             len(pages), len(chunks), cfg.chunk_size, cfg.chunk_overlap)

    report = IngestReport(path, False, len(chunks), 0.0)
    if cfg.contextual_chunks:
        from rag.contextualize import contextualize

        stats = contextualize(chunks, pages, cfg)
        report.context_calls, report.context_cache_hits = stats.calls, stats.cache_hits
        report.context_tokens, report.context_seconds = stats.tokens, stats.seconds
        report.context_full_tokens, report.context_full_seconds = stats.full_tokens, stats.full_seconds

    # Chroma stores the text it embeds, so we store the context-enriched text
    # there and keep the original chunk text in metadata for citations.
    store = _chroma(cfg, path)
    store.add_texts(
        texts=[embedding_text(c) for c in chunks],
        metadatas=[{**c.metadata, "text": c.page_content} for c in chunks],
        ids=[c.metadata["chunk_id"] for c in chunks],
    )
    with open(path / CHUNKS_FILE, "w") as fh:
        for c in chunks:
            fh.write(json.dumps({"text": c.page_content, "metadata": c.metadata}) + "\n")
    # The manifest is written last: if embedding fails half way, the next run
    # sees no manifest and rebuilds instead of trusting a partial index.
    tmp = path / (MANIFEST + ".tmp")
    tmp.write_text(json.dumps({
        "pdf": pdf.name,
        "pdf_sha256": pdf_hash,
        "fingerprint": cfg.index_fingerprint(),
        "embed_model": cfg.embed_name,
        "chunk_size": cfg.chunk_size,
        "chunk_overlap": cfg.chunk_overlap,
        "contextual_chunks": cfg.contextual_chunks,
        "context_model": cfg.effective_context_model if cfg.contextual_chunks else None,
        "max_pages": cfg.max_pages or None,
        "context_cost": {"llm_calls": report.n_chunks, "tokens": report.context_full_tokens,
                         "seconds": round(report.context_full_seconds, 1)} if cfg.contextual_chunks else None,
        "n_chunks": len(chunks),
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, indent=2))
    tmp.replace(path / MANIFEST)  # atomic: the manifest is either whole or absent
    report.seconds = time.perf_counter() - start
    log.info("Embedded and stored %d chunks in %s (%.1fs).", len(chunks), path, report.seconds)
    return report


def _forget_chroma_clients() -> None:
    from chromadb.api.client import SharedSystemClient

    SharedSystemClient.clear_system_cache()


def load_chunks(path: Path) -> list[Document]:
    with open(path / CHUNKS_FILE) as fh:
        return [Document(page_content=row["text"], metadata=row["metadata"]) for row in map(json.loads, fh)]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build the persistent index for the workshop PDF.")
    parser.add_argument("--force", action="store_true", help="rebuild even if the PDF is unchanged")
    parser.add_argument("--offline", action="store_true", help="use the offline stand-in models")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = cli_config(offline=args.offline)
    ingest(cfg, force=args.force)


if __name__ == "__main__":
    main()
