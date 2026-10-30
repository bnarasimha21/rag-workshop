"""Issue 1: chunking, metadata and the persistent index."""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys

from langchain_core.documents import Document

from rag.config import offline_config
from rag.ingest import index_path, ingest, load_pages, split_pages
from rag.retrievers import Index
from rag.text import clean_page_text, printed_page

from conftest import REPO


def test_splitter_keeps_page_metadata_and_respects_chunk_size():
    page_text = "\n".join(f"- Point {i:02d} on fog and edge." for i in range(60))
    page = Document(page_content=page_text, metadata={"source": "notes.pdf", "page": 11})
    cfg = offline_config(chunk_size=300, chunk_overlap=50)
    chunks = split_pages([page], cfg)
    assert len(chunks) > 5
    for i, c in enumerate(chunks):
        assert len(c.page_content) <= 300
        assert c.metadata["page"] == 11
        assert c.metadata["source"] == "notes.pdf"
        assert c.metadata["chunk_id"] == f"p12-c{i:02d}"
    # overlap: the start of each chunk repeats text from the end of the previous one
    assert chunks[1].page_content.split("\n")[0] in chunks[0].page_content


def test_clean_page_text_strips_running_header_and_bullet_glyphs():
    raw = "Module \n3\n \nMoModul\ne 3 \n Internet of Things[21CS735]     12 \n \nEdge: \n Offloadinghere"
    assert clean_page_text(raw) == "Edge: \n- Offloading here"


def test_real_pdf_pages_and_chunks_have_metadata(tmp_path):
    cfg = offline_config(index_dir=tmp_path)
    pages = load_pages(cfg.pdf_path)
    assert len(pages) == 16
    assert [p.metadata["page"] for p in pages] == list(range(16))
    assert "Internet of Things[21CS735]" not in pages[11].page_content
    assert pages[11].page_content.startswith("Edge:")
    chunks = split_pages(pages, cfg)
    ids = [c.metadata["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids))
    for c in chunks:
        assert {"page", "source", "chunk_id"} <= set(c.metadata)
        assert c.metadata["source"] == "3rd module notes.pdf"
        assert c.metadata["chunk_id"].startswith(f"p{printed_page(c.metadata['page']):02d}-")
        assert len(c.page_content) <= cfg.chunk_size


def test_ingest_persists_and_second_run_skips(tmp_path, caplog):
    cfg = offline_config(index_dir=tmp_path)
    caplog.set_level(logging.INFO, logger="rag.ingest")
    first = ingest(cfg)
    assert not first.skipped and first.n_chunks > 30
    second = ingest(cfg)
    assert second.skipped and second.n_chunks == first.n_chunks
    assert "skipping embedding" in caplog.text

    # Every chunk stored in Chroma carries page, source and chunk_id metadata.
    index = Index(cfg, second.path)
    stored = index.store.get(include=["metadatas"])
    assert len(stored["ids"]) == first.n_chunks
    for meta in stored["metadatas"]:
        assert {"page", "source", "chunk_id", "text"} <= set(meta)


def test_ingest_rebuilds_when_pdf_or_settings_change(tmp_path):
    pdf = tmp_path / "notes.pdf"
    shutil.copy(REPO / "3rd module notes.pdf", pdf)
    cfg = offline_config(index_dir=tmp_path / "idx", pdf_path=pdf)
    assert not ingest(cfg).skipped
    assert ingest(cfg).skipped
    with open(pdf, "ab") as fh:  # bytes after %%EOF: same text, new hash
        fh.write(b"\n% edited\n")
    assert not ingest(cfg).skipped
    # A different chunk size is a different index folder, built from scratch.
    other = cfg.with_overrides(chunk_size=500)
    assert index_path(other) != index_path(cfg)
    assert not ingest(other).skipped


def test_ingest_cli_twice(subprocess_env):
    cmd = [sys.executable, "-m", "rag.ingest"]
    first = subprocess.run(cmd, env=subprocess_env, cwd=REPO, capture_output=True, text=True, timeout=120)
    assert first.returncode == 0, first.stderr
    assert "Embedded and stored" in first.stderr
    second = subprocess.run(cmd, env=subprocess_env, cwd=REPO, capture_output=True, text=True, timeout=120)
    assert second.returncode == 0, second.stderr
    assert "skipping embedding" in second.stderr
