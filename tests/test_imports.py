"""Issue 1: importing the package has no side effects and uses current integrations."""

from __future__ import annotations

import subprocess
import sys
import warnings

from conftest import REPO

BLOCK_NETWORK = """
import socket, sys
def refuse(*a, **k):
    raise RuntimeError("network call during import")
socket.socket.connect = refuse
socket.create_connection = refuse
import rag.pipeline, rag.app, rag.ingest, rag.retrievers
import ollama_rag
assert "gradio" not in sys.modules, "gradio was imported at import time"
print("ok")
"""


def test_import_starts_no_server_and_makes_no_network_call():
    out = subprocess.run([sys.executable, "-c", BLOCK_NETWORK], cwd=REPO, capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "ok"


def test_no_langchain_community_imports_in_source():
    sources = list((REPO / "rag").glob("*.py")) + [REPO / "ollama_rag.py"]
    for path in sources:
        assert "langchain_community" not in path.read_text(), path


def test_real_integrations_construct_without_deprecation_warnings(tmp_path):
    from rag.config import RagConfig
    from rag.ingest import _chroma
    from rag.models import make_chat_model, make_embeddings

    cfg = RagConfig(index_dir=tmp_path)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        make_chat_model(cfg)
        make_embeddings(cfg)
        _chroma(cfg, tmp_path / "store")
    bad = [w for w in caught if issubclass(w.category, DeprecationWarning) and "langchain" in str(w.filename).lower()]
    assert not bad, [str(w.message) for w in bad]
