"""Shared fixtures.

Two kinds of tests live here:

- Offline tests (the default, and what CI runs): the real PDF, the real
  chunking, a real Chroma index on disk, but the deterministic stand-in models
  from rag/offline.py. No Ollama, no downloads.
- ``@pytest.mark.realmodel`` tests: the same flows against a running Ollama.
  They skip with a clear message when Ollama or a model is missing.
"""

from __future__ import annotations

import json
import os
import socket
import urllib.request
from pathlib import Path

import pytest

from rag.config import RagConfig, offline_config

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def shared_index_dir(tmp_path_factory) -> Path:
    """One offline index for the whole session; building it takes about a second."""
    return tmp_path_factory.mktemp("index")


@pytest.fixture
def offline_cfg(shared_index_dir):
    def make(**overrides) -> RagConfig:
        return offline_config(index_dir=shared_index_dir, **overrides)

    return make


def _ollama_models(base_url: str) -> set[str] | None:
    try:
        with urllib.request.urlopen(f"{base_url}/api/tags", timeout=2) as resp:
            data = json.load(resp)
    except OSError:
        return None
    names = set()
    for m in data.get("models", []):
        names.add(m["name"])
        names.add(m["name"].removesuffix(":latest"))
    return names


@pytest.fixture(scope="session")
def real_index_dir(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("real_index")


@pytest.fixture
def real_cfg(real_index_dir):
    """Factory for configs that use the real Ollama models; skips if unavailable."""
    base = RagConfig.from_env(index_dir=real_index_dir)
    models = _ollama_models(base.ollama_base_url)
    if models is None:
        pytest.skip(f"Ollama is not reachable at {base.ollama_base_url}; start it with `ollama serve` to run real-model tests")
    for needed in (base.chat_model, base.embed_model):
        if needed not in models:
            pytest.skip(f"Ollama model {needed!r} is not pulled; run `ollama pull {needed}`")

    def make(**overrides) -> RagConfig:
        return base.with_overrides(**overrides)

    return make


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def subprocess_env(tmp_path):
    env = dict(os.environ)
    env.update({
        "RAG_PROVIDER": "offline",
        "RAG_MIN_RELEVANCE": "0.08",
        "RAG_INDEX_DIR": str(tmp_path / "idx"),
        "GRADIO_ANALYTICS_ENABLED": "False",
        "PYTHONPATH": str(REPO),
    })
    return env
