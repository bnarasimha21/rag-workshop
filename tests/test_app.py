"""Issue 1: the Gradio app builds without launching, and `python ollama_rag.py` still serves it."""

from __future__ import annotations

import subprocess
import sys
import time
import urllib.request

from rag.app import build_ui, make_handler

from conftest import REPO, free_port


def test_handler_answers_without_a_server(offline_cfg):
    handle = make_handler(offline_cfg())
    assert handle("   ") == "Please type a question."
    assert handle("What are the two broad types of data?").strip()


def test_build_ui_does_not_launch(offline_cfg):
    ui = build_ui(offline_cfg())
    assert ui is not None
    assert not getattr(ui, "is_running", False)


def test_ollama_rag_py_launches_gradio(subprocess_env):
    port = free_port()
    env = dict(subprocess_env, GRADIO_SERVER_PORT=str(port), GRADIO_SERVER_NAME="127.0.0.1")
    proc = subprocess.Popen([sys.executable, "ollama_rag.py"], cwd=REPO, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        deadline = time.time() + 90
        status = None
        while time.time() < deadline and proc.poll() is None:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=2) as resp:
                    status = resp.status
                    break
            except OSError:
                time.sleep(0.5)
        assert status == 200, f"Gradio did not come up; exit code {proc.poll()}"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
