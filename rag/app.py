"""Step 4: a Gradio web UI over the pipeline.

Run with ``python -m rag.app`` (or ``python ollama_rag.py``). Importing this
module only defines functions; the server starts in ``main()``.
"""

from __future__ import annotations

import os

from rag.config import RagConfig
from rag.pipeline import Pipeline


def make_handler(cfg: RagConfig):
    pipeline = Pipeline(cfg)

    def handle(question: str) -> str:
        question = (question or "").strip()
        if not question:
            return "Please type a question."
        return pipeline.answer(question).answer

    return handle


def build_ui(cfg: RagConfig | None = None):
    import gradio as gr

    cfg = cfg or RagConfig.from_env()
    return gr.Interface(
        fn=make_handler(cfg),
        inputs=gr.Textbox(lines=2, placeholder="Enter your question here..."),
        outputs="text",
        title="RAG System",
        description="Ask a question and get an answer based on the document context.",
    )


def main() -> None:
    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
    build_ui().launch()


if __name__ == "__main__":
    main()
