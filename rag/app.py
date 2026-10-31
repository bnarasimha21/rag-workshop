"""Step 4: a Gradio web UI over the pipeline.

Run with ``python -m rag.app`` (or ``python ollama_rag.py``). Importing this
module only defines functions; the server starts in ``main()``.
"""

from __future__ import annotations

import os

from rag.config import RETRIEVAL_MODES, RagConfig
from rag.pipeline import Pipeline, RagResult

REFUSAL_REASONS = {
    "low_relevance": "No chunk in the document was close enough to the question, so the model was not asked.",
    "model": "The model found no answer in the retrieved chunks.",
    "no_citations": "The model answered without citing any retrieved chunk, so the answer was withheld.",
}


def snippet(text: str, limit: int = 1000) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + " ..."


def format_sources(result: RagResult) -> str:
    if result.refused:
        return f"_No sources._ {REFUSAL_REASONS.get(result.refusal_reason, '')}"
    if not result.citations:
        return "_This answer has no citations._"
    lines = []
    for c in result.citations:
        lines.append(f"**[{c.marker}] page {c.page}** (`{c.chunk_id}`)\n\n> {snippet(c.text)}")
    return "\n\n".join(lines)


def make_handler(cfg: RagConfig):
    """The function behind the Ask button.

    ``mode`` and ``rerank`` come from the UI controls. Each combination gets its
    own Pipeline, created on first use; they all share one index, because the
    retrieval mode does not change what is stored.
    """
    base = Pipeline(cfg)
    pipelines: dict[tuple[str, bool], Pipeline] = {(cfg.retrieval_mode, cfg.rerank): base}

    def pipeline_for(mode: str, rerank: bool) -> Pipeline:
        key = (mode, bool(rerank))
        if key not in pipelines:
            pipelines[key] = Pipeline(cfg.with_overrides(retrieval_mode=mode, rerank=bool(rerank)), index=base.index)
        return pipelines[key]

    def handle(question: str, mode: str | None = None, rerank: bool | None = None) -> tuple[str, str, str]:
        """Returns (answer, sources panel, trace panel), all Markdown."""
        question = (question or "").strip()
        if not question:
            return "Please type a question.", "", ""
        pipeline = pipeline_for(mode or cfg.retrieval_mode, cfg.rerank if rerank is None else rerank)
        result = pipeline.answer(question)
        return result.answer, format_sources(result), result.trace.as_markdown()

    return handle


def build_ui(cfg: RagConfig | None = None):
    import gradio as gr

    cfg = cfg or RagConfig.from_env()
    handle = make_handler(cfg)
    with gr.Blocks(title="RAG Workshop") as ui:
        gr.Markdown("# RAG System\nAsk a question about the IoT module notes. Answers cite the pages they come from.")
        question = gr.Textbox(lines=2, label="Question", placeholder="Enter your question here...")
        with gr.Row():
            mode = gr.Dropdown(list(RETRIEVAL_MODES), value=cfg.retrieval_mode, label="Retrieval mode")
            rerank = gr.Checkbox(value=cfg.rerank, label="Rerank with a cross-encoder")
        ask = gr.Button("Ask", variant="primary")
        answer = gr.Markdown(label="Answer")
        with gr.Accordion("Sources", open=True):
            sources = gr.Markdown()
        with gr.Accordion("Trace: time and tokens per stage", open=False):
            trace = gr.Markdown()
        inputs, outputs = [question, mode, rerank], [answer, sources, trace]
        ask.click(handle, inputs=inputs, outputs=outputs)
        question.submit(handle, inputs=inputs, outputs=outputs)
    return ui


def main() -> None:
    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
    build_ui().launch()


if __name__ == "__main__":
    main()
