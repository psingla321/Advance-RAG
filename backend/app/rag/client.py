"""``RAG`` — the synchronous, notebook-first facade.

    from app.rag import RAG

    rag = RAG("hybrid")                 # or RAG(retriever="mmr", llm={"type": "openai"})
    rag.add_directory("./docs")
    print(rag.ask("What is the hotel limit?"))

Everything underneath is async; this wrapper runs the loop for you, including
inside Jupyter where an event loop is already running.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import threading
from pathlib import Path
from typing import Any, Coroutine, TypeVar

from app.rag.config import RAGConfig
from app.rag.pipeline import RAGPipeline
from app.rag.types import Document, RAGResult

log = logging.getLogger(__name__)

T = TypeVar("T")

_loop: asyncio.AbstractEventLoop | None = None
_thread: threading.Thread | None = None
_lock = threading.Lock()


def _background_loop() -> asyncio.AbstractEventLoop:
    """A dedicated event loop on its own thread.

    Using one is what makes these helpers work identically in a script, in
    Jupyter (which already runs a loop) and inside a FastAPI request.
    """
    global _loop, _thread
    with _lock:
        if _loop is None or _loop.is_closed():
            _loop = asyncio.new_event_loop()
            _thread = threading.Thread(target=_loop.run_forever, name="advance-rag-loop", daemon=True)
            _thread.start()
    return _loop


def run_sync(coro: Coroutine[Any, Any, T], timeout: float | None = None) -> T:
    """Run a coroutine from synchronous code, in a notebook or not."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    future = asyncio.run_coroutine_threadsafe(coro, _background_loop())
    return future.result(timeout)


class RAG:
    """A ready-to-use RAG system.

    Accepts a preset name, a config file path, a :class:`RAGConfig`, or keyword
    overrides for any stage::

        RAG()                                     # offline defaults, no setup
        RAG("advanced")                           # a built-in recipe
        RAG("configs/mine.yaml")                  # your own file
        RAG(retriever={"type": "mmr", "lambda_mult": 0.7}, top_n=5)
    """

    def __init__(self, config: RAGConfig | dict[str, Any] | str | Path | None = None, **overrides: Any) -> None:
        self.config = self._resolve(config, overrides)
        self.pipeline = RAGPipeline(self.config)
        run_sync(self.pipeline.init())

    @staticmethod
    def _resolve(config: Any, overrides: dict[str, Any]) -> RAGConfig:
        if config is None:
            resolved = RAGConfig()
        elif isinstance(config, RAGConfig):
            resolved = config
        elif isinstance(config, dict):
            resolved = RAGConfig.from_dict(config)
        else:
            path = Path(str(config))
            resolved = RAGConfig.from_yaml(path) if path.exists() else RAGConfig.preset(str(config))
        return resolved.with_overrides(**overrides) if overrides else resolved

    # ------------------------------------------------------------ ingestion
    def add_directory(self, path: str | Path, recursive: bool = True, show_progress: bool = True) -> dict[str, Any]:
        """Index every supported file under ``path``."""
        return run_sync(self.pipeline.add_directory(path, recursive, show_progress=show_progress))

    def add_files(self, paths: list[str | Path], **kwargs: Any) -> dict[str, Any]:
        return run_sync(self.pipeline.add_files(paths, **kwargs))

    def add_texts(self, texts: list[str], metadatas: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        return run_sync(self.pipeline.add_texts(texts, metadatas))

    def add_documents(self, documents: list[Document], **kwargs: Any) -> dict[str, Any]:
        return run_sync(self.pipeline.add_documents(documents, **kwargs))

    def add_dataframe(self, df: Any, text_column: str, id_column: str | None = None) -> dict[str, Any]:
        """Index a pandas DataFrame; other columns become filterable metadata."""
        texts = df[text_column].astype(str).tolist()
        metadatas = [
            {**{k: v for k, v in row.items() if k != text_column},
             **({"id": str(row[id_column])} if id_column else {})}
            for row in df.to_dict("records")
        ]
        return self.add_texts(texts, metadatas)

    def delete(self, doc_ids: list[str]) -> None:
        run_sync(self.pipeline.delete(doc_ids))

    def count(self) -> int:
        """Number of indexed chunks."""
        return run_sync(self.pipeline.count())

    # ---------------------------------------------------------------- query
    def ask(self, question: str, **kwargs: Any) -> RAGResult:
        """Ask a question. ``print(result)`` shows the answer; the object carries everything else."""
        return run_sync(self.pipeline.query(question, **kwargs))

    def __call__(self, question: str, **kwargs: Any) -> RAGResult:
        return self.ask(question, **kwargs)

    def ask_many(self, questions: list[str], concurrency: int = 4, **kwargs: Any) -> list[RAGResult]:
        return run_sync(self.pipeline.batch_query(questions, concurrency, show_progress=True, **kwargs))

    def retrieve(self, question: str, top_n: int | None = None, **kwargs: Any) -> list[Any]:
        """Retrieval only — inspect what the LLM would have been given."""
        contexts, _, _ = run_sync(self.pipeline.retrieve(question, top_n=top_n, **kwargs))
        return contexts

    def stream(self, question: str, **kwargs: Any):
        """Yield streaming events synchronously (for a CLI or a Streamlit app)."""
        queue: "concurrent.futures.Future[Any]" = concurrent.futures.Future()  # noqa: F841
        import queue as queue_module

        events: queue_module.Queue[Any] = queue_module.Queue()
        sentinel = object()

        async def pump() -> None:
            try:
                async for event in self.pipeline.stream(question, **kwargs):
                    events.put(event)
            except Exception as exc:
                events.put({"type": "error", "message": str(exc)})
            finally:
                events.put(sentinel)

        asyncio.run_coroutine_threadsafe(pump(), _background_loop())
        while True:
            event = events.get()
            if event is sentinel:
                return
            yield event

    def print_answer(self, question: str, show_sources: bool = True, **kwargs: Any) -> RAGResult:
        """Ask and pretty-print answer, citations and timing. Handy in a notebook."""
        result = self.ask(question, **kwargs)
        print(f"Q: {question}\n")
        print(result.answer or "(no answer)")
        if show_sources and result.citations:
            print("\nSources:")
            for citation in result.citations:
                label = f"{citation.filename}" + (f" — {citation.heading}" if citation.heading else "")
                print(f"  [{citation.n}] {label}  (score {citation.score})")
        grounded = f", groundedness {result.groundedness}" if result.groundedness is not None else ""
        print(f"\n{result.latency_ms:.0f} ms{grounded}")
        return result

    def explain(self, question: str, **kwargs: Any) -> dict[str, Any]:
        """Full diagnostics for one query: variants searched, per-stage timing, scores.

        The first thing to run when an answer looks wrong.
        """
        result = self.ask(question, **kwargs)
        return {
            "question": question,
            "queries_searched": result.expanded_queries,
            "stages": result.trace,
            "contexts": [
                {"rank": i, "score": round(c.score, 4), "scores": c.scores,
                 "source": c.chunk.filename, "heading": c.chunk.heading, "preview": c.chunk.text[:200]}
                for i, c in enumerate(result.contexts, 1)
            ],
            "answer": result.answer,
            "groundedness": result.groundedness,
            "latency_ms": result.latency_ms,
        }

    # ----------------------------------------------------------- evaluation
    def evaluate(self, dataset: Any, metrics: list[str] | None = None, **kwargs: Any) -> Any:
        """Score this pipeline on a golden dataset. Returns an ``EvalReport``."""
        from app.rag.evaluation.runner import evaluate as _evaluate

        return run_sync(_evaluate(self.pipeline, dataset, metrics, **kwargs))

    def evaluate_retrieval(self, dataset: Any, k_values: list[int] | None = None) -> dict[str, Any]:
        from app.rag.evaluation.runner import evaluate_retrieval as _evaluate_retrieval

        return run_sync(_evaluate_retrieval(self.pipeline, dataset, k_values))

    def generate_testset(self, n_questions: int = 50, **kwargs: Any) -> Any:
        """Synthesise a golden dataset from the indexed corpus (needs a real LLM)."""
        from app.rag.evaluation.synth import generate_from_pipeline

        return run_sync(generate_from_pipeline(self.pipeline, n_questions=n_questions, **kwargs))

    # ------------------------------------------------------------ utilities
    def describe(self) -> dict[str, Any]:
        """The component graph actually in use."""
        return self.pipeline.describe()

    def save_config(self, path: str | Path) -> str:
        return self.config.to_yaml(path)

    @property
    def stats(self) -> dict[str, Any]:
        return self.pipeline.stats

    def close(self) -> None:
        run_sync(self.pipeline.close())

    def __enter__(self) -> "RAG":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        parts = self.describe()["components"]
        return (
            f"RAG({self.config.name!r}: chunker={parts['chunker']['type']}, "
            f"retriever={parts['retriever']['type']}, reranker={parts['reranker']['type']}, "
            f"generator={parts['generator']['type']}, llm={parts['llm']['type']})"
        )


def compare_configs(
    configs: list[str | dict[str, Any] | RAGConfig],
    dataset: Any,
    corpus: str | Path | list[str] | None = None,
    metrics: list[str] | None = None,
    **kwargs: Any,
) -> Any:
    """One-liner A/B test of several configurations.

        compare_configs(["baseline", "hybrid", "advanced"], dataset, corpus="./docs")
    """
    from app.rag.experiments.sweep import Sweep

    sweep = Sweep(dataset=dataset, corpus=corpus, metrics=metrics)
    for config in configs:
        sweep.add(config)
    return sweep.run(**kwargs)
