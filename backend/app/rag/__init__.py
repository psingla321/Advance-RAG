"""AdvanceRAG RAG Lab — a plug-and-play toolkit for building and evaluating RAG systems.

Quick start::

    from app.rag import RAG

    rag = RAG("hybrid")
    rag.add_directory("./my_docs")
    rag.print_answer("What is our hotel limit in high-cost cities?")

Compare approaches on your own data::

    from app.rag import Sweep, Dataset

    sweep = Sweep("hybrid", Dataset.load("golden.jsonl"), corpus="./my_docs")
    sweep.grid(**{"retriever.type": ["dense", "hybrid_rrf", "mmr"]})
    print(sweep.run().leaderboard())

Add your own component::

    from app.rag import register, Reranker

    @register("reranker", "recency")
    class RecencyReranker(Reranker):
        async def rerank(self, query, candidates, top_n):
            return sorted(candidates, key=lambda c: c.chunk.metadata.get("date", ""), reverse=True)[:top_n]

    RAG(reranker="recency")
"""

from app.rag import components  # noqa: F401  (registers all built-in components)
from app.rag.base import (
    Chunker,
    Component,
    Compressor,
    Embedder,
    Generator,
    LLM,
    Loader,
    Metric,
    QueryTransform,
    Reranker,
    Retriever,
    VectorStore,
)
from app.rag.client import RAG, compare_configs, run_sync
from app.rag.config import RAGConfig, available_presets
from app.rag.evaluation import (  # noqa: F401  (registers metrics)
    Dataset,
    EvalReport,
    compare,
    evaluate,
    evaluate_retrieval,
    generate_dataset,
)
from app.rag.experiments import Sweep, SweepResults
from app.rag.pipeline import RAGPipeline, build_pipeline
from app.rag.registry import ComponentError, build, list_components, names, register
from app.rag.types import Chunk, Citation, Document, EvalSample, RAGResult, ScoredChunk, Section

__version__ = "1.0.0"

__all__ = [
    # facade
    "RAG", "RAGConfig", "RAGPipeline", "build_pipeline", "compare_configs", "run_sync",
    # registry
    "register", "build", "list_components", "names", "ComponentError", "available_presets",
    # interfaces to subclass
    "Component", "LLM", "Embedder", "VectorStore", "Loader", "Chunker", "QueryTransform",
    "Retriever", "Reranker", "Compressor", "Generator", "Metric",
    # data types
    "Document", "Section", "Chunk", "ScoredChunk", "Citation", "RAGResult", "EvalSample",
    # evaluation and experiments
    "Dataset", "EvalReport", "evaluate", "evaluate_retrieval", "compare", "generate_dataset",
    "Sweep", "SweepResults",
    "__version__",
]
