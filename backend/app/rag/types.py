"""Core data types shared by every pluggable RAG component.

These are deliberately plain dataclasses: a data scientist can build them by hand
in a notebook without importing anything from the FastAPI application.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class Document:
    """A source document before chunking."""

    id: str
    text: str
    filename: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Section:
    """A structural unit of a document (heading + body) produced by a loader."""

    heading: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    """An indexable unit of text."""

    id: str
    text: str
    doc_id: str = ""
    filename: str = ""
    heading: str = ""
    ord: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    # Set by parent/sentence-window style chunkers: the larger block to show the LLM.
    parent_text: str | None = None

    @property
    def embed_text(self) -> str:
        return f"{self.heading}\n{self.text}" if self.heading else self.text

    @property
    def context_text(self) -> str:
        """What actually goes into the prompt (expanded window when available)."""
        return self.parent_text or self.text


@dataclass
class ScoredChunk:
    """A chunk with retrieval scores attached."""

    chunk: Chunk
    score: float = 0.0
    # Per-stage diagnostics, e.g. {"dense": 0.82, "bm25": 4.1, "rrf": 0.03, "rerank": 0.91}
    scores: dict[str, float] = field(default_factory=dict)
    rank: int = 0

    def with_score(self, score: float, name: str | None = None) -> "ScoredChunk":
        if name:
            self.scores[name] = score
        self.score = score
        return self

    @property
    def id(self) -> str:
        return self.chunk.id

    @property
    def text(self) -> str:
        return self.chunk.text


@dataclass
class Citation:
    n: int
    chunk_id: str
    doc_id: str
    filename: str
    heading: str
    snippet: str
    score: float = 0.0


@dataclass
class RAGResult:
    """Everything a query produced — answer, evidence and full diagnostics."""

    query: str
    answer: str = ""
    citations: list[Citation] = field(default_factory=list)
    contexts: list[ScoredChunk] = field(default_factory=list)
    # Query variants actually searched (after HyDE / multi-query / step-back).
    expanded_queries: list[str] = field(default_factory=list)
    groundedness: float | None = None
    abstained: bool = False
    latency_ms: float = 0.0
    # Per-stage timings and counters for the trace view and for sweeps.
    trace: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def context_texts(self) -> list[str]:
        return [c.chunk.context_text for c in self.contexts]

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "answer": self.answer,
            "citations": [c.__dict__ for c in self.citations],
            "contexts": [
                {"id": c.id, "score": round(c.score, 4), "scores": c.scores,
                 "filename": c.chunk.filename, "heading": c.chunk.heading, "text": c.chunk.text}
                for c in self.contexts
            ],
            "expanded_queries": self.expanded_queries,
            "groundedness": self.groundedness,
            "abstained": self.abstained,
            "latency_ms": self.latency_ms,
            "trace": self.trace,
            "usage": self.usage,
            "metadata": self.metadata,
        }

    def __repr__(self) -> str:  # notebook-friendly
        head = self.answer[:160] + ("…" if len(self.answer) > 160 else "")
        return f"RAGResult(answer={head!r}, contexts={len(self.contexts)}, latency_ms={self.latency_ms:.0f})"


@dataclass
class EvalSample:
    """One row of a golden evaluation set."""

    question: str
    ground_truth: str | None = None
    # Chunk ids or doc ids that *should* be retrieved.
    relevant_ids: list[str] = field(default_factory=list)
    # Free-text reference contexts, used when ids are unavailable.
    reference_contexts: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class SupportsClose(Protocol):
    async def close(self) -> None: ...
