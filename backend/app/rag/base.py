"""Abstract interfaces for every pluggable stage.

Subclass one of these, decorate it with ``@register(kind, name)``, and it becomes
usable from YAML config, the CLI, sweeps and the REST API with no other changes.
All methods are async; the ``RAG`` facade provides synchronous wrappers so the
same components work in a notebook.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from app.rag.types import Chunk, Document, RAGResult, ScoredChunk, Section


class Component:
    """Common base: carries the registered name and optional async cleanup."""

    component_name: str = "unnamed"
    component_kind: str = "unknown"

    def describe(self) -> dict[str, Any]:
        return {"type": self.component_name, "kind": self.component_kind}

    async def close(self) -> None:
        return None


class LLM(Component):
    """Text generation backend."""

    is_mock: bool = False

    async def complete(
        self, messages: list[dict[str, Any]], *, json_mode: bool = False,
        temperature: float = 0.1, max_tokens: int = 800,
    ) -> str:
        raise NotImplementedError

    async def stream(
        self, messages: list[dict[str, Any]], *, temperature: float = 0.1, max_tokens: int = 800
    ) -> AsyncIterator[str]:
        """Default: fall back to a single non-streamed completion."""
        yield await self.complete(messages, temperature=temperature, max_tokens=max_tokens)

    async def ask(self, prompt: str, *, system: str = "", **kwargs: Any) -> str:
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        return await self.complete(messages, **kwargs)


class Embedder(Component):
    """Maps text to vectors. ``dim`` must be stable for a given configuration."""

    dim: int = 0
    model_id: str = "unknown"

    async def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    async def embed_query(self, text: str) -> list[float]:
        """Override when the model has an asymmetric query prefix/instruction."""
        return (await self.embed([text]))[0]


class VectorStore(Component):
    """Durable (or in-memory) store for chunk vectors plus payloads."""

    async def init(self, dim: int) -> None:
        return None

    async def upsert(self, chunks: list[Chunk], vectors: list[list[float]], namespace: str = "default") -> None:
        raise NotImplementedError

    async def search(
        self, vector: list[float], k: int, namespace: str = "default", filters: dict[str, Any] | None = None
    ) -> list[ScoredChunk]:
        raise NotImplementedError

    async def delete(self, doc_ids: list[str], namespace: str = "default") -> None:
        raise NotImplementedError

    async def all_chunks(self, namespace: str = "default") -> list[Chunk]:
        """Used by keyword/BM25 retrievers and evaluation. May be expensive."""
        return []

    async def count(self, namespace: str = "default") -> int:
        return len(await self.all_chunks(namespace))


class Loader(Component):
    """Turns raw bytes into structured sections."""

    extensions: tuple[str, ...] = ()

    def load(self, filename: str, data: bytes) -> list[Section]:
        raise NotImplementedError


class Chunker(Component):
    """Splits sections into indexable chunks."""

    async def chunk(self, document: Document, sections: list[Section]) -> list[Chunk]:
        raise NotImplementedError


class QueryTransform(Component):
    """Rewrites / expands a query into one or more search queries."""

    async def transform(self, query: str, context: dict[str, Any] | None = None) -> list[str]:
        raise NotImplementedError


class Retriever(Component):
    """Fetches candidate chunks for one or more query variants."""

    async def retrieve(
        self, queries: list[str], k: int, namespace: str = "default", filters: dict[str, Any] | None = None
    ) -> list[ScoredChunk]:
        raise NotImplementedError

    async def index(self, chunks: list[Chunk], namespace: str = "default") -> None:
        """Hook for retrievers holding their own index (BM25, parent store)."""
        return None


class Reranker(Component):
    """Reorders candidates with a stronger (usually slower) signal."""

    async def rerank(self, query: str, candidates: list[ScoredChunk], top_n: int) -> list[ScoredChunk]:
        raise NotImplementedError


class Compressor(Component):
    """Shrinks the retrieved context to fit a token budget without losing answers."""

    async def compress(self, query: str, candidates: list[ScoredChunk], token_budget: int) -> list[ScoredChunk]:
        raise NotImplementedError


class Generator(Component):
    """Produces the final answer from the query and packed context."""

    async def generate(
        self, query: str, contexts: list[ScoredChunk], *, history: list[dict[str, str]] | None = None,
        system: str | None = None,
    ) -> RAGResult:
        raise NotImplementedError

    def stream(
        self, query: str, contexts: list[ScoredChunk], *, history: list[dict[str, str]] | None = None,
        system: str | None = None,
    ) -> AsyncIterator[str]:
        raise NotImplementedError


class Metric(Component):
    """A single evaluation measurement. Higher is always better, range [0, 1]."""

    needs_llm: bool = False
    needs_ground_truth: bool = False

    async def score(self, sample: Any, result: RAGResult) -> float:
        raise NotImplementedError
