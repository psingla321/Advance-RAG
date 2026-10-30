"""Plug in your own components.

    python examples/03_custom_components.py

Anything you register becomes available to YAML configs, the CLI, sweeps and the
REST API with no further wiring. That is the whole extension story.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Any

from app.rag import RAG, Chunker, Embedder, Metric, Reranker, Retriever, register
from app.rag.types import Chunk, Document, ScoredChunk, Section


# --- 1. A reranker that prefers recent documents ------------------------
@register("reranker", "recency_boost")
class RecencyBoostReranker(Reranker):
    """Blends relevance with document freshness — for corpora where stale answers are wrong."""

    def __init__(self, half_life_days: float = 365.0, weight: float = 0.3) -> None:
        self._half_life = half_life_days
        self._weight = weight

    async def rerank(self, query: str, candidates: list[ScoredChunk], top_n: int) -> list[ScoredChunk]:
        today = date.today()
        for item in candidates:
            published = item.chunk.metadata.get("published")
            freshness = 0.5
            if published:
                try:
                    age_days = (today - date.fromisoformat(str(published))) .days
                    freshness = 0.5 ** (max(age_days, 0) / self._half_life)
                except ValueError:
                    pass
            item.scores["freshness"] = round(freshness, 4)
            item.score = (1 - self._weight) * item.score + self._weight * freshness
        ranked = sorted(candidates, key=lambda i: i.score, reverse=True)[:top_n]
        for rank, item in enumerate(ranked, 1):
            item.rank = rank
        return ranked


# --- 2. A chunker for structured documents ------------------------------
@register("chunker", "clause")
class ClauseChunker(Chunker):
    """One chunk per numbered clause — contracts, policies, statutes."""

    PATTERN = re.compile(r"(?m)^(?=\s*(?:\d+(?:\.\d+)*\.?\s|Article\s+\d+|Section\s+\d+))")

    async def chunk(self, document: Document, sections: list[Section]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for section in sections:
            for part in self.PATTERN.split(section.text):
                if not part.strip():
                    continue
                label = part.strip().split("\n", 1)[0][:40]
                chunks.append(
                    Chunk(
                        id=f"{document.id}::{len(chunks)}", text=part.strip(), doc_id=document.id,
                        filename=document.filename, heading=section.heading or label, ord=len(chunks),
                        metadata={**document.metadata, "clause": label},
                    )
                )
        return chunks


# --- 3. A domain metric -------------------------------------------------
@register("metric", "cites_a_clause")
class CitesAClause(Metric):
    """Compliance requirement: every answer must point at a specific clause."""

    async def score(self, sample: Any, result: Any) -> float:
        if not result.answer.strip() or result.abstained:
            return 1.0 if result.abstained else 0.0
        return 1.0 if any(c.heading for c in result.citations) else 0.0


# --- 4. A retriever that wraps another one ------------------------------
@register("retriever", "always_include_glossary")
class GlossaryRetriever(Retriever):
    """Always prepends the glossary chunk — for jargon-heavy corpora."""

    def __init__(self, embedder: Embedder | None = None, vectorstore: Any = None,
                 glossary_filename: str = "glossary.md") -> None:
        from app.rag.registry import build

        self._inner = build("retriever", "hybrid_rrf", embedder=embedder, vectorstore=vectorstore)
        self._store = vectorstore
        self._filename = glossary_filename

    async def index(self, chunks: list[Chunk], namespace: str = "default") -> None:
        await self._inner.index(chunks, namespace)

    async def retrieve(self, queries, k, namespace="default", filters=None):
        results = await self._inner.retrieve(queries, k, namespace, filters)
        glossary = [c for c in await self._store.all_chunks(namespace) if c.filename == self._filename]
        extra = [ScoredChunk(chunk=c, score=0.01, scores={"glossary": 1.0}) for c in glossary[:1]]
        return extra + [r for r in results if r.chunk.filename != self._filename][: k - len(extra)]


CONTRACT = """1. Payment terms. Invoices are due within 30 days of receipt.
2. Late payment. Interest accrues at 1.5 percent per month on overdue amounts.
3. Termination. Either party may terminate with 60 days written notice.
4. Confidentiality. Both parties keep commercial terms confidential for 3 years."""


def main() -> None:
    # Your components are addressed by name, exactly like the built-ins.
    rag = RAG(chunker="clause", reranker={"type": "recency_boost", "weight": 0.2}, generator="extractive")
    rag.add_texts(
        [CONTRACT, "1. Payment terms. Invoices are due within 15 days."],
        metadatas=[{"filename": "contract_2024.md", "published": "2024-01-15"},
                   {"filename": "contract_2026.md", "published": "2026-02-01"}],
    )
    print(rag, "\n")

    result = rag.print_answer("When are invoices due?")
    print("\nFreshness scores applied by the custom reranker:")
    for item in result.contexts[:3]:
        print(f"  {item.chunk.filename:<20} relevance+freshness={item.score:.3f} "
              f"freshness={item.scores.get('freshness')}")

    # Custom metrics work in evaluation too.
    from app.rag import Dataset

    report = rag.evaluate(
        Dataset.from_pairs([("When are invoices due?", "within 15 days")]),
        metrics=["answer_f1", "cites_a_clause", "groundedness"],
        show_progress=False,
    )
    print("\n" + report.table())
    rag.close()


if __name__ == "__main__":
    main()
