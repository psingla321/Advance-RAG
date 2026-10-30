"""Rerankers: a second, more expensive pass over a small candidate set.

Reranking is usually the cheapest large quality win in a RAG system — retrieve
50 candidates cheaply, then spend real compute ordering only those.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

from app.rag.base import LLM, Embedder, Reranker
from app.rag.registry import register
from app.rag.types import ScoredChunk
from app.textutil import content_terms

log = logging.getLogger(__name__)


@register("reranker", "identity")
class IdentityReranker(Reranker):
    """Keeps retrieval order. The control condition for any reranking experiment."""

    async def rerank(self, query: str, candidates: list[ScoredChunk], top_n: int) -> list[ScoredChunk]:
        return candidates[:top_n]


@register("reranker", "keyword_overlap")
class KeywordOverlapReranker(Reranker):
    """Blends the retrieval score with query-term coverage. Free and surprisingly effective.

    This mirrors the heuristic the production Nexus retriever uses as its
    evidence gate.
    """

    def __init__(self, weight: float = 0.45) -> None:
        self._weight = weight

    async def rerank(self, query: str, candidates: list[ScoredChunk], top_n: int) -> list[ScoredChunk]:
        terms = content_terms(query)
        if not terms:
            return candidates[:top_n]
        for item in candidates:
            coverage = len(terms & content_terms(f"{item.chunk.heading} {item.chunk.text}")) / len(terms)
            item.scores["coverage"] = round(coverage, 4)
            base = item.scores.get("dense", item.score)
            item.score = (1 - self._weight) * base + self._weight * coverage
        ranked = sorted(candidates, key=lambda i: i.score, reverse=True)[:top_n]
        for rank, item in enumerate(ranked, 1):
            item.rank = rank
        return ranked


@register("reranker", "cross_encoder", requires=["sentence-transformers"])
class CrossEncoderReranker(Reranker):
    """A cross-encoder scores (query, passage) jointly — the quality gold standard.

    ``BAAI/bge-reranker-v2-m3`` and ``cross-encoder/ms-marco-MiniLM-L-6-v2`` are
    good starting points; both run locally on CPU.
    """

    def __init__(
        self, model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", device: str | None = None,
        batch_size: int = 16, max_length: int = 512,
    ) -> None:
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(model, device=device, max_length=max_length)
        self.model_id = model
        self._batch = batch_size

    async def rerank(self, query: str, candidates: list[ScoredChunk], top_n: int) -> list[ScoredChunk]:
        if not candidates:
            return []
        pairs = [(query, item.chunk.context_text) for item in candidates]
        scores = await asyncio.to_thread(self._model.predict, pairs, batch_size=self._batch, show_progress_bar=False)
        for item, score in zip(candidates, scores):
            item.scores["rerank"] = float(score)
            item.score = float(score)
        ranked = sorted(candidates, key=lambda i: i.score, reverse=True)[:top_n]
        for rank, item in enumerate(ranked, 1):
            item.rank = rank
        return ranked


@register("reranker", "cohere", requires=["cohere"])
class CohereReranker(Reranker):
    """Cohere Rerank — hosted, very strong, one API call for the whole candidate set."""

    def __init__(self, model: str = "rerank-english-v3.0", api_key: str | None = None) -> None:
        import os

        import cohere

        self._client = cohere.AsyncClient(api_key or os.getenv("COHERE_API_KEY"))
        self.model_id = model

    async def rerank(self, query: str, candidates: list[ScoredChunk], top_n: int) -> list[ScoredChunk]:
        if not candidates:
            return []
        response = await self._client.rerank(
            model=self.model_id, query=query,
            documents=[item.chunk.context_text for item in candidates], top_n=min(top_n, len(candidates)),
        )
        ranked = []
        for rank, result in enumerate(response.results, 1):
            item = candidates[result.index]
            item.scores["rerank"] = float(result.relevance_score)
            item.score = float(result.relevance_score)
            item.rank = rank
            ranked.append(item)
        return ranked


@register("reranker", "llm_pointwise")
class LLMPointwiseReranker(Reranker):
    """Ask the LLM to score each passage 0-10 for relevance, in parallel.

    Works with any model, no reranker to deploy — but costs one call per candidate,
    so keep the candidate set small or wrap the LLM in ``cached``.
    """

    PROMPT = (
        "Rate how useful this passage is for answering the question, from 0 (irrelevant) "
        "to 10 (contains the answer). Reply with the number only.\n\n"
        "Question: {query}\n\nPassage:\n{passage}\n\nScore:"
    )

    def __init__(self, llm: LLM | None = None, concurrency: int = 8, max_chars: int = 1500) -> None:
        self._llm = llm
        self._semaphore = asyncio.Semaphore(concurrency)
        self._max_chars = max_chars

    async def _score(self, query: str, item: ScoredChunk) -> float:
        async with self._semaphore:
            try:
                raw = await self._llm.ask(  # type: ignore[union-attr]
                    self.PROMPT.format(query=query, passage=item.chunk.context_text[: self._max_chars]),
                    max_tokens=8, temperature=0,
                )
            except Exception as exc:
                log.warning("llm rerank call failed: %s", exc)
                return item.score
        match = re.search(r"\d+(?:\.\d+)?", raw)
        return min(float(match.group()), 10.0) / 10.0 if match else 0.0

    async def rerank(self, query: str, candidates: list[ScoredChunk], top_n: int) -> list[ScoredChunk]:
        if self._llm is None or not candidates:
            return candidates[:top_n]
        scores = await asyncio.gather(*(self._score(query, item) for item in candidates))
        for item, score in zip(candidates, scores):
            item.scores["rerank"] = score
            item.score = score
        ranked = sorted(candidates, key=lambda i: i.score, reverse=True)[:top_n]
        for rank, item in enumerate(ranked, 1):
            item.rank = rank
        return ranked


@register("reranker", "llm_listwise")
class LLMListwiseReranker(Reranker):
    """One LLM call ranks the whole candidate list — cheaper than pointwise.

    The model sees all passages together, so it can judge relative usefulness,
    which pointwise scoring cannot.
    """

    PROMPT = (
        "Rank the passages by how well they help answer the question. "
        "Return JSON {{\"order\": [passage numbers, best first]}} and include only "
        "passages that genuinely help.\n\nQuestion: {query}\n\n{passages}"
    )

    def __init__(self, llm: LLM | None = None, max_chars: int = 700) -> None:
        self._llm = llm
        self._max_chars = max_chars

    async def rerank(self, query: str, candidates: list[ScoredChunk], top_n: int) -> list[ScoredChunk]:
        if self._llm is None or len(candidates) <= 1:
            return candidates[:top_n]
        rendered = "\n\n".join(
            f"[{i}] {item.chunk.context_text[: self._max_chars]}" for i, item in enumerate(candidates, 1)
        )
        try:
            raw = await self._llm.ask(
                self.PROMPT.format(query=query, passages=rendered), json_mode=True, max_tokens=200, temperature=0
            )
            order = [int(n) for n in json.loads(raw).get("order", []) if 1 <= int(n) <= len(candidates)]
        except Exception as exc:
            log.warning("listwise rerank failed: %s", exc)
            return candidates[:top_n]
        if not order:
            return candidates[:top_n]
        ranked: list[ScoredChunk] = []
        for rank, index in enumerate(dict.fromkeys(order), 1):
            item = candidates[index - 1]
            item.score = 1.0 - (rank - 1) / len(order)
            item.scores["rerank"] = item.score
            item.rank = rank
            ranked.append(item)
        return ranked[:top_n]


@register("reranker", "diversity")
class DiversityReranker(Reranker):
    """MMR applied as a reranking stage — drops near-duplicate passages.

    Cheaper than the MMR retriever because it reuses the candidates you already have.
    """

    def __init__(self, embedder: Embedder | None = None, lambda_mult: float = 0.7) -> None:
        self._embedder = embedder
        self._lambda = lambda_mult

    async def rerank(self, query: str, candidates: list[ScoredChunk], top_n: int) -> list[ScoredChunk]:
        if self._embedder is None or len(candidates) <= top_n:
            return candidates[:top_n]
        from app.rag.components.embedders import cosine

        vectors = await self._embedder.embed([item.chunk.embed_text for item in candidates])
        selected: list[int] = []
        remaining = list(range(len(candidates)))
        while remaining and len(selected) < top_n:
            best, best_score = remaining[0], -1e9
            for index in remaining:
                redundancy = max((cosine(vectors[index], vectors[s]) for s in selected), default=0.0)
                score = self._lambda * candidates[index].score - (1 - self._lambda) * redundancy
                if score > best_score:
                    best, best_score = index, score
            selected.append(best)
            remaining.remove(best)
        ranked = [candidates[i] for i in selected]
        for rank, item in enumerate(ranked, 1):
            item.rank = rank
        return ranked


@register("reranker", "chain")
class ChainReranker(Reranker):
    """Cascade: a cheap reranker narrows the field, then an expensive one orders it.

    ``{"type": "chain", "stages": [{"reranker": {"type": "keyword_overlap"}, "top_n": 20},
                                   {"reranker": {"type": "cross_encoder"}}]}``
    """

    def __init__(self, stages: list[dict[str, Any]] | None = None, llm: LLM | None = None, embedder: Embedder | None = None) -> None:
        from app.rag.registry import build

        self._stages = [
            (build("reranker", stage.get("reranker", stage), llm=llm, embedder=embedder), stage.get("top_n"))
            for stage in (stages or [])
        ]

    async def rerank(self, query: str, candidates: list[ScoredChunk], top_n: int) -> list[ScoredChunk]:
        current = candidates
        for reranker, stage_top_n in self._stages:
            current = await reranker.rerank(query, current, stage_top_n or top_n)
        return current[:top_n]

    def describe(self) -> dict[str, Any]:
        return {"type": "chain", "kind": "reranker",
                "stages": [{"reranker": r.describe(), "top_n": n} for r, n in self._stages]}
