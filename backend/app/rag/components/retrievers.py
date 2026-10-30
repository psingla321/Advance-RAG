"""Retrieval strategies: dense, sparse, hybrid, graph and ensembles.

Every retriever accepts a *list* of queries (query transforms may produce several)
and is responsible for fusing across them.
"""

from __future__ import annotations

import asyncio
import logging
import math
import re
from collections import defaultdict
from typing import Any

from app.rag.base import Embedder, Retriever, VectorStore
from app.rag.registry import build, register
from app.rag.types import Chunk, ScoredChunk
from app.textutil import content_term_list, content_terms

log = logging.getLogger(__name__)

RRF_K = 60


def reciprocal_rank_fusion(
    rankings: list[list[ScoredChunk]], k: int = RRF_K, weights: list[float] | None = None
) -> list[ScoredChunk]:
    """Rank-based fusion: robust because it never compares incomparable scores."""
    weights = weights or [1.0] * len(rankings)
    totals: dict[str, float] = defaultdict(float)
    best: dict[str, ScoredChunk] = {}
    for ranking, weight in zip(rankings, weights):
        for rank, item in enumerate(ranking, 1):
            totals[item.id] += weight / (k + rank)
            if item.id not in best:
                best[item.id] = item
            else:
                best[item.id].scores.update(item.scores)
    fused = []
    for chunk_id, score in sorted(totals.items(), key=lambda kv: kv[1], reverse=True):
        item = best[chunk_id]
        item.scores["rrf"] = round(score, 6)
        item.score = score
        fused.append(item)
    for rank, item in enumerate(fused, 1):
        item.rank = rank
    return fused


def _normalize(items: list[ScoredChunk], name: str) -> None:
    """Min-max normalize a score channel so weighted fusion is meaningful."""
    if not items:
        return
    values = [i.scores.get(name, i.score) for i in items]
    low, high = min(values), max(values)
    span = (high - low) or 1.0
    for item in items:
        item.scores[f"{name}_norm"] = (item.scores.get(name, item.score) - low) / span


@register("retriever", "dense")
class DenseRetriever(Retriever):
    """Pure vector similarity search. The baseline every other strategy beats."""

    def __init__(self, embedder: Embedder | None = None, vectorstore: VectorStore | None = None) -> None:
        self._embedder = embedder
        self._store = vectorstore

    async def retrieve(
        self, queries: list[str], k: int, namespace: str = "default", filters: dict[str, Any] | None = None
    ) -> list[ScoredChunk]:
        assert self._embedder is not None and self._store is not None
        vectors = await asyncio.gather(*(self._embedder.embed_query(q) for q in queries))
        rankings = await asyncio.gather(*(self._store.search(v, k, namespace, filters) for v in vectors))
        if len(rankings) == 1:
            return rankings[0]
        return reciprocal_rank_fusion(list(rankings))[:k]


@register("retriever", "bm25")
class BM25Retriever(Retriever):
    """Okapi BM25 over an in-process inverted index — exact keyword matching.

    Indispensable for product codes, error numbers and rare proper nouns, which
    embeddings routinely miss. Pure Python: no extra dependency.
    """

    def __init__(
        self, vectorstore: VectorStore | None = None, k1: float = 1.5, b: float = 0.75,
    ) -> None:
        self._store = vectorstore
        self._k1 = k1
        self._b = b
        self._postings: dict[str, dict[str, list[str]]] = {}
        self._chunks: dict[str, dict[str, Chunk]] = {}
        self._lengths: dict[str, dict[str, int]] = {}
        self._avg: dict[str, float] = {}

    async def index(self, chunks: list[Chunk], namespace: str = "default") -> None:
        postings = self._postings.setdefault(namespace, {})
        store = self._chunks.setdefault(namespace, {})
        lengths = self._lengths.setdefault(namespace, {})
        for chunk in chunks:
            if chunk.id in store:
                continue
            store[chunk.id] = chunk
            terms = content_term_list(chunk.embed_text)
            lengths[chunk.id] = len(terms)
            for term in set(terms):
                postings.setdefault(term, []).append(chunk.id)
        self._avg[namespace] = (sum(lengths.values()) / len(lengths)) if lengths else 1.0

    async def _ensure(self, namespace: str) -> None:
        if namespace in self._chunks or self._store is None:
            return
        await self.index(await self._store.all_chunks(namespace), namespace)

    async def retrieve(
        self, queries: list[str], k: int, namespace: str = "default", filters: dict[str, Any] | None = None
    ) -> list[ScoredChunk]:
        await self._ensure(namespace)
        store = self._chunks.get(namespace, {})
        if not store:
            return []
        rankings = [self._search(q, k, namespace, filters) for q in queries]
        return rankings[0] if len(rankings) == 1 else reciprocal_rank_fusion(rankings)[:k]

    def _search(self, query: str, k: int, namespace: str, filters: dict[str, Any] | None) -> list[ScoredChunk]:
        from app.rag.components.vectorstores import _matches, _payload

        postings = self._postings.get(namespace, {})
        store = self._chunks[namespace]
        lengths = self._lengths[namespace]
        total = len(store) or 1
        avg = self._avg.get(namespace, 1.0) or 1.0
        scores: dict[str, float] = defaultdict(float)
        for term in set(content_term_list(query)):
            ids = postings.get(term)
            if not ids:
                continue
            idf = math.log(1 + (total - len(ids) + 0.5) / (len(ids) + 0.5))
            for chunk_id in ids:
                # Term frequency is recomputed lazily to keep the index small.
                tf = content_term_list(store[chunk_id].embed_text).count(term)
                length_norm = 1 - self._b + self._b * (lengths[chunk_id] / avg)
                scores[chunk_id] += idf * (tf * (self._k1 + 1)) / (tf + self._k1 * length_norm)
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        out: list[ScoredChunk] = []
        for chunk_id, score in ranked:
            chunk = store[chunk_id]
            if filters and not _matches(_payload(chunk), filters):
                continue
            out.append(ScoredChunk(chunk=chunk, score=score, scores={"bm25": score}, rank=len(out) + 1))
            if len(out) >= k:
                break
        return out


@register("retriever", "hybrid_rrf")
class HybridRRFRetriever(Retriever):
    """Dense + BM25 fused with Reciprocal Rank Fusion. The best general default.

    Vectors catch paraphrase, BM25 catches exact identifiers; RRF needs no score
    calibration between them.
    """

    def __init__(
        self, embedder: Embedder | None = None, vectorstore: VectorStore | None = None,
        dense_weight: float = 1.0, sparse_weight: float = 1.0, rrf_k: int = RRF_K, overfetch: float = 2.0,
    ) -> None:
        self._dense = DenseRetriever(embedder, vectorstore)
        self._sparse = BM25Retriever(vectorstore)
        self._weights = [dense_weight, sparse_weight]
        self._rrf_k = rrf_k
        self._overfetch = overfetch

    async def index(self, chunks: list[Chunk], namespace: str = "default") -> None:
        await self._sparse.index(chunks, namespace)

    async def retrieve(
        self, queries: list[str], k: int, namespace: str = "default", filters: dict[str, Any] | None = None
    ) -> list[ScoredChunk]:
        fetch = int(k * self._overfetch)
        dense, sparse = await asyncio.gather(
            self._dense.retrieve(queries, fetch, namespace, filters),
            self._sparse.retrieve(queries, fetch, namespace, filters),
        )
        return reciprocal_rank_fusion([dense, sparse], self._rrf_k, self._weights)[:k]


@register("retriever", "hybrid_weighted")
class HybridWeightedRetriever(Retriever):
    """Dense + BM25 fused by normalized weighted score (convex combination).

    Use instead of RRF when you want to tune the dense/sparse balance explicitly.
    """

    def __init__(
        self, embedder: Embedder | None = None, vectorstore: VectorStore | None = None,
        alpha: float = 0.5, overfetch: float = 2.0,
    ) -> None:
        self._dense = DenseRetriever(embedder, vectorstore)
        self._sparse = BM25Retriever(vectorstore)
        self._alpha = alpha
        self._overfetch = overfetch

    async def index(self, chunks: list[Chunk], namespace: str = "default") -> None:
        await self._sparse.index(chunks, namespace)

    async def retrieve(
        self, queries: list[str], k: int, namespace: str = "default", filters: dict[str, Any] | None = None
    ) -> list[ScoredChunk]:
        fetch = int(k * self._overfetch)
        dense, sparse = await asyncio.gather(
            self._dense.retrieve(queries, fetch, namespace, filters),
            self._sparse.retrieve(queries, fetch, namespace, filters),
        )
        _normalize(dense, "dense")
        _normalize(sparse, "bm25")
        merged: dict[str, ScoredChunk] = {}
        for item in dense + sparse:
            existing = merged.setdefault(item.id, item)
            existing.scores.update(item.scores)
        for item in merged.values():
            item.score = (
                self._alpha * item.scores.get("dense_norm", 0.0)
                + (1 - self._alpha) * item.scores.get("bm25_norm", 0.0)
            )
        ranked = sorted(merged.values(), key=lambda i: i.score, reverse=True)[:k]
        for rank, item in enumerate(ranked, 1):
            item.rank = rank
        return ranked


@register("retriever", "mmr")
class MMRRetriever(Retriever):
    """Maximal Marginal Relevance: relevance minus redundancy.

    Stops the context window filling with five near-identical paragraphs — the
    single most common cause of "the answer was in the docs but not in my top-5".
    """

    def __init__(
        self, embedder: Embedder | None = None, vectorstore: VectorStore | None = None,
        inner: dict[str, Any] | str | None = None, lambda_mult: float = 0.5, fetch_k: int = 40,
    ) -> None:
        self._embedder = embedder
        self._inner: Retriever = build(
            "retriever", inner or {"type": "dense"}, embedder=embedder, vectorstore=vectorstore
        )
        self._lambda = lambda_mult
        self._fetch_k = fetch_k

    async def index(self, chunks: list[Chunk], namespace: str = "default") -> None:
        await self._inner.index(chunks, namespace)

    async def retrieve(
        self, queries: list[str], k: int, namespace: str = "default", filters: dict[str, Any] | None = None
    ) -> list[ScoredChunk]:
        from app.rag.components.embedders import cosine

        candidates = await self._inner.retrieve(queries, max(self._fetch_k, k), namespace, filters)
        if self._embedder is None or len(candidates) <= k:
            return candidates[:k]
        query_vector = await self._embedder.embed_query(queries[0])
        vectors = await self._embedder.embed([c.chunk.embed_text for c in candidates])
        relevance = [cosine(query_vector, v) for v in vectors]

        selected: list[int] = []
        remaining = list(range(len(candidates)))
        while remaining and len(selected) < k:
            best_index, best_score = remaining[0], -1e9
            for index in remaining:
                redundancy = max((cosine(vectors[index], vectors[s]) for s in selected), default=0.0)
                score = self._lambda * relevance[index] - (1 - self._lambda) * redundancy
                if score > best_score:
                    best_index, best_score = index, score
            selected.append(best_index)
            remaining.remove(best_index)
            candidates[best_index].scores["mmr"] = round(best_score, 4)
        picked = [candidates[i] for i in selected]
        for rank, item in enumerate(picked, 1):
            item.rank = rank
        return picked


@register("retriever", "parent_document")
class ParentDocumentRetriever(Retriever):
    """Match on small chunks, return the merged parent block.

    Pair with the ``parent_child`` or ``sentence_window`` chunker. Siblings from
    the same parent are collapsed into one context, so the LLM sees whole ideas.
    """

    def __init__(
        self, embedder: Embedder | None = None, vectorstore: VectorStore | None = None,
        inner: dict[str, Any] | str | None = None, merge_siblings: bool = True, overfetch: float = 3.0,
    ) -> None:
        self._inner: Retriever = build(
            "retriever", inner or {"type": "hybrid_rrf"}, embedder=embedder, vectorstore=vectorstore
        )
        self._merge = merge_siblings
        self._overfetch = overfetch

    async def index(self, chunks: list[Chunk], namespace: str = "default") -> None:
        await self._inner.index(chunks, namespace)

    async def retrieve(
        self, queries: list[str], k: int, namespace: str = "default", filters: dict[str, Any] | None = None
    ) -> list[ScoredChunk]:
        candidates = await self._inner.retrieve(queries, int(k * self._overfetch), namespace, filters)
        if not self._merge:
            return candidates[:k]
        seen: dict[str, ScoredChunk] = {}
        for item in candidates:
            key = item.chunk.metadata.get("parent_id") or item.chunk.parent_text or item.id
            existing = seen.get(key)
            if existing is None:
                seen[key] = item
            else:
                # Keep the best-scoring child but credit the parent for the extra hit.
                existing.scores["sibling_hits"] = existing.scores.get("sibling_hits", 1) + 1
                existing.score = max(existing.score, item.score)
        ranked = sorted(seen.values(), key=lambda i: i.score, reverse=True)[:k]
        for rank, item in enumerate(ranked, 1):
            item.rank = rank
        return ranked


@register("retriever", "ensemble")
class EnsembleRetriever(Retriever):
    """Fuse any number of arbitrary retrievers with per-retriever weights.

    ``{"type": "ensemble", "retrievers": [{"type": "dense"}, {"type": "bm25"}], "weights": [0.7, 0.3]}``
    """

    def __init__(
        self, retrievers: list[dict[str, Any] | str] | None = None, weights: list[float] | None = None,
        embedder: Embedder | None = None, vectorstore: VectorStore | None = None, overfetch: float = 2.0,
    ) -> None:
        specs = retrievers or [{"type": "dense"}, {"type": "bm25"}]
        self._retrievers: list[Retriever] = [
            build("retriever", spec, embedder=embedder, vectorstore=vectorstore) for spec in specs
        ]
        self._weights = weights or [1.0] * len(self._retrievers)
        self._overfetch = overfetch

    async def index(self, chunks: list[Chunk], namespace: str = "default") -> None:
        await asyncio.gather(*(r.index(chunks, namespace) for r in self._retrievers))

    async def retrieve(
        self, queries: list[str], k: int, namespace: str = "default", filters: dict[str, Any] | None = None
    ) -> list[ScoredChunk]:
        fetch = int(k * self._overfetch)
        rankings = await asyncio.gather(*(r.retrieve(queries, fetch, namespace, filters) for r in self._retrievers))
        return reciprocal_rank_fusion(list(rankings), weights=self._weights)[:k]

    def describe(self) -> dict[str, Any]:
        return {"type": "ensemble", "kind": "retriever",
                "retrievers": [r.describe() for r in self._retrievers], "weights": self._weights}


@register("retriever", "auto_merging")
class AutoMergingRetriever(Retriever):
    """Promote a parent when enough of its children are retrieved.

    If ≥ ``threshold`` of a parent's retrieved children appear in the candidate
    set, replace them with the single parent chunk — fewer, richer contexts.
    """

    def __init__(
        self, embedder: Embedder | None = None, vectorstore: VectorStore | None = None,
        inner: dict[str, Any] | str | None = None, threshold: float = 0.5, overfetch: float = 3.0,
    ) -> None:
        self._inner: Retriever = build(
            "retriever", inner or {"type": "hybrid_rrf"}, embedder=embedder, vectorstore=vectorstore
        )
        self._threshold = threshold
        self._overfetch = overfetch

    async def index(self, chunks: list[Chunk], namespace: str = "default") -> None:
        await self._inner.index(chunks, namespace)

    async def retrieve(
        self, queries: list[str], k: int, namespace: str = "default", filters: dict[str, Any] | None = None
    ) -> list[ScoredChunk]:
        candidates = await self._inner.retrieve(queries, int(k * self._overfetch), namespace, filters)
        groups: dict[str, list[ScoredChunk]] = defaultdict(list)
        for item in candidates:
            parent = item.chunk.metadata.get("parent_id")
            if parent:
                groups[parent].append(item)
        merged_parents = {p for p, items in groups.items() if len(items) >= max(2, self._threshold * len(candidates) / k)}
        out: list[ScoredChunk] = []
        emitted: set[str] = set()
        for item in candidates:
            parent = item.chunk.metadata.get("parent_id")
            if parent in merged_parents:
                if parent in emitted:
                    continue
                emitted.add(parent)
                item.chunk.parent_text = item.chunk.parent_text or item.chunk.text
                item.scores["merged_children"] = len(groups[parent])
            out.append(item)
            if len(out) >= k:
                break
        for rank, item in enumerate(out, 1):
            item.rank = rank
        return out


@register("retriever", "graph")
class GraphRetriever(Retriever):
    """Lightweight GraphRAG: expand hits along an entity co-occurrence graph.

    Builds an entity -> chunk index at ingest time, then pulls in chunks sharing
    entities with the top hits. Catches multi-hop questions where the answer is
    one document away from the best lexical match. No graph database required.
    """

    ENTITY = re.compile(r"\b(?:[A-Z][a-zA-Z0-9]+(?:\s+[A-Z][a-zA-Z0-9]+)*|[A-Z]{2,}|\d{4})\b")

    def __init__(
        self, embedder: Embedder | None = None, vectorstore: VectorStore | None = None,
        inner: dict[str, Any] | str | None = None, hops: int = 1, expand_from: int = 3,
        max_expansions: int = 6, min_entity_length: int = 3,
    ) -> None:
        self._inner: Retriever = build(
            "retriever", inner or {"type": "hybrid_rrf"}, embedder=embedder, vectorstore=vectorstore
        )
        self._store = vectorstore
        self._hops = hops
        self._expand_from = expand_from
        self._max_expansions = max_expansions
        self._min_len = min_entity_length
        self._entities: dict[str, dict[str, set[str]]] = {}
        self._chunks: dict[str, dict[str, Chunk]] = {}

    def _extract(self, text: str) -> set[str]:
        return {e.lower() for e in self.ENTITY.findall(text) if len(e) >= self._min_len}

    async def index(self, chunks: list[Chunk], namespace: str = "default") -> None:
        await self._inner.index(chunks, namespace)
        index = self._entities.setdefault(namespace, {})
        store = self._chunks.setdefault(namespace, {})
        for chunk in chunks:
            store[chunk.id] = chunk
            for entity in self._extract(f"{chunk.heading} {chunk.text}"):
                index.setdefault(entity, set()).add(chunk.id)

    async def retrieve(
        self, queries: list[str], k: int, namespace: str = "default", filters: dict[str, Any] | None = None
    ) -> list[ScoredChunk]:
        if namespace not in self._chunks and self._store is not None:
            await self.index(await self._store.all_chunks(namespace), namespace)
        seeds = await self._inner.retrieve(queries, k, namespace, filters)
        index = self._entities.get(namespace, {})
        store = self._chunks.get(namespace, {})
        if not index:
            return seeds

        found = {item.id: item for item in seeds}
        frontier = [item.chunk for item in seeds[: self._expand_from]]
        for hop in range(1, self._hops + 1):
            next_frontier: list[Chunk] = []
            for chunk in frontier:
                for entity in self._extract(f"{chunk.heading} {chunk.text}"):
                    for neighbour_id in list(index.get(entity, ()))[:5]:
                        if neighbour_id in found or neighbour_id not in store:
                            continue
                        neighbour = store[neighbour_id]
                        found[neighbour_id] = ScoredChunk(
                            chunk=neighbour,
                            score=0.1 / hop,
                            scores={"graph_hop": hop, "via_entity": 1.0},
                            rank=len(found) + 1,
                        )
                        next_frontier.append(neighbour)
                        if len(found) >= k + self._max_expansions:
                            break
            frontier = next_frontier
            if not frontier or len(found) >= k + self._max_expansions:
                break
        ranked = sorted(found.values(), key=lambda i: i.score, reverse=True)
        for rank, item in enumerate(ranked, 1):
            item.rank = rank
        return ranked


@register("retriever", "metadata_filtered")
class MetadataFilteredRetriever(Retriever):
    """Self-querying: let the LLM derive metadata filters from the question.

    "What did the 2024 security policy say about VPNs?" -> ``{"year": 2024}`` plus
    a semantic search for VPN. Needs ``schema`` describing the filterable fields.
    """

    PROMPT = (
        "Extract metadata filters from the question using only these fields:\n{schema}\n\n"
        "Question: {query}\n\n"
        'Return JSON {{"filters": {{...}}, "query": "the semantic part of the question"}}. '
        "Use an empty filters object when nothing applies."
    )

    def __init__(
        self, llm: Any = None, embedder: Embedder | None = None, vectorstore: VectorStore | None = None,
        schema: dict[str, str] | None = None, inner: dict[str, Any] | str | None = None,
    ) -> None:
        self._llm = llm
        self._schema = schema or {}
        self._inner: Retriever = build(
            "retriever", inner or {"type": "hybrid_rrf"}, embedder=embedder, vectorstore=vectorstore
        )

    async def index(self, chunks: list[Chunk], namespace: str = "default") -> None:
        await self._inner.index(chunks, namespace)

    async def retrieve(
        self, queries: list[str], k: int, namespace: str = "default", filters: dict[str, Any] | None = None
    ) -> list[ScoredChunk]:
        import json

        if self._llm is None or not self._schema:
            return await self._inner.retrieve(queries, k, namespace, filters)
        schema = "\n".join(f"- {name}: {kind}" for name, kind in self._schema.items())
        try:
            raw = await self._llm.ask(
                self.PROMPT.format(schema=schema, query=queries[0]), json_mode=True, max_tokens=200
            )
            parsed = json.loads(raw)
            derived = {k2: v for k2, v in (parsed.get("filters") or {}).items() if k2 in self._schema}
            semantic = (parsed.get("query") or queries[0]).strip()
        except Exception as exc:
            log.warning("self-query filter extraction failed: %s", exc)
            return await self._inner.retrieve(queries, k, namespace, filters)
        merged = {**(filters or {}), **derived}
        results = await self._inner.retrieve([semantic] + queries[1:], k, namespace, merged)
        if not results and derived:
            # Over-restrictive filters are worse than none: retry unfiltered.
            results = await self._inner.retrieve(queries, k, namespace, filters)
        for item in results:
            item.scores.setdefault("auto_filters", float(len(derived)))
        return results
