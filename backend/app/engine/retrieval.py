import asyncio
from typing import Any

from app.config import Settings
from app.engine.packing import pack_context
from app.services.db import Database
from app.services.embedder import CachedEmbedder
from app.services.vector_store import VectorStore
from app.textutil import content_terms, keyword_terms

RRF_K = 60
MIN_DENSE = 0.2
MIN_EVIDENCE_SCORE = 0.15
# At least one passage must cover this share of the query's content terms (or be a strong dense match).
GATE_COVERAGE = 0.4
GATE_DENSE = 0.5


class HybridRetriever:
    """Dense (Qdrant) + BM25-style keyword (PostgreSQL FTS) retrieval, RRF fusion, rerank and context packing."""

    def __init__(self, db: Database, vector_store: VectorStore, embedder: CachedEmbedder, settings: Settings) -> None:
        self._db = db
        self._vs = vector_store
        self._embedder = embedder
        self._k = settings.retrieval_top_k
        self._budget = settings.context_token_budget

    async def retrieve(self, query: str, tenant: str, groups: list[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        vector = await self._embedder.embed_one(query)
        dense, keyword = await asyncio.gather(
            self._vs.search_chunks(vector, tenant, groups, self._k),
            self._db.keyword_search(keyword_terms(query), tenant, groups, self._k),
        )

        fused: dict[str, dict[str, Any]] = {}
        for rank, hit in enumerate(dense, 1):
            item = fused.setdefault(hit["chunk_id"], self._base(hit))
            item["dense"] = hit["score"]
            item["rrf"] += 1 / (RRF_K + rank)
        for rank, hit in enumerate(keyword, 1):
            item = fused.setdefault(hit["chunk_id"], self._base(hit))
            item["keyword_rank"] = rank
            item["rrf"] += 1 / (RRF_K + rank)

        query_terms = content_terms(query)
        candidates = []
        for item in fused.values():
            dense_score = max(item.get("dense", 0.0), 0.0)
            coverage = len(query_terms & content_terms(f"{item['heading']} {item['content']}")) / max(1, len(query_terms))
            if item.get("keyword_rank") is None and (dense_score < MIN_DENSE or (coverage == 0 and dense_score < 0.35)):
                continue
            item["score"] = round(0.55 * dense_score + 0.45 * coverage, 4)
            item["coverage"] = coverage
            if item["score"] >= MIN_EVIDENCE_SCORE:
                candidates.append(item)

        if not any(c["coverage"] >= GATE_COVERAGE or c.get("dense", 0.0) >= GATE_DENSE for c in candidates):
            candidates = []
        candidates.sort(key=lambda c: (c["score"], c["rrf"]), reverse=True)
        packed = pack_context(candidates[:12], query, self._budget)
        for n, item in enumerate(packed, 1):
            item["n"] = n
        stats = {
            "dense_hits": len(dense),
            "keyword_hits": len(keyword),
            "candidates": len(candidates),
            "packed": len(packed),
            "top_score": packed[0]["score"] if packed else 0.0,
        }
        return packed, stats

    @staticmethod
    def _base(hit: dict[str, Any]) -> dict[str, Any]:
        return {
            "chunk_id": hit["chunk_id"],
            "doc_id": hit.get("doc_id", ""),
            "filename": hit.get("filename", ""),
            "heading": hit.get("heading", ""),
            "content": hit.get("content", ""),
            "rrf": 0.0,
        }
