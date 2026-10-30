import asyncio
import logging
import time
from typing import Any
from uuid import uuid4

from qdrant_client import AsyncQdrantClient, models

log = logging.getLogger(__name__)


def _match(key: str, value: Any) -> models.FieldCondition:
    return models.FieldCondition(key=key, match=models.MatchValue(value=value))


def _match_any(key: str, values: list[str]) -> models.FieldCondition:
    return models.FieldCondition(key=key, match=models.MatchAny(any=values))


class VectorStore:
    def __init__(self, url: str, dim: int) -> None:
        # ":memory:" runs Qdrant's embedded local mode (tests / quick demos).
        self.client = AsyncQdrantClient(location=url) if url == ":memory:" else AsyncQdrantClient(url=url, timeout=10)
        self.dim = dim
        self.chunks = f"chunks_{dim}"
        self.cache = f"semcache_{dim}"

    async def init(self, attempts: int = 30) -> None:
        for attempt in range(1, attempts + 1):
            try:
                await self._ensure_collections()
                return
            except Exception as exc:  # qdrant raises transport-specific errors while starting
                if attempt == attempts:
                    raise
                log.info("Waiting for Qdrant (%s/%s): %s", attempt, attempts, exc.__class__.__name__)
                await asyncio.sleep(1)

    async def _ensure_collections(self) -> None:
        specs = {
            self.chunks: {"tenant_id": "keyword", "acl": "keyword", "doc_id": "keyword"},
            self.cache: {"tenant_id": "keyword", "acl_hash": "keyword", "doc_ids": "keyword",
                         "corpus_ver": "integer", "created_at": "float"},
        }
        for name, fields in specs.items():
            if await self.client.collection_exists(name):
                continue
            await self.client.create_collection(
                name, vectors_config=models.VectorParams(size=self.dim, distance=models.Distance.COSINE)
            )
            for field, kind in fields.items():
                await self.client.create_payload_index(name, field, models.PayloadSchemaType(kind))

    async def ping(self) -> bool:
        await self.client.get_collections()
        return True

    async def close(self) -> None:
        await self.client.close()

    async def upsert_chunks(self, items: list[dict[str, Any]]) -> None:
        points = [models.PointStruct(id=i["id"], vector=i["vector"], payload=i["payload"]) for i in items]
        for start in range(0, len(points), 256):
            await self.client.upsert(self.chunks, points=points[start : start + 256], wait=True)

    async def search_chunks(self, vector: list[float], tenant: str, groups: list[str], k: int) -> list[dict[str, Any]]:
        result = await self.client.query_points(
            self.chunks,
            query=vector,
            query_filter=models.Filter(must=[_match("tenant_id", tenant), _match_any("acl", groups)]),
            limit=k,
            with_payload=True,
        )
        return [{**(p.payload or {}), "chunk_id": str(p.id), "score": float(p.score)} for p in result.points]

    async def delete_document(self, tenant: str, doc_id: str) -> None:
        await self.client.delete(
            self.chunks,
            points_selector=models.FilterSelector(filter=models.Filter(must=[_match("tenant_id", tenant), _match("doc_id", doc_id)])),
        )
        await self.client.delete(
            self.cache,
            points_selector=models.FilterSelector(filter=models.Filter(must=[_match("tenant_id", tenant), _match_any("doc_ids", [doc_id])])),
        )

    async def cache_search(
        self, vector: list[float], tenant: str, acl_hash: str, corpus_ver: int, max_age_seconds: int
    ) -> tuple[float, dict[str, Any]] | None:
        result = await self.client.query_points(
            self.cache,
            query=vector,
            query_filter=models.Filter(
                must=[
                    _match("tenant_id", tenant),
                    _match("acl_hash", acl_hash),
                    _match("corpus_ver", corpus_ver),
                    models.FieldCondition(key="created_at", range=models.Range(gte=time.time() - max_age_seconds)),
                ]
            ),
            limit=1,
            with_payload=True,
        )
        if not result.points:
            return None
        point = result.points[0]
        return float(point.score), point.payload or {}

    async def cache_store(self, vector: list[float], payload: dict[str, Any]) -> None:
        await self.client.upsert(self.cache, points=[models.PointStruct(id=str(uuid4()), vector=vector, payload=payload)])
