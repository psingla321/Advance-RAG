import json
import logging
import time
from typing import Any

from redis.asyncio import Redis
from redis.exceptions import RedisError

from app.config import Settings
from app.services.embedder import CachedEmbedder
from app.services.vector_store import VectorStore
from app.textutil import acl_hash, entity_signature, normalize_query, sha256_hex

log = logging.getLogger(__name__)


class AnswerCache:
    """L1 exact cache (Redis) + semantic cache (Qdrant), scoped by tenant, ACL hash and corpus version."""

    def __init__(self, redis: Redis, vector_store: VectorStore, embedder: CachedEmbedder, settings: Settings) -> None:
        self._redis = redis
        self._vs = vector_store
        self._embedder = embedder
        self._threshold = settings.semantic_cache_threshold
        self._ttl = settings.cache_ttl_seconds

    async def corpus_version(self, tenant: str) -> int:
        value = await self._redis.get(f"corpus_ver:{tenant}")
        return int(value) if value else 0

    async def bump_version(self, tenant: str) -> None:
        # Versioned namespace: every older cache entry for the tenant becomes unreachable in O(1).
        await self._redis.incr(f"corpus_ver:{tenant}")

    @staticmethod
    def _l1_key(tenant: str, acl: str, version: int, query: str) -> str:
        return f"l1:{tenant}:{acl}:{version}:{sha256_hex(normalize_query(query))}"

    async def lookup(self, query: str, tenant: str, groups: list[str]) -> dict[str, Any] | None:
        try:
            acl = acl_hash(groups)
            version = await self.corpus_version(tenant)
            raw = await self._redis.get(self._l1_key(tenant, acl, version, query))
            if raw:
                return {**json.loads(raw), "level": "L1 exact", "similarity": 1.0}
            vector = await self._embedder.embed_one(query)
            hit = await self._vs.cache_search(vector, tenant, acl, version, self._ttl)
            if not hit:
                return None
            score, payload = hit
            if score < self._threshold:
                return None
            if payload.get("signature") != sorted(entity_signature(query)):
                return None
            return {**payload["answer_payload"], "level": "L3 semantic", "similarity": round(score, 4)}
        except (RedisError, OSError) as exc:
            log.warning("Cache lookup failed, treating as miss: %s", exc.__class__.__name__)
            return None

    async def store(self, query: str, tenant: str, groups: list[str], payload: dict[str, Any], doc_ids: list[str]) -> None:
        try:
            acl = acl_hash(groups)
            version = await self.corpus_version(tenant)
            await self._redis.set(self._l1_key(tenant, acl, version, query), json.dumps(payload), ex=self._ttl)
            vector = await self._embedder.embed_one(query)
            await self._vs.cache_store(
                vector,
                {
                    "tenant_id": tenant,
                    "acl_hash": acl,
                    "corpus_ver": version,
                    "created_at": time.time(),
                    "doc_ids": doc_ids,
                    "signature": sorted(entity_signature(query)),
                    "query": query,
                    "answer_payload": payload,
                },
            )
        except Exception as exc:
            log.warning("Cache store failed: %s", exc.__class__.__name__)
