import logging
import struct

from redis.asyncio import Redis
from redis.exceptions import RedisError

from app.llm.client import LLMClient
from app.textutil import sha256_hex

log = logging.getLogger(__name__)


class CachedEmbedder:
    """L5 embedding cache: content-addressed vectors in Redis, model calls only for misses."""

    def __init__(self, llm: LLMClient, redis: Redis, model_id: str, dim: int, ttl_seconds: int = 30 * 86400) -> None:
        self._llm = llm
        self._redis = redis
        self._model_id = model_id
        self._dim = dim
        self._ttl = ttl_seconds
        self._format = f"{dim}f"

    def _key(self, text: str) -> str:
        return f"emb:{self._model_id}:{self._dim}:{sha256_hex(text)}"

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        keys = [self._key(t) for t in texts]
        try:
            cached = await self._redis.mget(keys)
        except RedisError:
            log.warning("Embedding cache unavailable; treating as miss")
            cached = [None] * len(texts)

        result: list[list[float] | None] = [None] * len(texts)
        missing: list[int] = []
        for i, raw in enumerate(cached):
            if raw is not None and len(raw) == 4 * self._dim:
                result[i] = list(struct.unpack(self._format, raw))
            else:
                missing.append(i)

        for start in range(0, len(missing), 64):
            batch = missing[start : start + 64]
            vectors = await self._llm.embed([texts[i] for i in batch])
            pipe = self._redis.pipeline()
            for i, vector in zip(batch, vectors):
                result[i] = vector
                pipe.set(keys[i], struct.pack(self._format, *vector), ex=self._ttl)
            try:
                await pipe.execute()
            except RedisError:
                log.warning("Failed to write embedding cache")

        return [v for v in result if v is not None]

    async def embed_one(self, text: str) -> list[float]:
        return (await self.embed([text]))[0]
