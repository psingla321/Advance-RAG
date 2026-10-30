import json
import logging
from typing import Any

from redis.asyncio import Redis
from redis.exceptions import ResponseError

log = logging.getLogger(__name__)

STREAM = "ingest:jobs"
GROUP = "ingest-workers"
BLOB_TTL_SECONDS = 6 * 3600


class IngestionQueue:
    """Durable job queue on Redis Streams with consumer groups (at-least-once, horizontally scalable workers)."""

    def __init__(self, redis: Redis) -> None:
        self.redis = redis

    @staticmethod
    def blob_key(doc_id: str) -> str:
        return f"ingest:blob:{doc_id}"

    async def ensure_group(self) -> None:
        try:
            await self.redis.xgroup_create(STREAM, GROUP, id="0", mkstream=True)
        except ResponseError as exc:
            if "BUSYGROUP" not in str(exc):
                raise

    async def enqueue(self, doc_id: str, tenant: str, filename: str, data: bytes, acl: list[str]) -> None:
        # Payload is staged in Redis for the demo; production stages it in Blob Storage and enqueues the URI.
        await self.redis.set(self.blob_key(doc_id), data, ex=BLOB_TTL_SECONDS)
        await self.redis.xadd(
            STREAM,
            {"doc_id": doc_id, "tenant": tenant, "filename": filename, "acl": json.dumps(acl)},
            maxlen=100_000,
            approximate=True,
        )

    async def depth(self) -> int:
        try:
            return int(await self.redis.xlen(STREAM))
        except ResponseError:
            return 0

    async def complete(self, message_id: Any, doc_id: str) -> None:
        pipe = self.redis.pipeline()
        pipe.xack(STREAM, GROUP, message_id)
        pipe.xdel(STREAM, message_id)
        pipe.delete(self.blob_key(doc_id))
        await pipe.execute()
