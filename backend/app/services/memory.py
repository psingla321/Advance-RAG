import json
from typing import Any

from redis.asyncio import Redis


class ConversationMemory:
    def __init__(self, redis: Redis, max_messages: int = 20, ttl_seconds: int = 7 * 86400) -> None:
        self._redis = redis
        self._max = max_messages
        self._ttl = ttl_seconds

    @staticmethod
    def _key(tenant: str, conversation_id: str) -> str:
        return f"conv:{tenant}:{conversation_id}"

    async def append(self, tenant: str, conversation_id: str, role: str, content: str, **meta: Any) -> None:
        key = self._key(tenant, conversation_id)
        pipe = self._redis.pipeline()
        pipe.rpush(key, json.dumps({"role": role, "content": content, **meta}))
        pipe.ltrim(key, -self._max, -1)
        pipe.expire(key, self._ttl)
        await pipe.execute()

    async def history(self, tenant: str, conversation_id: str, limit: int = 20) -> list[dict[str, Any]]:
        raw = await self._redis.lrange(self._key(tenant, conversation_id), -limit, -1)
        return [json.loads(item) for item in raw]
