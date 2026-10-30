import time
from dataclasses import dataclass

from redis.asyncio import Redis

# Atomic token bucket: refill by elapsed time, then try to take `cost` tokens.
_TOKEN_BUCKET = """
local capacity = tonumber(ARGV[1])
local rate = tonumber(ARGV[2])
local now = tonumber(ARGV[3])
local cost = tonumber(ARGV[4])
local state = redis.call('HMGET', KEYS[1], 'tokens', 'ts')
local tokens = tonumber(state[1]) or capacity
local ts = tonumber(state[2]) or now
tokens = math.min(capacity, tokens + math.max(0, now - ts) * rate)
local allowed = 0
local retry = 0
if tokens >= cost then
  tokens = tokens - cost
  allowed = 1
else
  retry = (cost - tokens) / rate
end
redis.call('HSET', KEYS[1], 'tokens', tokens, 'ts', now)
redis.call('EXPIRE', KEYS[1], math.ceil(capacity / rate) + 1)
return {allowed, tostring(retry), tostring(tokens)}
"""


@dataclass
class RateLimitResult:
    allowed: bool
    retry_after: float
    remaining: int


class RateLimiter:
    def __init__(self, redis: Redis, capacity: int, refill_per_sec: float) -> None:
        self._script = redis.register_script(_TOKEN_BUCKET)
        self._capacity = capacity
        self._rate = refill_per_sec

    async def acquire(self, key: str, cost: int = 1) -> RateLimitResult:
        allowed, retry, tokens = await self._script(keys=[f"rl:{key}"], args=[self._capacity, self._rate, time.time(), cost])
        return RateLimitResult(bool(int(allowed)), float(retry), int(float(tokens)))
