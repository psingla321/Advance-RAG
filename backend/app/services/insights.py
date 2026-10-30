import json
import time
from typing import Any

from redis.asyncio import Redis

from app.engine.guardrails import redact


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(pct / 100 * (len(ordered) - 1))))
    return round(ordered[index], 1)


class InsightsStore:
    """Per-tenant operational analytics in Redis (counters, latency samples, node timings, recent requests)."""

    def __init__(self, redis: Redis, max_samples: int = 2000, max_recent: int = 50) -> None:
        self._redis = redis
        self._max_samples = max_samples
        self._max_recent = max_recent

    async def record(self, tenant: str, record: dict[str, Any]) -> None:
        key = f"ins:{tenant}"
        pipe = self._redis.pipeline()
        pipe.hincrby(f"{key}:c", "total", 1)
        pipe.hincrby(f"{key}:c", f"mode:{record.get('mode') or 'none'}", 1)
        for route in record.get("routes") or []:
            pipe.hincrby(f"{key}:c", f"route:{route}", 1)
        for flag in ("cache_hit", "escalated", "abstained", "blocked"):
            if record.get(flag):
                pipe.hincrby(f"{key}:c", flag, 1)
        for metric in ("trust", "groundedness"):
            if record.get(metric) is not None:
                pipe.hincrbyfloat(f"{key}:c", f"{metric}_sum", record[metric])
                pipe.hincrby(f"{key}:c", f"{metric}_n", 1)
        for node, ms in (record.get("nodes") or {}).items():
            pipe.hincrbyfloat(f"{key}:node_ms", node, ms)
            pipe.hincrby(f"{key}:node_n", node, 1)
        pipe.lpush(f"{key}:lat", record["latency_ms"])
        pipe.ltrim(f"{key}:lat", 0, self._max_samples - 1)
        pipe.lpush(f"{key}:recent", json.dumps({**record, "query": redact(record.get("query", ""))[:160]}))
        pipe.ltrim(f"{key}:recent", 0, self._max_recent - 1)
        bucket = int(time.time() // 60)
        pipe.hincrby(f"{key}:qpm", str(bucket), 1)
        pipe.expire(f"{key}:qpm", 3 * 3600)
        await pipe.execute()

    async def record_feedback(self, tenant: str, request_id: str, rating: str, comment: str | None) -> None:
        key = f"ins:{tenant}"
        pipe = self._redis.pipeline()
        pipe.hincrby(f"{key}:c", f"feedback:{rating}", 1)
        pipe.lpush(f"{key}:feedback", json.dumps({"request_id": request_id, "rating": rating,
                                                   "comment": redact(comment or "")[:500], "ts": time.time()}))
        pipe.ltrim(f"{key}:feedback", 0, 199)
        await pipe.execute()

    async def summary(self, tenant: str) -> dict[str, Any]:
        key = f"ins:{tenant}"
        pipe = self._redis.pipeline()
        pipe.hgetall(f"{key}:c")
        pipe.lrange(f"{key}:lat", 0, -1)
        pipe.hgetall(f"{key}:node_ms")
        pipe.hgetall(f"{key}:node_n")
        pipe.lrange(f"{key}:recent", 0, -1)
        pipe.hgetall(f"{key}:qpm")
        counters_raw, latencies_raw, node_ms, node_n, recent_raw, qpm_raw = await pipe.execute()

        counters = {k.decode(): float(v) for k, v in counters_raw.items()}
        total = int(counters.get("total", 0))
        latencies = [float(v) for v in latencies_raw]

        def avg(name: str) -> float | None:
            n = counters.get(f"{name}_n", 0)
            return round(counters[f"{name}_sum"] / n, 3) if n else None

        now_bucket = int(time.time() // 60)
        qpm = {int(k): int(v) for k, v in qpm_raw.items()}
        timeline = [{"minute": b, "count": qpm.get(b, 0)} for b in range(now_bucket - 59, now_bucket + 1)]
        nodes = sorted(
            ({"node": k.decode(), "avg_ms": round(float(v) / max(1, int(node_n.get(k, 1))), 1),
              "count": int(node_n.get(k, 0))} for k, v in node_ms.items()),
            key=lambda n: n["avg_ms"],
            reverse=True,
        )
        return {
            "total": total,
            "cache_hit_rate": round(counters.get("cache_hit", 0) / total, 3) if total else 0.0,
            "escalation_rate": round(counters.get("escalated", 0) / total, 3) if total else 0.0,
            "abstention_rate": round(counters.get("abstained", 0) / total, 3) if total else 0.0,
            "blocked": int(counters.get("blocked", 0)),
            "avg_trust": avg("trust"),
            "avg_groundedness": avg("groundedness"),
            "latency": {"p50": _percentile(latencies, 50), "p95": _percentile(latencies, 95),
                        "p99": _percentile(latencies, 99), "samples": len(latencies)},
            "routes": {k.split(":", 1)[1]: int(v) for k, v in counters.items() if k.startswith("route:")},
            "modes": {k.split(":", 1)[1]: int(v) for k, v in counters.items() if k.startswith("mode:")},
            "feedback": {"up": int(counters.get("feedback:up", 0)), "down": int(counters.get("feedback:down", 0))},
            "nodes": nodes,
            "timeline": timeline,
            "recent": [json.loads(r) for r in recent_raw],
        }
