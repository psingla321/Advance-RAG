import asyncio
import json

from fakeredis import FakeAsyncRedis

from app.services.insights import InsightsStore
from app.services.jobs import IngestionQueue
from app.services.ratelimit import RateLimiter
from app.worker import IngestionWorker


def test_token_bucket_limits_and_reports_retry() -> None:
    async def scenario() -> None:
        limiter = RateLimiter(FakeAsyncRedis(), capacity=3, refill_per_sec=1.0)
        results = [await limiter.acquire("demo:alice") for _ in range(4)]
        assert [r.allowed for r in results] == [True, True, True, False]
        assert results[-1].retry_after > 0
        assert (await limiter.acquire("demo:bob")).allowed

    asyncio.run(scenario())


class _FakeDb:
    def __init__(self) -> None:
        self.statuses: dict[str, str] = {}

    async def set_status(self, doc_id: str, status: str, **_: object) -> None:
        self.statuses[doc_id] = status


class _FakePipeline:
    def __init__(self) -> None:
        self.ingested: list[tuple[str, str, bytes, list[str]]] = []

    async def ingest(self, doc_id: str, tenant: str, filename: str, data: bytes, acl: list[str]) -> None:
        self.ingested.append((doc_id, filename, data, acl))


class _FakeCore:
    def __init__(self, redis: FakeAsyncRedis) -> None:
        self.redis = redis
        self.queue = IngestionQueue(redis)
        self.pipeline = _FakePipeline()
        self.db = _FakeDb()


def test_queue_worker_processes_and_acks_jobs() -> None:
    async def scenario() -> None:
        redis = FakeAsyncRedis()
        core = _FakeCore(redis)
        await core.queue.ensure_group()
        for i in range(5):
            await core.queue.enqueue(f"doc-{i}", "demo", f"f{i}.md", f"# doc {i}".encode(), ["all"])
        assert await core.queue.depth() == 5

        stop = asyncio.Event()
        worker_task = asyncio.create_task(IngestionWorker(core, "test-consumer", concurrency=2).run(stop))
        for _ in range(100):
            if len(core.pipeline.ingested) == 5:
                break
            await asyncio.sleep(0.02)
        stop.set()
        await asyncio.wait_for(worker_task, timeout=5)

        assert sorted(d for d, *_ in core.pipeline.ingested) == [f"doc-{i}" for i in range(5)]
        assert await core.queue.depth() == 0
        assert await redis.get(IngestionQueue.blob_key("doc-0")) is None

    asyncio.run(scenario())


def test_insights_summary() -> None:
    async def scenario() -> None:
        store = InsightsStore(FakeAsyncRedis())
        for i, (mode, cache) in enumerate([("single", False), ("cache", True), ("parallel", False), ("cache", True)]):
            await store.record("demo", {
                "request_id": f"r{i}", "query": "mail bob@example.com", "mode": mode, "routes": ["B"],
                "cache_hit": cache, "trust": 0.8, "groundedness": 1.0, "latency_ms": 100.0 * (i + 1),
                "nodes": {"router": 10.0, "retrieval_agent": 50.0},
            })
        await store.record_feedback("demo", "r0", "up", None)
        summary = await store.summary("demo")
        assert summary["total"] == 4
        assert summary["cache_hit_rate"] == 0.5
        assert summary["routes"] == {"B": 4}
        assert summary["latency"]["p50"] in (200.0, 300.0) and summary["latency"]["p99"] == 400.0
        assert summary["feedback"] == {"up": 1, "down": 0}
        assert summary["nodes"][0]["node"] == "retrieval_agent"
        assert "[EMAIL]" in summary["recent"][0]["query"]
        assert sum(m["count"] for m in summary["timeline"]) == 4
        json.dumps(summary)

    asyncio.run(scenario())
