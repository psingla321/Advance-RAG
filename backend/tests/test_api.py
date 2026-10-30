"""Boots the real FastAPI app (lifespan, embedded worker, middleware, SSE, metrics) on in-memory infrastructure."""

import json
import os
import time
from typing import Any
from uuid import uuid4

os.environ.update({"SEED_SAMPLE_DOCS": "true", "JSON_LOGS": "false", "EMBEDDING_DIM": "256",
                   "AZURE_OPENAI_ENDPOINT": "", "OTEL_EXPORTER_OTLP_ENDPOINT": "", "RUN_EMBEDDED_WORKER": "true"})

import pytest  # noqa: E402
from fakeredis import FakeAsyncRedis  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.config import get_settings  # noqa: E402

get_settings.cache_clear()

import app.main as api  # noqa: E402
from app.bootstrap import Core  # noqa: E402
from app.ingestion.pipeline import IngestionPipeline  # noqa: E402
from app.llm.client import MockLLM  # noqa: E402
from app.services.answer_cache import AnswerCache  # noqa: E402
from app.services.embedder import CachedEmbedder  # noqa: E402
from app.services.insights import InsightsStore  # noqa: E402
from app.services.jobs import IngestionQueue  # noqa: E402
from app.services.ratelimit import RateLimiter  # noqa: E402
from app.services.vector_store import VectorStore  # noqa: E402
from tests.test_engine_graph import FakeDatabase  # noqa: E402


class FakeApiDatabase(FakeDatabase):
    def __init__(self) -> None:
        super().__init__()
        self.filenames: dict[str, str] = {}
        self.docs: dict[str, dict[str, Any]] = {}

    async def create_document(self, tenant: str, filename: str, content_hash: str, acl: list[str]) -> tuple[str, bool]:
        for doc_id, doc in self.docs.items():
            if doc["tenant"] == tenant and doc["hash"] == content_hash:
                return doc_id, False
        doc_id = str(uuid4())
        self.filenames[doc_id] = filename
        self.docs[doc_id] = {"tenant": tenant, "hash": content_hash, "acl": acl, "filename": filename}
        self.statuses[doc_id] = "RECEIVED"
        return doc_id, True

    async def set_status(self, doc_id: str, status: str, *, chunk_count: int | None = None, error: str | None = None) -> None:
        self.statuses[doc_id] = status

    async def count_documents(self, tenant: str) -> int:
        return sum(1 for d in self.docs.values() if d["tenant"] == tenant)

    async def list_documents(self, tenant: str, groups: list[str]) -> list[dict[str, Any]]:
        return [
            {"id": i, "filename": d["filename"], "status": self.statuses[i], "acl": d["acl"], "error": None,
             "created_at": "2026-01-01T00:00:00", "chunk_count": sum(1 for c in self.chunks if c["doc_id"] == i)}
            for i, d in self.docs.items() if d["tenant"] == tenant and set(d["acl"]) & set(groups)
        ]

    async def ping(self) -> bool:
        return True

    async def close(self) -> None:
        return None


async def fake_build_core(settings) -> Core:
    llm = MockLLM(settings.embedding_dim)
    redis = FakeAsyncRedis()
    db = FakeApiDatabase()
    vector_store = VectorStore(":memory:", settings.embedding_dim)
    await vector_store.init()
    embedder = CachedEmbedder(llm, redis, settings.embedding_model_id, settings.embedding_dim)
    cache = AnswerCache(redis, vector_store, embedder, settings)
    pipeline = IngestionPipeline(settings, db, vector_store, embedder, cache)
    queue = IngestionQueue(redis)
    await queue.ensure_group()
    return Core(settings, llm, redis, db, vector_store, embedder, cache, pipeline, queue, InsightsStore(redis))


def parse_sse(body: str) -> list[tuple[str, Any]]:
    events = []
    for block in body.strip().split("\n\n"):
        lines = block.split("\n")
        event = next(l[6:].strip() for l in lines if l.startswith("event:"))
        data = json.loads("".join(l[5:].strip() for l in lines if l.startswith("data:")))
        events.append((event, data))
    return events


@pytest.fixture(scope="module")
def client():
    original = api.build_core
    api.build_core = fake_build_core
    try:
        with TestClient(api.app) as c:
            deadline = time.time() + 20
            while time.time() < deadline:
                docs = c.get("/api/documents", headers={"X-User-Groups": "all,hr"}).json()["documents"]
                if len(docs) == 4 and all(d["status"] == "INDEXED" for d in docs):
                    break
                time.sleep(0.1)
            else:
                pytest.fail(f"Sample documents were not indexed: {docs}")
            yield c
    finally:
        api.build_core = original


def test_chat_streams_events_and_metadata(client: TestClient) -> None:
    response = client.post("/api/chat", json={"message": "What is the hotel limit in high-cost cities?"},
                           headers={"X-User-Id": "alice", "X-Request-Id": "req-test-1"})
    assert response.status_code == 200
    assert response.headers["x-request-id"] == "req-test-1"
    events = parse_sse(response.text)
    kinds = [e for e, _ in events]
    assert kinds[0] == "meta" and "token" in kinds and kinds[-1] == "final"
    meta = events[0][1]
    assert meta["request_id"] == "req-test-1" and len(meta["trace_id"]) == 32
    final = events[-1][1]
    assert final["request_id"] == "req-test-1"
    assert final["citations"] and "300" in final["answer"]


def test_feedback_insights_and_metrics(client: TestClient) -> None:
    assert client.post("/api/feedback", json={"request_id": "req-test-1", "rating": "up"}).status_code == 204
    insights = client.get("/api/insights").json()
    assert insights["total"] >= 1 and insights["feedback"]["up"] == 1
    assert insights["system"]["embedded_worker"] is True
    metrics = client.get("/metrics").text
    for name in ("nexus_chat_requests_total", "nexus_node_latency_seconds_bucket", "nexus_cache_lookups_total",
                 "nexus_ingestion_jobs_total", "nexus_llm_calls_total"):
        assert name in metrics


def test_upload_goes_through_queue(client: TestClient) -> None:
    files = {"file": ("faq.md", b"# FAQ\n\nThe cafeteria opens at 08:00 and closes at 16:00 on weekdays.", "text/markdown")}
    response = client.post("/api/documents", files=files, data={"acl": "all"})
    assert response.status_code == 202 and response.json()["status"] == "QUEUED"
    for _ in range(100):
        docs = {d["filename"]: d for d in client.get("/api/documents").json()["documents"]}
        if docs.get("faq.md", {}).get("status") == "INDEXED":
            break
        time.sleep(0.05)
    assert docs["faq.md"]["status"] == "INDEXED"
    duplicate = client.post("/api/documents", files=files, data={"acl": "all"})
    assert duplicate.json()["status"] == "DUPLICATE"


def test_rate_limit_returns_429(client: TestClient) -> None:
    services = api.app.state.services
    original = services.limiter
    services.limiter = RateLimiter(services.core.redis, capacity=2, refill_per_sec=0.01)
    try:
        codes = [client.post("/api/chat", json={"message": "hello"}, headers={"X-User-Id": "burst"}).status_code for _ in range(3)]
    finally:
        services.limiter = original
    assert codes == [200, 200, 429]


def test_health_endpoints(client: TestClient) -> None:
    assert client.get("/healthz").json() == {"status": "alive"}
    ready = client.get("/readyz")
    assert ready.status_code == 200 and ready.json()["ready"] is True
