import asyncio
import json
import logging
import re
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from uuid import UUID, uuid4

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from app.api.lab import router as lab_router
from app.api.lab import shutdown_sessions
from app.bootstrap import Core, build_core
from app.config import get_settings
from app.engine.graph import Engine
from app.ingestion.validation import UploadRejected, validate_upload
from app.observability import (
    FEEDBACK, INFLIGHT, QUEUE_DEPTH, RATE_LIMITED, SHED, request_id_var, setup_logging, setup_tracing,
    shutdown_tracing, tracer,
)
from app.services.ratelimit import RateLimiter
from app.textutil import sha256_hex
from app.worker import IngestionWorker

settings = get_settings()
setup_logging(settings)
setup_tracing(settings, "nexus-api")
log = logging.getLogger("nexus.api")

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "sample_data"
SAMPLE_ACL = {"compensation_bands.md": ["hr"]}
_IDENTIFIER = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


@dataclass
class Services:
    core: Core
    engine: Engine
    limiter: RateLimiter
    inflight: int = 0


async def _seed_samples(core: Core) -> None:
    if not SAMPLE_DIR.exists() or await core.db.count_documents("demo") > 0:
        return
    for path in sorted(SAMPLE_DIR.iterdir()):
        data = path.read_bytes()
        acl = SAMPLE_ACL.get(path.name, ["all"])
        doc_id, created = await core.db.create_document("demo", path.name, sha256_hex(data), acl)
        if created:
            await core.queue.enqueue(doc_id, "demo", path.name, data, acl)


@asynccontextmanager
async def lifespan(app: FastAPI):
    core = await build_core(settings)
    engine = Engine(
        settings=settings, llm=core.llm, redis=core.redis, db=core.db, vector_store=core.vector_store,
        embedder=core.embedder, cache=core.cache, insights=core.insights,
    )
    app.state.services = Services(core, engine, RateLimiter(core.redis, settings.rate_limit_capacity, settings.rate_limit_refill_per_sec))

    stop = asyncio.Event()
    worker_task = None
    if settings.run_embedded_worker:
        worker = IngestionWorker(core, f"api-embedded-{uuid4().hex[:6]}", settings.worker_concurrency)
        worker_task = asyncio.create_task(worker.run(stop))
    if settings.seed_sample_docs:
        await _seed_samples(core)
    log.info("Nexus API ready (mock LLM: %s, embedded worker: %s)", core.llm.is_mock, settings.run_embedded_worker)
    yield
    stop.set()
    if worker_task:
        await asyncio.wait_for(worker_task, timeout=10)
    await shutdown_sessions()
    await core.close()
    shutdown_tracing()


app = FastAPI(title=settings.app_name, version="0.2.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type", "X-Tenant-Id", "X-User-Groups", "X-User-Id", "X-Request-Id"],
    expose_headers=["X-Request-Id", "Retry-After", "X-RateLimit-Remaining"],
)
FastAPIInstrumentor.instrument_app(app, excluded_urls="metrics,healthz,readyz")

# The plug-and-play RAG lab: component catalogue, ad-hoc pipelines, evaluation and sweeps.
app.include_router(lab_router)


@app.middleware("http")
async def request_context(request: Request, call_next):
    incoming = request.headers.get("x-request-id", "")
    request_id = incoming if _IDENTIFIER.match(incoming) else uuid4().hex
    token = request_id_var.set(request_id)
    try:
        response = await call_next(request)
    finally:
        request_id_var.reset(token)
    response.headers["X-Request-Id"] = request_id
    return response


@dataclass
class Identity:
    tenant_id: str
    user_id: str
    groups: list[str]


def _parse_groups(raw: str) -> list[str]:
    groups = sorted({g.strip() for g in raw.split(",") if g.strip()})
    if not groups or len(groups) > 20 or not all(_IDENTIFIER.match(g) for g in groups):
        raise HTTPException(400, "Groups must be 1-20 comma-separated identifiers ([A-Za-z0-9_-]).")
    return groups


def get_identity(
    x_tenant_id: str = Header("demo"), x_user_groups: str = Header("all"), x_user_id: str = Header("anonymous")
) -> Identity:
    # Demo identity via headers; in production derive tenant, user and groups from a validated Entra ID token.
    if not _IDENTIFIER.match(x_tenant_id) or not _IDENTIFIER.match(x_user_id):
        raise HTTPException(400, "Invalid tenant or user id.")
    return Identity(x_tenant_id, x_user_id, _parse_groups(x_user_groups))


def get_services(request: Request) -> Services:
    return request.app.state.services


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    conversation_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_-]{1,64}$")


class FeedbackRequest(BaseModel):
    request_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")
    rating: Literal["up", "down"]
    comment: str | None = Field(default=None, max_length=1000)


def _sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


@app.post("/api/chat")
async def chat(body: ChatRequest, identity: Identity = Depends(get_identity), services: Services = Depends(get_services)):
    limit = await services.limiter.acquire(f"{identity.tenant_id}:{identity.user_id}")
    if not limit.allowed:
        RATE_LIMITED.inc()
        return JSONResponse(
            {"detail": "Rate limit exceeded. Please slow down."},
            status_code=429,
            headers={"Retry-After": str(max(1, round(limit.retry_after)))},
        )
    if services.inflight >= settings.max_inflight_requests:
        SHED.inc()
        return JSONResponse({"detail": "Server is busy. Please retry shortly."}, status_code=503, headers={"Retry-After": "2"})

    conversation_id = body.conversation_id or uuid4().hex
    request_id = request_id_var.get()
    state = {
        "request_id": request_id,
        "started_at": time.perf_counter(),
        "query": body.message.strip(),
        "tenant_id": identity.tenant_id,
        "groups": identity.groups,
        "conversation_id": conversation_id,
        "trace": [],
    }

    async def events():
        services.inflight += 1
        INFLIGHT.inc()
        token = request_id_var.set(request_id)
        try:
            with tracer.start_as_current_span("chat.request") as span:
                span.set_attribute("nexus.tenant", identity.tenant_id)
                span.set_attribute("nexus.conversation_id", conversation_id)
                yield _sse("meta", {"conversation_id": conversation_id, "request_id": request_id,
                                    "trace_id": format(span.get_span_context().trace_id, "032x")})
                async for mode, chunk in services.engine.graph.astream(state, stream_mode=["updates", "custom"]):
                    if mode == "custom":
                        yield _sse(chunk["type"], chunk)
                        continue
                    for node, update in chunk.items():
                        if not update:
                            continue
                        for entry in update.get("trace", []):
                            yield _sse("trace", entry)
                        if node == "route":
                            yield _sse("routing", update["routing"])
                        if node == "finalize":
                            final = update["final"]
                            final["latency_ms"] = round((time.perf_counter() - state["started_at"]) * 1000, 1)
                            span.set_attribute("nexus.latency_ms", final["latency_ms"])
                            yield _sse("final", final)
        except Exception:
            log.exception("Chat request failed")
            yield _sse("error", {"message": "The engine failed to process this request.", "request_id": request_id})
        finally:
            services.inflight -= 1
            INFLIGHT.dec()
            request_id_var.reset(token)

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "X-RateLimit-Remaining": str(limit.remaining)},
    )


@app.post("/api/feedback", status_code=204)
async def feedback(body: FeedbackRequest, identity: Identity = Depends(get_identity), services: Services = Depends(get_services)):
    FEEDBACK.labels(body.rating).inc()
    await services.core.insights.record_feedback(identity.tenant_id, body.request_id, body.rating, body.comment)


@app.get("/api/insights")
async def insights(identity: Identity = Depends(get_identity), services: Services = Depends(get_services)):
    summary = await services.core.insights.summary(identity.tenant_id)
    summary["system"] = {
        "queue_depth": await services.core.queue.depth(),
        "inflight": services.inflight,
        "max_inflight": settings.max_inflight_requests,
        "mock_llm": services.core.llm.is_mock,
        "embedded_worker": settings.run_embedded_worker,
    }
    return summary


@app.get("/api/conversations/{conversation_id}")
async def conversation(conversation_id: str, identity: Identity = Depends(get_identity), services: Services = Depends(get_services)):
    if not _IDENTIFIER.match(conversation_id):
        raise HTTPException(400, "Invalid conversation id.")
    return {"conversation_id": conversation_id,
            "messages": await services.engine.memory.history(identity.tenant_id, conversation_id)}


@app.post("/api/documents", status_code=202)
async def upload_document(
    file: UploadFile = File(...),
    acl: str = Form("all"),
    identity: Identity = Depends(get_identity),
    services: Services = Depends(get_services),
):
    max_bytes = settings.max_upload_mb * 1024 * 1024
    data = await file.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise HTTPException(413, f"File exceeds {settings.max_upload_mb} MB.")
    groups = _parse_groups(acl)
    if not set(groups) & set(identity.groups):
        raise HTTPException(400, "The document ACL must include at least one of your groups.")
    try:
        filename = validate_upload(file.filename or "upload.txt", data, max_bytes)
    except UploadRejected as exc:
        raise HTTPException(400, str(exc)) from exc
    core = services.core
    doc_id, created = await core.db.create_document(identity.tenant_id, filename, sha256_hex(data), groups)
    if created:
        await core.queue.enqueue(doc_id, identity.tenant_id, filename, data, groups)
        QUEUE_DEPTH.set(await core.queue.depth())
    return {"id": doc_id, "filename": filename, "status": "QUEUED" if created else "DUPLICATE"}


@app.get("/api/documents")
async def list_documents(identity: Identity = Depends(get_identity), services: Services = Depends(get_services)):
    return {
        "documents": await services.core.db.list_documents(identity.tenant_id, identity.groups),
        "queue_depth": await services.core.queue.depth(),
    }


@app.delete("/api/documents/{doc_id}", status_code=204)
async def delete_document(doc_id: UUID, identity: Identity = Depends(get_identity), services: Services = Depends(get_services)):
    core = services.core
    if not await core.db.delete_document(identity.tenant_id, identity.groups, str(doc_id)):
        raise HTTPException(404, "Document not found.")
    await core.vector_store.delete_document(identity.tenant_id, str(doc_id))
    await core.cache.bump_version(identity.tenant_id)


@app.get("/api/escalations")
async def escalations(identity: Identity = Depends(get_identity), services: Services = Depends(get_services)):
    raw = await services.core.redis.lrange(f"hitl:{identity.tenant_id}", 0, 49)
    return {"items": [json.loads(item) for item in raw]}


@app.get("/api/tickets")
async def tickets(identity: Identity = Depends(get_identity), services: Services = Depends(get_services)):
    raw = await services.core.redis.lrange(f"tickets:{identity.tenant_id}", 0, 49)
    return {"items": [json.loads(item) for item in raw]}


@app.get("/metrics", include_in_schema=False)
async def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/healthz", include_in_schema=False)
async def healthz():
    return {"status": "alive"}


async def _dependency_checks(services: Services) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    for name, probe in (
        ("redis", services.core.redis.ping),
        ("postgres", services.core.db.ping),
        ("qdrant", services.core.vector_store.ping),
    ):
        try:
            checks[name] = bool(await asyncio.wait_for(probe(), timeout=2))
        except Exception:
            checks[name] = False
    return checks


@app.get("/readyz", include_in_schema=False)
async def readyz(services: Services = Depends(get_services)):
    checks = await _dependency_checks(services)
    ready = all(checks.values()) and services.inflight < settings.max_inflight_requests
    return JSONResponse({"ready": ready, "checks": checks}, status_code=200 if ready else 503)


@app.get("/api/health")
async def health(services: Services = Depends(get_services)):
    checks = await _dependency_checks(services)
    return {"status": "ok" if all(checks.values()) else "degraded", "checks": checks, "mock_llm": settings.use_mock_llm}
