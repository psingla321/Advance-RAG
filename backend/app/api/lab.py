"""REST API for the RAG lab: browse components, build pipelines, run experiments.

Mounted at ``/api/lab`` by the main application. Pipelines created here live in
process memory and are keyed by a session id, so a data scientist can drive the
whole toolkit from the UI, a notebook or ``curl`` without touching the production
engine's state.

Sessions belong to the caller that created them (tenant + user), are capped per
caller and globally, and expire when idle. Compute-heavy calls share the chat
rate limiter. The ``/playground/*`` endpoints are stateless, single-stage probes
used by the Learn view to show what one technique does in isolation.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import time
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator

from app.api.auth import Identity, get_identity
from app.config import get_settings
from app.rag.config import RAGConfig, available_presets
from app.rag.evaluation.dataset import Dataset
from app.rag.evaluation.runner import evaluate, evaluate_retrieval
from app.rag.pipeline import INJECTION_PATTERNS, RAGPipeline
from app.rag.registry import ComponentError, build, list_components
from app.rag.types import Document, Section
from app.textutil import estimate_tokens

log = logging.getLogger("advance_rag.lab")

router = APIRouter(prefix="/api/lab", tags=["lab"], dependencies=[Depends(get_identity)])


@dataclass
class _Session:
    pipeline: RAGPipeline
    owner: str
    used: float


_SESSIONS: dict[str, _Session] = {}
# Strong references to fire-and-forget close() tasks so they are not garbage-collected mid-run.
_BACKGROUND: set[asyncio.Task[Any]] = set()


def _close_later(pipeline: RAGPipeline) -> None:
    task = asyncio.create_task(pipeline.close())
    _BACKGROUND.add(task)
    task.add_done_callback(_BACKGROUND.discard)


def _evict(owner: str) -> None:
    """Drop idle sessions, then make room for one more for ``owner``."""
    settings = get_settings()
    now = time.time()
    stale = {k for k, s in _SESSIONS.items() if now - s.used > settings.lab_session_ttl_seconds}

    def oldest(keys: list[str]) -> str:
        return min(keys, key=lambda k: _SESSIONS[k].used)

    mine = [k for k, s in _SESSIONS.items() if s.owner == owner and k not in stale]
    while len(mine) >= settings.lab_max_sessions_per_user:
        victim = oldest(mine)
        mine.remove(victim)
        stale.add(victim)
    live = [k for k in _SESSIONS if k not in stale]
    while len(live) >= settings.lab_max_sessions:
        victim = oldest(live)
        live.remove(victim)
        stale.add(victim)
    for key in stale:
        _close_later(_SESSIONS.pop(key).pipeline)


def _get(session_id: str, identity: Identity) -> RAGPipeline:
    session = _SESSIONS.get(session_id)
    # Someone else's session is indistinguishable from a missing one.
    if session is None or session.owner != identity.principal:
        raise HTTPException(404, "Unknown lab session. Create one with POST /api/lab/sessions.")
    session.used = time.time()
    return session.pipeline


async def rate_limited(request: Request, identity: Identity = Depends(get_identity)) -> Identity:
    """Charge compute-heavy lab calls against the caller's token bucket."""
    services = getattr(request.app.state, "services", None)
    if services is not None:
        limit = await services.limiter.acquire(f"lab:{identity.principal}")
        if not limit.allowed:
            raise HTTPException(429, "Rate limit exceeded. Please slow down.",
                                headers={"Retry-After": str(max(1, round(limit.retry_after)))})
    return identity


def _check_corpus(texts: list[str]) -> None:
    limit = get_settings().lab_max_corpus_chars
    if sum(len(t) for t in texts) > limit:
        raise HTTPException(413, f"Corpus exceeds {limit:,} characters. Use a smaller sample in the lab.")


class CreateSession(BaseModel):
    config: dict[str, Any] | str | None = Field(default=None, description="Preset name or full config object.")
    overrides: dict[str, Any] = Field(default_factory=dict, description="Dotted overrides, e.g. {'retriever.type': 'mmr'}.")


class AddTexts(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=2000)
    metadatas: list[dict[str, Any]] | None = None


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    filters: dict[str, Any] | None = None
    history: list[dict[str, str]] | None = Field(default=None, max_length=20)
    top_n: int | None = Field(default=None, ge=1, le=50)


class EvalRequest(BaseModel):
    samples: list[dict[str, Any]] = Field(min_length=1, max_length=500)
    metrics: list[str] | None = None
    retrieval_only: bool = False
    concurrency: int = Field(default=4, ge=1, le=16)


class SweepRequest(BaseModel):
    base: dict[str, Any] | str | None = None
    grid: dict[str, list[Any]] = Field(default_factory=dict)
    presets: list[str] = Field(default_factory=list)
    samples: list[dict[str, Any]] = Field(min_length=1, max_length=200)
    texts: list[str] = Field(default_factory=list, max_length=2000)
    metrics: list[str] | None = None
    max_configs: int = Field(default=12, ge=1, le=40)


def _samples(rows: list[dict[str, Any]]) -> Dataset:
    from app.rag.evaluation.dataset import _row_to_sample

    parsed = [s for s in (_row_to_sample(r) for r in rows) if s is not None]
    if not parsed:
        raise HTTPException(400, "No usable samples: each row needs at least a 'question' field.")
    return Dataset(parsed, "api")


def _resolve(config: dict[str, Any] | str | None, overrides: dict[str, Any]) -> RAGConfig:
    if isinstance(config, str):
        resolved = RAGConfig.preset(config)
    elif isinstance(config, dict):
        resolved = RAGConfig.from_dict(config)
    else:
        resolved = RAGConfig.preset("offline")
    return resolved.with_overrides(**overrides) if overrides else resolved


# --------------------------------------------------------------- catalogue
@router.get("/components")
async def components(kind: str | None = None) -> dict[str, Any]:
    """Every pluggable component, its parameters and any pip extras it needs."""
    try:
        return {"components": list_components(kind)}
    except ComponentError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.get("/presets")
async def presets() -> dict[str, Any]:
    items = []
    for name in available_presets():
        try:
            config = RAGConfig.preset(name)
            items.append({"name": name, "description": config.description, "config": config.model_dump()})
        except Exception as exc:
            log.warning("Preset %s failed to load: %s", name, exc)
    return {"presets": items}


@router.get("/metrics")
async def metrics_catalogue() -> dict[str, Any]:
    from app.rag.evaluation.metrics import DEFAULT_METRICS, OFFLINE_METRICS, RETRIEVAL_METRICS

    return {
        "available": list_components("metric")["metric"],
        "suites": {"default": DEFAULT_METRICS, "offline": OFFLINE_METRICS, "retrieval": RETRIEVAL_METRICS},
    }


# ----------------------------------------------------------------- sessions
@router.post("/sessions", status_code=201)
async def create_session(body: CreateSession, identity: Identity = Depends(rate_limited)) -> dict[str, Any]:
    """Build a pipeline from a config and keep it in memory for follow-up calls."""
    _evict(identity.principal)
    try:
        pipeline = await RAGPipeline(_resolve(body.config, body.overrides)).init()
    except (ComponentError, ValueError) as exc:
        raise HTTPException(400, str(exc)) from exc

    session_id = uuid4().hex
    _SESSIONS[session_id] = _Session(pipeline, identity.principal, time.time())
    return {"session_id": session_id, "pipeline": pipeline.describe()}


@router.get("/sessions")
async def list_sessions(identity: Identity = Depends(get_identity)) -> dict[str, Any]:
    now = time.time()
    return {
        "sessions": [
            {"session_id": key, "config": s.pipeline.config.name, "stats": s.pipeline.stats,
             "idle_seconds": round(now - s.used, 1)}
            for key, s in _SESSIONS.items() if s.owner == identity.principal
        ]
    }


@router.delete("/sessions/{session_id}", status_code=204)
async def delete_session(session_id: str, identity: Identity = Depends(get_identity)) -> None:
    _get(session_id, identity)
    await _SESSIONS.pop(session_id).pipeline.close()


@router.get("/sessions/{session_id}")
async def describe_session(session_id: str, identity: Identity = Depends(get_identity)) -> dict[str, Any]:
    pipeline = _get(session_id, identity)
    return {"pipeline": pipeline.describe(), "chunks": await pipeline.count(), "stats": pipeline.stats}


# ---------------------------------------------------------------- ingestion
@router.post("/sessions/{session_id}/texts")
async def add_texts(session_id: str, body: AddTexts, identity: Identity = Depends(rate_limited)) -> dict[str, Any]:
    pipeline = _get(session_id, identity)
    _check_corpus(body.texts)
    try:
        return await pipeline.add_texts(body.texts, body.metadatas)
    except Exception as exc:
        log.exception("Lab ingestion failed")
        raise HTTPException(500, f"Ingestion failed: {exc}") from exc


# -------------------------------------------------------------------- query
@router.post("/sessions/{session_id}/ask")
async def ask(session_id: str, body: AskRequest, identity: Identity = Depends(get_identity)) -> dict[str, Any]:
    pipeline = _get(session_id, identity)
    result = await pipeline.query(body.question, filters=body.filters, history=body.history, top_n=body.top_n)
    return result.to_dict()


@router.post("/sessions/{session_id}/explain")
async def explain(session_id: str, body: AskRequest, identity: Identity = Depends(get_identity)) -> dict[str, Any]:
    """Full run with the ranked list captured after every stage — the pipeline X-ray."""
    pipeline = _get(session_id, identity)
    return await pipeline.explain(body.question, filters=body.filters, history=body.history, top_n=body.top_n)


@router.post("/sessions/{session_id}/retrieve")
async def retrieve(session_id: str, body: AskRequest, identity: Identity = Depends(get_identity)) -> dict[str, Any]:
    """Retrieval only — inspect the context without paying for generation."""
    pipeline = _get(session_id, identity)
    contexts, queries, trace = await pipeline.retrieve(
        body.question, filters=body.filters, top_n=body.top_n, history=body.history
    )
    return {
        "queries": queries,
        "trace": trace,
        "contexts": [
            {"n": n, "id": c.id, "score": round(c.score, 4), "scores": c.scores, "filename": c.chunk.filename,
             "heading": c.chunk.heading, "text": c.chunk.context_text}
            for n, c in enumerate(contexts, 1)
        ],
    }


@router.post("/sessions/{session_id}/stream")
async def stream(session_id: str, body: AskRequest, identity: Identity = Depends(get_identity)):
    pipeline = _get(session_id, identity)

    async def events():
        try:
            async for event in pipeline.stream(body.question, filters=body.filters, history=body.history):
                yield f"event: {event['type']}\ndata: {json.dumps(event, default=str)}\n\n"
        except Exception:
            log.exception("Lab streaming failed")
            yield f"event: error\ndata: {json.dumps({'message': 'Streaming failed; see server logs.'})}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# --------------------------------------------------------------- evaluation
@router.post("/sessions/{session_id}/evaluate")
async def run_evaluation(session_id: str, body: EvalRequest, identity: Identity = Depends(rate_limited)) -> dict[str, Any]:
    pipeline = _get(session_id, identity)
    dataset = _samples(body.samples)
    try:
        if body.retrieval_only:
            return await evaluate_retrieval(pipeline, dataset)
        report = await evaluate(
            pipeline, dataset, body.metrics, concurrency=body.concurrency, show_progress=False, keep_contexts=True
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return report.to_dict()


@router.post("/sweep")
async def run_sweep(body: SweepRequest, identity: Identity = Depends(rate_limited)) -> dict[str, Any]:
    """Run a configuration sweep in one request.

    Capped by ``max_configs`` because each configuration re-indexes the corpus.
    """
    from app.rag.experiments.sweep import Sweep

    if not body.grid and not body.presets:
        raise HTTPException(400, "Provide 'grid' axes, 'presets', or both.")
    _check_corpus(body.texts)
    dataset = _samples(body.samples)
    sweep = Sweep(body.base or "offline", dataset, corpus=body.texts or None, metrics=body.metrics)
    if body.presets:
        sweep.presets(body.presets)
    if body.grid:
        sweep.grid(**body.grid)
    if len(sweep.configs) > body.max_configs:
        raise HTTPException(
            400,
            f"That grid expands to {len(sweep.configs)} configurations (limit {body.max_configs}). "
            "Narrow the axes or raise max_configs.",
        )
    results = await sweep.arun(save=False, show_progress=False)
    return {
        "runs": [
            {"name": r.config_name, "overall": r.overall, "scores": r.scores,
             "latency": r.latency, "pipeline": r.pipeline}
            for r in sorted(results.reports, key=lambda r: -r.overall)
        ],
        "failures": results.failures,
        "duration_seconds": results.duration_seconds,
    }


@router.post("/sessions/{session_id}/testset")
async def generate_testset(
    session_id: str, n_questions: int = 20, identity: Identity = Depends(rate_limited)
) -> dict[str, Any]:
    """Synthesise a golden dataset from the session's indexed corpus."""
    from app.rag.evaluation.synth import generate_from_pipeline

    if not 1 <= n_questions <= 200:
        raise HTTPException(400, "n_questions must be between 1 and 200.")
    pipeline = _get(session_id, identity)
    try:
        dataset = await generate_from_pipeline(pipeline, n_questions=n_questions, show_progress=False)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {
        "samples": [
            {"question": s.question, "ground_truth": s.ground_truth, "relevant_ids": s.relevant_ids,
             "metadata": s.metadata}
            for s in dataset
        ]
    }


# --------------------------------------------------------------- playground
# Stateless single-stage probes. Each builds only the component it demonstrates,
# on the offline LLM unless a spec says otherwise, so they are cheap and safe to
# call on every keystroke (the UI debounces).

MAX_PLAYGROUND_CHARS = 60_000
Spec = dict[str, Any] | str


def _spec_name(spec: Spec) -> str:
    return spec if isinstance(spec, str) else str(spec.get("type", "?"))


class ChunkRequest(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_PLAYGROUND_CHARS)
    filename: str = Field(default="document.md", max_length=200)
    chunkers: list[Spec] = Field(min_length=1, max_length=8)


class SimilarityRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    passages: list[str] = Field(min_length=1, max_length=50)
    embedders: list[Spec] = Field(default_factory=lambda: ["hash", "tfidf"], min_length=1, max_length=4)

    @field_validator("passages")
    @classmethod
    def _size(cls, passages: list[str]) -> list[str]:
        if sum(len(p) for p in passages) > MAX_PLAYGROUND_CHARS:
            raise ValueError(f"Passages exceed {MAX_PLAYGROUND_CHARS:,} characters in total.")
        return passages


class TransformRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    transforms: list[Spec] = Field(min_length=1, max_length=8)
    history: list[dict[str, str]] = Field(default_factory=list, max_length=20)


class GuardRequest(BaseModel):
    text: str = Field(min_length=1, max_length=8000)


class CompareVariant(BaseModel):
    label: str = Field(min_length=1, max_length=60)
    overrides: dict[str, Any] = Field(default_factory=dict)


class CompareRequest(BaseModel):
    base: dict[str, Any] | str | None = "offline"
    texts: list[str] = Field(min_length=1, max_length=500)
    question: str = Field(min_length=1, max_length=2000)
    variants: list[CompareVariant] = Field(min_length=1, max_length=4)


def _build(kind: str, spec: Spec, **deps: Any) -> Any:
    try:
        return build(kind, spec, **deps)
    except ComponentError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.post("/playground/chunk")
async def playground_chunk(body: ChunkRequest) -> dict[str, Any]:
    """Split one document with several chunkers so their behaviour can be compared side by side."""
    loader = _build("loader", "auto")
    try:
        sections = await asyncio.to_thread(loader.load, body.filename, body.text.encode())
    except Exception:
        sections = []
    llm = _build("llm", "mock")
    embedder = _build("embedder", "hash")
    document = Document(id="playground", text=body.text, filename=body.filename)

    results = []
    for spec in body.chunkers:
        chunker = _build("chunker", spec, llm=llm, embedder=embedder)
        started = time.perf_counter()
        try:
            chunks = await chunker.chunk(document, sections or [Section("", body.text)])
        except Exception as exc:
            results.append({"chunker": _spec_name(spec), "spec": spec, "error": str(exc), "chunks": []})
            continue
        tokens = [estimate_tokens(c.text) for c in chunks] or [0]
        results.append({
            "chunker": _spec_name(spec), "spec": spec, "describe": chunker.describe(),
            "ms": round((time.perf_counter() - started) * 1000, 2),
            "stats": {"count": len(chunks), "avg_tokens": round(sum(tokens) / len(tokens), 1),
                      "min_tokens": min(tokens), "max_tokens": max(tokens)},
            "chunks": [
                {"ord": c.ord, "heading": c.heading, "text": c.text, "tokens": estimate_tokens(c.text),
                 "embed_text": c.embed_text if c.embed_text != c.text else None,
                 "context_text": c.parent_text, "metadata": {k: v for k, v in c.metadata.items() if k != "_sections"}}
                for c in chunks
            ],
        })
    return {"sections": len(sections), "chunkers": results}


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


@router.post("/playground/similarity")
async def playground_similarity(body: SimilarityRequest) -> dict[str, Any]:
    """Cosine similarity of a query to each passage under different embedders."""
    from app.textutil import content_terms

    query_terms = content_terms(body.query)
    results = []
    for spec in body.embedders:
        embedder = _build("embedder", spec)
        if hasattr(embedder, "fit"):
            embedder.fit(body.passages)
        try:
            vectors = await embedder.embed(body.passages)
            query_vector = await embedder.embed_query(body.query)
        except Exception as exc:
            results.append({"embedder": _spec_name(spec), "error": str(exc), "scores": []})
            continue
        results.append({
            "embedder": _spec_name(spec), "dim": len(query_vector),
            "query_preview": [round(v, 3) for v in query_vector[:24]],
            "scores": [round(_cosine(query_vector, v), 4) for v in vectors],
        })
    return {
        "passages": [
            {"text": p, "shared_terms": sorted(query_terms & content_terms(p))} for p in body.passages
        ],
        "embedders": results,
    }


@router.post("/playground/transform")
async def playground_transform(body: TransformRequest) -> dict[str, Any]:
    """Show the search queries each transform produces for one question."""
    llm = build("llm", "mock") if get_settings().use_mock_llm else _lab_llm()
    results = []
    for spec in body.transforms:
        transform = _build("transform", spec, llm=llm)
        started = time.perf_counter()
        try:
            queries = await transform.transform(body.query, {"history": body.history})
        except Exception as exc:
            results.append({"transform": _spec_name(spec), "error": str(exc), "queries": []})
            continue
        results.append({"transform": _spec_name(spec), "queries": queries,
                        "ms": round((time.perf_counter() - started) * 1000, 2)})
    return {"offline": llm.is_mock, "transforms": results}


def _lab_llm() -> Any:
    settings = get_settings()
    return build("llm", {
        "type": "azure_openai", "endpoint": settings.azure_openai_endpoint,
        "deployment": settings.azure_openai_chat_deployment, "api_version": settings.azure_openai_api_version,
        "api_key": settings.azure_openai_api_key,
    })


@router.post("/playground/guard")
async def playground_guard(body: GuardRequest) -> dict[str, Any]:
    """Which input guardrails a piece of text trips: prompt injection patterns and PII."""
    injections = [p.pattern for p in INJECTION_PATTERNS if p.search(body.text)]
    return {"blocked": bool(injections), "injection_patterns": injections, "pii": RAGPipeline.detect_pii(body.text)}


@router.post("/playground/compare")
async def playground_compare(body: CompareRequest, identity: Identity = Depends(rate_limited)) -> dict[str, Any]:
    """Run one question through up to four pipeline variants over the same corpus."""
    _check_corpus(body.texts)

    async def run(variant: CompareVariant) -> dict[str, Any]:
        try:
            pipeline = await RAGPipeline(_resolve(body.base, variant.overrides)).init()
        except (ComponentError, ValueError) as exc:
            return {"label": variant.label, "overrides": variant.overrides, "error": str(exc)}
        try:
            await pipeline.add_texts(body.texts)
            return {"label": variant.label, "overrides": variant.overrides,
                    **await pipeline.explain(body.question)}
        except Exception as exc:
            log.exception("Compare variant %s failed", variant.label)
            return {"label": variant.label, "overrides": variant.overrides, "error": str(exc)}
        finally:
            await pipeline.close()

    return {"question": body.question, "variants": await asyncio.gather(*(run(v) for v in body.variants))}


async def shutdown_sessions() -> None:
    """Close every live lab pipeline. Called from the app's lifespan."""
    for session in list(_SESSIONS.values()):
        try:
            await session.pipeline.close()
        except Exception as exc:
            log.debug("Lab session cleanup failed: %s", exc)
    _SESSIONS.clear()
