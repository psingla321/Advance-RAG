"""REST API for the RAG lab: browse components, build pipelines, run experiments.

Mounted at ``/api/lab`` by the main application. Pipelines created here live in
process memory and are keyed by a session id, so a data scientist can drive the
whole toolkit from the UI, a notebook or ``curl`` without touching the production
engine's state.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.rag.config import RAGConfig, PRESETS, available_presets
from app.rag.evaluation.dataset import Dataset
from app.rag.evaluation.runner import evaluate, evaluate_retrieval
from app.rag.pipeline import RAGPipeline
from app.rag.registry import ComponentError, list_components
from app.rag.types import EvalSample

log = logging.getLogger("nexus.lab")

router = APIRouter(prefix="/api/lab", tags=["lab"])

# Session id -> (pipeline, last used). Bounded so a long-running server cannot
# accumulate indexes indefinitely.
_SESSIONS: dict[str, tuple[RAGPipeline, float]] = {}
MAX_SESSIONS = 12
SESSION_TTL_SECONDS = 3600


def _evict() -> None:
    now = time.time()
    stale = [k for k, (_, used) in _SESSIONS.items() if now - used > SESSION_TTL_SECONDS]
    while len(_SESSIONS) - len(stale) >= MAX_SESSIONS:
        oldest = min((k for k in _SESSIONS if k not in stale), key=lambda k: _SESSIONS[k][1])
        stale.append(oldest)
    for key in stale:
        pipeline, _ = _SESSIONS.pop(key)
        asyncio.create_task(pipeline.close())


def _get(session_id: str) -> RAGPipeline:
    entry = _SESSIONS.get(session_id)
    if entry is None:
        raise HTTPException(404, "Unknown lab session. Create one with POST /api/lab/sessions.")
    pipeline, _ = entry
    _SESSIONS[session_id] = (pipeline, time.time())
    return pipeline


class CreateSession(BaseModel):
    config: dict[str, Any] | str | None = Field(default=None, description="Preset name or full config object.")
    overrides: dict[str, Any] = Field(default_factory=dict, description="Dotted overrides, e.g. {'retriever.type': 'mmr'}.")


class AddTexts(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=2000)
    metadatas: list[dict[str, Any]] | None = None


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    filters: dict[str, Any] | None = None
    history: list[dict[str, str]] | None = None
    top_n: int | None = None


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
async def create_session(body: CreateSession) -> dict[str, Any]:
    """Build a pipeline from a config and keep it in memory for follow-up calls."""
    _evict()
    try:
        if isinstance(body.config, str):
            config = RAGConfig.preset(body.config)
        elif isinstance(body.config, dict):
            config = RAGConfig.from_dict(body.config)
        else:
            config = RAGConfig.preset("offline")
        if body.overrides:
            config = config.with_overrides(**body.overrides)
        pipeline = await RAGPipeline(config).init()
    except (ComponentError, ValueError) as exc:
        raise HTTPException(400, str(exc)) from exc

    session_id = uuid4().hex[:12]
    _SESSIONS[session_id] = (pipeline, time.time())
    return {"session_id": session_id, "pipeline": pipeline.describe()}


@router.get("/sessions")
async def list_sessions() -> dict[str, Any]:
    return {
        "sessions": [
            {"session_id": key, "config": pipeline.config.name, "stats": pipeline.stats,
             "idle_seconds": round(time.time() - used, 1)}
            for key, (pipeline, used) in _SESSIONS.items()
        ]
    }


@router.delete("/sessions/{session_id}", status_code=204)
async def delete_session(session_id: str) -> None:
    entry = _SESSIONS.pop(session_id, None)
    if entry is None:
        raise HTTPException(404, "Unknown lab session.")
    await entry[0].close()


@router.get("/sessions/{session_id}")
async def describe_session(session_id: str) -> dict[str, Any]:
    pipeline = _get(session_id)
    return {"pipeline": pipeline.describe(), "chunks": await pipeline.count(), "stats": pipeline.stats}


# ---------------------------------------------------------------- ingestion
@router.post("/sessions/{session_id}/texts")
async def add_texts(session_id: str, body: AddTexts) -> dict[str, Any]:
    pipeline = _get(session_id)
    try:
        return await pipeline.add_texts(body.texts, body.metadatas)
    except Exception as exc:
        log.exception("Lab ingestion failed")
        raise HTTPException(500, f"Ingestion failed: {exc}") from exc


# -------------------------------------------------------------------- query
@router.post("/sessions/{session_id}/ask")
async def ask(session_id: str, body: AskRequest) -> dict[str, Any]:
    pipeline = _get(session_id)
    result = await pipeline.query(body.question, filters=body.filters, history=body.history, top_n=body.top_n)
    return result.to_dict()


@router.post("/sessions/{session_id}/retrieve")
async def retrieve(session_id: str, body: AskRequest) -> dict[str, Any]:
    """Retrieval only — inspect the context without paying for generation."""
    pipeline = _get(session_id)
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
async def stream(session_id: str, body: AskRequest):
    pipeline = _get(session_id)

    async def events():
        try:
            async for event in pipeline.stream(body.question, filters=body.filters, history=body.history):
                yield f"event: {event['type']}\ndata: {json.dumps(event, default=str)}\n\n"
        except Exception as exc:
            log.exception("Lab streaming failed")
            yield f"event: error\ndata: {json.dumps({'message': str(exc)})}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# --------------------------------------------------------------- evaluation
@router.post("/sessions/{session_id}/evaluate")
async def run_evaluation(session_id: str, body: EvalRequest) -> dict[str, Any]:
    pipeline = _get(session_id)
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
async def run_sweep(body: SweepRequest) -> dict[str, Any]:
    """Run a configuration sweep in one request.

    Capped by ``max_configs`` because each configuration re-indexes the corpus.
    """
    from app.rag.experiments.sweep import Sweep

    if not body.grid and not body.presets:
        raise HTTPException(400, "Provide 'grid' axes, 'presets', or both.")
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
async def generate_testset(session_id: str, n_questions: int = 20) -> dict[str, Any]:
    """Synthesise a golden dataset from the session's indexed corpus."""
    from app.rag.evaluation.synth import generate_from_pipeline

    pipeline = _get(session_id)
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


async def shutdown_sessions() -> None:
    """Close every live lab pipeline. Called from the app's lifespan."""
    for pipeline, _ in list(_SESSIONS.values()):
        try:
            await pipeline.close()
        except Exception as exc:
            log.debug("Lab session cleanup failed: %s", exc)
    _SESSIONS.clear()
