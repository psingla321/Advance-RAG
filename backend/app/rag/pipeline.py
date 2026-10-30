"""The pipeline: builds components from a config and runs the RAG loop.

    ingest:  load -> chunk -> embed -> index
    query:   guard -> transform -> retrieve -> rerank -> compress -> generate -> verify

Every stage is swappable and every stage is traced, so a slow or weak step is
visible in ``result.trace`` rather than guessed at.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.rag.base import (
    LLM, Chunker, Compressor, Embedder, Generator, Loader, QueryTransform, Reranker, Retriever, VectorStore,
)
from app.rag.config import RAGConfig
from app.rag.registry import build
from app.rag.types import Chunk, Document, RAGResult, ScoredChunk, Section
from app.textutil import estimate_tokens, sha256_hex

log = logging.getLogger(__name__)

INJECTION_PATTERNS = [
    re.compile(p, re.I)
    for p in (
        r"ignore (all |any |the )?(previous|prior|above) instructions",
        r"disregard (your|the) (system )?(prompt|instructions)",
        r"reveal (your|the) (system )?prompt",
        r"you are now (in )?(developer|dan|god) mode",
        r"print (your|the) (instructions|system prompt)",
    )
]
PII_PATTERNS = {
    "email": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]{2,}\b"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "credit_card": re.compile(r"\b(?:\d[ -]?){13,16}\b"),
}


class GuardrailViolation(ValueError):
    """Raised when input screening rejects a query."""


class Timer:
    """Records per-stage wall time into a trace list."""

    def __init__(self, trace: list[dict[str, Any]], stage: str, **fields: Any) -> None:
        self._trace = trace
        self._stage = stage
        self._fields = fields

    def __enter__(self) -> "Timer":
        self._started = time.perf_counter()
        return self

    def set(self, **fields: Any) -> None:
        self._fields.update(fields)

    def __exit__(self, *exc: Any) -> None:
        self._trace.append(
            {"stage": self._stage, "ms": round((time.perf_counter() - self._started) * 1000, 2), **self._fields}
        )


class RAGPipeline:
    """An assembled, runnable RAG system.

    Build one from a config and use it directly in async code, or wrap it with
    :class:`app.rag.client.RAG` for a synchronous notebook-friendly API.
    """

    def __init__(self, config: RAGConfig) -> None:
        self.config = config
        self.llm: LLM = build("llm", config.llm)
        self.embedder: Embedder = build("embedder", config.embedder)
        self.vectorstore: VectorStore = build("vectorstore", config.vectorstore)
        self.loader: Loader = build("loader", config.loader)
        self.chunker: Chunker = build("chunker", config.chunker, llm=self.llm, embedder=self.embedder)
        self.transform: QueryTransform = build("transform", config.transform, llm=self.llm)
        self.retriever: Retriever = build(
            "retriever", config.retriever, llm=self.llm, embedder=self.embedder, vectorstore=self.vectorstore
        )
        self.reranker: Reranker = build("reranker", config.reranker, llm=self.llm, embedder=self.embedder)
        self.compressor: Compressor = build("compressor", config.compressor, llm=self.llm)
        self.generator: Generator = build("generator", config.generator, llm=self.llm)
        self._answer_cache: dict[str, tuple[float, RAGResult]] = {}
        self._initialized = False
        self.stats: dict[str, Any] = {"queries": 0, "documents": 0, "chunks": 0, "cache_hits": 0}

    # ------------------------------------------------------------------ setup
    async def init(self) -> "RAGPipeline":
        if not self._initialized:
            await self.vectorstore.init(self.embedder.dim)
            self._initialized = True
        return self

    async def close(self) -> None:
        for component in (self.llm, self.embedder, self.vectorstore, self.generator):
            try:
                await component.close()
            except Exception as exc:  # cleanup must never mask the real error
                log.debug("close() failed for %s: %s", component, exc)

    def describe(self) -> dict[str, Any]:
        """The full component graph — what a sweep report shows for each run."""
        return {
            "name": self.config.name,
            "fingerprint": self.config.fingerprint(),
            "components": {
                kind: getattr(self, kind).describe()
                for kind in ("llm", "embedder", "vectorstore", "chunker", "transform",
                             "retriever", "reranker", "compressor", "generator")
            },
            "knobs": {
                "top_k": self.config.top_k, "top_n": self.config.top_n,
                "context_token_budget": self.config.context_token_budget,
            },
        }

    # --------------------------------------------------------------- ingestion
    async def add_documents(
        self, documents: list[Document], namespace: str | None = None, show_progress: bool = False
    ) -> dict[str, Any]:
        """Chunk, embed and index already-loaded documents."""
        await self.init()
        namespace = namespace or self.config.namespace
        started = time.perf_counter()
        all_chunks: list[Chunk] = []
        for index, document in enumerate(documents, 1):
            sections = document.metadata.pop("_sections", None) or [Section("", document.text)]
            chunks = await self.chunker.chunk(document, sections)
            all_chunks.extend(chunks)
            if show_progress:
                print(f"  [{index}/{len(documents)}] {document.filename or document.id}: {len(chunks)} chunks", flush=True)
        if not all_chunks:
            return {"documents": len(documents), "chunks": 0, "seconds": 0.0}

        vectors = await self.embedder.embed([c.embed_text for c in all_chunks])
        await self.vectorstore.upsert(all_chunks, vectors, namespace)
        await self.retriever.index(all_chunks, namespace)
        self.stats["documents"] += len(documents)
        self.stats["chunks"] += len(all_chunks)
        return {
            "documents": len(documents), "chunks": len(all_chunks),
            "seconds": round(time.perf_counter() - started, 2),
            "avg_chunk_tokens": round(sum(estimate_tokens(c.text) for c in all_chunks) / len(all_chunks), 1),
        }

    async def add_texts(
        self, texts: list[str], metadatas: list[dict[str, Any]] | None = None, namespace: str | None = None
    ) -> dict[str, Any]:
        """Index raw strings — the fastest way to try the pipeline on your own data."""
        metadatas = metadatas or [{} for _ in texts]
        documents = [
            Document(
                id=meta.get("id") or sha256_hex(text)[:16],
                text=text,
                filename=meta.get("filename", f"text-{i}"),
                metadata={k: v for k, v in meta.items() if k not in {"id", "filename"}},
            )
            for i, (text, meta) in enumerate(zip(texts, metadatas), 1)
        ]
        return await self.add_documents(documents, namespace)

    async def add_files(
        self, paths: list[str | Path], namespace: str | None = None, metadata: dict[str, Any] | None = None,
        show_progress: bool = False,
    ) -> dict[str, Any]:
        """Load files of any supported type through the configured loader."""
        documents: list[Document] = []
        errors: list[dict[str, str]] = []
        for path in paths:
            path = Path(path)
            try:
                data = path.read_bytes()
                sections = await asyncio.to_thread(self.loader.load, path.name, data)
                text = "\n\n".join(s.text for s in sections)
                document = Document(
                    id=sha256_hex(f"{path.name}:{sha256_hex(data)}")[:16],
                    text=text, filename=path.name,
                    metadata={"source": str(path), "bytes": len(data), **(metadata or {}), "_sections": sections},
                )
                documents.append(document)
            except Exception as exc:
                log.warning("Could not load %s: %s", path, exc)
                errors.append({"file": str(path), "error": str(exc)})
        report = await self.add_documents(documents, namespace, show_progress)
        report["errors"] = errors
        return report

    async def add_directory(
        self, path: str | Path, recursive: bool = True, namespace: str | None = None, show_progress: bool = False
    ) -> dict[str, Any]:
        from app.rag.components.loaders import iter_files

        files = iter_files(path, recursive)
        if show_progress:
            print(f"Loading {len(files)} files from {path}", flush=True)
        return await self.add_files(files, namespace, show_progress=show_progress)

    async def delete(self, doc_ids: list[str], namespace: str | None = None) -> None:
        await self.vectorstore.delete(doc_ids, namespace or self.config.namespace)

    async def count(self, namespace: str | None = None) -> int:
        await self.init()
        return await self.vectorstore.count(namespace or self.config.namespace)

    # ------------------------------------------------------------------ query
    def _guard(self, query: str) -> None:
        if not self.config.guardrails:
            return
        for pattern in INJECTION_PATTERNS:
            if pattern.search(query):
                raise GuardrailViolation("This request looks like a prompt-injection attempt and was blocked.")

    @staticmethod
    def detect_pii(text: str) -> dict[str, list[str]]:
        """Report PII found in a string — call before logging or indexing user data."""
        return {name: pattern.findall(text) for name, pattern in PII_PATTERNS.items() if pattern.search(text)}

    def _cache_key(self, query: str, namespace: str, filters: dict[str, Any] | None) -> str:
        return hashlib.sha256(
            f"{self.config.fingerprint()}|{namespace}|{query.strip().lower()}|{sorted((filters or {}).items())}".encode()
        ).hexdigest()

    async def retrieve(
        self, query: str, namespace: str | None = None, filters: dict[str, Any] | None = None,
        top_n: int | None = None, history: list[dict[str, str]] | None = None,
    ) -> tuple[list[ScoredChunk], list[str], list[dict[str, Any]]]:
        """Run retrieval only. Returns (contexts, queries searched, trace).

        Exposed separately so you can evaluate and tune retrieval without paying
        for generation — which is where most RAG quality problems actually live.
        """
        await self.init()
        namespace = namespace or self.config.namespace
        trace: list[dict[str, Any]] = []

        with Timer(trace, "transform") as timer:
            queries = await self.transform.transform(query, {"history": history or []})
            queries = [q for q in queries if q.strip()] or [query]
            timer.set(queries=len(queries), variants=queries[:5])

        with Timer(trace, "retrieve") as timer:
            candidates = await self.retriever.retrieve(queries, self.config.top_k, namespace, filters)
            timer.set(candidates=len(candidates))

        with Timer(trace, "rerank") as timer:
            ranked = await self.reranker.rerank(query, candidates, top_n or self.config.top_n)
            if self.config.min_score:
                ranked = [r for r in ranked if r.score >= self.config.min_score]
            timer.set(kept=len(ranked), top_score=round(ranked[0].score, 4) if ranked else 0.0)

        with Timer(trace, "compress") as timer:
            packed = await self.compressor.compress(query, ranked, self.config.context_token_budget)
            timer.set(packed=len(packed), tokens=sum(estimate_tokens(p.chunk.context_text) for p in packed))

        return packed, queries, trace

    async def query(
        self, question: str, namespace: str | None = None, filters: dict[str, Any] | None = None,
        history: list[dict[str, str]] | None = None, top_n: int | None = None,
    ) -> RAGResult:
        """The full loop. Never raises on component failure — errors land in ``result.metadata``."""
        started = time.perf_counter()
        namespace = namespace or self.config.namespace
        self.stats["queries"] += 1

        try:
            self._guard(question)
        except GuardrailViolation as exc:
            return RAGResult(query=question, answer=str(exc), abstained=True, metadata={"blocked": True})

        key = self._cache_key(question, namespace, filters)
        if self.config.cache and key in self._answer_cache:
            cached_at, cached = self._answer_cache[key]
            if time.time() - cached_at < self.config.cache_ttl_seconds:
                self.stats["cache_hits"] += 1
                cached.metadata = {**cached.metadata, "cache": "hit"}
                return cached

        contexts, queries, trace = await self.retrieve(question, namespace, filters, top_n, history)

        if not contexts and self.config.abstain_when_empty:
            result = RAGResult(
                query=question,
                answer="I could not find anything in the knowledge base that answers this question.",
                abstained=True, expanded_queries=queries, trace=trace,
                latency_ms=round((time.perf_counter() - started) * 1000, 1),
            )
            return result

        with Timer(trace, "generate") as timer:
            result = await self.generator.generate(
                question, contexts, history=history, system=self.config.system_prompt
            )
            timer.set(answer_chars=len(result.answer), abstained=result.abstained)

        result.expanded_queries = queries
        result.trace = trace + result.trace
        result.latency_ms = round((time.perf_counter() - started) * 1000, 1)

        if self.config.verify and result.groundedness is None and not result.abstained:
            with Timer(result.trace, "verify") as timer:
                result.groundedness = await self._groundedness(result)
                timer.set(groundedness=result.groundedness)
            if result.groundedness < self.config.verify_threshold:
                result.metadata["low_groundedness"] = True

        if self.config.cache:
            self._answer_cache[key] = (time.time(), result)
        return result

    async def stream(
        self, question: str, namespace: str | None = None, filters: dict[str, Any] | None = None,
        history: list[dict[str, str]] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Token-by-token streaming with structured events: contexts, token, done."""
        started = time.perf_counter()
        try:
            self._guard(question)
        except GuardrailViolation as exc:
            yield {"type": "error", "message": str(exc)}
            return

        contexts, queries, trace = await self.retrieve(question, namespace, filters, None, history)
        yield {
            "type": "contexts",
            "queries": queries,
            "contexts": [
                {"n": n, "id": c.id, "score": round(c.score, 4), "filename": c.chunk.filename,
                 "heading": c.chunk.heading, "snippet": c.chunk.text[:200]}
                for n, c in enumerate(contexts, 1)
            ],
            "trace": trace,
        }
        if not contexts:
            yield {"type": "done", "answer": "No relevant context found.", "abstained": True}
            return

        buffer: list[str] = []
        try:
            async for token in self.generator.stream(
                question, contexts, history=history, system=self.config.system_prompt
            ):
                buffer.append(token)
                yield {"type": "token", "text": token}
        except NotImplementedError:
            result = await self.generator.generate(question, contexts, history=history)
            buffer = [result.answer]
            yield {"type": "token", "text": result.answer}

        from app.rag.components.generators import build_citations

        answer = "".join(buffer).strip()
        yield {
            "type": "done",
            "answer": answer,
            "citations": [c.__dict__ for c in build_citations(contexts, answer)],
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
        }

    async def batch_query(
        self, questions: list[str], concurrency: int = 4, namespace: str | None = None,
        filters: dict[str, Any] | None = None, show_progress: bool = False,
    ) -> list[RAGResult]:
        """Run many questions with bounded concurrency — the evaluation workhorse."""
        semaphore = asyncio.Semaphore(concurrency)
        done = 0

        async def run(question: str) -> RAGResult:
            nonlocal done
            async with semaphore:
                try:
                    result = await self.query(question, namespace, filters)
                except Exception as exc:
                    log.exception("Query failed: %s", question)
                    result = RAGResult(query=question, answer="", metadata={"error": str(exc)})
            done += 1
            if show_progress and done % max(1, len(questions) // 20) == 0:
                print(f"  {done}/{len(questions)} queries", flush=True)
            return result

        return list(await asyncio.gather(*(run(q) for q in questions)))

    # ---------------------------------------------------------------- verify
    async def _groundedness(self, result: RAGResult) -> float:
        """Share of answer sentences supported by the retrieved context.

        Uses the LLM when one is configured, otherwise a lexical overlap proxy —
        so this metric is always available, even offline.
        """
        from app.rag.evaluation.metrics import lexical_groundedness

        if self.llm.is_mock:
            return lexical_groundedness(result.answer, result.context_texts)
        from app.rag.components.generators import render_context

        prompt = (
            "Count how many sentences of the answer are fully supported by the context.\n\n"
            f"Context:\n{render_context(result.contexts)}\n\nAnswer:\n{result.answer}\n\n"
            'Return JSON: {"supported": <int>, "total": <int>}'
        )
        try:
            import json

            parsed = json.loads(await self.llm.ask(prompt, json_mode=True, max_tokens=100))
            total = max(1, int(parsed.get("total", 1)))
            return round(min(1.0, int(parsed.get("supported", 0)) / total), 3)
        except Exception as exc:
            log.warning("LLM groundedness check failed, using lexical fallback: %s", exc)
            return lexical_groundedness(result.answer, result.context_texts)


async def build_pipeline(config: RAGConfig | dict[str, Any] | str | None = None, **overrides: Any) -> RAGPipeline:
    """Convenience builder: accepts a config object, dict, preset name or nothing."""
    if config is None:
        resolved = RAGConfig()
    elif isinstance(config, RAGConfig):
        resolved = config
    elif isinstance(config, dict):
        resolved = RAGConfig.from_dict(config)
    elif Path(config).exists():
        resolved = RAGConfig.from_yaml(config)
    else:
        resolved = RAGConfig.preset(config)
    if overrides:
        resolved = resolved.with_overrides(**overrides)
    return await RAGPipeline(resolved).init()
