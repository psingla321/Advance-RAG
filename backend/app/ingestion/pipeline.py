import asyncio
import logging
import time
from uuid import uuid4

from app.config import Settings
from app.ingestion.chunker import chunk_sections, extract
from app.ingestion.validation import UploadRejected
from app.observability import INGEST_JOBS, INGEST_LATENCY, tracer
from app.services.answer_cache import AnswerCache
from app.services.db import Database
from app.services.embedder import CachedEmbedder
from app.services.vector_store import VectorStore

log = logging.getLogger(__name__)


class IngestionError(ValueError):
    pass


class IngestionPipeline:
    """Extract -> chunk -> embed (L5 cached) -> index (PostgreSQL FTS + Qdrant) -> invalidate answer caches."""

    def __init__(self, settings: Settings, db: Database, vector_store: VectorStore, embedder: CachedEmbedder, cache: AnswerCache) -> None:
        self._settings = settings
        self._db = db
        self._vs = vector_store
        self._embedder = embedder
        self._cache = cache

    async def ingest(self, doc_id: str, tenant: str, filename: str, data: bytes, acl: list[str]) -> None:
        started = time.perf_counter()
        with tracer.start_as_current_span("ingestion.document") as span:
            span.set_attribute("nexus.doc_id", doc_id)
            span.set_attribute("nexus.tenant", tenant)
            span.set_attribute("nexus.bytes", len(data))
            ok = await self._ingest(doc_id, tenant, filename, data, acl)
            span.set_attribute("nexus.status", "indexed" if ok else "failed")
        INGEST_JOBS.labels("indexed" if ok else "failed").inc()
        INGEST_LATENCY.observe(time.perf_counter() - started)

    async def _ingest(self, doc_id: str, tenant: str, filename: str, data: bytes, acl: list[str]) -> bool:
        try:
            await self._db.set_status(doc_id, "PROCESSING")
            sections = await asyncio.to_thread(extract, filename, data)
            chunks = chunk_sections(sections, self._settings.chunk_max_tokens, self._settings.chunk_overlap_sentences)
            if not chunks:
                raise IngestionError("No extractable text was found in the document.")

            texts = [f"{c.heading}\n{c.content}" if c.heading else c.content for c in chunks]
            vectors = await self._embedder.embed(texts)
            ids = [str(uuid4()) for _ in chunks]

            await self._db.insert_chunks(doc_id, tenant, acl, [(ids[i], c.ord, c.heading, c.content) for i, c in enumerate(chunks)])
            await self._vs.upsert_chunks(
                [
                    {
                        "id": ids[i],
                        "vector": vectors[i],
                        "payload": {"tenant_id": tenant, "doc_id": doc_id, "filename": filename, "heading": c.heading,
                                    "content": c.content, "acl": acl, "ord": c.ord},
                    }
                    for i, c in enumerate(chunks)
                ]
            )
            await self._db.set_status(doc_id, "INDEXED", chunk_count=len(chunks))
            await self._cache.bump_version(tenant)
            log.info("Indexed %s (%s chunks) for tenant %s", filename, len(chunks), tenant)
            return True
        except Exception as exc:
            log.exception("Ingestion failed for %s", doc_id)
            message = str(exc)[:300] if isinstance(exc, (IngestionError, UploadRejected)) else "Processing failed."
            await self._db.delete_chunks(doc_id)
            try:
                await self._vs.delete_document(tenant, doc_id)
            except Exception:
                log.warning("Vector cleanup failed for %s", doc_id)
            await self._db.set_status(doc_id, "FAILED", error=message)
            return False
