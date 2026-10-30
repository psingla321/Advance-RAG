"""Vector store backends. ``memory`` needs nothing installed and persists to disk on demand."""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

from app.rag.base import VectorStore
from app.rag.registry import register
from app.rag.types import Chunk, ScoredChunk

log = logging.getLogger(__name__)


def _matches(payload: dict[str, Any], filters: dict[str, Any] | None) -> bool:
    """Filter semantics: scalar means equality, list means 'any of'."""
    if not filters:
        return True
    for key, wanted in filters.items():
        value = payload.get(key)
        if isinstance(wanted, (list, tuple, set)):
            if isinstance(value, (list, tuple, set)):
                if not set(value) & set(wanted):
                    return False
            elif value not in wanted:
                return False
        elif isinstance(value, (list, tuple, set)):
            if wanted not in value:
                return False
        elif value != wanted:
            return False
    return True


def _as_chunk(data: dict[str, Any]) -> Chunk:
    return Chunk(
        id=str(data.get("id", "")),
        text=data.get("text", ""),
        doc_id=data.get("doc_id", ""),
        filename=data.get("filename", ""),
        heading=data.get("heading", ""),
        ord=int(data.get("ord", 0)),
        metadata=data.get("metadata", {}) or {},
        parent_text=data.get("parent_text"),
    )


def _payload(chunk: Chunk) -> dict[str, Any]:
    return {
        "id": chunk.id, "text": chunk.text, "doc_id": chunk.doc_id, "filename": chunk.filename,
        "heading": chunk.heading, "ord": chunk.ord, "metadata": chunk.metadata, "parent_text": chunk.parent_text,
        **{k: v for k, v in chunk.metadata.items() if isinstance(v, (str, int, float, bool, list))},
    }


@register("vectorstore", "memory")
class MemoryVectorStore(VectorStore):
    """Pure-Python in-process store — zero infrastructure, exact search.

    Fast enough for tens of thousands of chunks, which covers most experiments.
    ``persist_path`` makes it survive a kernel restart.
    """

    def __init__(self, persist_path: str | None = None) -> None:
        self._data: dict[str, dict[str, tuple[list[float], Chunk]]] = {}
        self._path = Path(persist_path) if persist_path else None
        if self._path and self._path.exists():
            self._load()

    async def init(self, dim: int) -> None:
        self.dim = dim

    async def upsert(self, chunks: list[Chunk], vectors: list[list[float]], namespace: str = "default") -> None:
        space = self._data.setdefault(namespace, {})
        for chunk, vector in zip(chunks, vectors):
            norm = math.sqrt(sum(v * v for v in vector)) or 1.0
            space[chunk.id] = ([v / norm for v in vector], chunk)
        self._save()

    async def search(
        self, vector: list[float], k: int, namespace: str = "default", filters: dict[str, Any] | None = None
    ) -> list[ScoredChunk]:
        space = self._data.get(namespace, {})
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        query = [v / norm for v in vector]
        scored: list[tuple[float, Chunk]] = []
        for stored, chunk in space.values():
            if not _matches(_payload(chunk), filters):
                continue
            scored.append((sum(a * b for a, b in zip(query, stored)), chunk))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            ScoredChunk(chunk=chunk, score=float(score), scores={"dense": float(score)}, rank=i)
            for i, (score, chunk) in enumerate(scored[:k], 1)
        ]

    async def delete(self, doc_ids: list[str], namespace: str = "default") -> None:
        space = self._data.get(namespace, {})
        targets = set(doc_ids)
        for key in [k for k, (_, chunk) in space.items() if chunk.doc_id in targets or chunk.id in targets]:
            space.pop(key, None)
        self._save()

    async def all_chunks(self, namespace: str = "default") -> list[Chunk]:
        return [chunk for _, chunk in self._data.get(namespace, {}).values()]

    def _save(self) -> None:
        if not self._path:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            ns: [{"vector": v, "chunk": _payload(c)} for v, c in space.values()]
            for ns, space in self._data.items()
        }
        self._path.write_text(json.dumps(payload), "utf-8")

    def _load(self) -> None:
        assert self._path is not None
        raw = json.loads(self._path.read_text("utf-8"))
        self._data = {
            ns: {item["chunk"]["id"]: (item["vector"], _as_chunk(item["chunk"])) for item in items}
            for ns, items in raw.items()
        }


@register("vectorstore", "qdrant", requires=["qdrant-client"])
class QdrantVectorStore(VectorStore):
    """Qdrant — the production store used by the Nexus engine.

    ``url=":memory:"`` runs the embedded local mode, so tests need no container.
    """

    def __init__(self, url: str = "http://localhost:6333", collection: str = "nexus_lab", api_key: str | None = None) -> None:
        from qdrant_client import AsyncQdrantClient

        self._client = (
            AsyncQdrantClient(location=url) if url == ":memory:" else AsyncQdrantClient(url=url, api_key=api_key, timeout=30)
        )
        self._collection = collection
        self.dim = 0

    def _name(self, namespace: str) -> str:
        return f"{self._collection}_{namespace}_{self.dim}"

    async def init(self, dim: int) -> None:
        self.dim = dim

    async def _ensure(self, namespace: str) -> str:
        from qdrant_client import models

        name = self._name(namespace)
        if not await self._client.collection_exists(name):
            await self._client.create_collection(
                name, vectors_config=models.VectorParams(size=self.dim, distance=models.Distance.COSINE)
            )
            for field in ("doc_id", "filename"):
                await self._client.create_payload_index(name, field, models.PayloadSchemaType.KEYWORD)
        return name

    async def upsert(self, chunks: list[Chunk], vectors: list[list[float]], namespace: str = "default") -> None:
        from uuid import uuid5, NAMESPACE_URL

        from qdrant_client import models

        name = await self._ensure(namespace)
        points = [
            models.PointStruct(id=str(uuid5(NAMESPACE_URL, c.id)), vector=v, payload=_payload(c))
            for c, v in zip(chunks, vectors)
        ]
        for start in range(0, len(points), 256):
            await self._client.upsert(name, points=points[start : start + 256], wait=True)

    async def search(
        self, vector: list[float], k: int, namespace: str = "default", filters: dict[str, Any] | None = None
    ) -> list[ScoredChunk]:
        from qdrant_client import models

        name = await self._ensure(namespace)
        conditions = []
        for key, wanted in (filters or {}).items():
            match = models.MatchAny(any=list(wanted)) if isinstance(wanted, (list, tuple, set)) else models.MatchValue(value=wanted)
            conditions.append(models.FieldCondition(key=key, match=match))
        result = await self._client.query_points(
            name, query=vector, limit=k, with_payload=True,
            query_filter=models.Filter(must=conditions) if conditions else None,
        )
        return [
            ScoredChunk(chunk=_as_chunk(p.payload or {}), score=float(p.score), scores={"dense": float(p.score)}, rank=i)
            for i, p in enumerate(result.points, 1)
        ]

    async def delete(self, doc_ids: list[str], namespace: str = "default") -> None:
        from qdrant_client import models

        name = await self._ensure(namespace)
        await self._client.delete(
            name,
            points_selector=models.FilterSelector(
                filter=models.Filter(must=[models.FieldCondition(key="doc_id", match=models.MatchAny(any=doc_ids))])
            ),
        )

    async def all_chunks(self, namespace: str = "default") -> list[Chunk]:
        name = await self._ensure(namespace)
        chunks: list[Chunk] = []
        offset = None
        while True:
            points, offset = await self._client.scroll(name, limit=512, offset=offset, with_payload=True, with_vectors=False)
            chunks.extend(_as_chunk(p.payload or {}) for p in points)
            if offset is None:
                break
        return chunks

    async def close(self) -> None:
        await self._client.close()


@register("vectorstore", "chroma", requires=["chromadb"])
class ChromaVectorStore(VectorStore):
    """ChromaDB — a popular local-first store, persists to a directory."""

    def __init__(self, path: str = ".nexus_cache/chroma", collection: str = "nexus_lab") -> None:
        import chromadb

        self._client = chromadb.PersistentClient(path=path)
        self._prefix = collection
        self._collections: dict[str, Any] = {}

    def _get(self, namespace: str):
        if namespace not in self._collections:
            self._collections[namespace] = self._client.get_or_create_collection(
                f"{self._prefix}_{namespace}", metadata={"hnsw:space": "cosine"}
            )
        return self._collections[namespace]

    async def upsert(self, chunks: list[Chunk], vectors: list[list[float]], namespace: str = "default") -> None:
        import asyncio

        collection = self._get(namespace)
        metadatas = [
            {k: v for k, v in _payload(c).items() if isinstance(v, (str, int, float, bool)) and k != "text"}
            for c in chunks
        ]
        await asyncio.to_thread(
            collection.upsert, ids=[c.id for c in chunks], embeddings=vectors,
            documents=[c.text for c in chunks], metadatas=metadatas,
        )

    async def search(
        self, vector: list[float], k: int, namespace: str = "default", filters: dict[str, Any] | None = None
    ) -> list[ScoredChunk]:
        import asyncio

        collection = self._get(namespace)
        where = {k: v for k, v in (filters or {}).items() if isinstance(v, (str, int, float, bool))} or None
        result = await asyncio.to_thread(
            collection.query, query_embeddings=[vector], n_results=k, where=where,
            include=["documents", "metadatas", "distances"],
        )
        out: list[ScoredChunk] = []
        for i, (doc_id, text, meta, distance) in enumerate(
            zip(result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0]), 1
        ):
            score = 1.0 - float(distance)
            out.append(ScoredChunk(chunk=_as_chunk({**(meta or {}), "id": doc_id, "text": text}),
                                   score=score, scores={"dense": score}, rank=i))
        return out

    async def delete(self, doc_ids: list[str], namespace: str = "default") -> None:
        import asyncio

        await asyncio.to_thread(self._get(namespace).delete, where={"doc_id": {"$in": doc_ids}})

    async def all_chunks(self, namespace: str = "default") -> list[Chunk]:
        import asyncio

        result = await asyncio.to_thread(self._get(namespace).get, include=["documents", "metadatas"])
        return [
            _as_chunk({**(meta or {}), "id": cid, "text": text})
            for cid, text, meta in zip(result["ids"], result["documents"], result["metadatas"])
        ]


@register("vectorstore", "faiss", requires=["faiss-cpu", "numpy"])
class FaissVectorStore(VectorStore):
    """FAISS flat-IP index — very fast exact search for large local corpora."""

    def __init__(self, persist_path: str | None = None) -> None:
        self._indexes: dict[str, Any] = {}
        self._chunks: dict[str, list[Chunk]] = {}
        self._path = Path(persist_path) if persist_path else None
        self.dim = 0

    async def init(self, dim: int) -> None:
        self.dim = dim

    def _index(self, namespace: str):
        import faiss

        if namespace not in self._indexes:
            self._indexes[namespace] = faiss.IndexFlatIP(self.dim)
            self._chunks[namespace] = []
        return self._indexes[namespace]

    async def upsert(self, chunks: list[Chunk], vectors: list[list[float]], namespace: str = "default") -> None:
        import numpy as np

        index = self._index(namespace)
        matrix = np.asarray(vectors, dtype="float32")
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        index.add(matrix / np.where(norms == 0, 1, norms))
        self._chunks[namespace].extend(chunks)

    async def search(
        self, vector: list[float], k: int, namespace: str = "default", filters: dict[str, Any] | None = None
    ) -> list[ScoredChunk]:
        import numpy as np

        index = self._index(namespace)
        if index.ntotal == 0:
            return []
        query = np.asarray([vector], dtype="float32")
        query /= np.linalg.norm(query) or 1.0
        # Over-fetch so post-filtering still returns k results.
        scores, ids = index.search(query, min(index.ntotal, k * 5 if filters else k))
        out: list[ScoredChunk] = []
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0:
                continue
            chunk = self._chunks[namespace][idx]
            if not _matches(_payload(chunk), filters):
                continue
            out.append(ScoredChunk(chunk=chunk, score=float(score), scores={"dense": float(score)}, rank=len(out) + 1))
            if len(out) >= k:
                break
        return out

    async def delete(self, doc_ids: list[str], namespace: str = "default") -> None:
        # Flat indexes cannot delete in place: rebuild without the removed docs.
        raise NotImplementedError("faiss store: rebuild the index to delete documents, or use 'memory'/'qdrant'.")

    async def all_chunks(self, namespace: str = "default") -> list[Chunk]:
        return list(self._chunks.get(namespace, []))


@register("vectorstore", "pgvector", requires=["asyncpg"])
class PgVectorStore(VectorStore):
    """PostgreSQL + pgvector — one database for both vectors and business data."""

    def __init__(self, dsn: str = "postgresql://nexus@localhost:5432/nexus", table: str = "nexus_lab_chunks") -> None:
        self._dsn = dsn
        self._table = table
        self._pool: Any = None
        self.dim = 0

    async def init(self, dim: int) -> None:
        import asyncpg

        self.dim = dim
        self._pool = await asyncpg.create_pool(self._dsn, min_size=1, max_size=5)
        async with self._pool.acquire() as conn:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            await conn.execute(
                f"""CREATE TABLE IF NOT EXISTS {self._table} (
                        id TEXT, namespace TEXT, doc_id TEXT, filename TEXT, heading TEXT,
                        ord INT, text TEXT, parent_text TEXT, metadata JSONB,
                        embedding vector({dim}), PRIMARY KEY (namespace, id))"""
            )
            await conn.execute(
                f"CREATE INDEX IF NOT EXISTS {self._table}_ns_idx ON {self._table} (namespace, doc_id)"
            )

    @staticmethod
    def _literal(vector: list[float]) -> str:
        return "[" + ",".join(f"{v:.6f}" for v in vector) + "]"

    async def upsert(self, chunks: list[Chunk], vectors: list[list[float]], namespace: str = "default") -> None:
        rows = [
            (c.id, namespace, c.doc_id, c.filename, c.heading, c.ord, c.text, c.parent_text,
             json.dumps(c.metadata), self._literal(v))
            for c, v in zip(chunks, vectors)
        ]
        async with self._pool.acquire() as conn:
            await conn.executemany(
                f"""INSERT INTO {self._table}
                    (id, namespace, doc_id, filename, heading, ord, text, parent_text, metadata, embedding)
                    VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9::jsonb,$10::vector)
                    ON CONFLICT (namespace, id) DO UPDATE SET text = EXCLUDED.text, embedding = EXCLUDED.embedding""",
                rows,
            )

    async def search(
        self, vector: list[float], k: int, namespace: str = "default", filters: dict[str, Any] | None = None
    ) -> list[ScoredChunk]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"""SELECT id, doc_id, filename, heading, ord, text, parent_text, metadata,
                           1 - (embedding <=> $1::vector) AS score
                    FROM {self._table} WHERE namespace = $2
                    ORDER BY embedding <=> $1::vector LIMIT $3""",
                self._literal(vector), namespace, k * 5 if filters else k,
            )
        out: list[ScoredChunk] = []
        for row in rows:
            data = dict(row)
            data["metadata"] = json.loads(data["metadata"] or "{}")
            score = float(data.pop("score"))
            chunk = _as_chunk(data)
            if not _matches(_payload(chunk), filters):
                continue
            out.append(ScoredChunk(chunk=chunk, score=score, scores={"dense": score}, rank=len(out) + 1))
            if len(out) >= k:
                break
        return out

    async def delete(self, doc_ids: list[str], namespace: str = "default") -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(f"DELETE FROM {self._table} WHERE namespace = $1 AND doc_id = ANY($2)", namespace, doc_ids)

    async def all_chunks(self, namespace: str = "default") -> list[Chunk]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"SELECT id, doc_id, filename, heading, ord, text, parent_text, metadata FROM {self._table} WHERE namespace = $1",
                namespace,
            )
        return [_as_chunk({**dict(r), "metadata": json.loads(r["metadata"] or "{}")}) for r in rows]

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
