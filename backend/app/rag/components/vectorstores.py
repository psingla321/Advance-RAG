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
    """Qdrant — the production store used by the AdvanceRAG engine.

    ``url=":memory:"`` runs the embedded local mode, so tests need no container.
    """

    def __init__(self, url: str = "http://localhost:6333", collection: str = "advance_rag_lab", api_key: str | None = None) -> None:
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

    def __init__(self, path: str = ".advance_rag_cache/chroma", collection: str = "advance_rag_lab") -> None:
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

    def __init__(self, dsn: str = "postgresql://advance_rag@localhost:5432/advance_rag", table: str = "advance_rag_lab_chunks") -> None:
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


# --------------------------------------------------------------------------
# Managed and server-backed stores.
#
# Each takes an optional ``client`` so the mapping logic can be exercised with a
# fake in tests; left as None the real SDK is imported lazily on construction.
# Namespaces map onto one index/collection per namespace, as elsewhere here.
# --------------------------------------------------------------------------


def _flat_metadata(chunk: Chunk) -> dict[str, Any]:
    """Payload for stores that only accept scalar (or scalar-list) metadata values."""
    payload = _payload(chunk)
    payload["metadata"] = json.dumps(chunk.metadata or {})
    return {
        k: v for k, v in payload.items()
        if isinstance(v, (str, int, float, bool)) or (isinstance(v, list) and all(isinstance(i, str) for i in v))
    }


def _unflatten(data: dict[str, Any]) -> Chunk:
    """Inverse of :func:`_flat_metadata` — restores the nested metadata dict."""
    restored = dict(data)
    raw = restored.get("metadata")
    if isinstance(raw, str):
        try:
            restored["metadata"] = json.loads(raw)
        except json.JSONDecodeError:
            restored["metadata"] = {}
    return _as_chunk(restored)


@register("vectorstore", "azure_ai_search", requires=["azure-search-documents"])
class AzureAISearchVectorStore(VectorStore):
    """Azure AI Search — the managed option when the rest of the stack is on Azure.

    Creates one index per namespace with an HNSW vector profile. Authenticates with
    an API key, or with Entra ID (managed identity) when ``api_key`` is omitted.
    """

    def __init__(
        self, endpoint: str | None = None, index: str = "advance-rag-lab", api_key: str | None = None, client: Any = None,
    ) -> None:
        self._endpoint = endpoint
        self._prefix = index
        self._api_key = api_key
        self._clients: dict[str, Any] = {}
        self._injected = client
        self.dim = 0

    def _name(self, namespace: str) -> str:
        return f"{self._prefix}-{namespace}".lower()

    async def init(self, dim: int) -> None:
        self.dim = dim

    def _client(self, namespace: str) -> Any:
        if self._injected is not None:
            return self._injected
        if namespace not in self._clients:
            from azure.search.documents.aio import SearchClient

            self._clients[namespace] = SearchClient(self._endpoint, self._name(namespace), self._credential())
        return self._clients[namespace]

    def _credential(self) -> Any:
        if self._api_key:
            from azure.core.credentials import AzureKeyCredential

            return AzureKeyCredential(self._api_key)
        from azure.identity.aio import DefaultAzureCredential

        return DefaultAzureCredential()

    async def upsert(self, chunks: list[Chunk], vectors: list[list[float]], namespace: str = "default") -> None:
        documents = [
            {**_flat_metadata(c), "id": c.id, "vector": v, "acl": c.metadata.get("acl", [])}
            for c, v in zip(chunks, vectors)
        ]
        await self._client(namespace).merge_or_upload_documents(documents)

    async def search(
        self, vector: list[float], k: int, namespace: str = "default", filters: dict[str, Any] | None = None
    ) -> list[ScoredChunk]:
        from azure.search.documents.models import VectorizedQuery

        query = VectorizedQuery(vector=vector, k_nearest_neighbors=k, fields="vector")
        results = await self._client(namespace).search(
            search_text=None, vector_queries=[query], filter=_odata_filter(filters), top=k,
        )
        out: list[ScoredChunk] = []
        async for rank, document in _aenumerate(results):
            out.append(ScoredChunk(_unflatten(dict(document)), float(document.get("@search.score", 0.0)), rank=rank))
        return out

    async def delete(self, doc_ids: list[str], namespace: str = "default") -> None:
        await self._client(namespace).delete_documents([{"id": doc_id} for doc_id in doc_ids])

    async def close(self) -> None:
        for client in self._clients.values():
            await client.close()


def _odata_filter(filters: dict[str, Any] | None) -> str | None:
    """Translate the generic filter dict into an OData clause."""
    if not filters:
        return None
    clauses: list[str] = []
    for key, wanted in filters.items():
        values = list(wanted) if isinstance(wanted, (list, tuple, set)) else [wanted]
        escaped = [str(v).replace("'", "''") for v in values]
        if key == "acl":  # collection field: match when any group overlaps
            clauses.append("(" + " or ".join(f"acl/any(g: g eq '{v}')" for v in escaped) + ")")
        else:
            clauses.append("(" + " or ".join(f"{key} eq '{v}'" for v in escaped) + ")")
    return " and ".join(clauses)


async def _aenumerate(iterable: Any, start: int = 1) -> Any:
    index = start
    async for item in iterable:
        yield index, item
        index += 1


@register("vectorstore", "pinecone", requires=["pinecone"])
class PineconeVectorStore(VectorStore):
    """Pinecone — fully managed, serverless. One Pinecone namespace per lab namespace."""

    def __init__(
        self, index: str = "advance-rag-lab", api_key: str | None = None, cloud: str = "aws", region: str = "us-east-1",
        client: Any = None,
    ) -> None:
        self._index_name = index
        self._cloud = cloud
        self._region = region
        self._index = client
        self._api_key = api_key
        self.dim = 0

    async def init(self, dim: int) -> None:
        import asyncio

        self.dim = dim
        if self._index is not None:
            return
        from pinecone import Pinecone, ServerlessSpec

        pinecone = Pinecone(api_key=self._api_key)
        existing = {i["name"] for i in await asyncio.to_thread(pinecone.list_indexes)}
        if self._index_name not in existing:
            await asyncio.to_thread(
                pinecone.create_index, name=self._index_name, dimension=dim, metric="cosine",
                spec=ServerlessSpec(cloud=self._cloud, region=self._region),
            )
        self._index = pinecone.Index(self._index_name)

    async def upsert(self, chunks: list[Chunk], vectors: list[list[float]], namespace: str = "default") -> None:
        import asyncio

        payload = [
            {"id": c.id, "values": v, "metadata": _flat_metadata(c)} for c, v in zip(chunks, vectors)
        ]
        await asyncio.to_thread(self._index.upsert, vectors=payload, namespace=namespace)

    async def search(
        self, vector: list[float], k: int, namespace: str = "default", filters: dict[str, Any] | None = None
    ) -> list[ScoredChunk]:
        import asyncio

        response = await asyncio.to_thread(
            self._index.query, vector=vector, top_k=k, namespace=namespace,
            include_metadata=True, filter=_pinecone_filter(filters),
        )
        matches = response["matches"] if isinstance(response, dict) else response.matches
        out: list[ScoredChunk] = []
        for rank, match in enumerate(matches, 1):
            data = dict(match["metadata"] if isinstance(match, dict) else match.metadata or {})
            data["id"] = match["id"] if isinstance(match, dict) else match.id
            score = match["score"] if isinstance(match, dict) else match.score
            out.append(ScoredChunk(_unflatten(data), float(score), rank=rank))
        return out

    async def delete(self, doc_ids: list[str], namespace: str = "default") -> None:
        import asyncio

        await asyncio.to_thread(self._index.delete, filter={"doc_id": {"$in": doc_ids}}, namespace=namespace)


def _pinecone_filter(filters: dict[str, Any] | None) -> dict[str, Any] | None:
    """Generic filters -> Pinecone's Mongo-like filter language."""
    if not filters:
        return None
    clause: dict[str, Any] = {}
    for key, wanted in filters.items():
        clause[key] = {"$in": list(wanted)} if isinstance(wanted, (list, tuple, set)) else {"$eq": wanted}
    return clause


@register("vectorstore", "elasticsearch", requires=["elasticsearch"])
class ElasticsearchVectorStore(VectorStore):
    """Elasticsearch 8 kNN — the option when the organisation already runs Elastic.

    Set ``opensearch=True`` to target OpenSearch, whose kNN query shape differs.
    """

    def __init__(
        self, url: str = "http://localhost:9200", index: str = "advance-rag-lab", api_key: str | None = None,
        opensearch: bool = False, client: Any = None,
    ) -> None:
        self._prefix = index
        self._opensearch = opensearch
        self._client = client
        self._url = url
        self._api_key = api_key
        self.dim = 0

    def _name(self, namespace: str) -> str:
        return f"{self._prefix}-{namespace}".lower()

    async def init(self, dim: int) -> None:
        self.dim = dim
        if self._client is None:
            from elasticsearch import AsyncElasticsearch

            self._client = AsyncElasticsearch(self._url, api_key=self._api_key)

    async def _ensure(self, namespace: str) -> str:
        name = self._name(namespace)
        if not await self._client.indices.exists(index=name):
            vector_field = (
                {"type": "knn_vector", "dimension": self.dim} if self._opensearch
                else {"type": "dense_vector", "dims": self.dim, "index": True, "similarity": "cosine"}
            )
            await self._client.indices.create(
                index=name,
                body={"mappings": {"properties": {
                    "vector": vector_field, "text": {"type": "text"}, "doc_id": {"type": "keyword"},
                    "filename": {"type": "keyword"}, "heading": {"type": "text"}, "acl": {"type": "keyword"},
                }}},
            )
        return name

    async def upsert(self, chunks: list[Chunk], vectors: list[list[float]], namespace: str = "default") -> None:
        name = await self._ensure(namespace)
        operations: list[dict[str, Any]] = []
        for chunk, vector in zip(chunks, vectors):
            operations.append({"index": {"_index": name, "_id": chunk.id}})
            operations.append({**_flat_metadata(chunk), "vector": vector, "acl": chunk.metadata.get("acl", [])})
        await self._client.bulk(operations=operations, refresh=True)

    async def search(
        self, vector: list[float], k: int, namespace: str = "default", filters: dict[str, Any] | None = None
    ) -> list[ScoredChunk]:
        name = await self._ensure(namespace)
        terms = [
            {"terms": {key: [str(v) for v in wanted]}} if isinstance(wanted, (list, tuple, set))
            else {"term": {key: wanted}}
            for key, wanted in (filters or {}).items()
        ]
        if self._opensearch:
            query: dict[str, Any] = {"knn": {"vector": {"vector": vector, "k": k}}}
            body = {"query": {"bool": {"must": [query], "filter": terms}} if terms else query}
            response = await self._client.search(index=name, size=k, body=body)
        else:
            knn = {"field": "vector", "query_vector": vector, "k": k, "num_candidates": max(k * 4, 50)}
            if terms:
                knn["filter"] = terms
            response = await self._client.search(index=name, knn=knn, size=k)
        hits = response["hits"]["hits"]
        return [
            ScoredChunk(_unflatten(hit["_source"]), float(hit.get("_score", 0.0)), rank=rank)
            for rank, hit in enumerate(hits, 1)
        ]

    async def delete(self, doc_ids: list[str], namespace: str = "default") -> None:
        await self._client.delete_by_query(
            index=self._name(namespace), body={"query": {"terms": {"doc_id": doc_ids}}}, refresh=True,
        )

    async def count(self, namespace: str = "default") -> int:
        response = await self._client.count(index=self._name(namespace))
        return int(response["count"])

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()


@register("vectorstore", "weaviate", requires=["weaviate-client"])
class WeaviateVectorStore(VectorStore):
    """Weaviate (v4 client) — OSS vector database with a managed cloud option."""

    def __init__(
        self, url: str = "http://localhost:8080", collection: str = "AdvanceRagLab", api_key: str | None = None,
        client: Any = None,
    ) -> None:
        self._prefix = collection
        self._url = url
        self._api_key = api_key
        self._client = client
        self.dim = 0

    def _name(self, namespace: str) -> str:
        # Weaviate class names must be TitleCase and alphanumeric.
        cleaned = "".join(part.title() for part in str(namespace).replace("-", "_").split("_") if part)
        return f"{self._prefix}{cleaned or 'Default'}"

    async def init(self, dim: int) -> None:
        self.dim = dim
        if self._client is None:
            import weaviate

            self._client = weaviate.use_async_with_local(host=self._url) if not self._api_key else (
                weaviate.use_async_with_weaviate_cloud(cluster_url=self._url, auth_credentials=self._api_key)
            )
            await self._client.connect()

    async def _collection(self, namespace: str) -> Any:
        from weaviate.classes.config import Configure, DataType, Property

        name = self._name(namespace)
        if not await self._client.collections.exists(name):
            await self._client.collections.create(
                name,
                vectorizer_config=Configure.Vectorizer.none(),
                properties=[
                    Property(name="text", data_type=DataType.TEXT),
                    Property(name="doc_id", data_type=DataType.TEXT),
                    Property(name="filename", data_type=DataType.TEXT),
                    Property(name="heading", data_type=DataType.TEXT),
                    Property(name="metadata", data_type=DataType.TEXT),
                ],
            )
        return self._client.collections.get(name)

    async def upsert(self, chunks: list[Chunk], vectors: list[list[float]], namespace: str = "default") -> None:
        from weaviate.classes.data import DataObject
        from weaviate.util import generate_uuid5

        collection = await self._collection(namespace)
        await collection.data.insert_many([
            DataObject(properties={**_flat_metadata(c), "chunk_id": c.id}, vector=v, uuid=generate_uuid5(c.id))
            for c, v in zip(chunks, vectors)
        ])

    async def search(
        self, vector: list[float], k: int, namespace: str = "default", filters: dict[str, Any] | None = None
    ) -> list[ScoredChunk]:
        from weaviate.classes.query import Filter, MetadataQuery

        collection = await self._collection(namespace)
        where = None
        for key, wanted in (filters or {}).items():
            values = list(wanted) if isinstance(wanted, (list, tuple, set)) else [wanted]
            clause = Filter.any_of([Filter.by_property(key).equal(v) for v in values])
            where = clause if where is None else where & clause
        response = await collection.query.near_vector(
            near_vector=vector, limit=k, filters=where, return_metadata=MetadataQuery(distance=True),
        )
        out: list[ScoredChunk] = []
        for rank, item in enumerate(response.objects, 1):
            data = dict(item.properties)
            data["id"] = data.pop("chunk_id", str(item.uuid))
            distance = getattr(item.metadata, "distance", None)
            out.append(ScoredChunk(_unflatten(data), 1.0 - float(distance or 0.0), rank=rank))
        return out

    async def delete(self, doc_ids: list[str], namespace: str = "default") -> None:
        from weaviate.classes.query import Filter

        collection = await self._collection(namespace)
        await collection.data.delete_many(where=Filter.by_property("doc_id").contains_any(doc_ids))

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()


@register("vectorstore", "milvus", requires=["pymilvus"])
class MilvusVectorStore(VectorStore):
    """Milvus / Zilliz — built for very large collections."""

    def __init__(
        self, uri: str = "http://localhost:19530", collection: str = "advance_rag_lab", token: str | None = None,
        client: Any = None,
    ) -> None:
        self._prefix = collection
        self._uri = uri
        self._token = token
        self._client = client
        self.dim = 0

    def _name(self, namespace: str) -> str:
        return f"{self._prefix}_{str(namespace).replace('-', '_')}"

    async def init(self, dim: int) -> None:
        self.dim = dim
        if self._client is None:
            from pymilvus import MilvusClient

            self._client = MilvusClient(uri=self._uri, token=self._token)

    async def _ensure(self, namespace: str) -> str:
        import asyncio

        name = self._name(namespace)
        if not await asyncio.to_thread(self._client.has_collection, name):
            await asyncio.to_thread(
                self._client.create_collection, collection_name=name, dimension=self.dim,
                metric_type="COSINE", auto_id=False, id_type="string", max_length=512, enable_dynamic_field=True,
            )
        return name

    async def upsert(self, chunks: list[Chunk], vectors: list[list[float]], namespace: str = "default") -> None:
        import asyncio

        name = await self._ensure(namespace)
        rows = [{**_flat_metadata(c), "id": c.id, "vector": v} for c, v in zip(chunks, vectors)]
        await asyncio.to_thread(self._client.upsert, collection_name=name, data=rows)

    async def search(
        self, vector: list[float], k: int, namespace: str = "default", filters: dict[str, Any] | None = None
    ) -> list[ScoredChunk]:
        import asyncio

        name = await self._ensure(namespace)
        results = await asyncio.to_thread(
            self._client.search, collection_name=name, data=[vector], limit=k,
            filter=_milvus_filter(filters), output_fields=["*"],
        )
        hits = results[0] if results else []
        out: list[ScoredChunk] = []
        for rank, hit in enumerate(hits, 1):
            data = dict(hit.get("entity", {}))
            data.setdefault("id", hit.get("id"))
            out.append(ScoredChunk(_unflatten(data), float(hit.get("distance", 0.0)), rank=rank))
        return out

    async def delete(self, doc_ids: list[str], namespace: str = "default") -> None:
        import asyncio

        quoted = ", ".join(f'"{d}"' for d in doc_ids)
        await asyncio.to_thread(
            self._client.delete, collection_name=self._name(namespace), filter=f"doc_id in [{quoted}]",
        )


def _milvus_filter(filters: dict[str, Any] | None) -> str:
    """Generic filters -> a Milvus boolean expression."""
    if not filters:
        return ""
    clauses: list[str] = []
    for key, wanted in filters.items():
        values = list(wanted) if isinstance(wanted, (list, tuple, set)) else [wanted]
        quoted = ", ".join(f'"{v}"' for v in values)
        clauses.append(f"{key} in [{quoted}]")
    return " and ".join(clauses)


@register("vectorstore", "mongodb", requires=["pymongo"])
class MongoVectorStore(VectorStore):
    """MongoDB Atlas Vector Search — keeps vectors beside the documents they came from.

    The Atlas vector index must exist; Atlas creates it asynchronously and the
    driver cannot create it on a shared tier, so ``index_name`` is assumed present.
    """

    def __init__(
        self, uri: str = "mongodb://localhost:27017", database: str = "advance_rag", collection: str = "chunks",
        index_name: str = "vector_index", client: Any = None,
    ) -> None:
        self._uri = uri
        self._database = database
        self._prefix = collection
        self._index_name = index_name
        self._client = client
        self.dim = 0

    async def init(self, dim: int) -> None:
        self.dim = dim
        if self._client is None:
            from pymongo import AsyncMongoClient

            self._client = AsyncMongoClient(self._uri)

    def _collection(self, namespace: str) -> Any:
        return self._client[self._database][f"{self._prefix}_{namespace}"]

    async def upsert(self, chunks: list[Chunk], vectors: list[list[float]], namespace: str = "default") -> None:
        from pymongo import ReplaceOne

        collection = self._collection(namespace)
        await collection.bulk_write([
            ReplaceOne({"_id": c.id}, {**_flat_metadata(c), "_id": c.id, "vector": v}, upsert=True)
            for c, v in zip(chunks, vectors)
        ])

    async def search(
        self, vector: list[float], k: int, namespace: str = "default", filters: dict[str, Any] | None = None
    ) -> list[ScoredChunk]:
        stage: dict[str, Any] = {
            "index": self._index_name, "path": "vector", "queryVector": vector,
            "numCandidates": max(k * 10, 100), "limit": k,
        }
        if filters:
            stage["filter"] = _pinecone_filter(filters)  # Atlas uses the same $in/$eq shape
        pipeline = [{"$vectorSearch": stage}, {"$set": {"score": {"$meta": "vectorSearchScore"}}}]
        out: list[ScoredChunk] = []
        rank = 0
        async for document in await self._collection(namespace).aggregate(pipeline):
            rank += 1
            data = dict(document)
            data["id"] = data.pop("_id")
            out.append(ScoredChunk(_unflatten(data), float(data.pop("score", 0.0)), rank=rank))
        return out

    async def delete(self, doc_ids: list[str], namespace: str = "default") -> None:
        await self._collection(namespace).delete_many({"doc_id": {"$in": doc_ids}})

    async def count(self, namespace: str = "default") -> int:
        return int(await self._collection(namespace).count_documents({}))

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()


@register("vectorstore", "lancedb", requires=["lancedb"])
class LanceDBVectorStore(VectorStore):
    """LanceDB — embedded and columnar, like FAISS but with metadata filtering built in."""

    def __init__(self, path: str = ".advance_rag_cache/lancedb", table: str = "advance_rag_lab", client: Any = None) -> None:
        self._path = path
        self._prefix = table
        self._db = client
        self.dim = 0

    def _name(self, namespace: str) -> str:
        return f"{self._prefix}_{namespace}"

    async def init(self, dim: int) -> None:
        import asyncio

        self.dim = dim
        if self._db is None:
            import lancedb

            self._db = await asyncio.to_thread(lancedb.connect, self._path)

    async def upsert(self, chunks: list[Chunk], vectors: list[list[float]], namespace: str = "default") -> None:
        import asyncio

        name = self._name(namespace)
        rows = [{**_flat_metadata(c), "id": c.id, "vector": v} for c, v in zip(chunks, vectors)]
        names = await asyncio.to_thread(self._db.table_names)
        if name in names:
            table = await asyncio.to_thread(self._db.open_table, name)
            await asyncio.to_thread(table.delete, f"id IN ({', '.join(repr(c.id) for c in chunks)})")
            await asyncio.to_thread(table.add, rows)
        else:
            await asyncio.to_thread(self._db.create_table, name, rows)

    async def search(
        self, vector: list[float], k: int, namespace: str = "default", filters: dict[str, Any] | None = None
    ) -> list[ScoredChunk]:
        import asyncio

        def run() -> list[dict[str, Any]]:
            table = self._db.open_table(self._name(namespace))
            query = table.search(vector).limit(k)
            where = _milvus_filter(filters).replace(" and ", " AND ")
            return query.where(where).to_list() if where else query.to_list()

        rows = await asyncio.to_thread(run)
        out: list[ScoredChunk] = []
        for rank, row in enumerate(rows, 1):
            data = {k: v for k, v in row.items() if k != "vector"}
            # LanceDB reports L2 distance; smaller is better.
            out.append(ScoredChunk(_unflatten(data), 1.0 / (1.0 + float(row.get("_distance", 0.0))), rank=rank))
        return out

    async def delete(self, doc_ids: list[str], namespace: str = "default") -> None:
        import asyncio

        def run() -> None:
            table = self._db.open_table(self._name(namespace))
            table.delete(f"doc_id IN ({', '.join(repr(d) for d in doc_ids)})")

        await asyncio.to_thread(run)
