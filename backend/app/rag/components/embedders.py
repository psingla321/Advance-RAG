"""Embedding backends, from a zero-dependency hashing baseline to hosted APIs."""

from __future__ import annotations

import asyncio
import logging
import math
import os
import struct
from typing import Any

from app.rag.base import Embedder
from app.rag.registry import build, register
from app.textutil import hash_embed, sha256_hex

log = logging.getLogger(__name__)


@register("embedder", "hash")
class HashEmbedder(Embedder):
    """Deterministic feature-hashing embedding — no model, no network, no cost.

    Surprisingly decent on keyword-ish corpora and perfect for CI, tests and
    "does my pipeline wire up?" checks.
    """

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim
        self.model_id = f"hash-{dim}"

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [hash_embed(t, self.dim) for t in texts]


@register("embedder", "tfidf")
class TfidfEmbedder(Embedder):
    """Classic TF-IDF projected onto a fixed-size hashed space; fit-free and local."""

    def __init__(self, dim: int = 1024) -> None:
        self.dim = dim
        self.model_id = f"tfidf-{dim}"
        self._df: dict[str, int] = {}
        self._n = 0

    def fit(self, texts: list[str]) -> "TfidfEmbedder":
        from app.textutil import content_terms

        for text in texts:
            self._n += 1
            for term in content_terms(text):
                self._df[term] = self._df.get(term, 0) + 1
        return self

    async def embed(self, texts: list[str]) -> list[list[float]]:
        from app.textutil import content_term_list

        vectors = []
        for text in texts:
            terms = content_term_list(text)
            counts: dict[str, int] = {}
            for term in terms:
                counts[term] = counts.get(term, 0) + 1
            vector = [0.0] * self.dim
            for term, count in counts.items():
                idf = math.log((1 + self._n) / (1 + self._df.get(term, 0))) + 1
                index = int(sha256_hex(term)[:8], 16) % self.dim
                vector[index] += (count / len(terms)) * idf
            norm = math.sqrt(sum(v * v for v in vector)) or 1.0
            vectors.append([v / norm for v in vector])
        return vectors


@register("embedder", "sentence_transformers", requires=["sentence-transformers"])
class SentenceTransformerEmbedder(Embedder):
    """Local open-source embeddings (BGE, E5, GTE, MiniLM…) — free and private.

    Set ``query_prefix``/``doc_prefix`` for asymmetric models, e.g. E5 wants
    ``"query: "`` and ``"passage: "``.
    """

    def __init__(
        self, model: str = "sentence-transformers/all-MiniLM-L6-v2", device: str | None = None,
        normalize: bool = True, batch_size: int = 32, query_prefix: str = "", doc_prefix: str = "",
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model, device=device)
        self.dim = int(self._model.get_sentence_embedding_dimension())
        self.model_id = model
        self._normalize = normalize
        self._batch = batch_size
        self._query_prefix = query_prefix
        self._doc_prefix = doc_prefix

    async def embed(self, texts: list[str]) -> list[list[float]]:
        prefixed = [self._doc_prefix + t for t in texts]
        return await asyncio.to_thread(self._encode, prefixed)

    async def embed_query(self, text: str) -> list[float]:
        return (await asyncio.to_thread(self._encode, [self._query_prefix + text]))[0]

    def _encode(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(
            texts, batch_size=self._batch, normalize_embeddings=self._normalize, show_progress_bar=False
        )
        return [list(map(float, v)) for v in vectors]


@register("embedder", "openai", requires=["openai"])
class OpenAIEmbedder(Embedder):
    """OpenAI embeddings (or any OpenAI-compatible embedding endpoint)."""

    def __init__(
        self, model: str = "text-embedding-3-small", dim: int = 1536, api_key: str | None = None,
        base_url: str | None = None, batch_size: int = 128,
    ) -> None:
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY", "not-set"),
            base_url=base_url or os.getenv("OPENAI_BASE_URL") or None,
        )
        self.dim = dim
        self.model_id = model
        self._batch = batch_size

    async def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for start in range(0, len(texts), self._batch):
            batch = texts[start : start + self._batch]
            response = await self._client.embeddings.create(model=self.model_id, input=batch, dimensions=self.dim)
            out.extend(item.embedding for item in response.data)
        return out

    async def close(self) -> None:
        await self._client.close()


@register("embedder", "azure_openai", requires=["openai", "azure-identity"])
class AzureOpenAIEmbedder(Embedder):
    """Azure OpenAI embeddings, key or Entra ID auth (matches the production app)."""

    def __init__(
        self, deployment: str | None = None, dim: int = 1024, endpoint: str | None = None,
        api_key: str | None = None, api_version: str = "2024-10-21", batch_size: int = 64,
    ) -> None:
        from openai import AsyncAzureOpenAI

        endpoint = endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        if not endpoint:
            raise ValueError("azure_openai embedder needs 'endpoint' or AZURE_OPENAI_ENDPOINT.")
        self.model_id = deployment or os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-large")
        self.dim = dim
        self._batch = batch_size
        kwargs: dict[str, Any] = {"azure_endpoint": endpoint, "api_version": api_version}
        self._credential = None
        key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
        if key:
            kwargs["api_key"] = key
        else:
            from azure.identity.aio import DefaultAzureCredential, get_bearer_token_provider

            self._credential = DefaultAzureCredential()
            kwargs["azure_ad_token_provider"] = get_bearer_token_provider(
                self._credential, "https://cognitiveservices.azure.com/.default"
            )
        self._client = AsyncAzureOpenAI(**kwargs)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for start in range(0, len(texts), self._batch):
            response = await self._client.embeddings.create(
                model=self.model_id, input=texts[start : start + self._batch], dimensions=self.dim
            )
            out.extend(item.embedding for item in response.data)
        return out

    async def close(self) -> None:
        await self._client.close()
        if self._credential is not None:
            await self._credential.close()


@register("embedder", "ollama", requires=["httpx"])
class OllamaEmbedder(Embedder):
    """Local embeddings served by Ollama (nomic-embed-text, mxbai-embed-large…)."""

    def __init__(self, model: str = "nomic-embed-text", base_url: str = "http://localhost:11434", dim: int = 768) -> None:
        import httpx

        self._client = httpx.AsyncClient(base_url=base_url.rstrip("/"), timeout=120.0)
        self.model_id = model
        self.dim = dim

    async def embed(self, texts: list[str]) -> list[list[float]]:
        response = await self._client.post("/api/embed", json={"model": self.model_id, "input": texts})
        response.raise_for_status()
        vectors = response.json()["embeddings"]
        self.dim = len(vectors[0]) if vectors else self.dim
        return vectors

    async def close(self) -> None:
        await self._client.aclose()


@register("embedder", "cohere", requires=["cohere"])
class CohereEmbedder(Embedder):
    """Cohere embed-v3 with proper query/document input types."""

    def __init__(self, model: str = "embed-english-v3.0", api_key: str | None = None, dim: int = 1024) -> None:
        import cohere

        self._client = cohere.AsyncClient(api_key or os.getenv("COHERE_API_KEY"))
        self.model_id = model
        self.dim = dim

    async def _call(self, texts: list[str], input_type: str) -> list[list[float]]:
        response = await self._client.embed(texts=texts, model=self.model_id, input_type=input_type)
        return [list(map(float, v)) for v in response.embeddings]

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return await self._call(texts, "search_document")

    async def embed_query(self, text: str) -> list[float]:
        return (await self._call([text], "search_query"))[0]


@register("embedder", "cached")
class CachedEmbedderWrapper(Embedder):
    """Content-addressed cache in front of any embedder (Redis or local files).

    Re-indexing the same corpus after a chunker change then costs almost nothing.
    """

    def __init__(
        self, inner: dict[str, Any] | str, backend: str = "file", path: str = ".advance_rag_cache/emb",
        redis_url: str = "redis://localhost:6379/0", ttl_seconds: int = 30 * 86400,
    ) -> None:
        from pathlib import Path

        self._inner: Embedder = build("embedder", inner)
        self.dim = self._inner.dim
        self.model_id = f"cached:{self._inner.model_id}"
        self._backend = backend
        self._ttl = ttl_seconds
        self._dir = Path(path)
        self._redis = None
        if backend == "redis":
            from redis.asyncio import Redis

            self._redis = Redis.from_url(redis_url, decode_responses=False)
        else:
            self._dir.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0

    def _key(self, text: str) -> str:
        return f"{self._inner.model_id}:{self.dim}:{sha256_hex(text)}"

    async def _get(self, key: str) -> list[float] | None:
        fmt = f"{self.dim}f"
        if self._redis is not None:
            raw = await self._redis.get(f"emb:{key}")
        else:
            path = self._dir / f"{sha256_hex(key)}.vec"
            raw = path.read_bytes() if path.exists() else None
        if raw is None or len(raw) != 4 * self.dim:
            return None
        return list(struct.unpack(fmt, raw))

    async def _put(self, key: str, vector: list[float]) -> None:
        raw = struct.pack(f"{self.dim}f", *vector)
        if self._redis is not None:
            await self._redis.set(f"emb:{key}", raw, ex=self._ttl)
        else:
            (self._dir / f"{sha256_hex(key)}.vec").write_bytes(raw)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        keys = [self._key(t) for t in texts]
        results: list[list[float] | None] = list(await asyncio.gather(*(self._get(k) for k in keys)))
        missing = [i for i, v in enumerate(results) if v is None]
        self.hits += len(texts) - len(missing)
        self.misses += len(missing)
        if missing:
            fresh = await self._inner.embed([texts[i] for i in missing])
            for i, vector in zip(missing, fresh):
                results[i] = vector
            await asyncio.gather(*(self._put(keys[i], results[i]) for i in missing))  # type: ignore[arg-type]
        return [v for v in results if v is not None]

    def describe(self) -> dict[str, Any]:
        return {"type": "cached", "kind": "embedder", "inner": self._inner.describe(),
                "hits": self.hits, "misses": self.misses}

    async def close(self) -> None:
        await self._inner.close()
        if self._redis is not None:
            await self._redis.aclose()


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)
