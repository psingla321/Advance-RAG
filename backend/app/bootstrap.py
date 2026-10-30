import asyncio
from dataclasses import dataclass

from redis.asyncio import Redis

from app.config import Settings
from app.ingestion.pipeline import IngestionPipeline
from app.llm.client import LLMClient, create_llm
from app.services.answer_cache import AnswerCache
from app.services.db import Database
from app.services.embedder import CachedEmbedder
from app.services.insights import InsightsStore
from app.services.jobs import IngestionQueue
from app.services.vector_store import VectorStore


@dataclass
class Core:
    settings: Settings
    llm: LLMClient
    redis: Redis
    db: Database
    vector_store: VectorStore
    embedder: CachedEmbedder
    cache: AnswerCache
    pipeline: IngestionPipeline
    queue: IngestionQueue
    insights: InsightsStore

    async def close(self) -> None:
        await self.llm.close()
        await self.vector_store.close()
        await self.db.close()
        await self.redis.aclose()


async def build_core(settings: Settings) -> Core:
    """Shared wiring for the API and the ingestion worker processes."""
    llm = create_llm(settings)
    redis = Redis.from_url(settings.redis_url, decode_responses=False, health_check_interval=30)
    db = Database(settings.postgres_dsn, settings.db_pool_min, settings.db_pool_max)
    vector_store = VectorStore(settings.qdrant_url, settings.embedding_dim)
    await asyncio.gather(db.connect(), vector_store.init())
    embedder = CachedEmbedder(llm, redis, settings.embedding_model_id, settings.embedding_dim)
    cache = AnswerCache(redis, vector_store, embedder, settings)
    pipeline = IngestionPipeline(settings, db, vector_store, embedder, cache)
    queue = IngestionQueue(redis)
    await queue.ensure_group()
    return Core(settings, llm, redis, db, vector_store, embedder, cache, pipeline, queue, InsightsStore(redis))
