"""Ingestion worker: `python -m app.worker`. Run N replicas; Redis consumer groups distribute the jobs."""

import asyncio
import json
import logging
import os
import signal
import socket
from typing import Any

from prometheus_client import start_http_server

from app.bootstrap import Core, build_core
from app.config import get_settings
from app.observability import QUEUE_DEPTH, setup_logging, setup_tracing, shutdown_tracing
from app.services.jobs import GROUP, STREAM

log = logging.getLogger("nexus.worker")

RECLAIM_IDLE_MS = 5 * 60 * 1000
RECLAIM_EVERY_SECONDS = 30


def _decode(fields: dict[Any, Any]) -> dict[str, str]:
    return {(k.decode() if isinstance(k, bytes) else k): (v.decode() if isinstance(v, bytes) else v) for k, v in fields.items()}


class IngestionWorker:
    def __init__(self, core: Core, consumer: str, concurrency: int) -> None:
        self._core = core
        self._consumer = consumer
        self._semaphore = asyncio.Semaphore(concurrency)
        self._concurrency = concurrency
        self._tasks: set[asyncio.Task] = set()

    async def run(self, stop: asyncio.Event) -> None:
        redis = self._core.redis
        await self._core.queue.ensure_group()
        log.info("Worker %s consuming %s (concurrency %s)", self._consumer, STREAM, self._concurrency)
        last_reclaim = 0.0
        loop = asyncio.get_running_loop()
        while not stop.is_set():
            try:
                if loop.time() - last_reclaim > RECLAIM_EVERY_SECONDS:
                    last_reclaim = loop.time()
                    # Recover jobs from crashed consumers (at-least-once delivery).
                    _, claimed, *_ = await redis.xautoclaim(STREAM, GROUP, self._consumer, RECLAIM_IDLE_MS, "0-0", count=10)
                    for message_id, fields in claimed:
                        await self._dispatch(message_id, fields)
                QUEUE_DEPTH.set(await self._core.queue.depth())
                free = max(1, self._concurrency - len(self._tasks))
                response = await redis.xreadgroup(GROUP, self._consumer, {STREAM: ">"}, count=free, block=2000)
                if not response:
                    # Guarantees the loop yields even if the server returns early from a blocking read.
                    await asyncio.sleep(0.05)
                    continue
                for _, messages in response:
                    for message_id, fields in messages:
                        await self._dispatch(message_id, fields)
            except asyncio.CancelledError:
                break
            except Exception:
                log.exception("Worker loop error; backing off")
                await asyncio.sleep(2)
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

    async def _dispatch(self, message_id: Any, fields: dict[Any, Any]) -> None:
        await self._semaphore.acquire()
        task = asyncio.create_task(self._process(message_id, _decode(fields)))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _process(self, message_id: Any, job: dict[str, str]) -> None:
        try:
            doc_id = job["doc_id"]
            data = await self._core.redis.get(self._core.queue.blob_key(doc_id))
            if data is None:
                await self._core.db.set_status(doc_id, "FAILED", error="Upload payload expired before processing.")
            else:
                await self._core.pipeline.ingest(doc_id, job["tenant"], job["filename"], data, json.loads(job["acl"]))
            await self._core.queue.complete(message_id, doc_id)
        except Exception:
            log.exception("Job %s failed; it will be retried after the idle timeout", message_id)
        finally:
            self._semaphore.release()


async def main() -> None:
    settings = get_settings()
    setup_logging(settings)
    setup_tracing(settings, "nexus-worker")
    start_http_server(settings.worker_metrics_port)
    core = await build_core(settings)
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except NotImplementedError:  # Windows event loop
            pass
    consumer = f"{socket.gethostname()}-{os.getpid()}"
    try:
        await IngestionWorker(core, consumer, settings.worker_concurrency).run(stop)
    finally:
        await core.close()
        shutdown_tracing()


if __name__ == "__main__":
    asyncio.run(main())
