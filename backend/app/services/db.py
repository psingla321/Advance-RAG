import asyncio
import logging
import random
from datetime import date, datetime
from decimal import Decimal
from typing import Any
from uuid import UUID, uuid4

import asyncpg

log = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    filename TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    acl TEXT[] NOT NULL,
    status TEXT NOT NULL,
    chunk_count INT NOT NULL DEFAULT 0,
    error TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (tenant_id, content_hash)
);

CREATE TABLE IF NOT EXISTS chunks (
    id UUID PRIMARY KEY,
    doc_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    tenant_id TEXT NOT NULL,
    ord INT NOT NULL,
    heading TEXT NOT NULL DEFAULT '',
    content TEXT NOT NULL,
    acl TEXT[] NOT NULL,
    tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', heading || ' ' || content)) STORED
);
CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING GIN (tsv);
CREATE INDEX IF NOT EXISTS chunks_tenant_idx ON chunks (tenant_id);
CREATE INDEX IF NOT EXISTS chunks_acl_idx ON chunks USING GIN (acl);

CREATE TABLE IF NOT EXISTS sales (
    id SERIAL PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    region TEXT NOT NULL,
    product TEXT NOT NULL,
    year INT NOT NULL,
    quarter TEXT NOT NULL,
    revenue NUMERIC(14, 2) NOT NULL,
    units INT NOT NULL
);

DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'nexus_reader') THEN
        CREATE ROLE nexus_reader NOLOGIN;
    END IF;
END
$$;
GRANT SELECT ON sales TO nexus_reader;
ALTER TABLE sales ENABLE ROW LEVEL SECURITY;
ALTER TABLE sales FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS tenant_isolation ON sales;
CREATE POLICY tenant_isolation ON sales USING (tenant_id = current_setting('app.tenant_id', true));
"""

REGION_BASE = {"North America": 1200, "EMEA": 950, "APAC": 800, "LATAM": 400}
PRODUCT_PRICE = {"Analytics Suite": 249.0, "Cloud Storage": 89.0, "Security Pro": 399.0}
PRODUCT_MIX = {"Analytics Suite": 1.0, "Cloud Storage": 2.2, "Security Pro": 0.6}


def _jsonable(value: Any) -> Any:
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if value is None or isinstance(value, (int, float, str, bool)):
        return value
    return str(value)


SCHEMA_LOCK_ID = 727274


class Database:
    def __init__(self, dsn: str, min_size: int = 1, max_size: int = 10) -> None:
        self._dsn = dsn
        self._min = min_size
        self._max = max_size
        self.pool: asyncpg.Pool | None = None

    async def connect(self, attempts: int = 30) -> None:
        for attempt in range(1, attempts + 1):
            try:
                self.pool = await asyncpg.create_pool(self._dsn, min_size=self._min, max_size=self._max)
                break
            except (OSError, asyncpg.PostgresError) as exc:
                if attempt == attempts:
                    raise
                log.info("Waiting for PostgreSQL (%s/%s): %s", attempt, attempts, exc.__class__.__name__)
                await asyncio.sleep(1)
        async with self.pool.acquire() as conn:
            # Serialize schema migration/seeding across replicas starting at the same time.
            await conn.execute("SELECT pg_advisory_lock($1)", SCHEMA_LOCK_ID)
            try:
                await conn.execute(SCHEMA)
                await self._seed_sales(conn)
            finally:
                await conn.execute("SELECT pg_advisory_unlock($1)", SCHEMA_LOCK_ID)

    async def close(self) -> None:
        if self.pool:
            await self.pool.close()

    async def ping(self) -> bool:
        return await self.pool.fetchval("SELECT 1") == 1

    async def _seed_sales(self, conn: asyncpg.Connection) -> None:
        if await conn.fetchval("SELECT EXISTS (SELECT 1 FROM sales WHERE tenant_id = 'demo')"):
            return
        rng = random.Random(42)
        rows = []
        for year in (2024, 2025):
            growth = 1.12 if year == 2025 else 1.0
            for q_index, quarter in enumerate(("Q1", "Q2", "Q3", "Q4")):
                seasonality = 1.0 + 0.06 * q_index
                for region, base in REGION_BASE.items():
                    for product, mix in PRODUCT_MIX.items():
                        units = int(base * mix * growth * seasonality * rng.uniform(0.85, 1.2))
                        revenue = Decimal(str(round(units * PRODUCT_PRICE[product], 2)))
                        rows.append(("demo", region, product, year, quarter, revenue, units))
        await conn.copy_records_to_table(
            "sales",
            records=rows,
            columns=["tenant_id", "region", "product", "year", "quarter", "revenue", "units"],
        )
        log.info("Seeded %s demo sales rows", len(rows))

    async def create_document(self, tenant: str, filename: str, content_hash: str, acl: list[str]) -> tuple[str, bool]:
        row = await self.pool.fetchrow(
            """INSERT INTO documents (id, tenant_id, filename, content_hash, acl, status)
               VALUES ($1, $2, $3, $4, $5, 'RECEIVED')
               ON CONFLICT (tenant_id, content_hash) DO NOTHING RETURNING id""",
            uuid4(), tenant, filename, content_hash, acl,
        )
        if row:
            return str(row["id"]), True
        existing = await self.pool.fetchval(
            "SELECT id FROM documents WHERE tenant_id = $1 AND content_hash = $2", tenant, content_hash
        )
        return str(existing), False

    async def set_status(self, doc_id: str, status: str, *, chunk_count: int | None = None, error: str | None = None) -> None:
        await self.pool.execute(
            """UPDATE documents SET status = $2, chunk_count = COALESCE($3, chunk_count), error = $4
               WHERE id = $1""",
            UUID(doc_id), status, chunk_count, error,
        )

    async def insert_chunks(self, doc_id: str, tenant: str, acl: list[str], chunks: list[tuple[str, int, str, str]]) -> None:
        await self.pool.executemany(
            "INSERT INTO chunks (id, doc_id, tenant_id, ord, heading, content, acl) VALUES ($1, $2, $3, $4, $5, $6, $7)",
            [(UUID(cid), UUID(doc_id), tenant, ord_, heading, content, acl) for cid, ord_, heading, content in chunks],
        )

    async def delete_chunks(self, doc_id: str) -> None:
        await self.pool.execute("DELETE FROM chunks WHERE doc_id = $1", UUID(doc_id))

    async def list_documents(self, tenant: str, groups: list[str]) -> list[dict[str, Any]]:
        rows = await self.pool.fetch(
            """SELECT id, filename, status, chunk_count, acl, error, created_at FROM documents
               WHERE tenant_id = $1 AND acl && $2::text[] ORDER BY created_at DESC LIMIT 500""",
            tenant, groups,
        )
        return [{k: _jsonable(v) for k, v in dict(r).items()} | {"id": str(r["id"]), "acl": list(r["acl"])} for r in rows]

    async def delete_document(self, tenant: str, groups: list[str], doc_id: str) -> bool:
        result = await self.pool.execute(
            "DELETE FROM documents WHERE id = $1 AND tenant_id = $2 AND acl && $3::text[]", UUID(doc_id), tenant, groups
        )
        return result.endswith(" 1")

    async def count_documents(self, tenant: str) -> int:
        return await self.pool.fetchval("SELECT count(*) FROM documents WHERE tenant_id = $1", tenant)

    async def chunk_count(self, tenant: str, groups: list[str]) -> int:
        return await self.pool.fetchval(
            "SELECT count(*) FROM chunks WHERE tenant_id = $1 AND acl && $2::text[]", tenant, groups
        )

    async def keyword_search(self, terms: list[str], tenant: str, groups: list[str], k: int) -> list[dict[str, Any]]:
        if not terms:
            return []
        # Terms are restricted to [a-z0-9]+ by the tokenizer, so the OR-query is syntactically safe.
        rows = await self.pool.fetch(
            """SELECT c.id, c.doc_id, c.heading, c.content, d.filename, ts_rank_cd(c.tsv, q) AS rank
               FROM chunks c JOIN documents d ON d.id = c.doc_id, to_tsquery('english', $1) q
               WHERE c.tenant_id = $2 AND c.acl && $3::text[] AND d.status = 'INDEXED' AND c.tsv @@ q
               ORDER BY rank DESC LIMIT $4""",
            " | ".join(terms), tenant, groups, k,
        )
        return [
            {
                "chunk_id": str(r["id"]),
                "doc_id": str(r["doc_id"]),
                "heading": r["heading"],
                "content": r["content"],
                "filename": r["filename"],
                "rank": float(r["rank"]),
            }
            for r in rows
        ]

    async def run_readonly_sql(self, tenant: str, sql: str, *, timeout_ms: int = 5000, max_rows: int = 200) -> dict[str, Any]:
        async with self.pool.acquire() as conn:
            async with conn.transaction(readonly=True):
                # Least-privilege role + RLS tenant scoping + timeout, all transaction-local.
                await conn.execute("SET LOCAL ROLE nexus_reader")
                await conn.execute(f"SET LOCAL statement_timeout = {int(timeout_ms)}")
                await conn.execute("SELECT set_config('app.tenant_id', $1, true)", tenant)
                statement = await conn.prepare(sql)
                records = await statement.fetch()
                columns = [attr.name for attr in statement.get_attributes()]
        rows = [[_jsonable(v) for v in record] for record in records[:max_rows]]
        return {"columns": columns, "rows": rows, "row_count": len(records)}
