import json
import logging
import re
from typing import Any

import asyncpg
import sqlglot
from sqlglot import exp

from app.llm.client import LLMClient
from app.services.db import Database

log = logging.getLogger(__name__)

ALLOWED_TABLES = frozenset({"sales"})
FORBIDDEN_FUNCTIONS = frozenset({
    "pg_sleep", "set_config", "current_setting", "pg_read_file", "pg_read_binary_file", "pg_ls_dir",
    "dblink", "lo_import", "lo_export", "pg_terminate_backend", "pg_cancel_backend", "version", "pg_stat_file",
})
FORBIDDEN_NODES = tuple(
    getattr(exp, name)
    for name in ("Insert", "Update", "Delete", "Drop", "Create", "Alter", "AlterTable", "Command", "Into", "Merge", "Grant")
    if hasattr(exp, name)
)
MAX_ROWS = 200

REGIONS = {"north america": "North America", "emea": "EMEA", "apac": "APAC", "latam": "LATAM"}
PRODUCTS = {"analytics suite": "Analytics Suite", "cloud storage": "Cloud Storage", "security pro": "Security Pro"}

SQL_PROMPT = """You translate questions into a single read-only PostgreSQL SELECT statement.
Schema: sales(region TEXT -- one of 'North America','EMEA','APAC','LATAM';
              product TEXT -- one of 'Analytics Suite','Cloud Storage','Security Pro';
              year INT -- 2024 or 2025; quarter TEXT -- 'Q1'..'Q4'; revenue NUMERIC -- USD; units INT)
Rules: SELECT only; never reference tenant_id (row-level security handles tenancy); use clear column aliases;
aggregate when the question asks for totals, averages or rankings.
Return JSON: {"sql": "<query>"}"""


class SqlValidationError(ValueError):
    pass


def validate_sql(sql: str, max_rows: int = MAX_ROWS) -> str:
    try:
        statements = [s for s in sqlglot.parse(sql, read="postgres") if s is not None]
    except sqlglot.errors.SqlglotError as exc:
        raise SqlValidationError("The generated SQL could not be parsed.") from exc
    if len(statements) != 1:
        raise SqlValidationError("Exactly one SQL statement is allowed.")
    statement = statements[0]
    if not isinstance(statement, exp.Select):
        raise SqlValidationError("Only SELECT queries are allowed.")
    if any(True for _ in statement.find_all(*FORBIDDEN_NODES)):
        raise SqlValidationError("Data-modifying or administrative statements are not allowed.")

    cte_names = {cte.alias_or_name.lower() for cte in statement.find_all(exp.CTE)}
    tables = {table.name.lower() for table in statement.find_all(exp.Table)}
    unknown = tables - cte_names - ALLOWED_TABLES
    if unknown:
        raise SqlValidationError(f"Access to table(s) {sorted(unknown)} is not allowed.")

    for func in statement.find_all(exp.Func):
        name = (func.name if isinstance(func, exp.Anonymous) else func.key).lower()
        if name in FORBIDDEN_FUNCTIONS:
            raise SqlValidationError(f"Function {name} is not allowed.")

    inner = statement.sql(dialect="postgres")
    return f"SELECT * FROM ({inner}) AS q LIMIT {int(max_rows)}"


def heuristic_sql(question: str) -> str:
    """Deterministic NL-to-SQL for offline mode; literals come only from fixed allow-lists."""
    q = question.lower()
    metric = "units" if re.search(r"\bunits?\b", q) else "revenue"
    agg = "AVG" if re.search(r"\b(average|avg|mean)\b", q) else "SUM"
    alias = f"{'avg' if agg == 'AVG' else 'total'}_{metric}"

    dims: list[str] = []
    for dim in ("region", "product", "year", "quarter"):
        if re.search(rf"\b(by|per|each|every|across)\s+{dim}s?\b", q) or re.search(rf"\bwhich\s+{dim}s?\b", q):
            dims.append(dim)
    top_match = re.search(r"\btop\s+(\d+)(?:\s+(region|product)s?)?", q)
    best = re.search(r"\b(highest|best|most|largest|biggest)\b", q)
    worst = re.search(r"\b(lowest|worst|least|smallest)\b", q)
    if top_match and top_match.group(2) and top_match.group(2) not in dims:
        dims.append(top_match.group(2))
    if re.search(r"\b(trend|over time|quarterly)\b", q) and not dims:
        dims = ["year", "quarter"]
    if (top_match or best or worst) and not dims:
        dims = ["product"] if "product" in q else ["region"]

    where: list[str] = []
    years = sorted(set(re.findall(r"\b(20\d\d)\b", q)))
    if years:
        where.append(f"year IN ({', '.join(years)})")
    quarters = sorted({f"'Q{n}'" for n in re.findall(r"\bq([1-4])\b", q)})
    if quarters:
        where.append(f"quarter IN ({', '.join(quarters)})")
    regions = [v for k, v in REGIONS.items() if k in q]
    if regions:
        where.append("region IN (" + ", ".join(f"'{r}'" for r in regions) + ")")
    products = [v for k, v in PRODUCTS.items() if k in q]
    if products:
        where.append("product IN (" + ", ".join(f"'{p}'" for p in products) + ")")

    select = dims + [f"ROUND({agg}({metric}), 2) AS {alias}"]
    sql = f"SELECT {', '.join(select)} FROM sales"
    if where:
        sql += " WHERE " + " AND ".join(where)
    if dims:
        sql += f" GROUP BY {', '.join(dims)}"
        if top_match or best:
            sql += f" ORDER BY {alias} DESC"
        elif worst:
            sql += f" ORDER BY {alias} ASC"
        else:
            sql += f" ORDER BY {', '.join(dims)}"
    if top_match:
        sql += f" LIMIT {min(int(top_match.group(1)), 50)}"
    elif best or worst:
        sql += " LIMIT 1"
    return sql


class SqlAgent:
    def __init__(self, llm: LLMClient, db: Database) -> None:
        self._llm = llm
        self._db = db

    async def run(self, question: str, tenant: str) -> dict[str, Any]:
        sql: str | None = None
        error: str | None = None
        attempts = 1 if self._llm.is_mock else 2
        for attempt in range(1, attempts + 1):
            try:
                candidate = await self._generate(question, sql, error)
                sql = validate_sql(candidate)
                result = await self._db.run_readonly_sql(tenant, sql)
                return {"sql": sql, **result, "error": None, "attempts": attempt}
            except SqlValidationError as exc:
                error = str(exc)
            except asyncpg.PostgresError as exc:
                log.info("SQL execution failed: %s", exc.__class__.__name__)
                error = f"Database rejected the query ({exc.__class__.__name__})."
        return {"sql": sql, "columns": [], "rows": [], "row_count": 0, "error": error, "attempts": attempts}

    async def _generate(self, question: str, previous_sql: str | None, previous_error: str | None) -> str:
        if self._llm.is_mock:
            return heuristic_sql(question)
        messages = [{"role": "system", "content": SQL_PROMPT}, {"role": "user", "content": question}]
        if previous_error:
            messages.append({"role": "user", "content": f"The previous query failed: {previous_error}\nQuery: {previous_sql}\nFix it."})
        raw = await self._llm.complete(messages, json_mode=True, temperature=0, max_tokens=400)
        try:
            return str(json.loads(raw)["sql"])
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise SqlValidationError("The model did not return a SQL query.") from exc
