import pytest

from app.engine.sql_agent import SqlValidationError, heuristic_sql, validate_sql


def test_select_is_wrapped_with_limit() -> None:
    sql = validate_sql("SELECT region, SUM(revenue) FROM sales GROUP BY region")
    assert sql.startswith("SELECT * FROM (") and sql.endswith("LIMIT 200")


@pytest.mark.parametrize(
    "bad_sql",
    [
        "DELETE FROM sales",
        "DROP TABLE sales",
        "UPDATE sales SET revenue = 0",
        "SELECT * FROM pg_user",
        "SELECT * FROM sales; DROP TABLE sales",
        "SELECT pg_sleep(10)",
        "SELECT current_setting('app.tenant_id')",
        "SELECT * INTO copy_sales FROM sales",
        "SELECT * FROM information_schema.tables",
    ],
)
def test_unsafe_sql_rejected(bad_sql: str) -> None:
    with pytest.raises(SqlValidationError):
        validate_sql(bad_sql)


def test_cte_on_allowed_table_passes() -> None:
    validate_sql("WITH t AS (SELECT * FROM sales) SELECT region FROM t")


def test_heuristic_sql_group_and_filters() -> None:
    sql = heuristic_sql("Total revenue by region in 2025 for Q3")
    assert "GROUP BY region" in sql
    assert "year IN (2025)" in sql
    assert "quarter IN ('Q3')" in sql
    validate_sql(sql)


def test_heuristic_sql_top_n_products() -> None:
    sql = heuristic_sql("top 2 products by units")
    assert "GROUP BY product" in sql and "LIMIT 2" in sql and "SUM(units)" in sql


def test_heuristic_sql_best_region() -> None:
    sql = heuristic_sql("Which region had the highest revenue in 2024?")
    assert "ORDER BY total_revenue DESC" in sql and sql.endswith("LIMIT 1")
