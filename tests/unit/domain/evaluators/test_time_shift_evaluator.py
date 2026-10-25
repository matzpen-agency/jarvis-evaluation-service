"""
test_time_shift_evaluator.py — Unit tests for TimeShiftEvaluator.

Updated to reflect the new CTE-based temporal shifting approach via
_dynamically_wrap_with_yaml_cte / _sort_dataframe, replacing the old
_inject_date_offset (CURRENT_DATE regex patching) approach.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from src.domain.entities.evaluation_context import EvaluationContext
from src.domain.entities.query_result import QueryResult
from src.domain.evaluators.sql_comparison_utils import dynamically_wrap_with_yaml_cte
from src.domain.evaluators.time_shift_evaluator import TimeShiftEvaluator

# ── dynamically_wrap_with_yaml_cte unit tests ─────────────────────────────────


def test_cte_wrap_zero_shift_returns_original():
    """shift_days=0 must return the query unchanged."""
    sql = "SELECT * FROM orders WHERE order_date = CURRENT_DATE"
    assert dynamically_wrap_with_yaml_cte(sql, 0, {}) == sql


def test_cte_wrap_no_schema_map_returns_original():
    """With an empty schema map no CTEs are generated — original returned."""
    sql = "SELECT * FROM orders WHERE order_date = CURRENT_DATE"
    result = dynamically_wrap_with_yaml_cte(sql, -7, {})
    assert result == sql


def test_cte_wrap_with_date_column():
    """When a table has temporal columns, they are shifted according to their DB type."""
    sql = "SELECT * FROM orders"
    schema_map = {
        "orders": {
            "order_date": "date", 
            "created_at": "timestamp", 
            "updated_ts": "unix_seconds",
            "iso_time": "iso_timestamp",
            "customer_id": "varchar", 
            "amount": "double"
        }
    }
    result = dynamically_wrap_with_yaml_cte(sql, -7, schema_map)
    result_lower = result.lower()
    
    assert "with" in result_lower
    assert "__shifted_orders_1" in result_lower
    
    # date type -> cast(date_add('day', -7, cast(order_date as date)) as date)
    assert "cast(date_add('day', -7, cast(order_date as date)) as date)" in result_lower
    
    # timestamp type -> cast(date_add('day', -7, cast(created_at as timestamp)) as timestamp)
    assert "cast(date_add('day', -7, cast(created_at as timestamp)) as timestamp)" in result_lower
    
    # unix_seconds type -> (updated_ts + (-7 * 86400))
    assert "(updated_ts + (-7 * 86400))" in result_lower
    
    # iso_timestamp type -> cast(date_add('day', -7, try_cast(iso_time as timestamp)) as varchar)
    assert "cast(date_add('day', -7, try_cast(iso_time as timestamp)) as varchar)" in result_lower


def test_cte_wrap_skips_table_with_no_date_columns():
    """Tables with no date-heuristic columns are not wrapped."""
    sql = "SELECT * FROM products"
    schema_map = {"products": {"sku": "varchar", "name": "varchar", "price": "double"}}
    result = dynamically_wrap_with_yaml_cte(sql, -7, schema_map)
    # No CTE because no date columns
    assert "WITH" not in result
    assert result == sql


# ── TimeShiftEvaluator behavior tests ────────────────────────────────────────

ORDERS_SCHEMA = {"orders": {"order_date": "date", "amount": "double"}}


def _ctx(sample_dataset_item, expected_sql="SELECT order_date FROM orders",
         generated_sql="SELECT order_date FROM orders", schema=ORDERS_SCHEMA,
         dialect="trino") -> EvaluationContext:
    ctx = EvaluationContext(
        dataset_item=sample_dataset_item,
        run_id="run-x",
        query="test",
        expected_sql=expected_sql,
        allowed_tables=[],
        connection_id=6,
        dialect=dialect,
        schema_map=schema,
    )
    ctx.generated_sql = generated_sql
    return ctx


def _rows(*values) -> QueryResult:
    return QueryResult(success=True, rows=[[v] for v in values], columns=["v"], row_count=len(values))


@pytest.mark.unit
@pytest.mark.asyncio
async def test_missing_sql(sample_dataset_item):
    """Without golden SQL the metric can't be judged."""
    executor = AsyncMock()
    evaluator = TimeShiftEvaluator(query_executor=executor)

    ctx = _ctx(sample_dataset_item, expected_sql="", generated_sql="SELECT 1")

    result = await evaluator.evaluate(ctx)
    assert result.score == 0.0
    assert result.passed is False
    assert result.applicable is False
    assert result.details["reason"] == "missing_sql"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_no_generated_sql_scores_zero(sample_dataset_item):
    executor = AsyncMock()
    evaluator = TimeShiftEvaluator(query_executor=executor)

    result = await evaluator.evaluate(_ctx(sample_dataset_item, generated_sql=None))
    assert result.score == 0.0
    assert result.applicable is True
    executor.execute.assert_not_called()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_all_shifts_pass(sample_dataset_item):
    """When all query executions succeed and return same rows, score should be 1.0."""
    executor = AsyncMock()
    executor.execute = AsyncMock(return_value=_rows(10, 20))

    evaluator = TimeShiftEvaluator(query_executor=executor, offsets_days=[-1, -7])

    result = await evaluator.evaluate(_ctx(sample_dataset_item))
    assert result.score == 1.0
    assert result.passed is True
    assert len(result.details["shift_results"]) == 2
    # 2 offsets × 2 queries (expected + generated), all on the item's connection
    assert executor.execute.call_count == 4
    assert all(c.kwargs["connection_id"] == 6 for c in executor.execute.call_args_list)
    assert all("__shifted_orders" in c.args[0] for c in executor.execute.call_args_list)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_some_shifts_fail(sample_dataset_item):
    """When one shift returns mismatched rows, the overall score should reflect this."""
    executor = AsyncMock()

    async def mock_execute(sql: str, connection_id: int | None = None) -> QueryResult:
        # Generated SQL hard-codes a date: same rows today, wrong once shifted by -7
        if "2026-09-01" in sql and "-7" in sql:
            return _rows(99)
        return _rows(10)

    executor.execute = AsyncMock(side_effect=mock_execute)
    evaluator = TimeShiftEvaluator(query_executor=executor, offsets_days=[-1, -7])

    ctx = _ctx(
        sample_dataset_item,
        expected_sql="SELECT order_date FROM orders WHERE order_date >= CURRENT_DATE - INTERVAL '30' DAY",
        generated_sql="SELECT order_date FROM orders WHERE order_date >= DATE '2026-09-01'",
    )

    result = await evaluator.evaluate(ctx)
    assert result.score == 0.5
    assert result.passed is False
    statuses = [r["status"] for r in result.details["shift_results"]]
    assert statuses == ["passed", "failed"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_shifted_golden_sql_failure_is_inconclusive(sample_dataset_item):
    """If the shifted golden SQL fails, the shift says nothing about the agent."""
    executor = AsyncMock()
    executor.execute = AsyncMock(return_value=QueryResult.failure("Trino error"))

    evaluator = TimeShiftEvaluator(query_executor=executor, offsets_days=[-1])

    result = await evaluator.evaluate(_ctx(sample_dataset_item))
    assert result.score == 0.0
    assert result.passed is False
    assert result.applicable is False
    assert result.details["shift_results"][0]["expected_error"] == "Trino error"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_shifted_generated_sql_failure_counts_as_failed(sample_dataset_item):
    async def mock_execute(sql: str, connection_id: int | None = None) -> QueryResult:
        if "generated_col" in sql:
            return QueryResult.failure("boom")
        return _rows(1)

    executor = AsyncMock()
    executor.execute = AsyncMock(side_effect=mock_execute)
    evaluator = TimeShiftEvaluator(query_executor=executor, offsets_days=[-1])

    ctx = _ctx(sample_dataset_item, generated_sql="SELECT order_date AS generated_col FROM orders")
    result = await evaluator.evaluate(ctx)
    assert result.applicable is True
    assert result.score == 0.0
    assert result.details["shift_results"][0]["status"] == "failed"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_exception_handling(sample_dataset_item):
    """If a shift task raises an exception, gather catches it and reports without crashing."""
    executor = AsyncMock()
    executor.execute = AsyncMock(side_effect=ValueError("Boom"))

    evaluator = TimeShiftEvaluator(query_executor=executor, offsets_days=[-1])

    result = await evaluator.evaluate(_ctx(sample_dataset_item))
    assert result.score == 0.0
    assert result.passed is False
    assert result.error is None
    assert result.details["shift_results"][0]["error"] == "Boom"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_no_temporal_columns_is_not_applicable(sample_dataset_item):
    """Shifting tables without date columns changes nothing — don't re-run identical SQL."""
    executor = AsyncMock()
    evaluator = TimeShiftEvaluator(query_executor=executor, offsets_days=[-1, -7])

    ctx = _ctx(sample_dataset_item, schema={"orders": {"id": "integer"}})
    result = await evaluator.evaluate(ctx)

    assert result.applicable is False
    assert result.details["reason"] == "no_temporal_columns_to_shift"
    executor.execute.assert_not_called()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_golden_sql_failure_is_not_applicable(sample_dataset_item):
    executor = AsyncMock()
    evaluator = TimeShiftEvaluator(query_executor=executor)
    ctx = _ctx(sample_dataset_item)
    ctx.expected_result = QueryResult.failure("relation does not exist")

    result = await evaluator.evaluate(ctx)
    assert result.applicable is False
    executor.execute.assert_not_called()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_shift_respects_expected_order_by(sample_dataset_item):
    executor = AsyncMock()

    async def mock_execute(sql: str, connection_id: int | None = None) -> QueryResult:
        if "generated" in sql:
            return QueryResult(success=True, rows=[[2], [1]], columns=["generated"])
        return QueryResult(success=True, rows=[[1], [2]], columns=["order_date"])

    executor.execute = AsyncMock(side_effect=mock_execute)

    class AsksForOrder:
        async def requires_order(self, question):
            from src.ports.order_judge import OrderDecision

            return OrderDecision(True, "asks for chronological order", "llm")

    evaluator = TimeShiftEvaluator(
        query_executor=executor, offsets_days=[-1], order_judge=AsksForOrder()
    )
    ctx = _ctx(
        sample_dataset_item,
        expected_sql="SELECT order_date FROM orders ORDER BY order_date",
        generated_sql="SELECT order_date AS generated FROM orders",
    )

    result = await evaluator.evaluate(ctx)
    assert result.score == 0.0
    assert result.details["shift_results"][0]["reason"] == "row_order_mismatch"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_discriminative_flag(sample_dataset_item):
    """discriminative is True when a shift actually changed the golden result."""
    calls = {"n": 0}

    async def mock_execute(sql: str, connection_id: int | None = None) -> QueryResult:
        calls["n"] += 1
        return _rows(5)

    executor = AsyncMock()
    executor.execute = AsyncMock(side_effect=mock_execute)
    evaluator = TimeShiftEvaluator(query_executor=executor, offsets_days=[-1])

    ctx = _ctx(sample_dataset_item)
    ctx.expected_result = _rows(5)
    result = await evaluator.evaluate(ctx)
    assert result.details["discriminative"] is False

    ctx.expected_result = _rows(4)
    result = await evaluator.evaluate(ctx)
    assert result.details["discriminative"] is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_shift_timeout_is_inconclusive(sample_dataset_item):
    import asyncio

    async def slow_execute(sql: str, connection_id: int | None = None) -> QueryResult:
        await asyncio.sleep(1)
        return _rows(1)

    executor = AsyncMock()
    executor.execute = AsyncMock(side_effect=slow_execute)
    evaluator = TimeShiftEvaluator(
        query_executor=executor, offsets_days=[-1], shift_timeout_seconds=0.05
    )

    result = await evaluator.evaluate(_ctx(sample_dataset_item))
    assert result.applicable is False
    assert result.details["shift_results"][0]["status"] == "inconclusive"


@pytest.mark.unit
def test_cte_wrap_postgres_uses_interval_arithmetic():
    sql = "SELECT SUM(f.amount) FROM orders f WHERE f.order_date >= CURRENT_DATE - INTERVAL '30 days'"
    result = dynamically_wrap_with_yaml_cte(
        sql, -7, {"orders": {"order_date": "timestamp without time zone", "amount": "numeric"}}, "postgres"
    ).lower()
    assert "date_add" not in result
    assert "order_date + interval '-7 day' as order_date" in result
    assert "from __shifted_orders_1 as f" in result


@pytest.mark.unit
def test_cte_wrap_prepends_before_query_ctes():
    """The query's own CTEs must be able to read the shifted table."""
    sql = "WITH recent AS (SELECT * FROM orders) SELECT * FROM recent"
    result = dynamically_wrap_with_yaml_cte(sql, -1, ORDERS_SCHEMA, "trino").lower()
    assert result.index("__shifted_orders_1 as (") < result.index("recent as (")


@pytest.mark.unit
def test_cte_wrap_reuses_one_cte_per_table():
    sql = "SELECT * FROM orders a JOIN orders b ON a.amount = b.amount"
    result = dynamically_wrap_with_yaml_cte(sql, -1, ORDERS_SCHEMA, "trino").lower()
    assert result.count("__shifted_orders_1 as (") == 1
    assert "__shifted_orders_2" not in result


@pytest.mark.unit
@pytest.mark.asyncio
async def test_unhandled_exception_returns_error_result():
    """If an unhandled exception occurs outside the tasks, the evaluator catches it."""
    executor = AsyncMock()
    evaluator = TimeShiftEvaluator(query_executor=executor)

    # Passing None as context causes AttributeError
    result = await evaluator.evaluate(None)  # type: ignore[arg-type]
    assert result.score == 0.0
    assert result.passed is False
    assert result.error is not None
    assert "expected_sql" in result.error
