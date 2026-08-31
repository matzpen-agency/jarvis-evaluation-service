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
from src.domain.evaluators.time_shift_evaluator import TimeShiftEvaluator
from src.domain.evaluators.sql_comparison_utils import dynamically_wrap_with_yaml_cte


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


@pytest.mark.unit
@pytest.mark.asyncio
async def test_missing_sql(sample_dataset_item):
    """If expected or generated SQL is missing, score should be 0.0."""
    executor = AsyncMock()
    evaluator = TimeShiftEvaluator(query_executor=executor)

    ctx = EvaluationContext(
        dataset_item=sample_dataset_item,
        run_id="run-x",
        query="test",
        expected_sql="",
        allowed_tables=[],
    )
    ctx.generated_sql = "SELECT 1"

    result = await evaluator.evaluate(ctx)
    assert result.score == 0.0
    assert result.passed is False
    assert result.details["reason"] == "missing_sql"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_all_shifts_pass(sample_dataset_item):
    """When all query executions succeed and return same rows, score should be 1.0."""
    executor = AsyncMock()
    # Mock execute to return a successful QueryResult with identical rows
    executor.execute = AsyncMock(return_value=QueryResult(
        success=True,
        rows=[[10], [20]],
        columns=["val"],
        row_count=2,
    ))

    evaluator = TimeShiftEvaluator(query_executor=executor, offsets_days=[-1, -7])

    ctx = EvaluationContext(
        dataset_item=sample_dataset_item,
        run_id="run-x",
        query="test",
        expected_sql="SELECT order_date FROM orders",
        allowed_tables=[],
    )
    ctx.generated_sql = "SELECT order_date FROM orders"

    result = await evaluator.evaluate(ctx)
    assert result.score == 1.0
    assert result.passed is True
    assert len(result.details["shift_results"]) == 2
    # 2 offsets × 2 queries (expected + generated)
    assert executor.execute.call_count == 4


@pytest.mark.unit
@pytest.mark.asyncio
async def test_some_shifts_fail(sample_dataset_item):
    """When one shift returns mismatched rows, the overall score should reflect this."""
    executor = AsyncMock()

    call_count = {"n": 0}

    async def mock_execute(sql: str) -> QueryResult:
        call_count["n"] += 1
        # Every other call returns different rows to simulate a mismatch
        if call_count["n"] % 2 == 0:
            return QueryResult(success=True, rows=[[20]], columns=["v"], row_count=1)
        return QueryResult(success=True, rows=[[10]], columns=["v"], row_count=1)

    executor.execute = AsyncMock(side_effect=mock_execute)
    evaluator = TimeShiftEvaluator(query_executor=executor, offsets_days=[-1, -7])

    ctx = EvaluationContext(
        dataset_item=sample_dataset_item,
        run_id="run-x",
        query="test",
        expected_sql="SELECT order_date FROM orders",
        allowed_tables=[],
    )
    ctx.generated_sql = "SELECT order_date FROM orders"

    result = await evaluator.evaluate(ctx)
    # Both shifts mismatched → score 0.0
    assert result.score == 0.0
    assert result.passed is False


@pytest.mark.unit
@pytest.mark.asyncio
async def test_execution_failure_returns_zero(sample_dataset_item):
    """If Trino execution fails, that shift score is 0.0."""
    executor = AsyncMock()
    executor.execute = AsyncMock(return_value=QueryResult.failure("Trino error"))

    evaluator = TimeShiftEvaluator(query_executor=executor, offsets_days=[-1])

    ctx = EvaluationContext(
        dataset_item=sample_dataset_item,
        run_id="run-x",
        query="test",
        expected_sql="SELECT order_date FROM orders",
        allowed_tables=[],
    )
    ctx.generated_sql = "SELECT order_date FROM orders"

    result = await evaluator.evaluate(ctx)
    assert result.score == 0.0
    assert result.passed is False
    assert result.details["shift_results"][0]["expected_error"] == "Trino error"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_exception_handling(sample_dataset_item):
    """If a shift task raises an exception, gather catches it and reports without crashing."""
    executor = AsyncMock()
    executor.execute = AsyncMock(side_effect=ValueError("Boom"))

    evaluator = TimeShiftEvaluator(query_executor=executor, offsets_days=[-1])

    ctx = EvaluationContext(
        dataset_item=sample_dataset_item,
        run_id="run-x",
        query="test",
        expected_sql="SELECT order_date FROM orders",
        allowed_tables=[],
    )
    ctx.generated_sql = "SELECT order_date FROM orders"

    result = await evaluator.evaluate(ctx)
    assert result.score == 0.0
    assert result.passed is False
    assert result.error is None
    assert result.details["shift_results"][0]["error"] == "Boom"


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
