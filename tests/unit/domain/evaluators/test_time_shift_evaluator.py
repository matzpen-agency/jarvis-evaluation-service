"""
test_time_shift_evaluator.py — Unit tests for TimeShiftEvaluator.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from src.domain.entities.evaluation_context import EvaluationContext
from src.domain.entities.query_result import QueryResult
from src.domain.evaluators.time_shift_evaluator import (
    TimeShiftEvaluator,
    _inject_date_offset,
)


def test_inject_date_offset():
    """Test standard date replacements using regex."""
    assert _inject_date_offset("SELECT * FROM orders WHERE created_at = CURRENT_DATE", -7) == (
        "SELECT * FROM orders WHERE created_at = DATE_ADD('day', -7, CURRENT_DATE)"
    )
    assert _inject_date_offset("SELECT * FROM orders WHERE created_at = current_date", -7) == (
        "SELECT * FROM orders WHERE created_at = DATE_ADD('day', -7, CURRENT_DATE)"
    )
    assert _inject_date_offset("SELECT * FROM orders", -7) == "SELECT * FROM orders"


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
        expected_sql="SELECT CURRENT_DATE",
        allowed_tables=[],
    )
    ctx.generated_sql = "SELECT CURRENT_DATE"

    result = await evaluator.evaluate(ctx)
    assert result.score == 1.0
    assert result.passed is True
    assert len(result.details["shift_results"]) == 2
    assert executor.execute.call_count == 4  # 2 offsets * 2 queries (expected + generated)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_some_shifts_fail(sample_dataset_item):
    """When one shift returns mismatched rows, the overall score should reflect this."""
    executor = AsyncMock()

    # We will control the return value based on SQL string passed
    async def mock_execute(sql: str) -> QueryResult:
        # If it contains -7 offset and + 1 (the generated query), return mismatched rows.
        if "-7" in sql and "+ 1" in sql:
            return QueryResult(success=True, rows=[[20]], columns=["v"], row_count=1)
        # default matching rows
        return QueryResult(success=True, rows=[[10]], columns=["v"], row_count=1)

    executor.execute = AsyncMock(side_effect=mock_execute)
    evaluator = TimeShiftEvaluator(query_executor=executor, offsets_days=[-1, -7])

    ctx = EvaluationContext(
        dataset_item=sample_dataset_item,
        run_id="run-x",
        query="test",
        expected_sql="SELECT CURRENT_DATE",
        allowed_tables=[],
    )
    ctx.generated_sql = "SELECT CURRENT_DATE + 1"

    result = await evaluator.evaluate(ctx)
    # offset -1: score 1.0 (both return [[10]])
    # offset -7: SELECT CURRENT_DATE returns [[10]], SELECT CURRENT_DATE + 1 returns [[20]], score 0.0
    # Average score: 0.5. Since 0.5 < 0.8, passed should be False
    assert result.score == 0.5
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
        expected_sql="SELECT CURRENT_DATE",
        allowed_tables=[],
    )
    ctx.generated_sql = "SELECT CURRENT_DATE"

    result = await evaluator.evaluate(ctx)
    assert result.score == 0.0
    assert result.passed is False
    assert result.details["shift_results"][0]["expected_error"] == "Trino error"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_exception_handling(sample_dataset_item):
    """If a shift task raises an exception, the gather should catch it, and report it in details without crashing."""
    executor = AsyncMock()
    executor.execute = AsyncMock(side_effect=ValueError("Boom"))

    evaluator = TimeShiftEvaluator(query_executor=executor, offsets_days=[-1])

    ctx = EvaluationContext(
        dataset_item=sample_dataset_item,
        run_id="run-x",
        query="test",
        expected_sql="SELECT CURRENT_DATE",
        allowed_tables=[],
    )
    ctx.generated_sql = "SELECT CURRENT_DATE"

    result = await evaluator.evaluate(ctx)
    assert result.score == 0.0
    assert result.passed is False
    assert result.error is None
    assert result.details["shift_results"][0]["error"] == "Boom"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_unhandled_exception_returns_error_result():
    """If an unhandled exception occurs outside the tasks, the evaluator catches it and returns error."""
    executor = AsyncMock()
    evaluator = TimeShiftEvaluator(query_executor=executor)

    # Passing None as context causes AttributeError
    result = await evaluator.evaluate(None)  # type: ignore[arg-type]
    assert result.score == 0.0
    assert result.passed is False
    assert result.error is not None
    assert "expected_sql" in result.error

