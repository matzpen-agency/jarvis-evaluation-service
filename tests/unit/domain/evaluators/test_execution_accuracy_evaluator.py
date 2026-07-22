"""
test_execution_accuracy_evaluator.py — Unit tests for ExecutionAccuracyEvaluator.
"""

from __future__ import annotations

import pytest

from src.domain.entities.evaluation_context import EvaluationContext
from src.domain.entities.query_result import QueryResult
from src.domain.evaluators.execution_accuracy_evaluator import (
    ExecutionAccuracyEvaluator,
)


@pytest.fixture
def evaluator() -> ExecutionAccuracyEvaluator:
    return ExecutionAccuracyEvaluator()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_exact_match_scores_1(evaluator, sample_context):
    """Identical result sets should score 1.0."""
    result = await evaluator.evaluate(sample_context)
    assert result.score == 1.0
    assert result.passed is True
    assert result.evaluator_name == "execution_accuracy"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_order_independent_match(evaluator, sample_context, sample_dataset_item):
    """Results in different row order should still score 1.0.

    After _sort_dataframe normalization, both sides are sorted identically,
    so order_independent_match and exact_ordered_match are both True.
    This is the desired behaviour: the evaluator is now fully order-invariant.
    """
    ctx = EvaluationContext(
        dataset_item=sample_dataset_item,
        run_id="run-x",
        query="test",
        expected_sql="SELECT 1",
        allowed_tables=[],
    )
    ctx.expected_result = QueryResult(
        success=True, rows=[[1], [2], [3]], columns=["v"], row_count=3
    )
    ctx.generated_result = QueryResult(
        success=True, rows=[[3], [1], [2]], columns=["v"], row_count=3
    )
    result = await evaluator.evaluate(ctx)
    assert result.score == 1.0
    assert result.details["order_independent_match"] is True
    assert result.details["exact_ordered_match"] is False


@pytest.mark.unit
@pytest.mark.asyncio
async def test_mismatch_scores_0(evaluator, sample_context, sample_dataset_item):
    """Different row sets should score 0.0."""
    ctx = EvaluationContext(
        dataset_item=sample_dataset_item,
        run_id="run-x",
        query="test",
        expected_sql="SELECT 1",
        allowed_tables=[],
    )
    ctx.expected_result = QueryResult(success=True, rows=[[1]], columns=["v"], row_count=1)
    ctx.generated_result = QueryResult(success=True, rows=[[2]], columns=["v"], row_count=1)
    result = await evaluator.evaluate(ctx)
    assert result.score == 0.0
    assert result.passed is False


@pytest.mark.unit
@pytest.mark.asyncio
async def test_failed_expected_execution(evaluator, sample_context, sample_dataset_item):
    """When expected SQL fails, score should be 0.0."""
    ctx = EvaluationContext(
        dataset_item=sample_dataset_item,
        run_id="run-x",
        query="test",
        expected_sql="SELECT 1",
        allowed_tables=[],
    )
    ctx.expected_result = QueryResult.failure("Table not found")
    ctx.generated_result = QueryResult(success=True, rows=[[1]], columns=["v"], row_count=1)
    result = await evaluator.evaluate(ctx)
    assert result.score == 0.0
    assert "expected_sql_execution_failed" in result.details.get("reason", "")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_none_results(evaluator, sample_context, sample_dataset_item):
    """When results are None, score should be 0.0."""
    ctx = EvaluationContext(
        dataset_item=sample_dataset_item,
        run_id="run-x",
        query="test",
        expected_sql="SELECT 1",
        allowed_tables=[],
    )
    # No results set
    result = await evaluator.evaluate(ctx)
    assert result.score == 0.0
