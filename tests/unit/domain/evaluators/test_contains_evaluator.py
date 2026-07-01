"""
test_contains_evaluator.py — Unit tests for ContainsEvaluator.
"""

from __future__ import annotations

import pytest

from src.domain.entities.evaluation_context import EvaluationContext
from src.domain.entities.query_result import QueryResult
from src.domain.evaluators.contains_evaluator import ContainsEvaluator


@pytest.fixture
def evaluator() -> ContainsEvaluator:
    return ContainsEvaluator()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_ex_contains_perfect_match(evaluator, sample_dataset_item):
    """When generated has same rows and same columns, score = 1.0."""
    ctx = EvaluationContext(
        dataset_item=sample_dataset_item, run_id="r", query="q",
        expected_sql="SELECT 1", allowed_tables=[]
    )
    ctx.expected_result = QueryResult(success=True, rows=[[1], [2]], columns=["v"], row_count=2)
    ctx.generated_result = QueryResult(success=True, rows=[[1], [2]], columns=["v"], row_count=2)

    result = await evaluator.evaluate(ctx)
    assert result.score == 1.0
    assert result.passed is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_ex_contains_extra_columns_allowed(evaluator, sample_dataset_item):
    """When generated has extra columns, but matches on expected columns, score = 1.0."""
    ctx = EvaluationContext(
        dataset_item=sample_dataset_item, run_id="r", query="q",
        expected_sql="SELECT 1", allowed_tables=[]
    )
    ctx.expected_result = QueryResult(success=True, rows=[[1, "a"], [2, "b"]], columns=["id", "name"], row_count=2)
    ctx.generated_result = QueryResult(success=True, rows=[[99, 1, "a"], [100, 2, "b"]], columns=["extra", "id", "name"], row_count=2)

    result = await evaluator.evaluate(ctx)
    assert result.score == 1.0
    assert result.passed is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_ex_contains_missing_columns_fails(evaluator, sample_dataset_item):
    """When generated is missing expected columns, score = 0.0."""
    ctx = EvaluationContext(
        dataset_item=sample_dataset_item, run_id="r", query="q",
        expected_sql="SELECT 1", allowed_tables=[]
    )
    ctx.expected_result = QueryResult(success=True, rows=[[1, "a"]], columns=["id", "name"], row_count=1)
    ctx.generated_result = QueryResult(success=True, rows=[[1]], columns=["id"], row_count=1)

    result = await evaluator.evaluate(ctx)
    assert result.score == 0.0
    assert result.passed is False


@pytest.mark.unit
@pytest.mark.asyncio
async def test_ex_contains_row_count_mismatch_fails(evaluator, sample_dataset_item):
    """When row count differs, score = 0.0."""
    ctx = EvaluationContext(
        dataset_item=sample_dataset_item, run_id="r", query="q",
        expected_sql="SELECT 1", allowed_tables=[]
    )
    ctx.expected_result = QueryResult(success=True, rows=[[1], [2]], columns=["v"], row_count=2)
    ctx.generated_result = QueryResult(success=True, rows=[[1], [2], [3]], columns=["v"], row_count=3)

    result = await evaluator.evaluate(ctx)
    assert result.score == 0.0
    assert result.passed is False
