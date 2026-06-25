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
async def test_full_subset_scores_1(evaluator, sample_dataset_item):
    """When generated contains all expected rows, score = 1.0."""
    ctx = EvaluationContext(
        dataset_item=sample_dataset_item, run_id="r", query="q",
        expected_sql="SELECT 1", allowed_tables=[]
    )
    ctx.expected_result = QueryResult(success=True, rows=[[1], [2]], columns=["v"], row_count=2)
    ctx.generated_result = QueryResult(success=True, rows=[[1], [2], [3]], columns=["v"], row_count=3)

    result = await evaluator.evaluate(ctx)
    assert result.score == 1.0
    assert result.passed is True
    assert result.details["matched_row_count"] == 2


@pytest.mark.unit
@pytest.mark.asyncio
async def test_partial_match(evaluator, sample_dataset_item):
    """When only some expected rows are present, score = fraction."""
    ctx = EvaluationContext(
        dataset_item=sample_dataset_item, run_id="r", query="q",
        expected_sql="SELECT 1", allowed_tables=[]
    )
    ctx.expected_result = QueryResult(success=True, rows=[[1], [2], [3]], columns=["v"], row_count=3)
    ctx.generated_result = QueryResult(success=True, rows=[[1], [3]], columns=["v"], row_count=2)

    result = await evaluator.evaluate(ctx)
    assert abs(result.score - 2 / 3) < 0.01
    assert result.passed is False


@pytest.mark.unit
@pytest.mark.asyncio
async def test_no_expected_rows_vacuously_true(evaluator, sample_dataset_item):
    """When expected is empty, result should be 1.0."""
    ctx = EvaluationContext(
        dataset_item=sample_dataset_item, run_id="r", query="q",
        expected_sql="SELECT 1", allowed_tables=[]
    )
    ctx.expected_result = QueryResult(success=True, rows=[], columns=[], row_count=0)
    ctx.generated_result = QueryResult(success=True, rows=[[1]], columns=["v"], row_count=1)

    result = await evaluator.evaluate(ctx)
    assert result.score == 1.0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_no_match_scores_0(evaluator, sample_dataset_item):
    """When no expected rows are in generated, score = 0.0."""
    ctx = EvaluationContext(
        dataset_item=sample_dataset_item, run_id="r", query="q",
        expected_sql="SELECT 1", allowed_tables=[]
    )
    ctx.expected_result = QueryResult(success=True, rows=[[1], [2]], columns=["v"], row_count=2)
    ctx.generated_result = QueryResult(success=True, rows=[[99], [100]], columns=["v"], row_count=2)

    result = await evaluator.evaluate(ctx)
    assert result.score == 0.0
