"""
test_sql_exact_match_evaluator.py — Unit tests for SqlExactMatchEvaluator.
"""

from __future__ import annotations

import pytest

from src.domain.entities.evaluation_context import EvaluationContext
from src.domain.evaluators.sql_exact_match_evaluator import SqlExactMatchEvaluator


@pytest.fixture
def evaluator() -> SqlExactMatchEvaluator:
    return SqlExactMatchEvaluator()


def _ctx(sample_dataset_item, expected_sql: str, generated_sql: str) -> EvaluationContext:
    ctx = EvaluationContext(
        dataset_item=sample_dataset_item,
        run_id="r",
        query="q",
        expected_sql=expected_sql,
        allowed_tables=[],
    )
    ctx.generated_sql = generated_sql
    return ctx


@pytest.mark.unit
@pytest.mark.asyncio
async def test_identical_sql_scores_high(evaluator, sample_dataset_item):
    sql = "SELECT COUNT(*) FROM orders"
    ctx = _ctx(sample_dataset_item, sql, sql)
    result = await evaluator.evaluate(ctx)
    assert result.score >= 0.9
    assert result.passed is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_whitespace_difference_normalized(evaluator, sample_dataset_item):
    ctx = _ctx(
        sample_dataset_item,
        "SELECT COUNT(*)  FROM  orders",
        "SELECT COUNT(*) FROM orders",
    )
    result = await evaluator.evaluate(ctx)
    # Normalized and AST levels should match
    assert result.details["normalized_score"] == 1.0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_completely_different_sql(evaluator, sample_dataset_item):
    ctx = _ctx(
        sample_dataset_item,
        "SELECT id FROM customers",
        "SELECT revenue FROM products",
    )
    result = await evaluator.evaluate(ctx)
    assert result.score < 0.5
    assert result.passed is False


@pytest.mark.unit
@pytest.mark.asyncio
async def test_missing_generated_sql(evaluator, sample_dataset_item):
    ctx = EvaluationContext(
        dataset_item=sample_dataset_item,
        run_id="r",
        query="q",
        expected_sql="SELECT 1",
        allowed_tables=[],
    )
    # No generated_sql set
    result = await evaluator.evaluate(ctx)
    assert result.score == 0.0
    assert result.details.get("reason") == "no_generated_sql"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_case_insensitive_match(evaluator, sample_dataset_item):
    ctx = _ctx(
        sample_dataset_item,
        "SELECT COUNT(*) FROM orders",
        "select count(*) from orders",
    )
    result = await evaluator.evaluate(ctx)
    assert result.details["raw_score"] == 1.0
