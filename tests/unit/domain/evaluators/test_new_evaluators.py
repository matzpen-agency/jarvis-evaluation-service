"""
test_new_evaluators.py — Unit tests for the new evaluators.
"""

from __future__ import annotations

import pytest

from src.domain.entities.evaluation_context import EvaluationContext
from src.domain.entities.query_result import QueryResult
from src.domain.evaluators.component_match_evaluator import ComponentMatchEvaluator
from src.domain.evaluators.schema_hallucination_evaluator import SchemaHallucinationEvaluator
from src.domain.evaluators.dialect_error_evaluator import DialectErrorEvaluator


class MockTableResolver:
    async def get_table_schema_map(self) -> dict[str, set[str]]:
        return {
            "orders": {"order_id", "customer_name", "total_amount"},
            "customers": {"customer_id", "first_name", "last_name", "country"},
        }


@pytest.fixture
def comp_evaluator() -> ComponentMatchEvaluator:
    return ComponentMatchEvaluator()


@pytest.fixture
def schema_evaluator() -> SchemaHallucinationEvaluator:
    return SchemaHallucinationEvaluator(table_resolver=MockTableResolver())


@pytest.fixture
def dialect_evaluator() -> DialectErrorEvaluator:
    return DialectErrorEvaluator()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_component_match_perfect(comp_evaluator, sample_dataset_item):
    ctx = EvaluationContext(
        dataset_item=sample_dataset_item,
        run_id="r",
        query="q",
        expected_sql="SELECT order_id FROM orders WHERE total_amount > 100",
        allowed_tables=["orders"],
    )
    ctx.generated_sql = "SELECT order_id FROM orders WHERE total_amount > 100"

    result = await comp_evaluator.evaluate(ctx)
    assert result.score == 1.0
    assert result.passed is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_component_match_partial(comp_evaluator, sample_dataset_item):
    ctx = EvaluationContext(
        dataset_item=sample_dataset_item,
        run_id="r",
        query="q",
        expected_sql="SELECT order_id FROM orders WHERE total_amount > 100",
        allowed_tables=["orders"],
    )
    ctx.generated_sql = "SELECT order_id FROM orders WHERE total_amount > 200"

    result = await comp_evaluator.evaluate(ctx)
    assert result.score < 1.0
    # SELECT, FROM, GROUP, ORDER, JOIN should match (5 out of 6), WHERE mismatches
    assert result.score == round(5 / 6, 4)
    assert result.passed is False


@pytest.mark.unit
@pytest.mark.asyncio
async def test_schema_hallucination_none(schema_evaluator, sample_dataset_item):
    ctx = EvaluationContext(
        dataset_item=sample_dataset_item,
        run_id="r",
        query="q",
        expected_sql="",
        allowed_tables=["orders"],
    )
    ctx.generated_sql = "SELECT order_id, total_amount FROM orders"

    result = await schema_evaluator.evaluate(ctx)
    assert result.score == 1.0
    assert result.passed is True
    assert len(result.details["hallucinated_tables"]) == 0
    assert len(result.details["hallucinated_columns"]) == 0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_schema_hallucination_detected(schema_evaluator, sample_dataset_item):
    ctx = EvaluationContext(
        dataset_item=sample_dataset_item,
        run_id="r",
        query="q",
        expected_sql="",
        allowed_tables=["orders"],
    )
    # references non_existent column 'salary' and non_allowed table 'employees'
    ctx.generated_sql = "SELECT salary FROM orders JOIN employees ON orders.id = employees.id"

    result = await schema_evaluator.evaluate(ctx)
    assert result.score == 0.0
    assert result.passed is False
    assert "employees" in result.details["hallucinated_tables"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_dialect_error_none(dialect_evaluator, sample_dataset_item):
    ctx = EvaluationContext(
        dataset_item=sample_dataset_item,
        run_id="r",
        query="q",
        expected_sql="",
        allowed_tables=["orders"],
    )
    ctx.generated_sql = "SELECT * FROM orders LIMIT 10"

    result = await dialect_evaluator.evaluate(ctx)
    assert result.score == 1.0
    assert result.passed is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_dialect_error_detected(dialect_evaluator, sample_dataset_item):
    ctx = EvaluationContext(
        dataset_item=sample_dataset_item,
        run_id="r",
        query="q",
        expected_sql="",
        allowed_tables=["orders"],
    )
    # invalid SQL syntax
    ctx.generated_sql = "SELECT FROM WHERE SELECT"

    result = await dialect_evaluator.evaluate(ctx)
    assert result.score == 0.0
    assert result.passed is False
    assert "error" in result.details
