"""
test_metrics_aggregator.py — Unit tests for MetricsAggregator.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from src.domain.entities.agent_response import AgentResponse
from src.domain.entities.evaluation_context import EvaluationContext
from src.domain.entities.evaluation_result import EvaluationResult
from src.domain.entities.query_result import QueryResult
from src.domain.services.metrics_aggregator import MetricsAggregator, SampleRecord


@pytest.fixture
def aggregator() -> MetricsAggregator:
    return MetricsAggregator()


@pytest.mark.unit
def test_aggregate_empty(aggregator):
    start = datetime.now(tz=UTC)
    run = aggregator.aggregate(
        dataset_name="empty-dataset",
        run_id="run-empty",
        run_name="Run Empty",
        records=[],
        started_at=start,
    )

    assert run.total_cases == 0
    assert run.passed == 0
    assert run.failed == 0
    assert run.failure_rate == 0.0
    assert run.latency.total_samples == 0
    assert run.accuracy.execution_accuracy == 0.0
    assert run.accuracy.composite_score == 0.0
    assert run.failure_analysis.total_failures == 0


@pytest.mark.unit
def test_aggregate_successful_and_failed_records(aggregator, sample_dataset_item):
    # Setup successful context
    ctx_success = EvaluationContext(
        dataset_item=sample_dataset_item,
        run_id="run-1",
        query="q",
        expected_sql="SELECT 1",
        allowed_tables=[],
    )
    ctx_success.agent_response = AgentResponse(thread_id="t1", status="completed", sql_query="SELECT 1")
    ctx_success.generated_sql = "SELECT 1"
    ctx_success.expected_result = QueryResult(success=True, rows=[[1]], columns=["c"], row_count=1)
    ctx_success.generated_result = QueryResult(success=True, rows=[[1]], columns=["c"], row_count=1)
    ctx_success.total_latency_ms = 100.0

    results_success = [
        EvaluationResult(evaluator_name="execution_accuracy", score=1.0, passed=True),
        EvaluationResult(evaluator_name="contains_accuracy", score=1.0, passed=True),
        EvaluationResult(evaluator_name="sql_exact_match", score=1.0, passed=True),
        EvaluationResult(evaluator_name="time_shift", score=1.0, passed=True),
    ]

    # Setup failed context (agent crash)
    ctx_fail = EvaluationContext(
        dataset_item=sample_dataset_item,
        run_id="run-1",
        query="q",
        expected_sql="SELECT 1",
        allowed_tables=[],
    )
    ctx_fail.agent_crashed = True
    ctx_fail.total_latency_ms = 200.0

    results_fail = [
        EvaluationResult(evaluator_name="execution_accuracy", score=0.0, passed=False),
        EvaluationResult(evaluator_name="contains_accuracy", score=0.0, passed=False),
        EvaluationResult(evaluator_name="sql_exact_match", score=0.0, passed=False),
        EvaluationResult(evaluator_name="time_shift", score=0.0, passed=False),
    ]

    records = [
        SampleRecord(context=ctx_success, results=results_success),
        SampleRecord(context=ctx_fail, results=results_fail),
    ]

    start = datetime.now(tz=UTC)
    run = aggregator.aggregate(
        dataset_name="test-dataset",
        run_id="run-123",
        run_name="Run 123",
        records=records,
        started_at=start,
    )

    assert run.dataset_name == "test-dataset"
    assert run.run_id == "run-123"
    assert run.run_name == "Run 123"
    assert run.total_cases == 2
    assert run.passed == 1
    assert run.failed == 1
    assert run.failure_rate == 0.5

    assert run.latency.total_samples == 2
    assert run.latency.minimum == 100.0
    assert run.latency.maximum == 200.0
    assert run.latency.average == 150.0

    assert run.accuracy.execution_accuracy == 0.5
    assert run.accuracy.contains_accuracy == 0.5
    assert run.accuracy.sql_exact_match == 0.5
    assert run.accuracy.time_shift_score == 0.5

    # composite score should be: 0.6 * 0.5 + 0.15 * 0.5 + 0.15 * 0.5 + 0.1 * 0.5 = 0.5
    assert run.accuracy.composite_score == 0.5

    assert run.failure_analysis.total_failures == 1
    assert run.failure_analysis.agent_crash_count == 1
