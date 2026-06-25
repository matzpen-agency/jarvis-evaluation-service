"""
test_langfuse_reporter.py — Integration tests for LangfuseReporter with mocked Langfuse SDK.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.domain.entities.agent_response import AgentResponse
from src.domain.entities.dataset_run import (
    AccuracyStats,
    DatasetRun,
    FailureAnalysis,
    LatencyStats,
)
from src.domain.entities.evaluation_context import EvaluationContext
from src.domain.entities.evaluation_result import EvaluationResult
from src.domain.entities.query_result import QueryResult
from src.infrastructure.langfuse.langfuse_reporter import LangfuseReporter


@pytest.fixture
def mock_langfuse_sdk_client() -> MagicMock:
    client = MagicMock()
    # Mock trace method to return a mock trace object with an id
    mock_trace = MagicMock()
    mock_trace.id = "mocked-trace-id"
    client.trace.return_value = mock_trace
    return client


@pytest.mark.integration
@pytest.mark.asyncio
async def test_reporter_disabled():
    reporter = LangfuseReporter(client=None)
    trace_id = await reporter.create_dataset_run_trace("d1", "r1", {})
    assert trace_id == "r1"  # Returns default run_id when disabled

    # Should not throw any exceptions when calling methods
    await reporter.report_sample(
        context=MagicMock(),
        results=[],
        parent_trace_id="p1",
        dataset_item_id="i1",
        run_name="rn",
    )
    await reporter.report_run_summary(
        run=MagicMock(),
        trace_id="t1",
    )
    await reporter.flush()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_create_dataset_run_trace(mock_langfuse_sdk_client):
    reporter = LangfuseReporter(client=mock_langfuse_sdk_client)
    trace_id = await reporter.create_dataset_run_trace(
        dataset_name="my-dataset",
        run_id="run-123",
        metadata={"user": "tester"},
    )

    assert trace_id == "mocked-trace-id"
    mock_langfuse_sdk_client.trace.assert_called_once_with(
        name="eval-run:my-dataset",
        id="run-123",
        input={"dataset_name": "my-dataset"},
        metadata={"user": "tester"},
        tags=["evaluation", "text-to-sql"],
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_report_sample(mock_langfuse_sdk_client, sample_dataset_item):
    reporter = LangfuseReporter(client=mock_langfuse_sdk_client)

    ctx = EvaluationContext(
        dataset_item=sample_dataset_item,
        run_id="run-123",
        query="show orders",
        expected_sql="SELECT * FROM orders",
        allowed_tables=["orders"],
    )
    ctx.agent_response = AgentResponse(thread_id="t-1", status="completed", sql_query="SELECT * FROM orders")
    ctx.generated_sql = "SELECT * FROM orders"
    ctx.expected_result = QueryResult(success=True, rows=[], columns=[], row_count=0)
    ctx.generated_result = QueryResult(success=True, rows=[], columns=[], row_count=0)

    results = [
        EvaluationResult(evaluator_name="execution_accuracy", score=1.0, passed=True),
    ]

    await reporter.report_sample(
        context=ctx,
        results=results,
        parent_trace_id="parent-trace-456",
        dataset_item_id="item-001",
        run_name="my-run-name",
    )

    mock_langfuse_sdk_client.trace.assert_called_with(
        name="eval-sample:item-001",
        input={
            "query": "show orders",
            "expected_sql": "SELECT * FROM orders",
            "allowed_tables": ["orders"],
        },
        output={
            "generated_sql": "SELECT * FROM orders",
            "succeeded": True,
            "scores": {"execution_accuracy": 1.0},
        },
        metadata={
            "total_latency_ms": 0.0,
            "agent_latency_ms": 0.0,
            "agent_crashed": False,
            "timed_out": False,
            "error": None,
            "expected_row_count": 0,
            "generated_row_count": 0,
        },
        tags=["evaluation", "text-to-sql", "sample"],
    )

    # Verify spans and scores created
    mock_langfuse_sdk_client.span.assert_any_call(
        trace_id="mocked-trace-id",
        name="agent_execution",
        input={"query": "show orders", "allowed_tables": ["orders"]},
        output={"sql_query": "SELECT * FROM orders", "status": "completed"},
        metadata={"latency_ms": 0.0, "agent_crashed": False, "timed_out": False},
    )

    mock_langfuse_sdk_client.score.assert_called_with(
        trace_id="mocked-trace-id",
        name="execution_accuracy",
        value=1.0,
        comment="passed=True",
        data_type="NUMERIC",
    )

    # Verify linking request is made
    mock_langfuse_sdk_client.client.dataset_run_items.create.assert_called_once()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_report_run_summary(mock_langfuse_sdk_client):
    reporter = LangfuseReporter(client=mock_langfuse_sdk_client)

    run = DatasetRun(
        dataset_name="my-dataset",
        run_id="run-123",
        run_name="Run Name",
        started_at=MagicMock(),
        ended_at=MagicMock(),
        duration_seconds=10.0,
        total_cases=10,
        passed=8,
        failed=2,
        failure_rate=0.2,
        latency=LatencyStats(p50=1.0, p95=2.0, p99=3.0, average=1.5, minimum=1.0, maximum=3.0, total_samples=10),
        accuracy=AccuracyStats(
            execution_accuracy=0.8,
            contains_accuracy=0.8,
            sql_exact_match=0.8,
            time_shift_score=0.8,
            composite_score=0.8,
        ),
        failure_analysis=FailureAnalysis(
            total_failures=2,
            failure_rate=0.2,
            agent_crash_count=0,
            agent_crash_rate=0.0,
            categories=[],
            failure_timestamps=[],
        ),
    )

    await reporter.report_run_summary(run=run, trace_id="run-trace-id")

    # Verify score updates for run stats
    mock_langfuse_sdk_client.score.assert_any_call(
        trace_id="run-trace-id",
        name="composite_score",
        value=0.8,
        data_type="NUMERIC",
    )
    mock_langfuse_sdk_client.score.assert_any_call(
        trace_id="run-trace-id",
        name="failure_rate",
        value=0.2,
        data_type="NUMERIC",
    )

    # Verify trace update call
    mock_langfuse_sdk_client.trace.assert_called_with(
        id="run-trace-id",
        output={
            "total_cases": 10,
            "passed": 8,
            "failed": 2,
            "failure_rate": 0.2,
            "composite_score": 0.8,
            "duration_seconds": 10.0,
        },
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_flush(mock_langfuse_sdk_client):
    reporter = LangfuseReporter(client=mock_langfuse_sdk_client)
    await reporter.flush()
    mock_langfuse_sdk_client.flush.assert_called_once()
