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
from src.infrastructure.langfuse.langfuse_reporter import (
    AGENT_SQL_GENERATION_SCORE,
    LangfuseReporter,
)


@pytest.fixture
def mock_langfuse_sdk_client() -> MagicMock:
    client = MagicMock()
    # Mock start_span method to return a mock span object
    mock_span = MagicMock()
    mock_span.id = "mocked-span-id"
    mock_span.trace_id = "mocked-trace-id"
    client.start_span.return_value = mock_span
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

    assert trace_id == "run123"
    mock_langfuse_sdk_client.start_span.assert_called_once_with(
        name="eval-run:my-dataset",
        trace_context={"trace_id": "run123"},
        input={"dataset_name": "my-dataset"},
        metadata={"user": "tester"},
    )
    # Check that tag is added
    mock_span = mock_langfuse_sdk_client.start_span.return_value
    mock_span.update_trace.assert_called_once_with(tags=["evaluation", "text-to-sql"])


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

    sample_call = next(
        c for c in mock_langfuse_sdk_client.start_span.call_args_list
        if c.kwargs.get("name") == "eval-sample:item-001"
    )
    trace_id = sample_call.kwargs["trace_context"]["trace_id"]
    assert sample_call.kwargs["input"] == {
        "query": "show orders",
        "expected_sql": "SELECT * FROM orders",
        "allowed_tables": ["orders"],
        "connection_id": None,
    }
    output = sample_call.kwargs["output"]
    assert output["generated_sql"] == "SELECT * FROM orders"
    assert output["expected_sql"] == "SELECT * FROM orders"
    assert output["generated_output"]["success"] is True
    assert output["scores"] == {"execution_accuracy": 1.0}
    assert output["evaluations"]["execution_accuracy"]["passed"] is True
    assert sample_call.kwargs["metadata"]["agent_crashed"] is False

    # Child spans hang off the same trace
    agent_call = next(
        c for c in mock_langfuse_sdk_client.start_span.call_args_list
        if c.kwargs.get("name") == "agent_execution"
    )
    assert agent_call.kwargs["trace_context"] == {
        "trace_id": trace_id,
        "parent_span_id": "mocked-span-id",
    }
    assert agent_call.kwargs["output"]["sql_query"] == "SELECT * FROM orders"

    mock_langfuse_sdk_client.create_score.assert_called_with(
        trace_id=trace_id,
        name="execution_accuracy",
        value=1.0,
        comment="passed=True",
        data_type="NUMERIC",
    )

    # Verify linking request is made
    mock_langfuse_sdk_client.api.dataset_run_items.create.assert_called_once()
    request = mock_langfuse_sdk_client.api.dataset_run_items.create.call_args.kwargs["request"]
    assert request.dataset_item_id == "item-001"
    assert request.trace_id == trace_id


def _sample_context(sample_dataset_item) -> EvaluationContext:
    ctx = EvaluationContext(
        dataset_item=sample_dataset_item,
        run_id="run-123",
        query="show orders",
        expected_sql="SELECT * FROM orders",
        allowed_tables=[],
    )
    ctx.generated_sql = "SELECT * FROM orders"
    return ctx


class _ApiError(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__(f"status_code: {status_code}")
        self.status_code = status_code


@pytest.mark.integration
@pytest.mark.asyncio
async def test_link_is_retried_after_transient_failure(mock_langfuse_sdk_client, sample_dataset_item):
    create = mock_langfuse_sdk_client.api.dataset_run_items.create
    create.side_effect = [ConnectionError("Name or service not known"), None]
    reporter = LangfuseReporter(client=mock_langfuse_sdk_client, link_retry_base_seconds=0)

    await reporter.report_sample(
        context=_sample_context(sample_dataset_item),
        results=[],
        parent_trace_id="p",
        dataset_item_id="item-001",
        run_name="run",
    )

    assert create.call_count == 2
    assert reporter._failed_links == []


@pytest.mark.integration
@pytest.mark.asyncio
async def test_link_not_retried_on_client_error(mock_langfuse_sdk_client, sample_dataset_item):
    create = mock_langfuse_sdk_client.api.dataset_run_items.create
    create.side_effect = _ApiError(404)
    reporter = LangfuseReporter(client=mock_langfuse_sdk_client, link_retry_base_seconds=0)

    await reporter.report_sample(
        context=_sample_context(sample_dataset_item),
        results=[],
        parent_trace_id="p",
        dataset_item_id="item-001",
        run_name="run",
    )

    assert create.call_count == 1
    assert reporter._failed_links == []


@pytest.mark.integration
@pytest.mark.asyncio
async def test_failed_links_are_retried_on_flush(mock_langfuse_sdk_client, sample_dataset_item):
    create = mock_langfuse_sdk_client.api.dataset_run_items.create
    create.side_effect = [ConnectionError("down")] * 2 + [None]
    reporter = LangfuseReporter(
        client=mock_langfuse_sdk_client, link_max_attempts=2, link_retry_base_seconds=0
    )

    await reporter.report_sample(
        context=_sample_context(sample_dataset_item),
        results=[],
        parent_trace_id="p",
        dataset_item_id="item-001",
        run_name="run",
    )
    assert len(reporter._failed_links) == 1

    await reporter.flush()
    assert create.call_count == 3
    assert reporter._failed_links == []


@pytest.mark.integration
@pytest.mark.asyncio
async def test_item_is_linked_even_if_span_creation_fails(mock_langfuse_sdk_client, sample_dataset_item):
    mock_langfuse_sdk_client.start_span.side_effect = TypeError("not JSON serialisable")
    reporter = LangfuseReporter(client=mock_langfuse_sdk_client, link_retry_base_seconds=0)

    await reporter.report_sample(
        context=_sample_context(sample_dataset_item),
        results=[],
        parent_trace_id="p",
        dataset_item_id="item-001",
        run_name="run",
    )

    mock_langfuse_sdk_client.api.dataset_run_items.create.assert_called_once()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_not_applicable_results_are_not_scored(mock_langfuse_sdk_client, sample_dataset_item):
    reporter = LangfuseReporter(client=mock_langfuse_sdk_client)
    results = [
        EvaluationResult.not_applicable("time_shift", "no_temporal_columns_to_shift"),
        EvaluationResult.from_error("component_match", "boom"),
    ]

    await reporter.report_sample(
        context=_sample_context(sample_dataset_item),
        results=results,
        parent_trace_id="p",
        dataset_item_id="item-001",
        run_name="run",
    )

    scored = [c.kwargs["name"] for c in mock_langfuse_sdk_client.create_score.call_args_list]
    # errors count as 0, N/A is skipped
    assert scored == ["agent_sql_generation_seconds", "component_match"]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_report_run_summary(mock_langfuse_sdk_client):
    reporter = LangfuseReporter(client=mock_langfuse_sdk_client)
    mock_span = MagicMock()
    reporter._run_spans["run-trace-id"] = mock_span

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
    mock_langfuse_sdk_client.create_score.assert_any_call(
        trace_id="run-trace-id",
        name="composite_score",
        value=0.8,
        data_type="NUMERIC",
    )
    mock_langfuse_sdk_client.create_score.assert_any_call(
        trace_id="run-trace-id",
        name="failure_rate",
        value=0.2,
        data_type="NUMERIC",
    )

    # Verify span update and end call
    mock_span.update.assert_called_with(
        output={
            "total_cases": 10,
            "passed": 8,
            "failed": 2,
            "failure_rate": 0.2,
            "composite_score": 0.8,
            "duration_seconds": 10.0,
            "performance": {
                "average_total_execution_time_ms": 0.0,
                "average_time_to_first_row_ms": 0.0,
                "average_token_usage": 0.0,
            },
            "iterations": {
                "average_iterations": 0.0,
            }
        },
    )
    mock_span.end.assert_called_once()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_flush(mock_langfuse_sdk_client):
    reporter = LangfuseReporter(client=mock_langfuse_sdk_client)
    await reporter.flush()
    mock_langfuse_sdk_client.flush.assert_called_once()



def _score_calls(client, name):
    return [c.kwargs for c in client.create_score.call_args_list if c.kwargs["name"] == name]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_agent_sql_generation_time_is_scored_in_seconds(
    mock_langfuse_sdk_client, sample_dataset_item
):
    reporter = LangfuseReporter(client=mock_langfuse_sdk_client)
    ctx = _sample_context(sample_dataset_item)
    ctx.agent_latency_ms = 12_345.0

    await reporter.report_sample(
        context=ctx, results=[], parent_trace_id="p", dataset_item_id="item-001", run_name="run"
    )

    [score] = _score_calls(mock_langfuse_sdk_client, AGENT_SQL_GENERATION_SCORE)
    assert score["value"] == 12.35
    assert score["data_type"] == "NUMERIC"
    agent_span = next(
        c for c in mock_langfuse_sdk_client.start_span.call_args_list
        if c.kwargs["name"] == "agent_execution"
    )
    assert agent_span.kwargs["metadata"]["sql_generation_seconds"] == 12.35


@pytest.mark.integration
@pytest.mark.asyncio
async def test_no_generation_time_score_without_sql(mock_langfuse_sdk_client, sample_dataset_item):
    reporter = LangfuseReporter(client=mock_langfuse_sdk_client)
    ctx = _sample_context(sample_dataset_item)
    ctx.generated_sql = None
    ctx.agent_latency_ms = 120_000.0  # e.g. the agent timed out

    await reporter.report_sample(
        context=ctx, results=[], parent_trace_id="p", dataset_item_id="item-001", run_name="run"
    )

    assert _score_calls(mock_langfuse_sdk_client, AGENT_SQL_GENERATION_SCORE) == []


@pytest.mark.integration
@pytest.mark.asyncio
async def test_run_summary_has_agent_generation_time(mock_langfuse_sdk_client):
    reporter = LangfuseReporter(client=mock_langfuse_sdk_client)
    run = DatasetRun(
        dataset_name="d",
        run_id="r",
        run_name="r",
        started_at=MagicMock(),
        total_cases=2,
        agent_latency=LatencyStats(p50=20_000.0, p95=41_000.0, average=25_500.0, total_samples=2),
    )

    await reporter.report_run_summary(run=run, trace_id="t")

    values = {
        c.kwargs["name"]: c.kwargs["value"]
        for c in mock_langfuse_sdk_client.create_score.call_args_list
    }
    assert values["avg_agent_sql_generation_seconds"] == 25.5
    assert values["p50_agent_sql_generation_seconds"] == 20.0
    assert values["p95_agent_sql_generation_seconds"] == 41.0
