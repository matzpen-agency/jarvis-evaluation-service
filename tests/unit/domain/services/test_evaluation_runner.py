"""
test_evaluation_runner.py — Unit tests for EvaluationRunner.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.domain.entities.agent_response import AgentResponse
from src.domain.entities.dataset_run import DatasetRun
from src.domain.services.evaluation_engine import EvaluationEngine
from src.domain.services.evaluation_runner import EvaluationRunner
from src.domain.services.metrics_aggregator import MetricsAggregator


@pytest.fixture
def mock_engine() -> EvaluationEngine:
    engine = AsyncMock(spec=EvaluationEngine)
    engine.run_all = AsyncMock(return_value=[])
    return engine


@pytest.fixture
def mock_aggregator() -> MetricsAggregator:
    aggregator = MagicMock(spec=MetricsAggregator)
    # Return a dummy DatasetRun
    dummy_run = MagicMock(spec=DatasetRun)
    dummy_run.total_cases = 1
    dummy_run.passed = 1
    dummy_run.duration_seconds = 1.2
    dummy_run.accuracy = MagicMock()
    dummy_run.accuracy.composite_score = 0.95
    aggregator.aggregate.return_value = dummy_run
    return aggregator


@pytest.mark.unit
@pytest.mark.asyncio
async def test_run_success(
    mock_dataset_provider,
    mock_agent_client,
    mock_query_executor,
    mock_reporter,
    mock_engine,
    mock_aggregator,
):
    runner = EvaluationRunner(
        dataset_provider=mock_dataset_provider,
        agent_client=mock_agent_client,
        query_executor=mock_query_executor,
        reporter=mock_reporter,
        evaluation_engine=mock_engine,
        metrics_aggregator=mock_aggregator,
        max_concurrency=2,
    )

    run_result = await runner.run(
        dataset_name="test-dataset",
        allowed_tables=["orders"],
        run_name="my-run",
    )

    assert run_result.total_cases == 1
    assert run_result.passed == 1

    mock_dataset_provider.get_dataset.assert_called_once_with("test-dataset")
    mock_reporter.create_dataset_run_trace.assert_called_once()
    mock_agent_client.run.assert_called_once_with(
        query="How many orders were placed last month?",
        allowed_tables=["orders"],
    )
    # Execution should run on both expected and generated SQL
    assert mock_query_executor.execute.call_count == 2
    mock_engine.run_all.assert_called_once()
    mock_reporter.report_sample.assert_called_once()
    mock_aggregator.aggregate.assert_called_once()
    mock_reporter.report_run_summary.assert_called_once()
    mock_reporter.flush.assert_called_once()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_run_with_agent_failure(
    mock_dataset_provider,
    mock_query_executor,
    mock_reporter,
    mock_engine,
    mock_aggregator,
):
    # Agent fails to return SQL
    failed_agent = AsyncMock()
    failed_agent.run = AsyncMock(return_value=AgentResponse(
        thread_id="t",
        status="error",
        error="LLM failed",
    ))

    runner = EvaluationRunner(
        dataset_provider=mock_dataset_provider,
        agent_client=failed_agent,
        query_executor=mock_query_executor,
        reporter=mock_reporter,
        evaluation_engine=mock_engine,
        metrics_aggregator=mock_aggregator,
    )

    await runner.run(
        dataset_name="test-dataset",
        allowed_tables=["orders"],
    )

    # Should execute expected SQL but not generated (since agent failed to produce it)
    assert mock_query_executor.execute.call_count == 1

    # Engine is still called with the context containing the error
    mock_engine.run_all.assert_called_once()
    ctx = mock_engine.run_all.call_args[0][0]
    assert ctx.error == "LLM failed"
    assert ctx.agent_response.succeeded is False


@pytest.mark.unit
@pytest.mark.asyncio
async def test_run_with_agent_exception(
    mock_dataset_provider,
    mock_query_executor,
    mock_reporter,
    mock_engine,
    mock_aggregator,
):
    # Agent throws exception
    failing_agent = AsyncMock()
    failing_agent.run = AsyncMock(side_effect=RuntimeError("Connection refused"))

    runner = EvaluationRunner(
        dataset_provider=mock_dataset_provider,
        agent_client=failing_agent,
        query_executor=mock_query_executor,
        reporter=mock_reporter,
        evaluation_engine=mock_engine,
        metrics_aggregator=mock_aggregator,
    )

    await runner.run(
        dataset_name="test-dataset",
        allowed_tables=["orders"],
    )

    mock_engine.run_all.assert_called_once()
    ctx = mock_engine.run_all.call_args[0][0]
    assert ctx.agent_crashed is True
    assert ctx.error == "Connection refused"

    # Only the golden SQL runs (to tell a broken golden question from an agent
    # failure); there is no generated SQL to execute
    assert mock_query_executor.execute.call_count == 1
    assert mock_query_executor.execute.call_args.args[0] == ctx.expected_sql
    mock_reporter.report_sample.assert_called_once()


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("max_concurrency", [1, 4])
async def test_questions_run_at_the_same_time_up_to_the_limit(
    max_concurrency,
    successful_agent_response,
    mock_query_executor,
    mock_reporter,
    mock_engine,
    mock_aggregator,
):
    """Agent calls overlap, max_concurrency at a time and no more."""
    import asyncio

    from src.domain.entities.dataset_item import DatasetItem

    items = [
        DatasetItem(id=f"q{i}", input={"query": f"q{i}?"}, expected_output={"sql": "SELECT 1"})
        for i in range(8)
    ]
    provider = AsyncMock()
    provider.get_dataset = AsyncMock(return_value=items)

    running = peak = 0

    async def slow_agent(**_):
        nonlocal running, peak
        running += 1
        peak = max(peak, running)
        await asyncio.sleep(0.05)
        running -= 1
        return successful_agent_response

    agent = AsyncMock()
    agent.run = AsyncMock(side_effect=slow_agent)
    runner = EvaluationRunner(
        dataset_provider=provider,
        agent_client=agent,
        query_executor=mock_query_executor,
        reporter=mock_reporter,
        evaluation_engine=mock_engine,
        metrics_aggregator=mock_aggregator,
        max_concurrency=max_concurrency,
    )

    await runner.run(dataset_name="d", allowed_tables=[], run_name="r")

    assert agent.run.call_count == 8
    assert peak == max_concurrency


def _unfinished_agent() -> AsyncMock:
    """An agent that gave SQL but didn't finish normally (e.g. stopped after a Trino error)."""
    agent = AsyncMock()
    agent.run = AsyncMock(
        return_value=AgentResponse(
            thread_id="t", status="error", sql_query="SELECT 2", error="stopped"
        )
    )
    return agent


@pytest.mark.unit
@pytest.mark.asyncio
async def test_unfinished_agent_whose_sql_runs_has_no_error(
    mock_dataset_provider, mock_query_executor, mock_reporter, mock_engine, mock_aggregator
):
    runner = EvaluationRunner(
        dataset_provider=mock_dataset_provider,
        agent_client=_unfinished_agent(),
        query_executor=mock_query_executor,
        reporter=mock_reporter,
        evaluation_engine=mock_engine,
        metrics_aggregator=mock_aggregator,
    )

    await runner.run(dataset_name="test-dataset", allowed_tables=["orders"])

    ctx = mock_engine.run_all.call_args[0][0]
    assert ctx.generated_result.success
    assert ctx.error is None
    assert ctx.metadata["agent_status"] == "error"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_unfinished_agent_whose_sql_fails_keeps_the_error(
    mock_dataset_provider,
    successful_query_result,
    mock_reporter,
    mock_engine,
    mock_aggregator,
):
    from src.domain.entities.query_result import QueryResult

    async def execute(sql, *args, **kwargs):
        if sql == "SELECT 2":
            return QueryResult(success=False, error="line 1: syntax error")
        return successful_query_result

    executor = AsyncMock()
    executor.execute = AsyncMock(side_effect=execute)
    runner = EvaluationRunner(
        dataset_provider=mock_dataset_provider,
        agent_client=_unfinished_agent(),
        query_executor=executor,
        reporter=mock_reporter,
        evaluation_engine=mock_engine,
        metrics_aggregator=mock_aggregator,
    )

    await runner.run(dataset_name="test-dataset", allowed_tables=["orders"])

    ctx = mock_engine.run_all.call_args[0][0]
    assert ctx.error == "sql_execution_failure"
