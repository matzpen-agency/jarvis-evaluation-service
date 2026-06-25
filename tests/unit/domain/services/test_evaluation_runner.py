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

    # Query execution is skipped entirely because the agent client call threw an exception
    assert mock_query_executor.execute.call_count == 0

    mock_engine.run_all.assert_called_once()
    ctx = mock_engine.run_all.call_args[0][0]
    assert ctx.agent_crashed is True
    assert ctx.error == "Connection refused"
