"""
conftest.py — Shared pytest fixtures for all test levels.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from src.domain.entities.agent_response import AgentResponse
from src.domain.entities.dataset_item import DatasetItem
from src.domain.entities.evaluation_context import EvaluationContext
from src.domain.entities.query_result import QueryResult

# ── Dataset fixtures ───────────────────────────────────────────────────────────


@pytest.fixture
def sample_dataset_item() -> DatasetItem:
    return DatasetItem(
        id="item-001",
        input={"query": "How many orders were placed last month?"},
        expected_output={"sql": "SELECT COUNT(*) FROM orders WHERE month = CURRENT_DATE"},
        metadata={"difficulty": "simple"},
    )


@pytest.fixture
def successful_query_result() -> QueryResult:
    return QueryResult(
        success=True,
        rows=[[100], [200], [300]],
        columns=["value"],
        row_count=3,
        execution_time_ms=42.0,
    )


@pytest.fixture
def failed_query_result() -> QueryResult:
    return QueryResult.failure(error="Table not found", execution_time_ms=5.0)


@pytest.fixture
def successful_agent_response() -> AgentResponse:
    return AgentResponse(
        thread_id="thread-abc",
        status="completed",
        sql_query="SELECT COUNT(*) FROM orders WHERE month = CURRENT_DATE",
        sql_explanation="Counts orders for the current month",
    )


@pytest.fixture
def failed_agent_response() -> AgentResponse:
    return AgentResponse(
        thread_id="thread-xyz",
        status="error",
        error="LLM call failed",
    )


@pytest.fixture
def sample_context(
    sample_dataset_item: DatasetItem,
    successful_agent_response: AgentResponse,
    successful_query_result: QueryResult,
) -> EvaluationContext:
    ctx = EvaluationContext(
        dataset_item=sample_dataset_item,
        run_id="run-001",
        query=sample_dataset_item.query,
        expected_sql=sample_dataset_item.expected_sql,
        allowed_tables=["orders"],
    )
    ctx.agent_response = successful_agent_response
    ctx.generated_sql = successful_agent_response.sql_query
    ctx.expected_result = successful_query_result
    ctx.generated_result = successful_query_result
    ctx.agent_latency_ms = 150.0
    ctx.total_latency_ms = 300.0
    return ctx


@pytest.fixture
def failed_context(sample_dataset_item: DatasetItem) -> EvaluationContext:
    ctx = EvaluationContext(
        dataset_item=sample_dataset_item,
        run_id="run-002",
        query=sample_dataset_item.query,
        expected_sql=sample_dataset_item.expected_sql,
        allowed_tables=["orders"],
    )
    ctx.agent_crashed = True
    ctx.error = "LLM call failed"
    return ctx


# ── Mock infrastructure fixtures ───────────────────────────────────────────────


@pytest.fixture
def mock_agent_client(successful_agent_response: AgentResponse) -> AsyncMock:
    client = AsyncMock()
    client.run = AsyncMock(return_value=successful_agent_response)
    return client


@pytest.fixture
def mock_query_executor(successful_query_result: QueryResult) -> AsyncMock:
    executor = AsyncMock()
    executor.execute = AsyncMock(return_value=successful_query_result)
    return executor


@pytest.fixture
def mock_reporter() -> AsyncMock:
    reporter = AsyncMock()
    reporter.create_dataset_run_trace = AsyncMock(return_value="trace-001")
    reporter.report_sample = AsyncMock()
    reporter.report_run_summary = AsyncMock()
    reporter.flush = AsyncMock()
    return reporter


@pytest.fixture
def mock_dataset_provider(sample_dataset_item: DatasetItem) -> AsyncMock:
    provider = AsyncMock()
    provider.get_dataset = AsyncMock(return_value=[sample_dataset_item])
    return provider
