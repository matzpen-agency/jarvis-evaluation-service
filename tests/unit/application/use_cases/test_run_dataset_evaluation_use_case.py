"""
test_run_dataset_evaluation_use_case.py — Unit tests for RunDatasetEvaluationUseCase.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
import respx
from httpx import Response

from src.application.dto.run_dataset_request import RunDatasetRequest
from src.application.use_cases.run_dataset_evaluation_use_case import (
    BackendTableResolver,
    RunDatasetEvaluationUseCase,
)
from src.domain.entities.dataset_run import (
    AccuracyStats,
    DatasetRun,
    FailureAnalysis,
    LatencyStats,
)


@pytest.fixture
def mock_dataset_run() -> DatasetRun:
    return DatasetRun(
        dataset_name="test-dataset",
        run_id="run-123",
        run_name="Run 123",
        started_at=AsyncMock(),
        ended_at=AsyncMock(),
        duration_seconds=5.5,
        total_cases=10,
        passed=8,
        failed=2,
        failure_rate=0.2,
        latency=LatencyStats(
            p50=100.0,
            p95=200.0,
            p99=250.0,
            average=120.0,
            minimum=50.0,
            maximum=300.0,
            total_samples=10,
        ),
        accuracy=AccuracyStats(
            execution_accuracy=0.8,
            contains_accuracy=0.85,
            sql_exact_match=0.7,
            time_shift_score=0.9,
            composite_score=0.81,
        ),
        failure_analysis=FailureAnalysis(
            total_failures=2,
            failure_rate=0.2,
            agent_crash_count=1,
            agent_crash_rate=0.1,
            validation_failure_count=1,
            validation_failure_rate=0.1,
            categories=[],
            failure_timestamps=[],
        ),
        langfuse_trace_id="trace-xyz",
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_use_case_execute(mock_dataset_run):
    mock_runner = AsyncMock()
    mock_runner.run = AsyncMock(return_value=mock_dataset_run)

    mock_resolver = AsyncMock(spec=BackendTableResolver)
    mock_resolver.get_production_tables = AsyncMock(return_value=["orders", "users"])

    use_case = RunDatasetEvaluationUseCase(
        runner=mock_runner,
        backend_table_resolver=mock_resolver,
    )

    req = RunDatasetRequest(
        dataset_name="test-dataset",
        additional_tables=["products"],
    )

    resp = await use_case.execute(req)

    # Verify tables resolved as union of production + additional
    # Union of {"orders", "users"} and {"products"} -> {"orders", "users", "products"}
    called_allowed_tables = mock_runner.run.call_args[1]["allowed_tables"]
    assert set(called_allowed_tables) == {"orders", "users", "products"}

    assert resp.dataset_name == "test-dataset"
    assert resp.run_id == "run-123"
    assert resp.total_cases == 10
    assert resp.passed == 8
    assert resp.failed == 2
    assert resp.failure_rate == 0.2
    assert resp.latency.p50 == 100.0
    assert resp.accuracy.composite_score == 0.81
    assert resp.failure_analysis.agent_crash_count == 1


@pytest.mark.unit
@pytest.mark.asyncio
@respx.mock
async def test_resolver_get_production_tables_list_of_strings():
    """Test when backend returns simple list of strings."""
    respx.get("http://localhost:8000/api/agent/tables?status=production").mock(
        return_value=Response(200, json=["orders", "users"])
    )

    resolver = BackendTableResolver(backend_url="http://localhost:8000")
    tables = await resolver.get_production_tables()
    assert tables == ["orders", "users"]


@pytest.mark.unit
@pytest.mark.asyncio
@respx.mock
async def test_resolver_get_production_tables_list_of_dicts():
    """Test when backend returns list of dictionaries with name key."""
    respx.get("http://localhost:8000/api/agent/tables?status=production").mock(
        return_value=Response(200, json=[{"name": "orders"}, {"name": "users"}])
    )

    resolver = BackendTableResolver(backend_url="http://localhost:8000")
    tables = await resolver.get_production_tables()
    assert tables == ["orders", "users"]


@pytest.mark.unit
@pytest.mark.asyncio
@respx.mock
async def test_resolver_get_production_tables_fallback():
    """Test resolver gracefully handles errors and returns empty list."""
    respx.get("http://localhost:8000/api/agent/tables?status=production").mock(
        return_value=Response(500)
    )

    resolver = BackendTableResolver(backend_url="http://localhost:8000")
    tables = await resolver.get_production_tables()
    assert tables == []
