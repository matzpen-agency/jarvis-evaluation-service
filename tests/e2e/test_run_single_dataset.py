"""
test_run_single_dataset.py — End-to-end tests for the evaluation run endpoint.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from src.api.dependencies.container import get_run_dataset_use_case
from src.application.dto.run_dataset_response import (
    AccuracyStatsDTO,
    FailureAnalysisDTO,
    LatencyStatsDTO,
    PerformanceStatsDTO,
    RunDatasetResponse,
)
from src.main import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.mark.e2e
def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


@pytest.mark.e2e
def test_run_single_dataset_success(client):
    mock_use_case = AsyncMock()

    # Construct a valid RunDatasetResponse mock DTO
    expected_response = RunDatasetResponse(
        dataset_name="my-dataset",
        run_id="run-123",
        total_cases=5,
        passed=4,
        failed=1,
        failure_rate=0.2,
        duration_seconds=3.5,
        langfuse_trace_id="trace-abc",
        latency=LatencyStatsDTO(
            p50=120.0,
            p95=200.0,
            p99=250.0,
            average=130.0,
            minimum=80.0,
            maximum=260.0,
            total_samples=5,
        ),
        accuracy=AccuracyStatsDTO(
            execution_accuracy=0.8,
            contains_accuracy=0.8,
            sql_exact_match=0.6,
            time_shift_score=0.7,
            component_match=0.0,
            schema_hallucination=0.0,
            dialect_error=0.0,
            composite_score=0.75,
        ),
        failure_analysis=FailureAnalysisDTO(
            total_failures=1,
            failure_rate=0.2,
            validation_failure_count=1,
            validation_failure_rate=0.2,
            categories=[],
        ),
        performance=PerformanceStatsDTO(
            average_total_execution_time_ms=0.0,
            average_time_to_first_row_ms=0.0,
            total_token_usage=0,
            average_token_usage=0.0,
            average_refiner_iterations=0.0,
        ),
    )
    mock_use_case.execute = AsyncMock(return_value=expected_response)

    # Apply dependency override
    app.dependency_overrides[get_run_dataset_use_case] = lambda: mock_use_case

    try:
        payload = {
            "dataset_name": "my-dataset",
            "additional_tables": ["products"],
        }
        response = client.post("/text-to-sql/evaluation/run-single-dataset", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert data["dataset_name"] == "my-dataset"
        assert data["run_id"] == "run-123"
        assert data["total_cases"] == 5
        assert data["latency"]["p50"] == 120.0
        assert data["accuracy"]["composite_score"] == 0.75

        mock_use_case.execute.assert_called_once()
        call_arg = mock_use_case.execute.call_args[0][0]
        assert call_arg.dataset_name == "my-dataset"
        assert call_arg.additional_tables == ["products"]

    finally:
        # Clear dependency overrides to prevent leakage
        app.dependency_overrides.clear()


@pytest.mark.e2e
def test_run_single_dataset_validation_error(client):
    # Pass an invalid payload (missing dataset_name)
    payload = {
        "additional_tables": ["products"],
    }
    response = client.post("/text-to-sql/evaluation/run-single-dataset", json=payload)

    assert response.status_code == 422
