"""
test_backend_query_executor.py — Integration tests for BackendQueryExecutor using respx.
"""

from __future__ import annotations

import httpx
import pytest
import respx

from src.infrastructure.trino.backend_query_executor import BackendQueryExecutor


@pytest.fixture
def executor() -> BackendQueryExecutor:
    return BackendQueryExecutor(
        backend_url="http://localhost:8000",
        query_endpoint="/api/query/execute",
        timeout=1.0,
        enabled=True,
    )


@pytest.mark.integration
@pytest.mark.asyncio
@respx.mock
async def test_execute_success(executor):
    payload = {
        "success": True,
        "rows": [[10], [20]],
        "columns": ["val"],
        "row_count": 2,
        "execution_time_ms": 15.4,
        "error": None,
    }
    respx.post("http://localhost:8000/api/query/execute").mock(
        return_value=httpx.Response(200, json=payload)
    )

    result = await executor.execute("SELECT * FROM my_table")
    assert result.success is True
    assert result.rows == [[10], [20]]
    assert result.columns == ["val"]
    assert result.row_count == 2
    assert result.execution_time_ms == 15.4
    assert result.error is None


@pytest.mark.integration
@pytest.mark.asyncio
@respx.mock
async def test_execute_database_failure(executor):
    payload = {
        "success": False,
        "rows": [],
        "columns": [],
        "row_count": 0,
        "execution_time_ms": 5.0,
        "error": "Table not found",
    }
    respx.post("http://localhost:8000/api/query/execute").mock(
        return_value=httpx.Response(200, json=payload)
    )

    result = await executor.execute("SELECT * FROM my_table")
    assert result.success is False
    assert result.error == "Table not found"
    assert result.row_count == 0


@pytest.mark.integration
@pytest.mark.asyncio
@respx.mock
async def test_execute_http_error(executor):
    respx.post("http://localhost:8000/api/query/execute").mock(
        return_value=httpx.Response(500, text="Internal Server Error")
    )

    result = await executor.execute("SELECT * FROM my_table")
    assert result.success is False
    assert "HTTP 500" in result.error


@pytest.mark.integration
@pytest.mark.asyncio
@respx.mock
async def test_execute_timeout(executor):
    respx.post("http://localhost:8000/api/query/execute").mock(
        side_effect=httpx.TimeoutException("Timeout")
    )

    result = await executor.execute("SELECT * FROM my_table")
    assert result.success is False
    assert "timed out" in result.error


@pytest.mark.integration
@pytest.mark.asyncio
@respx.mock
async def test_execute_disabled():
    executor = BackendQueryExecutor(
        backend_url="http://localhost:8000",
        enabled=False,
    )
    result = await executor.execute("SELECT * FROM my_table")
    assert result.success is True
    assert result.rows == []
    assert "Query execution disabled" in result.error
