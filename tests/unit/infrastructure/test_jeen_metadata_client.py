"""
test_jeen_metadata_client.py — Jeen run_query / get_columns mapping and retries.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import httpx
import pytest

from src.infrastructure.jeen.jeen_metadata_client import (
    JeenMetadataClient,
    JeenQueryExecutor,
)
from src.infrastructure.mcp.mcp_http_client import McpHttpClient, McpToolResult


def _ok(payload) -> McpToolResult:
    return McpToolResult(is_error=False, text=json.dumps(payload), raw={})


@pytest.fixture
def client() -> JeenMetadataClient:
    c = JeenMetadataClient(url="http://jeen/api/mcp", api_key="k", max_retries=3)
    c._mcp = AsyncMock()
    return c


@pytest.mark.unit
@pytest.mark.asyncio
async def test_run_query_maps_object_rows_by_column_order(client):
    client._mcp.call_tool = AsyncMock(
        return_value=_ok(
            {
                "columns": ["promotion", "lines"],
                "rows": [{"lines": "58247", "promotion": "No discount"}],
                "truncated": True,
            }
        )
    )
    result = await client.run_query(6, "SELECT 1;")

    assert result.success is True
    assert result.columns == ["promotion", "lines"]
    assert result.rows == [["No discount", "58247"]]
    assert result.truncated is True
    args = client._mcp.call_tool.await_args.args
    assert args[0] == "run_query"
    assert args[1] == {"connection_id": 6, "query": "SELECT 1", "max_rows": 1000}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_sql_errors_are_returned_not_raised(client):
    client._mcp.call_tool = AsyncMock(
        return_value=McpToolResult(is_error=True, text='Query failed: relation "x" does not exist', raw={})
    )
    result = await client.run_query(6, "SELECT * FROM x")
    assert result.success is False
    assert "does not exist" in result.error
    assert client._mcp.call_tool.await_count == 1  # SQL errors are not retried


@pytest.mark.unit
@pytest.mark.asyncio
async def test_transport_errors_are_retried(client, monkeypatch):
    monkeypatch.setattr("asyncio.sleep", AsyncMock())
    client._mcp.call_tool = AsyncMock(
        side_effect=[httpx.ConnectError("dns"), _ok({"columns": ["a"], "rows": [{"a": 1}]})]
    )
    result = await client.run_query(6, "SELECT 1")
    assert result.success is True
    assert client._mcp.call_tool.await_count == 2


@pytest.mark.unit
@pytest.mark.asyncio
async def test_schema_map_and_dialect(client):
    async def call_tool(name, arguments):
        if name == "list_connections":
            return _ok({"connections": [{"connection_id": 6, "dialect": "postgresql"}]})
        return _ok(
            {
                "columns": [
                    {"schema": "public", "catalog": None, "table": "FactSales",
                     "column_name": "OrderDate", "data_type": "timestamp without time zone"},
                ]
            }
        )

    client._mcp.call_tool = AsyncMock(side_effect=call_tool)

    assert await client.get_dialect(6) == "postgres"
    schema = await client.get_table_schema_map(6)
    # table keys are lower-case; column names keep their case for quoting
    assert schema["factsales"] == {"OrderDate": "timestamp without time zone"}
    assert schema["public.factsales"] == {"OrderDate": "timestamp without time zone"}

    await client.get_table_schema_map(6)
    assert client._mcp.call_tool.await_count == 2  # cached


@pytest.mark.unit
@pytest.mark.asyncio
async def test_executor_falls_back_without_connection(client):
    fallback = AsyncMock()
    executor = JeenQueryExecutor(client, fallback=fallback)
    await executor.execute("SELECT 1")
    fallback.execute.assert_awaited_once_with("SELECT 1")


@pytest.mark.unit
def test_sse_parsing_skips_notifications():
    body = (
        'event: message\ndata: {"jsonrpc":"2.0","method":"notifications/message","params":{}}\n\n'
        'event: message\ndata: {"jsonrpc":"2.0","id":"req-1","result":{"content":[]}}\n\n'
    )
    messages = McpHttpClient.parse_sse_messages(body)
    assert [m.get("id") for m in messages] == [None, "req-1"]
