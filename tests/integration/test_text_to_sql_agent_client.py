"""
test_text_to_sql_agent_client.py — Integration tests for TextToSqlAgentClient.
"""

from __future__ import annotations

import httpx
import pytest
import respx

from src.config.settings import Settings
from src.infrastructure.text_to_sql_agent.text_to_sql_agent_client import (
    AgentError,
    TextToSqlAgentClient,
)


@pytest.fixture
def agent_settings() -> Settings:
    return Settings(
        AGENT_URL="http://localhost:8001",
        AGENT_ENDPOINT="/api/agent/chat",
        AGENT_TIMEOUT=1.0,
    )


@pytest.fixture
def client(agent_settings) -> TextToSqlAgentClient:
    return TextToSqlAgentClient(settings=agent_settings)


@pytest.mark.integration
@pytest.mark.asyncio
@respx.mock
async def test_run_success(client, agent_settings):
    payload = {
        "thread_id": "t-1",
        "status": "completed",
        "sql_query": "SELECT * FROM orders",
        "sql_explanation": "returns all orders",
    }
    respx.post(agent_settings.agent_chat_url).mock(
        return_value=httpx.Response(200, json=payload)
    )

    response = await client.run(query="get orders", allowed_tables=["orders"])
    assert response.succeeded is True
    assert response.sql_query == "SELECT * FROM orders"
    assert response.thread_id == "t-1"


@pytest.mark.integration
@pytest.mark.asyncio
@respx.mock
async def test_run_http_error(client, agent_settings):
    respx.post(agent_settings.agent_chat_url).mock(
        return_value=httpx.Response(500, text="Internal Server Error")
    )

    with pytest.raises(AgentError, match="HTTP 500"):
        await client.run(query="get orders", allowed_tables=["orders"])


@pytest.mark.integration
@pytest.mark.asyncio
@respx.mock
async def test_run_timeout(client, agent_settings):
    respx.post(agent_settings.agent_chat_url).mock(
        side_effect=httpx.TimeoutException("Timeout")
    )

    with pytest.raises(TimeoutError, match="timed out"):
        await client.run(query="get orders", allowed_tables=["orders"])
