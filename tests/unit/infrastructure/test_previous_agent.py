"""
test_previous_agent.py — the previous TextToSQL agent can answer an evaluation.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.application.dto.run_mcp_dataset_request import RunMcpDatasetRequest
from src.application.use_cases.run_mcp_dataset_evaluation_use_case import (
    RunMcpDatasetEvaluationUseCase,
)
from src.config.settings import Settings
from src.infrastructure.mcp.mcp_http_client import McpToolResult
from src.infrastructure.text_to_sql_agent.agent_router import AgentRouter
from src.infrastructure.text_to_sql_agent.previous_agent_client import (
    PreviousAgentClient,
    parse_answer,
)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_calls_texttosql_with_query_and_databases(monkeypatch):
    client = PreviousAgentClient(Settings(PREVIOUS_AGENT_DATABASES="home_command"))
    call = AsyncMock(
        return_value=McpToolResult(
            is_error=False, text='{"sql_query": "SELECT 1"}', raw={}
        )
    )
    monkeypatch.setattr(client._mcp, "call_tool", call)
    response = await client.run(
        "How many orders?", [], connection_id=6, model="ignored"
    )
    assert call.await_args.args == (
        "TextToSQL",
        {"query": "How many orders?", "databases": ["home_command"]},
    )
    assert response.succeeded and response.sql_query == "SELECT 1"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_tool_error_is_an_agent_error(monkeypatch):
    client = PreviousAgentClient(Settings())
    monkeypatch.setattr(
        client._mcp,
        "call_tool",
        AsyncMock(return_value=McpToolResult(is_error=True, text="boom", raw={})),
    )
    response = await client.run("q", [])
    assert response.status == "error" and "boom" in response.error


@pytest.mark.unit
@pytest.mark.parametrize(
    ("text", "sql"),
    [
        ('{"sql": "SELECT a FROM t"}', "SELECT a FROM t"),
        ('{"result": {"generated_sql": "WITH x AS (SELECT 1) SELECT * FROM x"}}',
         "WITH x AS (SELECT 1) SELECT * FROM x"),
        ('{"answer": "Here:\\n```sql\\nSELECT 2\\n```"}', "SELECT 2"),
        ("The query is:\n```sql\nSELECT 3 FROM t\n```\nDone.", "SELECT 3 FROM t"),
        ("SELECT 4", "SELECT 4"),
        ('{"query": "How many?", "sql_query": "SELECT 5"}', "SELECT 5"),
    ],
)
def test_sql_is_found_in_any_answer_shape(text, sql):
    response = parse_answer(text, "t")
    assert response.sql_query == sql
    assert response.succeeded


@pytest.mark.unit
@pytest.mark.parametrize("text", ["I could not answer that.", '{"error": "no table"}', None])
def test_answer_without_sql_fails(text):
    response = parse_answer(text, "t")
    assert response.sql_query is None
    assert not response.succeeded
    assert response.error


@pytest.mark.unit
@pytest.mark.asyncio
async def test_router_sends_each_question_to_the_chosen_agent():
    current, previous = MagicMock(), MagicMock()
    current.run = AsyncMock(return_value="current")
    previous.run = AsyncMock(return_value="previous")
    router = AgentRouter({"current": current, "previous": previous})
    assert await router.run("q", [], connection_id=6) == "current"
    assert await router.run("q", [], connection_id=6, agent="previous") == "previous"
    assert "agent" not in previous.run.await_args.kwargs
    with pytest.raises(ValueError):
        await router.run("q", [], agent="other")


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("agent", "model", "expected"),
    [
        ("previous", "gpt-oss:20b", {"connection_id": 6, "agent": "previous"}),
        ("current", "gpt-oss:20b", {"connection_id": 6, "model": "gpt-oss:20b"}),
        ("current", None, {"connection_id": 6}),
    ],
)
async def test_use_case_forwards_the_agent(agent, model, expected):
    runner = MagicMock()
    runner.run = AsyncMock(side_effect=RuntimeError("stop"))
    use_case = RunMcpDatasetEvaluationUseCase(runner=runner)
    request = RunMcpDatasetRequest(
        dataset_name="ds", connection_id=6, model=model, agent=agent
    )
    with pytest.raises(RuntimeError):
        await use_case.execute(request)
    assert runner.run.await_args.kwargs["agent_kwargs"] == expected
