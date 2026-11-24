"""
test_agent_choice.py — the previous TextToSQL agent and the Deep Agent can answer an evaluation.
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
@pytest.mark.asyncio
async def test_a_missing_ca_file_says_which_setting_and_path(monkeypatch):
    client = PreviousAgentClient(Settings(PREVIOUS_AGENT_VERIFY="/no/such/ca.crt"))
    call = AsyncMock()
    monkeypatch.setattr(client._mcp, "call_tool", call)
    response = await client.run("q", [])
    assert response.status == "error"
    assert "PREVIOUS_AGENT_VERIFY" in response.error
    assert "/no/such/ca.crt" in response.error
    call.assert_not_awaited()


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("verify", "expected"),
    [("true", True), ("false", False), (" False ", False), ("0", False)],
)
async def test_true_or_false_verify_is_not_a_file(monkeypatch, verify, expected):
    monkeypatch.setenv("PREVIOUS_AGENT_VERIFY", verify)
    settings = Settings()
    assert settings.PREVIOUS_AGENT_VERIFY is expected
    client = PreviousAgentClient(settings)
    assert client._mcp._verify is expected
    call = AsyncMock(
        return_value=McpToolResult(is_error=False, text="SELECT 1", raw={})
    )
    monkeypatch.setattr(client._mcp, "call_tool", call)
    assert (await client.run("q", [])).succeeded
    call.assert_awaited_once()


@pytest.mark.unit
@pytest.mark.parametrize(
    ("text", "sql"),
    [
        ('{"sql": "SELECT a FROM t"}', "SELECT a FROM t"),
        (
            '{"result": {"generated_sql": "WITH x AS (SELECT 1) SELECT * FROM x"}}',
            "WITH x AS (SELECT 1) SELECT * FROM x",
        ),
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
@pytest.mark.parametrize(
    "text", ["I could not answer that.", '{"error": "no table"}', None]
)
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


# ── The Deep Agent (deep_agent tool on the current agent's MCP server) ──────


def _deep_client(monkeypatch):
    from src.infrastructure.text_to_sql_agent.mcp_agent_client import McpAgentClient

    client = McpAgentClient.deep_agent(Settings(AGENT_TIMEOUT=300))
    call = AsyncMock(
        return_value={
            "result": {
                "content": [
                    {
                        "type": "text",
                        "text": '{"status": "completed", "sql_query": "SELECT 1", '
                        '"engine": "deep", "verification": "matched", '
                        '"confidence": "high", "executed": true}',
                    }
                ]
            }
        }
    )
    monkeypatch.setattr(client, "_call_mcp", call)
    return client, call


@pytest.mark.unit
@pytest.mark.asyncio
async def test_deep_agent_calls_its_own_tool(monkeypatch):
    client, call = _deep_client(monkeypatch)
    response = await client.run("How many orders?", [], connection_id=6)
    params = call.await_args.args[0]["params"]
    assert params["name"] == "deep_agent"
    arguments = params["arguments"]
    # its threads must start with deep-, and it stops before our timeout
    assert arguments["thread_id"].startswith("deep-")
    assert arguments["timeout_s"] == 290
    assert arguments["connection_id"] == 6 and arguments["hitl_enabled"] is False
    assert "model" not in arguments
    assert response.succeeded and response.sql_query == "SELECT 1"
    assert response.metadata["verification"] == "matched"
    assert response.metadata["confidence"] == "high"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_deep_agent_gets_the_chosen_model(monkeypatch):
    client, call = _deep_client(monkeypatch)
    await client.run("q", [], connection_id=6, model="gpt-oss:20b")
    assert call.await_args.args[0]["params"]["arguments"]["model"] == "gpt-oss:20b"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_use_case_forwards_the_deep_agent_and_its_model():
    runner = MagicMock()
    runner.run = AsyncMock(side_effect=RuntimeError("stop"))
    use_case = RunMcpDatasetEvaluationUseCase(runner=runner)
    request = RunMcpDatasetRequest(
        dataset_name="ds", connection_id=6, model="gpt-oss:20b", agent="deep"
    )
    with pytest.raises(RuntimeError):
        await use_case.execute(request)
    assert runner.run.await_args.kwargs["agent_kwargs"] == {
        "connection_id": 6,
        "model": "gpt-oss:20b",
        "agent": "deep",
    }
