"""
test_agent_model_selection.py — the chosen LLM model reaches the agent call.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.application.dto.run_mcp_dataset_request import RunMcpDatasetRequest
from src.application.use_cases.run_mcp_dataset_evaluation_use_case import (
    RunMcpDatasetEvaluationUseCase,
)
from src.config.settings import Settings
from src.infrastructure.text_to_sql_agent.mcp_agent_client import McpAgentClient


def _client(monkeypatch) -> tuple[McpAgentClient, AsyncMock]:
    client = McpAgentClient(Settings())
    call = AsyncMock(return_value={})
    monkeypatch.setattr(client, "_call_mcp", call)
    monkeypatch.setattr(client, "_map_response", lambda result, thread_id: "response")
    return client, call


@pytest.mark.unit
@pytest.mark.asyncio
async def test_chosen_model_is_sent_to_the_agent(monkeypatch):
    client, call = _client(monkeypatch)
    await client.run("How many orders?", [], connection_id=6, model="gpt-oss:20b")
    arguments = call.await_args.args[0]["params"]["arguments"]
    assert arguments["model"] == "gpt-oss:20b"
    assert arguments["connection_id"] == 6


@pytest.mark.unit
@pytest.mark.asyncio
async def test_no_model_keeps_the_agents_default(monkeypatch):
    client, call = _client(monkeypatch)
    await client.run("How many orders?", [], connection_id=6)
    assert "model" not in call.await_args.args[0]["params"]["arguments"]


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("model", "expected"),
    [("gpt-oss:20b", {"connection_id": 6, "model": "gpt-oss:20b"}), (None, {"connection_id": 6})],
)
async def test_use_case_forwards_the_model(model, expected):
    runner = MagicMock()
    runner.run = AsyncMock(side_effect=RuntimeError("stop"))
    use_case = RunMcpDatasetEvaluationUseCase(runner=runner)
    request = RunMcpDatasetRequest(dataset_name="ds", connection_id=6, model=model)
    with pytest.raises(RuntimeError):
        await use_case.execute(request)
    assert runner.run.await_args.kwargs["agent_kwargs"] == expected
