"""
test_mcp_handshake.py — the MCP client completes the handshake a strict server
(like the Legacy agent's) needs: initialize, notifications/initialized and
tools/list in one session, then tools/call.
"""

from __future__ import annotations

import json

import httpx
import pytest

from src.infrastructure.mcp.mcp_http_client import McpHttpClient, McpProtocolError

SESSION = "session-123"
PROTOCOL = "2025-06-18"


class StrictServer:
    """Answers only when each request comes in order, in the session."""

    def __init__(self, tools=("TextToSQL",)) -> None:
        self.tools = list(tools)
        self.steps: list[str] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.method == "DELETE":
            self.steps.append("DELETE")
            return httpx.Response(200)
        body = json.loads(request.content)
        method = body["method"]
        self.steps.append(method)
        if method == "initialize":
            return httpx.Response(
                200,
                headers={"mcp-session-id": SESSION},
                json={
                    "jsonrpc": "2.0",
                    "id": body["id"],
                    "result": {"protocolVersion": PROTOCOL},
                },
            )
        if request.headers.get("mcp-session-id") != SESSION:
            return httpx.Response(400, text="Bad Request: No valid session ID provided")
        if request.headers.get("mcp-protocol-version") != PROTOCOL:
            return httpx.Response(400, text="Bad Request: Unsupported protocol version")
        if method == "notifications/initialized":
            return httpx.Response(202)
        expected_before = {
            "tools/list": ["initialize", "notifications/initialized"],
            "tools/call": ["initialize", "notifications/initialized", "tools/list"],
        }[method]
        if self.steps[: len(expected_before)] != expected_before:
            return httpx.Response(400, text=f"Bad Request: {method} out of order")
        if method == "tools/list":
            result = {"tools": [{"name": name} for name in self.tools]}
        else:
            result = {
                "content": [{"type": "text", "text": '{"sql_query": "SELECT 1"}'}]
            }
        sse = f"event: message\ndata: {json.dumps({'jsonrpc': '2.0', 'id': body['id'], 'result': result})}\n\n"
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, text=sse
        )


@pytest.fixture
def strict(monkeypatch):
    def use(server: StrictServer) -> None:
        real = httpx.AsyncClient

        def client(*args, **kwargs):
            kwargs.pop("verify", None)
            return real(*args, transport=httpx.MockTransport(server), **kwargs)

        monkeypatch.setattr(httpx, "AsyncClient", client)

    return use


@pytest.mark.unit
@pytest.mark.asyncio
async def test_the_full_handshake_comes_before_the_call(strict):
    server = StrictServer()
    strict(server)
    client = McpHttpClient("https://agent.test/mcp", list_tools_first=True)
    result = await client.call_tool(
        "TextToSQL", {"query": "q", "databases": ["home_command"]}
    )
    assert not result.is_error and result.json() == {"sql_query": "SELECT 1"}
    assert server.steps == [
        "initialize",
        "notifications/initialized",
        "tools/list",
        "tools/call",
        "DELETE",
    ]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_without_tools_list_a_strict_server_refuses_the_call(strict):
    strict(StrictServer())
    client = McpHttpClient("https://agent.test/mcp")
    with pytest.raises(httpx.HTTPStatusError):
        await client.call_tool("TextToSQL", {"query": "q"})


@pytest.mark.unit
@pytest.mark.asyncio
async def test_a_missing_tool_names_the_tools_the_server_has(strict):
    strict(StrictServer(tools=["Other", "Another"]))
    client = McpHttpClient("https://agent.test/mcp", list_tools_first=True)
    with pytest.raises(McpProtocolError, match=r"no tool 'TextToSQL'.*Other, Another"):
        await client.call_tool("TextToSQL", {"query": "q"})


@pytest.mark.unit
@pytest.mark.asyncio
async def test_list_tools(strict):
    strict(StrictServer(tools=["TextToSQL", "Other"]))
    assert await McpHttpClient("https://agent.test/mcp").list_tools() == [
        "TextToSQL",
        "Other",
    ]


@pytest.mark.unit
def test_the_legacy_agent_lists_tools_first():
    from src.config.settings import Settings
    from src.infrastructure.text_to_sql_agent.previous_agent_client import (
        PreviousAgentClient,
    )

    assert PreviousAgentClient(Settings())._mcp._list_tools_first is True
