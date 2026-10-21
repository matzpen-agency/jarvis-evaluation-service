"""
mcp_agent_client.py — HTTP client for the updated Text-to-SQL agent MCP server.

The new agent exposes a FastMCP server at /sse (Server-Sent Events transport).
We call the `chat_with_agent` tool, which accepts:
  - connection_id: int  (replaces allowed_tables — the agent resolves tables itself)
  - query: str
  - hitl_enabled: bool  (always False for automated evaluation)

Maps the JSON response to the shared AgentResponse entity so the rest of the
evaluation pipeline remains unchanged.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any

import httpx
import structlog

from src.config.settings import Settings
from src.domain.entities.agent_response import AgentResponse
from src.infrastructure.mcp.mcp_http_client import McpHttpClient
from src.ports.agent_client import AgentClient

logger = structlog.get_logger(__name__)


class McpAgentError(Exception):
    """Raised when the MCP agent call fails."""


class McpAgentClient(AgentClient):
    """
    Calls the text2sql-onboarding agent via its MCP JSON-RPC over HTTP/SSE.

    The MCP server (FastMCP) handles two transports:
      - stdio  (local dev)
      - SSE    (HTTP — what we use here)

    FastMCP exposes:
      POST /messages  — send a JSON-RPC request
      GET  /sse       — receive responses as SSE events

    For evaluation we use a simple synchronous-style request:
      1. POST /messages with a `tools/call` JSON-RPC payload
      2. Collect the first non-ping SSE event that carries the result
    """

    def __init__(self, settings: Settings) -> None:
        self._base_url = settings.MCP_AGENT_URL.rstrip("/")
        self._connection_id = settings.MCP_AGENT_CONNECTION_ID
        self._timeout = settings.AGENT_TIMEOUT
        self._max_retries = 3

    async def run(
        self,
        query: str,
        allowed_tables: list[str],  # Unused — connection_id replaces it
        **kwargs: Any,
    ) -> AgentResponse:
        """Call the MCP agent and return a normalised AgentResponse."""
        thread_id = kwargs.get("thread_id") or str(uuid.uuid4())

        connection_id = kwargs.get("connection_id")
        if connection_id is None:
            connection_id = self._connection_id

        tool_arguments = {
            "connection_id": connection_id,
            "query": query,
            "hitl_enabled": False,
            "thread_id": thread_id,
        }

        payload = {
            "jsonrpc": "2.0",
            "id": thread_id,
            "method": "tools/call",
            "params": {
                "name": "chat_with_agent",
                "arguments": tool_arguments,
            },
        }

        logger.debug(
            "mcp_agent_client.calling",
            base_url=self._base_url,
            query_preview=query[:100],
            connection_id=connection_id,
        )

        last_error: Exception | None = None

        for attempt in range(1, self._max_retries + 1):
            try:
                result = await self._call_mcp(payload)
                return self._map_response(result, thread_id)

            except httpx.TimeoutException:
                logger.error(
                    "mcp_agent_client.timeout",
                    base_url=self._base_url,
                    attempt=attempt,
                )
                last_error = TimeoutError(
                    f"MCP agent call timed out after {self._timeout}s"
                )
                break  # Don't retry long timeouts

            except httpx.HTTPStatusError as exc:
                status_code = exc.response.status_code
                error_body = exc.response.text[:300]
                last_error = McpAgentError(
                    f"MCP agent returned HTTP {status_code}: {error_body}"
                )
                if status_code in (502, 503, 504) and attempt < self._max_retries:
                    wait = attempt * 1.5
                    logger.warning(
                        "mcp_agent_client.transient_error_retry",
                        status=status_code,
                        attempt=attempt,
                        next_retry_in_s=wait,
                    )
                    await asyncio.sleep(wait)
                    continue

                logger.error(
                    "mcp_agent_client.http_error",
                    status=status_code,
                    text=error_body,
                )
                raise last_error from exc

            except Exception as exc:
                logger.error(
                    "mcp_agent_client.unexpected_error",
                    error=str(exc),
                    attempt=attempt,
                )
                last_error = McpAgentError(f"MCP agent call failed: {exc}")
                if attempt < self._max_retries:
                    await asyncio.sleep(attempt * 1.5)
                    continue
                raise last_error from exc

        if last_error:
            raise last_error
        raise McpAgentError("MCP agent call failed after retries")

    async def _call_mcp(self, payload: dict) -> dict:
        """
        POST the JSON-RPC payload to /messages and collect the SSE response.

        FastMCP (SSE transport) works as follows:
          1. Client connects to GET /sse — receives a session endpoint URL.
          2. Client POSTs JSON-RPC messages to that session URL.
          3. Responses are streamed back on the SSE connection.

        For simplicity (and to avoid managing a persistent SSE connection per
        evaluation item), we use the simpler streamable HTTP transport that
        FastMCP also supports: POST directly to /mcp and receive the result
        as a JSON response body (available in FastMCP >= 2.x with
        transport="streamable-http").

        If the agent is running the older SSE-only mode, set
        MCP_AGENT_USE_SSE=true and we fall back to the two-step SSE approach.
        """
        messages_url = f"{self._base_url}/mcp"
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            # 1. Initialize session with FastMCP Streamable HTTP
            init_payload = {
                "jsonrpc": "2.0",
                "id": "init-1",
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "eval-client", "version": "1.0"},
                },
            }
            init_res = await client.post(messages_url, json=init_payload, headers=headers)
            init_res.raise_for_status()

            session_id = init_res.headers.get("mcp-session-id")
            session_headers = (
                {**headers, "mcp-session-id": session_id} if session_id else headers
            )

            # 2. Send initialized notification
            await client.post(
                messages_url,
                json={"jsonrpc": "2.0", "method": "notifications/initialized"},
                headers=session_headers,
            )

            # 3. Call tool
            response = await client.post(
                messages_url, json=payload, headers=session_headers
            )
            response.raise_for_status()

            content_type = response.headers.get("content-type", "")
            if "text/event-stream" in content_type:
                # SSE fallback: parse the first data event
                return self._parse_sse_response(response.text)
            else:
                return response.json()

    @staticmethod
    def _parse_sse_response(text: str) -> dict:
        """Extract the JSON-RPC response from a raw SSE body (skipping notifications)."""
        messages = McpHttpClient.parse_sse_messages(text)
        for message in messages:
            if "result" in message or "error" in message:
                return message
        raise McpAgentError("No valid data event found in SSE response")

    @staticmethod
    def _map_response(raw: dict, thread_id: str) -> AgentResponse:
        """Map MCP JSON-RPC response to AgentResponse entity."""
        # JSON-RPC wraps the result under 'result.content[0].text' (FastMCP convention)
        result = raw.get("result", raw)
        content = result.get("content", [])
        text_content: str | None = None
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text_content = block.get("text")
                break

        if result.get("isError"):
            # The tool raised: FastMCP returns the exception text, not JSON
            return AgentResponse(
                thread_id=thread_id,
                status="error",
                error=f"agent tool error: {(text_content or 'unknown error')[:1000]}",
            )

        if text_content:
            try:
                data = json.loads(text_content)
            except (json.JSONDecodeError, TypeError):
                data = {}
        else:
            # Some MCP servers return result directly
            data = result if isinstance(result, dict) else {}

        error_msg = data.get("error") or (raw.get("error") or {}).get("message")
        status = data.get("status", "completed")
        succeeded = status == "completed" and not error_msg

        return AgentResponse(
            thread_id=data.get("thread_id", thread_id),
            status=status,
            sql_query=data.get("sql_query"),
            sql_explanation=data.get("sql_explanation"),
            schema_plan=data.get("schema_plan"),
            summary=data.get("summary"),
            raw_data_ref=data.get("raw_data_ref"),
            error=error_msg,
            metadata={
                "execution_path": data.get("execution_path", []),
                "is_unanswerable": data.get("is_unanswerable", False),
                "trace_id": data.get("trace_id"),
            },
        )
