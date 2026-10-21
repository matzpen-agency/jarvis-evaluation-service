"""
mcp_http_client.py — Minimal MCP client for the Streamable HTTP transport.

Each call opens a short-lived session:
  1. POST initialize                → InitializeResult (+ mcp-session-id header)
  2. POST notifications/initialized → 202, empty body
  3. POST tools/call                → JSON body or an SSE stream of messages
  4. DELETE the session (best effort)

Responses may arrive as a single JSON body or as SSE events; servers can emit
notifications before the result, so the result is matched by JSON-RPC id.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from typing import Any

import httpx
import structlog

logger = structlog.get_logger(__name__)

_PROTOCOL_VERSION = "2025-03-26"


class McpProtocolError(Exception):
    """Raised when the server response is not a usable JSON-RPC message."""


@dataclass
class McpToolResult:
    """Outcome of a tools/call request."""

    is_error: bool
    text: str | None
    raw: dict[str, Any]

    def json(self) -> Any:
        """Parse the text content as JSON (raises ValueError if it isn't)."""
        if self.text is None:
            raise ValueError("tool result has no text content")
        return json.loads(self.text)


class McpHttpClient:
    """Calls tools on an MCP server exposed over Streamable HTTP."""

    def __init__(
        self,
        url: str,
        headers: dict[str, str] | None = None,
        timeout: float = 60.0,
        client_name: str = "jarvis-evaluation-service",
    ) -> None:
        self._url = url
        self._timeout = timeout
        self._client_name = client_name
        self._headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            **(headers or {}),
        }

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> McpToolResult:
        """
        Call one tool and return its result.

        Tool-level failures (isError=true) and JSON-RPC errors (e.g. argument
        validation) are returned with is_error=True. Transport failures raise
        httpx exceptions so callers can decide whether to retry.
        """
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            session_headers = await self._initialize(client)
            try:
                request_id = str(uuid.uuid4())
                response = await client.post(
                    self._url,
                    json={
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "method": "tools/call",
                        "params": {"name": name, "arguments": arguments},
                    },
                    headers=session_headers,
                )
                response.raise_for_status()
                message = self._extract_message(response, request_id)
            finally:
                await self._close_session(client, session_headers)

        return self._to_tool_result(message)

    async def _initialize(self, client: httpx.AsyncClient) -> dict[str, str]:
        init_id = str(uuid.uuid4())
        response = await client.post(
            self._url,
            json={
                "jsonrpc": "2.0",
                "id": init_id,
                "method": "initialize",
                "params": {
                    "protocolVersion": _PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {"name": self._client_name, "version": "1.0"},
                },
            },
            headers=self._headers,
        )
        response.raise_for_status()
        message = self._extract_message(response, init_id)
        if "error" in message:
            raise McpProtocolError(f"MCP initialize failed: {message['error']}")

        session_id = response.headers.get("mcp-session-id")
        session_headers = (
            {**self._headers, "mcp-session-id": session_id}
            if session_id
            else dict(self._headers)
        )
        notified = await client.post(
            self._url,
            json={"jsonrpc": "2.0", "method": "notifications/initialized"},
            headers=session_headers,
        )
        notified.raise_for_status()
        return session_headers

    async def _close_session(
        self, client: httpx.AsyncClient, session_headers: dict[str, str]
    ) -> None:
        if "mcp-session-id" not in session_headers:
            return
        try:
            await client.delete(self._url, headers=session_headers)
        except Exception as exc:  # stateless servers may not support DELETE
            logger.debug("mcp_http_client.close_session_failed", error=str(exc))

    @classmethod
    def _extract_message(cls, response: httpx.Response, request_id: str) -> dict:
        """Return the JSON-RPC message answering request_id."""
        content_type = response.headers.get("content-type", "")
        if "text/event-stream" in content_type:
            messages = cls.parse_sse_messages(response.text)
        else:
            try:
                body = response.json()
            except ValueError as exc:
                raise McpProtocolError(
                    f"Non-JSON MCP response: {response.text[:200]}"
                ) from exc
            messages = body if isinstance(body, list) else [body]

        for message in messages:
            if isinstance(message, dict) and message.get("id") == request_id:
                return message
        # Some servers don't echo the id; fall back to the first response-shaped message
        for message in messages:
            if isinstance(message, dict) and ("result" in message or "error" in message):
                return message
        raise McpProtocolError("No JSON-RPC response found in MCP reply")

    @staticmethod
    def parse_sse_messages(text: str) -> list[dict]:
        """Parse every `data:` event of an SSE body into JSON objects."""
        messages: list[dict] = []
        data_lines: list[str] = []

        def _flush() -> None:
            if not data_lines:
                return
            payload = "\n".join(data_lines).strip()
            data_lines.clear()
            if not payload or payload == "[DONE]":
                return
            try:
                parsed = json.loads(payload)
            except json.JSONDecodeError:
                return
            if isinstance(parsed, dict):
                messages.append(parsed)

        for line in text.splitlines():
            if line.startswith("data:"):
                data_lines.append(line[len("data:"):].strip())
            elif not line.strip():
                _flush()
        _flush()
        return messages

    @staticmethod
    def _to_tool_result(message: dict) -> McpToolResult:
        if "error" in message:
            error = message.get("error") or {}
            text = error.get("message") if isinstance(error, dict) else str(error)
            return McpToolResult(is_error=True, text=text, raw=message)

        result = message.get("result") or {}
        text: str | None = None
        for block in result.get("content") or []:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text")
                break
        return McpToolResult(
            is_error=bool(result.get("isError")), text=text, raw=message
        )
