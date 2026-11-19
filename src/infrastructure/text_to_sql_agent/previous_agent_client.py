"""
previous_agent_client.py — The previous Text-to-SQL agent, over MCP.

The earlier agent answers through one MCP tool (PREVIOUS_AGENT_TOOL_NAME,
"TextToSQL") with the arguments:
  - query: the question
  - databases: the databases to look in (PREVIOUS_AGENT_DATABASES)

It works by database rather than by Jeen connection, so the evaluation's
connection only decides where the golden and generated SQL run; the agent
always gets the configured databases. It has no model choice either.

Its answer may be JSON or plain text; the SQL is taken from the first field
that holds it (sql_query, sql, ...), or from a ```sql block in the text.
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Any

import httpx
import structlog

from src.config.settings import Settings
from src.domain.entities.agent_response import AgentResponse
from src.infrastructure.mcp.mcp_http_client import McpHttpClient
from src.ports.agent_client import AgentClient

logger = structlog.get_logger(__name__)

_SQL_KEYS = ("sql_query", "sql", "generated_sql", "query_sql", "SQL", "query")
_NESTED_KEYS = ("result", "data", "output", "response", "answer")
_TEXT_KEYS = ("summary", "answer", "explanation", "message", "text")
_SQL_BLOCK = re.compile(r"```(?:sql)?\s*\n?(.*?)```", re.DOTALL | re.IGNORECASE)
_SQL_START = re.compile(r"^\s*(SELECT|WITH)\b", re.IGNORECASE)


class PreviousAgentClient(AgentClient):
    """Calls the previous agent's TextToSQL tool and maps its answer."""

    def __init__(self, settings: Settings) -> None:
        token = settings.PREVIOUS_AGENT_TOKEN.get_secret_value()
        self._tool = settings.PREVIOUS_AGENT_TOOL_NAME
        self._databases = [
            name.strip()
            for name in settings.PREVIOUS_AGENT_DATABASES.split(",")
            if name.strip()
        ]
        self._mcp = McpHttpClient(
            settings.PREVIOUS_AGENT_MCP_URL,
            headers={"Authorization": f"Bearer {token}"} if token else None,
            timeout=settings.AGENT_TIMEOUT,
            verify=settings.PREVIOUS_AGENT_VERIFY,
        )
        self._max_attempts = 3

    async def run(
        self,
        query: str,
        allowed_tables: list[str],  # Unused: the agent works by database
        **kwargs: Any,
    ) -> AgentResponse:
        thread_id = kwargs.get("thread_id") or str(uuid.uuid4())
        arguments = {"query": query, "databases": self._databases}
        for attempt in range(1, self._max_attempts + 1):
            try:
                result = await self._mcp.call_tool(self._tool, arguments)
                break
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code not in (502, 503, 504) or (
                    attempt == self._max_attempts
                ):
                    raise
                logger.warning(
                    "previous_agent.transient_error_retry",
                    status=exc.response.status_code,
                    attempt=attempt,
                )
        if result.is_error:
            return AgentResponse(
                thread_id=thread_id,
                status="error",
                error=f"agent tool error: {(result.text or 'unknown error')[:1000]}",
            )
        return parse_answer(result.text, thread_id)


def parse_answer(text: str | None, thread_id: str) -> AgentResponse:
    """Map the tool's text (JSON or prose) to an AgentResponse."""
    data: Any = None
    if text:
        try:
            data = json.loads(text)
        except ValueError:
            data = None

    if isinstance(data, dict):
        sql = _find_sql(data)
        error = data.get("error") if isinstance(data.get("error"), str) else None
        summary = _first_text(data)
        status = data.get("status") if isinstance(data.get("status"), str) else None
    else:
        sql = _sql_from_text(text or "")
        error = None
        summary = text
        status = None

    if sql is None and error is None:
        error = "agent returned no SQL"
    return AgentResponse(
        thread_id=thread_id,
        status=status or ("completed" if sql and not error else "error"),
        sql_query=sql,
        summary=summary[:4000] if isinstance(summary, str) else None,
        error=error,
        metadata={"agent": "previous"},
    )


def _find_sql(data: dict, depth: int = 0) -> str | None:
    for key in _SQL_KEYS:
        value = data.get(key)
        if isinstance(value, str) and _looks_like_sql(value):
            return _sql_from_text(value) or value.strip()
    if depth < 3:
        for key in _NESTED_KEYS:
            value = data.get(key)
            if isinstance(value, dict):
                found = _find_sql(value, depth + 1)
                if found:
                    return found
            elif isinstance(value, str):
                found = _sql_from_text(value)
                if found:
                    return found
    return None


def _first_text(data: dict) -> str | None:
    for key in _TEXT_KEYS:
        if isinstance(data.get(key), str):
            return data[key]
    return None


def _looks_like_sql(value: str) -> bool:
    return bool(_SQL_START.match(value) or _SQL_BLOCK.search(value))


def _sql_from_text(text: str) -> str | None:
    """A ```sql block, or the whole text when it is a bare query."""
    block = _SQL_BLOCK.search(text)
    if block and block.group(1).strip():
        return block.group(1).strip()
    if _SQL_START.match(text):
        return text.strip()
    return None
