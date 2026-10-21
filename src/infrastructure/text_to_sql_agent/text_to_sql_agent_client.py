"""
text_to_sql_agent_client.py — HTTP client for the Text-to-SQL agent.

Calls POST /api/v1/agent/chat (or /agent/chat) with HITL disabled and maps
the response to the generic AgentResponse entity.
Includes retry backoff for transient MCP 503/502 errors and normalizes table names.
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import structlog

from src.config.settings import Settings
from src.domain.entities.agent_response import AgentResponse
from src.ports.agent_client import AgentClient

logger = structlog.get_logger(__name__)


class AgentError(Exception):
    """Raised when the agent call fails."""


class TextToSqlAgentClient(AgentClient):
    """
    Calls the Text-to-SQL agent REST API.

    Uses httpx.AsyncClient with configurable timeout. HITL is always
    disabled for automated evaluation runs.
    """

    def __init__(self, settings: Settings) -> None:
        self._url = settings.agent_chat_url
        self._timeout = settings.AGENT_TIMEOUT
        self._hitl_enabled = settings.EVALUATION_HITL_ENABLED
        self._max_retries = 3

    async def run(
        self,
        query: str,
        allowed_tables: list[str],
        **kwargs: Any,
    ) -> AgentResponse:
        """Call the agent and return a normalised AgentResponse."""
        payload = {
            "query": query,
            "allowed_tables": allowed_tables,
            "hitl_enabled": self._hitl_enabled,
        }

        logger.debug(
            "agent_client.calling",
            url=self._url,
            query_preview=query[:100],
            table_count=len(allowed_tables),
        )

        last_error: Exception | None = None

        for attempt in range(1, self._max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    response = await client.post(self._url, json=payload)
                    response.raise_for_status()
                    data = response.json()

                return self._map_response(data)

            except httpx.TimeoutException as exc:
                logger.error("agent_client.timeout", url=self._url, attempt=attempt)
                last_error = TimeoutError(f"Agent call timed out after {self._timeout}s")
                break  # Don't retry long timeouts

            except httpx.HTTPStatusError as exc:
                status_code = exc.response.status_code
                error_body = exc.response.text[:300]
                last_error = AgentError(f"Agent returned HTTP {status_code}: {error_body}")

                if status_code in (502, 503, 504) and attempt < self._max_retries:
                    wait_time = attempt * 1.5
                    logger.warning(
                        "agent_client.transient_mcp_error_retine",
                        status=status_code,
                        attempt=attempt,
                        next_retry_in_s=wait_time,
                    )
                    await asyncio.sleep(wait_time)
                    continue

                logger.error("agent_client.http_error", status=status_code, text=error_body)
                raise last_error from exc

            except Exception as exc:
                logger.error("agent_client.unexpected_error", error=str(exc), attempt=attempt)
                last_error = AgentError(f"Agent call failed: {exc}")
                if attempt < self._max_retries:
                    await asyncio.sleep(attempt * 1.5)
                    continue
                raise last_error from exc

        if last_error:
            raise last_error
        raise AgentError("Agent call failed after retries")

    @staticmethod
    def _map_response(data: dict) -> AgentResponse:
        """Map raw JSON response to AgentResponse entity."""
        return AgentResponse(
            thread_id=data.get("thread_id", ""),
            status=data.get("status", "error"),
            sql_query=data.get("sql_query"),
            sql_explanation=data.get("sql_explanation"),
            schema_plan=data.get("schema_plan"),
            summary=data.get("summary"),
            raw_data_ref=data.get("raw_data_ref"),
        )
