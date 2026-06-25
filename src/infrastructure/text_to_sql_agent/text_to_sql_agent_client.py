"""
text_to_sql_agent_client.py — HTTP client for the Text-to-SQL agent.

Calls POST /api/v1/agent/chat with HITL disabled and maps the response
to the generic AgentResponse entity.
"""

from __future__ import annotations

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

    Future: Add retry/backoff logic here without changing the AgentClient interface.
    """

    def __init__(self, settings: Settings) -> None:
        self._url = settings.agent_chat_url
        self._timeout = settings.AGENT_TIMEOUT
        self._hitl_enabled = settings.EVALUATION_HITL_ENABLED

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

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(self._url, json=payload)
                response.raise_for_status()
                data = response.json()

            return self._map_response(data)

        except httpx.TimeoutException as exc:
            logger.error("agent_client.timeout", url=self._url)
            raise TimeoutError(f"Agent call timed out after {self._timeout}s") from exc

        except httpx.HTTPStatusError as exc:
            error_msg = f"Agent returned HTTP {exc.response.status_code}: {exc.response.text[:200]}"
            logger.error("agent_client.http_error", status=exc.response.status_code)
            raise AgentError(error_msg) from exc

        except Exception as exc:
            logger.error("agent_client.unexpected_error", error=str(exc))
            raise AgentError(f"Agent call failed: {exc}") from exc

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
