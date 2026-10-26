"""
agent_client.py — Abstract interface for calling AI agent backends.

Implementations: TextToSqlAgentClient
Future:         OpenAIAgentClient, LangGraphAgentClient, ...
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from src.domain.entities.agent_response import AgentResponse


class AgentClient(ABC):
    """
    Generic interface for invoking an AI agent with a natural-language query.

    Concrete implementations are responsible for:
      - Transport (HTTP, gRPC, in-process)
      - Authentication
      - Retry / backoff logic
      - Mapping raw responses to AgentResponse
    """

    @abstractmethod
    async def run(
        self,
        query: str,
        allowed_tables: list[str],
        **kwargs: Any,
    ) -> AgentResponse:
        """
        Execute the agent with the given query and table scope.

        Args:
            query: Natural-language question to pass to the agent.
            allowed_tables: List of table names the agent may use.
            **kwargs: Implementation-specific overrides (e.g., thread_id, timeout).

        Returns:
            AgentResponse with generated SQL and metadata.

        Raises:
            AgentError: If the agent call fails after retries.
        """
        ...
