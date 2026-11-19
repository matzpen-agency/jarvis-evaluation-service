"""
agent_router.py — Sends each question to the agent the evaluation chose.

agent_kwargs["agent"] names it: "current" (default, the connection-based
agent) or "previous" (the earlier TextToSQL agent).
"""

from __future__ import annotations

from typing import Any

from src.domain.entities.agent_response import AgentResponse
from src.ports.agent_client import AgentClient

CURRENT_AGENT = "current"
PREVIOUS_AGENT = "previous"
AGENTS = (CURRENT_AGENT, PREVIOUS_AGENT)


class AgentRouter(AgentClient):
    def __init__(self, agents: dict[str, AgentClient]) -> None:
        self._agents = agents

    async def run(
        self, query: str, allowed_tables: list[str], **kwargs: Any
    ) -> AgentResponse:
        name = kwargs.pop("agent", None) or CURRENT_AGENT
        agent = self._agents.get(name)
        if agent is None:
            raise ValueError(f"Unknown agent {name!r}; choose one of {sorted(self._agents)}")
        return await agent.run(query=query, allowed_tables=allowed_tables, **kwargs)
