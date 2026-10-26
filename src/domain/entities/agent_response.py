"""
agent_response.py — Domain entity for Text-to-SQL agent responses.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AgentResponse:
    """Normalised response from any agent backend."""

    thread_id: str
    status: str  # "completed" | "interrupted" | "error"
    sql_query: str | None = None
    sql_explanation: str | None = None
    schema_plan: str | None = None
    summary: str | None = None
    raw_data_ref: str | None = None
    error: str | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def succeeded(self) -> bool:
        """True when agent completed without error and produced SQL."""
        return self.status == "completed" and self.sql_query is not None

    @property
    def was_interrupted(self) -> bool:
        """True when agent requested human-in-the-loop approval."""
        return self.status == "interrupted"
