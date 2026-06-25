"""
evaluation_context.py — Snapshot of all data available during evaluation.

EvaluationContext is the single object passed to every BaseEvaluator.
It contains everything an evaluator may need, enabling evaluators to be
stateless and independently testable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.domain.entities.agent_response import AgentResponse
from src.domain.entities.dataset_item import DatasetItem
from src.domain.entities.query_result import QueryResult


@dataclass
class EvaluationContext:
    """
    Immutable snapshot of all data for a single evaluation sample.

    Passed to every BaseEvaluator. Evaluators must NOT mutate this object.
    """

    # ── Identity ──────────────────────────────────────────────────────────────
    dataset_item: DatasetItem
    run_id: str

    # ── Inputs ────────────────────────────────────────────────────────────────
    query: str
    expected_sql: str
    allowed_tables: list[str]

    # ── Agent output ──────────────────────────────────────────────────────────
    agent_response: AgentResponse | None = None
    generated_sql: str | None = None

    # ── Execution results ─────────────────────────────────────────────────────
    expected_result: QueryResult | None = None
    generated_result: QueryResult | None = None

    # ── Timing ────────────────────────────────────────────────────────────────
    agent_latency_ms: float = 0.0
    total_latency_ms: float = 0.0

    # ── Error state ───────────────────────────────────────────────────────────
    error: str | None = None
    agent_crashed: bool = False
    timed_out: bool = False

    # ── Extensible metadata ───────────────────────────────────────────────────
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def succeeded(self) -> bool:
        """True when agent produced SQL and both Trino executions succeeded."""
        return (
            self.agent_response is not None
            and self.agent_response.succeeded
            and self.expected_result is not None
            and self.expected_result.success
            and self.generated_result is not None
            and self.generated_result.success
        )

    @property
    def failed(self) -> bool:
        """True when any required step failed."""
        return not self.succeeded
