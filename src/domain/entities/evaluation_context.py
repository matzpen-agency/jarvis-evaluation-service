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
    connection_id: int | None = None

    # ── Connection metadata (resolved once per run) ───────────────────────────
    # sqlglot dialect of the connection's database (expected SQL is written in it)
    dialect: str | None = None
    # sqlglot dialect the agent writes its SQL in
    agent_dialect: str = "trino"
    # table → {column: db_type}, keyed by "table", "schema.table" (lower-case)
    schema_map: dict[str, dict[str, str]] = field(default_factory=dict)

    # The form of expected_sql that was actually executed (it differs when it
    # had to be rewritten to run, e.g. money columns cast to numeric)
    executed_expected_sql: str | None = None

    # ── Agent output ──────────────────────────────────────────────────────────
    agent_response: AgentResponse | None = None
    generated_sql: str | None = None
    # The form of generated_sql that was actually executed (it differs when the
    # agent's SQL had to be transpiled to the connection's dialect)
    executed_generated_sql: str | None = None
    # True when generated SQL runs on the agent's own engine (see
    # AgentSqlExecutor) instead of the connection: it stays in agent_dialect
    generated_on_agent_engine: bool = False

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
    agent_call_timed_out: bool = False

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
