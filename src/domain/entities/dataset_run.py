"""
dataset_run.py — Aggregate result of a full dataset evaluation run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class LatencyStats:
    """Latency distribution statistics in milliseconds."""

    p50: float = 0.0
    p95: float = 0.0
    p99: float = 0.0
    average: float = 0.0
    minimum: float = 0.0
    maximum: float = 0.0
    total_samples: int = 0


@dataclass
class AccuracyStats:
    """Per-evaluator accuracy scores (0.0-1.0)."""

    execution_accuracy: float = 0.0
    contains_accuracy: float = 0.0
    sql_exact_match: float = 0.0
    time_shift_score: float = 0.0
    component_match: float = 0.0
    schema_hallucination: float = 0.0
    dialect_error: float = 0.0
    composite_score: float = 0.0


@dataclass
class FailureCategory:
    """Breakdown of a specific failure category."""

    category: str
    count: int
    rate: float
    timestamps: list[datetime] = field(default_factory=list)


@dataclass
class FailureAnalysis:
    """Comprehensive failure breakdown for the run."""

    total_failures: int = 0
    failure_rate: float = 0.0
    agent_crash_count: int = 0
    agent_crash_rate: float = 0.0
    sql_execution_failure_count: int = 0
    sql_execution_failure_rate: float = 0.0
    trino_failure_count: int = 0
    trino_failure_rate: float = 0.0
    timeout_count: int = 0
    timeout_rate: float = 0.0
    validation_failure_count: int = 0
    validation_failure_rate: float = 0.0
    categories: list[FailureCategory] = field(default_factory=list)
    failure_timestamps: list[datetime] = field(default_factory=list)
    retry_count: int = 0
    retry_success_rate: float = 0.0


@dataclass
class IterationStats:
    """Statistics about agent graph iterations (for future LangGraph metadata)."""

    average_iterations: float = 0.0
    max_iterations: int = 0
    total_iterations: int = 0
    iteration_distribution: dict[int, int] = field(default_factory=dict)
    loop_counts_by_node: dict[str, int] = field(default_factory=dict)


@dataclass
class PerformanceStats:
    """Performance statistics collected during run."""

    average_total_execution_time_ms: float = 0.0
    average_time_to_first_row_ms: float = 0.0
    total_token_usage: int = 0
    average_token_usage: float = 0.0


@dataclass
class DatasetRun:
    """
    Complete result of evaluating a dataset against an AI agent.

    This is the canonical aggregate output of EvaluationRunner.run().
    It is used to:
      - Build the API response
      - Write summary metrics to Langfuse
    """

    # ── Identity ──────────────────────────────────────────────────────────────
    dataset_name: str
    run_id: str
    run_name: str

    # ── Timing ────────────────────────────────────────────────────────────────
    started_at: datetime
    ended_at: datetime | None = None
    duration_seconds: float = 0.0

    # ── Counts ────────────────────────────────────────────────────────────────
    total_cases: int = 0
    passed: int = 0
    failed: int = 0
    failure_rate: float = 0.0

    # ── Metrics ───────────────────────────────────────────────────────────────
    latency: LatencyStats = field(default_factory=LatencyStats)
    accuracy: AccuracyStats = field(default_factory=AccuracyStats)
    failure_analysis: FailureAnalysis = field(default_factory=FailureAnalysis)
    iteration_stats: IterationStats = field(default_factory=IterationStats)
    performance: PerformanceStats = field(default_factory=PerformanceStats)

    # ── Tracing ───────────────────────────────────────────────────────────────
    langfuse_trace_id: str | None = None
