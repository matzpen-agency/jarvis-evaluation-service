"""
run_dataset_response.py — API response DTO for dataset evaluation results.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class LatencyStatsDTO(BaseModel):
    """Latency distribution in milliseconds."""

    p50: float = 0.0
    p95: float = 0.0
    p99: float = 0.0
    average: float = 0.0
    minimum: float = 0.0
    maximum: float = 0.0
    total_samples: int = 0


class AccuracyStatsDTO(BaseModel):
    """Per-evaluator accuracy scores (0.0-1.0)."""

    execution_accuracy: float = 0.0
    contains_accuracy: float = 0.0
    sql_exact_match: float = 0.0
    time_shift_score: float = 0.0
    composite_score: float = 0.0


class FailureCategoryDTO(BaseModel):
    """Single failure category summary."""

    category: str
    count: int
    rate: float


class FailureAnalysisDTO(BaseModel):
    """Comprehensive failure analysis."""

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
    categories: list[FailureCategoryDTO] = Field(default_factory=list)


class RunDatasetResponse(BaseModel):
    """
    Response body for POST /text-to-sql/evaluation/run-single-dataset.

    Matches the specification schema exactly.
    """

    dataset_name: str
    run_id: str
    total_cases: int
    passed: int
    failed: int
    failure_rate: float
    latency: LatencyStatsDTO
    accuracy: AccuracyStatsDTO
    failure_analysis: FailureAnalysisDTO
    langfuse_trace_id: str | None = None
    duration_seconds: float = 0.0
