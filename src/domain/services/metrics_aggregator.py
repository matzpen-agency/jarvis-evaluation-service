"""
metrics_aggregator.py — Aggregates per-item data into dataset-level metrics.

Takes all EvaluationContexts and EvaluationResults from a run and produces
a complete DatasetRun with all statistics populated.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import NamedTuple

import structlog

from src.domain.entities.dataset_run import AccuracyStats, DatasetRun
from src.domain.entities.evaluation_context import EvaluationContext
from src.domain.entities.evaluation_result import EvaluationResult
from src.domain.metrics.composite_score_calculator import (
    CompositeScoreCalculator,
    CompositeWeights,
)
from src.domain.metrics.failure_analyzer import FailureAnalyzer
from src.domain.metrics.latency_calculator import LatencyCalculator

logger = structlog.get_logger(__name__)


class SampleRecord(NamedTuple):
    """Pairing of context and its evaluator results."""

    context: EvaluationContext
    results: list[EvaluationResult]


class MetricsAggregator:
    """
    Computes dataset-level metrics from a collection of per-sample records.

    Delegates to:
      - LatencyCalculator   (p50/p95/p99/avg/min/max)
      - FailureAnalyzer     (failure categorisation)
      - CompositeScoreCalculator (weighted composite score)
    """

    def __init__(
        self,
        weights: CompositeWeights | None = None,
    ) -> None:
        self._latency_calc = LatencyCalculator()
        self._failure_analyzer = FailureAnalyzer()
        self._composite_calc = CompositeScoreCalculator(weights)

    def aggregate(
        self,
        dataset_name: str,
        run_id: str,
        run_name: str,
        records: list[SampleRecord],
        started_at: datetime,
        langfuse_trace_id: str | None = None,
    ) -> DatasetRun:
        """
        Produce a complete DatasetRun from all sample records.
        """
        ended_at = datetime.now(tz=UTC)
        duration = (ended_at - started_at).total_seconds()

        contexts = [r.context for r in records]
        all_results = [r.results for r in records]

        # ── Counts ────────────────────────────────────────────────────────────
        total = len(contexts)
        passed = sum(1 for ctx in contexts if ctx.succeeded)
        failed = total - passed
        failure_rate = round(failed / total, 4) if total > 0 else 0.0

        # ── Latency ───────────────────────────────────────────────────────────
        latencies = [ctx.total_latency_ms for ctx in contexts]
        latency_stats = self._latency_calc.calculate(latencies)

        # ── Accuracy (per-evaluator averages) ─────────────────────────────────
        accuracy = self._compute_accuracy_stats(all_results)

        # ── Composite score ───────────────────────────────────────────────────
        accuracy.composite_score = self._composite_calc.calculate(accuracy)

        # ── Failures ──────────────────────────────────────────────────────────
        failure_analysis = self._failure_analyzer.calculate(contexts)

        logger.info(
            "metrics_aggregator.done",
            dataset_name=dataset_name,
            total=total,
            passed=passed,
            failed=failed,
            composite=accuracy.composite_score,
            duration_s=round(duration, 1),
        )

        return DatasetRun(
            dataset_name=dataset_name,
            run_id=run_id,
            run_name=run_name,
            started_at=started_at,
            ended_at=ended_at,
            duration_seconds=round(duration, 2),
            total_cases=total,
            passed=passed,
            failed=failed,
            failure_rate=failure_rate,
            latency=latency_stats,
            accuracy=accuracy,
            failure_analysis=failure_analysis,
            langfuse_trace_id=langfuse_trace_id,
        )

    def _compute_accuracy_stats(
        self, all_results: list[list[EvaluationResult]]
    ) -> AccuracyStats:
        """Average each evaluator's score across all samples."""

        def _avg(evaluator_name: str) -> float:
            scores = [
                r.score
                for results in all_results
                for r in results
                if r.evaluator_name == evaluator_name
            ]
            return round(sum(scores) / len(scores), 4) if scores else 0.0

        return AccuracyStats(
            execution_accuracy=_avg("execution_accuracy"),
            contains_accuracy=_avg("contains_accuracy"),
            sql_exact_match=_avg("sql_exact_match"),
            time_shift_score=_avg("time_shift"),
            composite_score=0.0,  # filled in by caller
        )
