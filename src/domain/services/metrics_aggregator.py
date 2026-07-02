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

        # ── Performance Stats ─────────────────────────────────────────────────
        total_exec_times = []
        time_to_first_rows = []
        for ctx in contexts:
            if "total_execution_time_ms" in ctx.metadata:
                total_exec_times.append(ctx.metadata["total_execution_time_ms"])
            elif ctx.generated_result and ctx.generated_result.success:
                total_exec_times.append(ctx.generated_result.execution_time_ms)
            else:
                total_exec_times.append(0.0)

            if "time_to_first_row_ms" in ctx.metadata:
                time_to_first_rows.append(ctx.metadata["time_to_first_row_ms"])
            elif ctx.generated_result and ctx.generated_result.success:
                time_to_first_rows.append(ctx.generated_result.execution_time_ms)
            else:
                time_to_first_rows.append(0.0)

        token_usages = [ctx.metadata.get("token_usage", 0) for ctx in contexts]
        total_tokens = sum(token_usages)
        avg_tokens = round(total_tokens / len(token_usages), 2) if token_usages else 0.0
        avg_total_exec_time = round(sum(total_exec_times) / len(total_exec_times), 2) if total_exec_times else 0.0
        avg_time_to_first_row = round(sum(time_to_first_rows) / len(time_to_first_rows), 2) if time_to_first_rows else 0.0

        from src.domain.entities.dataset_run import PerformanceStats
        performance_stats = PerformanceStats(
            average_total_execution_time_ms=avg_total_exec_time,
            average_time_to_first_row_ms=avg_time_to_first_row,
            total_token_usage=total_tokens,
            average_token_usage=avg_tokens,
        )

        # ── Iteration Stats ───────────────────────────────────────────────────
        iterations = [ctx.metadata.get("refiner_iteration_count", 0) for ctx in contexts]
        total_iter = sum(iterations)
        avg_iter = round(total_iter / len(iterations), 2) if iterations else 0.0
        max_iter = max(iterations) if iterations else 0
        iter_dist = {}
        for it in iterations:
            iter_dist[it] = iter_dist.get(it, 0) + 1

        from src.domain.entities.dataset_run import IterationStats
        iteration_stats = IterationStats(
            average_iterations=avg_iter,
            max_iterations=max_iter,
            total_iterations=total_iter,
            iteration_distribution=iter_dist,
        )

        # ── Cases ─────────────────────────────────────────────────────────────
        from src.domain.entities.dataset_run import DatasetCaseResult
        cases = [
            DatasetCaseResult(
                question_id=r.context.dataset_item.id,
                generated_sql=r.context.generated_sql,
                expected_sql=r.context.expected_sql,
                succeeded=r.context.succeeded,
                error=r.context.error,
                scores={res.evaluator_name: res.score for res in r.results},
            )
            for r in records
        ]

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
            iteration_stats=iteration_stats,
            performance=performance_stats,
            langfuse_trace_id=langfuse_trace_id,
            cases=cases,
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
            component_match=_avg("component_match"),
            schema_hallucination=_avg("schema_hallucination"),
            dialect_error=_avg("dialect_error"),
            composite_score=0.0,  # filled in by caller
        )
