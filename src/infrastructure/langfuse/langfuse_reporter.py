"""
langfuse_reporter.py — Langfuse implementation of TraceReporter.

Creates structured traces, spans, and scores in Langfuse for every
evaluation run and per-sample evaluation result.

Trace hierarchy:
  DatasetRun trace (top-level)
    └── Per-sample span
          ├── agent_execution span
          ├── trino_expected_execution span
          ├── trino_generated_execution span
          └── evaluation span (one per evaluator)
                  scores via langfuse.create_score()
"""

from __future__ import annotations

from typing import Any

import langfuse as lf_sdk
import structlog
from langfuse.api.resources.dataset_run_items.types.create_dataset_run_item_request import (
    CreateDatasetRunItemRequest,
)

from src.domain.entities.dataset_run import DatasetRun
from src.domain.entities.evaluation_context import EvaluationContext
from src.domain.entities.evaluation_result import EvaluationResult
from src.ports.trace_reporter import TraceReporter

logger = structlog.get_logger(__name__)


class LangfuseReporter(TraceReporter):
    """
    Publishes evaluation traces, spans, and scores to Langfuse.

    Gracefully degrades when the Langfuse client is None (disabled).
    """

    def __init__(self, client: lf_sdk.Langfuse | None) -> None:
        self._client = client
        self._run_spans: dict[str, Any] = {}

    @property
    def _enabled(self) -> bool:
        return self._client is not None

    async def create_dataset_run_trace(
        self,
        dataset_name: str,
        run_id: str,
        metadata: dict,
    ) -> str:
        """Create the top-level Langfuse trace for the evaluation run."""
        if not self._enabled:
            return run_id

        try:
            trace_id = run_id.replace("-", "")
            span = self._client.start_span(  # type: ignore[union-attr]
                name=f"eval-run:{dataset_name}",
                trace_context={"trace_id": trace_id},
                input={"dataset_name": dataset_name},
                metadata=metadata,
            )
            span.update_trace(tags=["evaluation", "text-to-sql"])
            self._run_spans[trace_id] = span
            logger.info(
                "langfuse_reporter.trace_created",
                trace_id=trace_id,
                dataset_name=dataset_name,
            )
            return trace_id
        except Exception as exc:
            logger.error("langfuse_reporter.create_trace_failed", error=str(exc))
            return run_id

    async def report_sample(
        self,
        context: EvaluationContext,
        results: list[EvaluationResult],
        parent_trace_id: str,
        dataset_item_id: str,
        run_name: str,
    ) -> None:
        """Report a single sample's complete evaluation to Langfuse."""
        if not self._enabled:
            return

        try:
            # ── Sample-level trace ────────────────────────────────────────────
            sample_span = self._client.start_span(  # type: ignore[union-attr]
                name=f"eval-sample:{context.dataset_item.id}",
                input={
                    "query": context.query,
                    "expected_sql": context.expected_sql,
                    "allowed_tables": context.allowed_tables,
                },
                output=self._build_sample_output(context, results),
                metadata=self._build_sample_metadata(context),
            )
            status_tag = "passed" if context.succeeded else "failed"
            sample_span.update_trace(
                tags=["evaluation", "text-to-sql", "sample", status_tag],
                output={
                    "generated_sql": context.generated_sql,
                    "succeeded": context.succeeded,
                    "scores": {r.evaluator_name: r.score for r in results},
                },
            )
            trace_id = sample_span.trace_id

            # ── Agent execution span ──────────────────────────────────────────
            agent_span = self._client.start_span(  # type: ignore[union-attr]
                trace_context={"trace_id": trace_id, "parent_span_id": sample_span.id},
                name="agent_execution",
                input={"query": context.query, "allowed_tables": context.allowed_tables},
                output={
                    "sql_query": context.generated_sql,
                    "status": context.agent_response.status if context.agent_response else "error",
                },
                metadata={
                    "latency_ms": context.agent_latency_ms,
                    "agent_crashed": context.agent_crashed,
                    "timed_out": context.timed_out,
                },
            )
            agent_span.end()

            # ── Trino execution spans ─────────────────────────────────────────
            if context.expected_result is not None:
                exp_span = self._client.start_span(  # type: ignore[union-attr]
                    trace_context={"trace_id": trace_id, "parent_span_id": sample_span.id},
                    name="trino_expected_execution",
                    input={"sql": context.expected_sql},
                    output={
                        "success": context.expected_result.success,
                        "row_count": context.expected_result.row_count,
                        "error": context.expected_result.error,
                    },
                    metadata={"execution_time_ms": context.expected_result.execution_time_ms},
                )
                exp_span.end()

            if context.generated_result is not None:
                gen_span = self._client.start_span(  # type: ignore[union-attr]
                    trace_context={"trace_id": trace_id, "parent_span_id": sample_span.id},
                    name="trino_generated_execution",
                    input={"sql": context.generated_sql},
                    output={
                        "success": context.generated_result.success,
                        "row_count": context.generated_result.row_count,
                        "error": context.generated_result.error,
                    },
                    metadata={"execution_time_ms": context.generated_result.execution_time_ms},
                )
                gen_span.end()

            # ── Evaluator scores ──────────────────────────────────────────────
            for result in results:
                if result.error is None:
                    self._client.create_score(  # type: ignore[union-attr]
                        trace_id=trace_id,
                        name=result.evaluator_name,
                        value=result.score,
                        comment=f"passed={result.passed}",
                        data_type="NUMERIC",
                    )

            sample_span.end()

            # ── Link sample trace to dataset run ──────────────────────────────
            self._link_to_dataset_run(
                dataset_item_id=dataset_item_id,
                trace_id=trace_id,
                run_name=run_name,
                metadata={"parent_run_trace_id": parent_trace_id},
                observation_id=sample_span.id,
            )

        except Exception as exc:
            logger.error(
                "langfuse_reporter.report_sample_failed",
                item_id=context.dataset_item.id,
                error=str(exc),
            )

    async def report_run_summary(self, run: DatasetRun, trace_id: str) -> None:
        """Write aggregate run metrics as scores on the top-level trace."""
        if not self._enabled:
            return

        try:
            summary_scores: list[tuple[str, float]] = [
                ("composite_score", run.accuracy.composite_score),
                ("execution_accuracy", run.accuracy.execution_accuracy),
                ("contains_accuracy", run.accuracy.contains_accuracy),
                ("sql_exact_match", run.accuracy.sql_exact_match),
                ("time_shift_score", run.accuracy.time_shift_score),
                ("component_match", run.accuracy.component_match),
                ("schema_hallucination", run.accuracy.schema_hallucination),
                ("dialect_error", run.accuracy.dialect_error),
                ("failure_rate", run.failure_analysis.failure_rate),
                ("pass_rate", run.passed / run.total_cases if run.total_cases > 0 else 0.0),
                ("p95_latency_ms", run.latency.p95),
                ("average_total_execution_time_ms", run.performance.average_total_execution_time_ms),
                ("average_time_to_first_row_ms", run.performance.average_time_to_first_row_ms),
                ("average_token_usage", run.performance.average_token_usage),
                ("average_iterations", run.iteration_stats.average_iterations),
            ]

            for name, value in summary_scores:
                self._client.create_score(  # type: ignore[union-attr]
                    trace_id=trace_id,
                    name=name,
                    value=value,
                    data_type="NUMERIC",
                )

            # Update trace output with summary and end the root span
            span = self._run_spans.pop(trace_id, None)
            if span is not None:
                span.update(
                    output={
                        "total_cases": run.total_cases,
                        "passed": run.passed,
                        "failed": run.failed,
                        "failure_rate": run.failure_rate,
                        "composite_score": run.accuracy.composite_score,
                        "duration_seconds": run.duration_seconds,
                        "performance": {
                            "average_total_execution_time_ms": run.performance.average_total_execution_time_ms,
                            "average_time_to_first_row_ms": run.performance.average_time_to_first_row_ms,
                            "average_token_usage": run.performance.average_token_usage,
                        },
                        "iterations": {
                            "average_iterations": run.iteration_stats.average_iterations,
                        }
                    }
                )
                span.end()
            else:
                logger.warning("langfuse_reporter.run_span_not_found", trace_id=trace_id)

            logger.info(
                "langfuse_reporter.run_summary_written",
                trace_id=trace_id,
                composite=run.accuracy.composite_score,
            )
        except Exception as exc:
            logger.error("langfuse_reporter.summary_failed", error=str(exc))

    async def flush(self) -> None:
        """Flush all pending Langfuse events."""
        if not self._enabled:
            return
        try:
            self._client.flush()  # type: ignore[union-attr]
            logger.debug("langfuse_reporter.flushed")
        except Exception as exc:
            logger.warning("langfuse_reporter.flush_failed", error=str(exc))

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _build_sample_output(
        self,
        context: EvaluationContext,
        results: list[EvaluationResult],
    ) -> dict[str, Any]:
        return {
            "generated_sql": context.generated_sql,
            "succeeded": context.succeeded,
            "scores": {r.evaluator_name: r.score for r in results},
        }

    def _build_sample_metadata(self, context: EvaluationContext) -> dict[str, Any]:
        meta = {
            "total_latency_ms": context.total_latency_ms,
            "agent_latency_ms": context.agent_latency_ms,
            "agent_crashed": context.agent_crashed,
            "timed_out": context.timed_out,
            "error": context.error,
            "expected_row_count": (
                context.expected_result.row_count if context.expected_result else None
            ),
            "generated_row_count": (
                context.generated_result.row_count if context.generated_result else None
            ),
        }
        for key in [
            "time_to_first_row_ms",
            "total_execution_time_ms",
            "token_usage",
            "refiner_iteration_count",
            "model_version",
            "agent_version",
            "prompt_version",
            "feature_flags",
            "evaluation_config",
        ]:
            if key in context.metadata:
                meta[key] = context.metadata[key]
        return meta

    def _link_to_dataset_run(
        self,
        dataset_item_id: str,
        trace_id: str,
        run_name: str,
        metadata: dict,
        observation_id: str | None = None,
    ) -> None:
        """Link a sample trace to the Langfuse dataset run."""
        try:
            request = CreateDatasetRunItemRequest(
                runName=run_name,
                datasetItemId=dataset_item_id,
                traceId=trace_id,
                observationId=observation_id,
                metadata=metadata,
            )
            self._client.api.dataset_run_items.create(request=request)  # type: ignore[union-attr]
        except Exception as exc:
            logger.warning(
                "langfuse_reporter.link_failed",
                item_id=dataset_item_id,
                error=str(exc),
            )

