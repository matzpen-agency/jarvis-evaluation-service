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

import asyncio
import uuid
from typing import Any

import langfuse as lf_sdk
import structlog
from langfuse.api.resources.dataset_run_items.types.create_dataset_run_item_request import (
    CreateDatasetRunItemRequest,
)

from src.domain.entities.dataset_run import DatasetRun
from src.domain.entities.evaluation_context import EvaluationContext
from src.domain.entities.evaluation_result import EvaluationResult
from src.domain.entities.query_result import QueryResult
from src.ports.trace_reporter import TraceReporter

logger = structlog.get_logger(__name__)


class LangfuseReporter(TraceReporter):
    """
    Publishes evaluation traces, spans, and scores to Langfuse.

    Gracefully degrades when the Langfuse client is None (disabled).
    """

    def __init__(
        self,
        client: lf_sdk.Langfuse | None,
        link_max_attempts: int = 4,
        link_retry_base_seconds: float = 1.0,
    ) -> None:
        self._client = client
        self._run_spans: dict[str, Any] = {}
        self._link_max_attempts = link_max_attempts
        self._link_retry_base = link_retry_base_seconds
        # Links that still failed after retries; retried once more on flush()
        self._failed_links: list[CreateDatasetRunItemRequest] = []

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

        # Fixed up front so the item can be linked to the dataset run even if
        # building the spans fails — every item must appear in the experiment.
        trace_id = uuid.uuid4().hex
        observation_id: str | None = None
        try:
            # ── Sample-level trace ────────────────────────────────────────────
            sample_output = self._build_sample_output(context, results)
            sample_span = self._client.start_span(  # type: ignore[union-attr]
                name=f"eval-sample:{context.dataset_item.id}",
                trace_context={"trace_id": trace_id},
                input={
                    "query": context.query,
                    "expected_sql": context.expected_sql,
                    "allowed_tables": context.allowed_tables,
                    "connection_id": context.connection_id,
                },
                output=sample_output,
                metadata=self._build_sample_metadata(context),
            )
            observation_id = sample_span.id
            status_tag = "passed" if context.succeeded else "failed"
            sample_span.update_trace(
                tags=["evaluation", "text-to-sql", "sample", status_tag],
                output=sample_output,
            )

            # ── Agent execution span ──────────────────────────────────────────
            agent_span = self._client.start_span(  # type: ignore[union-attr]
                trace_context={"trace_id": trace_id, "parent_span_id": sample_span.id},
                name="agent_execution",
                input={"query": context.query, "allowed_tables": context.allowed_tables},
                output={
                    "sql_query": context.generated_sql,
                    "status": context.agent_response.status if context.agent_response else "error",
                    "error": context.error,
                },
                metadata={
                    "latency_ms": context.agent_latency_ms,
                    "agent_crashed": context.agent_crashed,
                    "timed_out": context.timed_out,
                },
            )
            agent_span.end()

            # ── SQL execution spans ───────────────────────────────────────────
            if context.expected_result is not None:
                exp_span = self._client.start_span(  # type: ignore[union-attr]
                    trace_context={"trace_id": trace_id, "parent_span_id": sample_span.id},
                    name="expected_sql_execution",
                    input={"sql": context.expected_sql},
                    output=self._format_query_result(context.expected_result),
                    metadata={"execution_time_ms": context.expected_result.execution_time_ms},
                )
                exp_span.end()

            if context.generated_result is not None:
                gen_span = self._client.start_span(  # type: ignore[union-attr]
                    trace_context={"trace_id": trace_id, "parent_span_id": sample_span.id},
                    name="generated_sql_execution",
                    input={"sql": context.executed_generated_sql or context.generated_sql},
                    output=self._format_query_result(context.generated_result),
                    metadata={"execution_time_ms": context.generated_result.execution_time_ms},
                )
                gen_span.end()

            # ── Evaluator scores ──────────────────────────────────────────────
            for result in results:
                if not result.applicable:
                    continue  # excluded from averages, so not scored either
                comment = f"passed={result.passed}"
                if result.error:
                    comment = f"evaluator error: {result.error}"[:500]
                elif result.details.get("reason"):
                    comment += f"; reason={result.details['reason']}"
                self._client.create_score(  # type: ignore[union-attr]
                    trace_id=trace_id,
                    name=result.evaluator_name,
                    value=result.score,
                    comment=comment,
                    data_type="NUMERIC",
                )

            sample_span.end()
        except Exception as exc:
            logger.error(
                "langfuse_reporter.report_sample_failed",
                item_id=context.dataset_item.id,
                error=str(exc),
            )

        # ── Link sample trace to dataset run ──────────────────────────────────
        await self._link_to_dataset_run(
            dataset_item_id=dataset_item_id,
            trace_id=trace_id,
            run_name=run_name,
            metadata={"parent_run_trace_id": parent_trace_id},
            observation_id=observation_id,
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
                ("schema_validity", run.accuracy.schema_validity),
                ("dialect_validity", run.accuracy.dialect_validity),
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
        """Flush all pending Langfuse events and retry links that failed."""
        if not self._enabled:
            return
        if self._failed_links:
            pending, self._failed_links = self._failed_links, []
            logger.info("langfuse_reporter.retrying_failed_links", count=len(pending))
            for request in pending:
                if not await self._create_run_item(request):
                    self._failed_links.append(request)
            if self._failed_links:
                logger.error(
                    "langfuse_reporter.links_permanently_failed",
                    item_ids=[r.dataset_item_id for r in self._failed_links],
                )
        try:
            self._client.flush()  # type: ignore[union-attr]
            logger.debug("langfuse_reporter.flushed")
        except Exception as exc:
            logger.warning("langfuse_reporter.flush_failed", error=str(exc))

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _format_query_result(
        result: QueryResult | None, max_rows: int = 100
    ) -> dict[str, Any] | None:
        """Format a QueryResult for Langfuse output, serializing data safely."""
        if result is None:
            return None

        formatted_rows: list[list[Any]] = []
        for row in result.rows[:max_rows]:
            formatted_row: list[Any] = []
            for cell in row:
                if cell is None or isinstance(cell, (int, float, bool, str)):
                    formatted_row.append(cell)
                else:
                    formatted_row.append(str(cell))
            formatted_rows.append(formatted_row)

        data: dict[str, Any] = {
            "success": result.success,
            "columns": result.columns,
            "rows": formatted_rows,
            "row_count": result.row_count,
            "error": result.error,
        }
        if result.truncated:
            data["truncated_by_executor"] = True
        if len(result.rows) > max_rows:
            data["truncated"] = True
            data["total_rows"] = len(result.rows)
        return data

    def _build_sample_output(
        self,
        context: EvaluationContext,
        results: list[EvaluationResult],
    ) -> dict[str, Any]:
        output: dict[str, Any] = {
            "generated_sql": context.generated_sql,
            "expected_sql": context.expected_sql,
            "generated_output": self._format_query_result(context.generated_result),
            "expected_output": self._format_query_result(context.expected_result),
            "succeeded": context.succeeded,
            "error": context.error,
            "scores": {r.evaluator_name: r.score for r in results if r.applicable},
            "evaluations": {
                r.evaluator_name: {
                    "score": r.score,
                    "passed": r.passed,
                    "applicable": r.applicable,
                    "error": r.error,
                    "details": r.details,
                }
                for r in results
            },
        }
        if (
            context.executed_generated_sql
            and context.executed_generated_sql != context.generated_sql
        ):
            output["executed_generated_sql"] = context.executed_generated_sql
        return output

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
            "total_execution_time_ms",
            "generated_sql_transpiled",
            "generated_sql_raw_error",
            "expected_sql_error",
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

    async def _link_to_dataset_run(
        self,
        dataset_item_id: str,
        trace_id: str,
        run_name: str,
        metadata: dict,
        observation_id: str | None = None,
    ) -> None:
        """Link a sample trace to the Langfuse dataset run, retrying on failure."""
        request = CreateDatasetRunItemRequest(
            runName=run_name,
            datasetItemId=dataset_item_id,
            traceId=trace_id,
            observationId=observation_id,
            metadata=metadata,
        )
        if not await self._create_run_item(request):
            self._failed_links.append(request)

    async def _create_run_item(self, request: CreateDatasetRunItemRequest) -> bool:
        for attempt in range(1, self._link_max_attempts + 1):
            try:
                await asyncio.to_thread(
                    self._client.api.dataset_run_items.create,  # type: ignore[union-attr]
                    request=request,
                )
                return True
            except Exception as exc:
                status_code = getattr(exc, "status_code", None)
                permanent = (
                    isinstance(status_code, int)
                    and 400 <= status_code < 500
                    and status_code != 429
                )
                logger.warning(
                    "langfuse_reporter.link_failed",
                    item_id=request.dataset_item_id,
                    attempt=attempt,
                    status_code=status_code,
                    error=str(exc),
                )
                if permanent:
                    # e.g. the item is not in the dataset — retrying won't help
                    return True
                if attempt < self._link_max_attempts:
                    await asyncio.sleep(self._link_retry_base * 2 ** (attempt - 1))
        return False
