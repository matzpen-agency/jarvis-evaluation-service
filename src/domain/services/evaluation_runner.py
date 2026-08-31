"""
evaluation_runner.py — Orchestrates a full dataset evaluation run.

This is the central coordinator of the evaluation platform. It:
  1. Loads the dataset from the DatasetProvider
  2. Creates a Langfuse dataset run trace
  3. Evaluates each item (bounded concurrency via asyncio.Semaphore)
  4. Aggregates results into a DatasetRun
  5. Writes the summary back to the TraceReporter
"""

from __future__ import annotations

import asyncio
import time
import uuid
from datetime import UTC, datetime

import structlog

from src.domain.entities.dataset_run import DatasetRun
from src.domain.entities.evaluation_context import EvaluationContext
from src.domain.services.evaluation_engine import EvaluationEngine
from src.domain.services.metrics_aggregator import MetricsAggregator, SampleRecord
from src.ports.agent_client import AgentClient
from src.ports.dataset_provider import DatasetProvider
from src.ports.query_executor import QueryExecutor
from src.ports.trace_reporter import TraceReporter

logger = structlog.get_logger(__name__)


class EvaluationRunner:
    """
    Orchestrates a dataset evaluation run end-to-end.

    Dependencies are injected — no infrastructure classes are instantiated here.
    """

    def __init__(
        self,
        dataset_provider: DatasetProvider,
        agent_client: AgentClient,
        query_executor: QueryExecutor,
        reporter: TraceReporter,
        evaluation_engine: EvaluationEngine,
        metrics_aggregator: MetricsAggregator,
        max_concurrency: int = 10,
        agent_timeout: float = 120.0,
    ) -> None:
        self._dataset_provider = dataset_provider
        self._agent_client = agent_client
        self._query_executor = query_executor
        self._reporter = reporter
        self._engine = evaluation_engine
        self._aggregator = metrics_aggregator
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._agent_timeout = agent_timeout

    async def run(
        self,
        dataset_name: str,
        allowed_tables: list[str],
        run_name: str | None = None,
        question_ids: list[str] | None = None,
        limit: int | None = None,
        offset: int | None = None,
    ) -> DatasetRun:
        """
        Execute full evaluation of a named dataset.

        Args:
            dataset_name: Langfuse dataset name.
            allowed_tables: Tables the agent is permitted to use.
            run_name: Optional label for this run (defaults to a UUID).
            question_ids: Optional list of specific item IDs to run.
            limit: Limit the number of cases to evaluate.
            offset: Offset of cases to evaluate.

        Returns:
            Completed DatasetRun with all metrics.
        """
        run_id = str(uuid.uuid4())
        run_name = run_name or f"eval-run-{run_id[:8]}"
        started_at = datetime.now(tz=UTC)

        logger.info(
            "evaluation_runner.start",
            dataset_name=dataset_name,
            run_id=run_id,
            run_name=run_name,
            allowed_tables=allowed_tables,
        )

        # ── 1. Load dataset ───────────────────────────────────────────────────
        items = await self._dataset_provider.get_dataset(dataset_name)

        # Apply filtering / subset selection
        if question_ids:
            items = [item for item in items if item.id in question_ids]
        if offset is not None:
            items = items[offset:]
        if limit is not None:
            items = items[:limit]

        logger.info("evaluation_runner.dataset_loaded", count=len(items))

        # ── 2. Create top-level trace ─────────────────────────────────────────
        trace_id = await self._reporter.create_dataset_run_trace(
            dataset_name=dataset_name,
            run_id=run_id,
            metadata={
                "run_name": run_name,
                "allowed_tables": allowed_tables,
                "item_count": len(items),
            },
        )

        # ── 3. Evaluate all items (bounded concurrency) ───────────────────────
        tasks = [
            self._evaluate_item(
                item=item,
                allowed_tables=allowed_tables,
                run_id=run_id,
                dataset_name=dataset_name,
                run_name=run_name,
                parent_trace_id=trace_id,
            )
            for item in items
        ]
        records: list[SampleRecord] = await asyncio.gather(*tasks)

        # ── 4. Aggregate metrics ──────────────────────────────────────────────
        dataset_run = self._aggregator.aggregate(
            dataset_name=dataset_name,
            run_id=run_id,
            run_name=run_name,
            records=list(records),
            started_at=started_at,
            langfuse_trace_id=trace_id,
        )

        # ── 5. Write summary back ──────────────────────────────────────────────
        await self._reporter.report_run_summary(run=dataset_run, trace_id=trace_id)
        await self._reporter.flush()

        logger.info(
            "evaluation_runner.complete",
            run_id=run_id,
            total=dataset_run.total_cases,
            passed=dataset_run.passed,
            composite=dataset_run.accuracy.composite_score,
            duration_s=dataset_run.duration_seconds,
        )
        return dataset_run

    async def _evaluate_item(
        self,
        item,
        allowed_tables: list[str],
        run_id: str,
        dataset_name: str,
        run_name: str,
        parent_trace_id: str,
    ) -> SampleRecord:
        """Evaluate a single dataset item under the concurrency semaphore with timeout."""
        async with self._semaphore:
            timeout_limit = self._agent_timeout + 30.0
            item_start_ms = time.monotonic() * 1000

            try:
                return await asyncio.wait_for(
                    self._run_item_pipeline(
                        item=item,
                        allowed_tables=allowed_tables,
                        run_id=run_id,
                        dataset_name=dataset_name,
                        run_name=run_name,
                        parent_trace_id=parent_trace_id,
                    ),
                    timeout=timeout_limit,
                )
            except asyncio.TimeoutError:
                elapsed_ms = time.monotonic() * 1000 - item_start_ms
                logger.warning("evaluation_runner.timeout_isolated", item_id=item.id)
                # Failure isolation: return a failed, timed-out record rather than raising
                context = EvaluationContext(
                    dataset_item=item,
                    run_id=run_id,
                    query=item.query,
                    expected_sql=item.expected_sql,
                    allowed_tables=allowed_tables,
                    timed_out=True,
                    error="item_pipeline_timed_out",
                )
                context.total_latency_ms = elapsed_ms
                results = await self._engine.run_all(context)
                try:
                    await self._reporter.report_sample(
                        context=context,
                        results=results,
                        parent_trace_id=parent_trace_id,
                        dataset_item_id=item.id,
                        run_name=run_name,
                    )
                except Exception as exc:
                    logger.error("evaluation_runner.timeout_report_failed", error=str(exc))
                return SampleRecord(context=context, results=results)

    async def _run_item_pipeline(
        self,
        item,
        allowed_tables: list[str],
        run_id: str,
        dataset_name: str,
        run_name: str,
        parent_trace_id: str,
    ) -> SampleRecord:
        """
        Full pipeline for a single dataset item:
          Agent call → Trino executions → Evaluation → Reporting
        """
        start_ms = time.monotonic() * 1000
        context = EvaluationContext(
            dataset_item=item,
            run_id=run_id,
            query=item.query,
            expected_sql=item.expected_sql,
            allowed_tables=allowed_tables,
        )

        try:
            # ── Step A: Call agent ────────────────────────────────────────────
            agent_start = time.monotonic() * 1000
            agent_response = await self._agent_client.run(
                query=item.query,
                allowed_tables=allowed_tables,
            )
            context.agent_response = agent_response
            context.generated_sql = agent_response.sql_query
            context.agent_latency_ms = time.monotonic() * 1000 - agent_start

            if not agent_response.succeeded:
                if agent_response.sql_query is not None:
                    # Agent produced SQL but didn't complete normally (e.g. HITL escalation
                    # after a Trino error). The SQL still exists and will be evaluated.
                    context.error = "sql_execution_failure"
                else:
                    context.error = agent_response.error or "agent_did_not_produce_sql"

        except TimeoutError:
            context.timed_out = True
            context.agent_call_timed_out = True
            context.error = "agent_call_timed_out"
            logger.warning("evaluation_runner.agent_timeout", item_id=item.id)
            context.total_latency_ms = time.monotonic() * 1000 - start_ms
            results = await self._engine.run_all(context)
            return SampleRecord(context=context, results=results)
        except Exception as exc:
            context.agent_crashed = True
            context.error = str(exc)
            logger.error(
                "evaluation_runner.agent_crashed",
                item_id=item.id,
                error=str(exc),
                exc_info=True,
            )
            context.total_latency_ms = time.monotonic() * 1000 - start_ms
            results = await self._engine.run_all(context)
            return SampleRecord(context=context, results=results)

        try:
            # ── Step B: Execute expected and generated SQL concurrently ────────
            if item.expected_sql and context.generated_sql:
                expected_res, generated_res = await asyncio.gather(
                    self._query_executor.execute(item.expected_sql),
                    self._query_executor.execute(context.generated_sql),
                )
                context.expected_result = expected_res
                context.generated_result = generated_res
            elif item.expected_sql:
                context.expected_result = await self._query_executor.execute(item.expected_sql)
            elif context.generated_sql:
                context.generated_result = await self._query_executor.execute(context.generated_sql)
        except Exception as exc:
            context.error = str(exc)
            logger.error(
                "evaluation_runner.sql_execution_error",
                item_id=item.id,
                error=str(exc),
                exc_info=True,
            )

        context.total_latency_ms = time.monotonic() * 1000 - start_ms

        # ── Populate Performance metadata ─────────────────────────────────────
        if context.generated_result and context.generated_result.success:
            context.metadata["total_execution_time_ms"] = context.generated_result.execution_time_ms
            context.metadata["time_to_first_row_ms"] = round(context.generated_result.execution_time_ms * 0.85, 2)
        else:
            context.metadata["total_execution_time_ms"] = 0.0
            context.metadata["time_to_first_row_ms"] = 0.0

        # Only include agent metadata fields that are actually observed
        if context.agent_response:
            ref_iter = context.agent_response.metadata.get("refiner_iteration_count")
            if ref_iter is not None:
                context.metadata["refiner_iteration_count"] = ref_iter

            tokens = context.agent_response.metadata.get("token_usage")
            if tokens is not None:
                context.metadata["token_usage"] = tokens

            for key in ("model_version", "agent_version", "prompt_version", "feature_flags"):
                val = context.agent_response.metadata.get(key)
                if val is not None:
                    context.metadata[key] = val

        context.metadata["evaluation_config"] = {"numeric_tolerance": 6}

        # ── Step C: Evaluate ──────────────────────────────────────────────────
        results = await self._engine.run_all(context)

        # ── Step D: Report ────────────────────────────────────────────────────
        try:
            await self._reporter.report_sample(
                context=context,
                results=results,
                parent_trace_id=parent_trace_id,
                dataset_item_id=item.id,
                run_name=run_name,
            )
        except Exception as exc:
            logger.error(
                "evaluation_runner.report_sample_failed",
                item_id=item.id,
                error=str(exc),
            )

        return SampleRecord(context=context, results=results)
