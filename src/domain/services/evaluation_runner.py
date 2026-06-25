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
    ) -> None:
        self._dataset_provider = dataset_provider
        self._agent_client = agent_client
        self._query_executor = query_executor
        self._reporter = reporter
        self._engine = evaluation_engine
        self._aggregator = metrics_aggregator
        self._semaphore = asyncio.Semaphore(max_concurrency)

    async def run(
        self,
        dataset_name: str,
        allowed_tables: list[str],
        run_name: str | None = None,
    ) -> DatasetRun:
        """
        Execute full evaluation of a named dataset.

        Args:
            dataset_name: Langfuse dataset name.
            allowed_tables: Tables the agent is permitted to use.
            run_name: Optional label for this run (defaults to a UUID).

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
        """Evaluate a single dataset item under the concurrency semaphore."""
        async with self._semaphore:
            return await self._run_item_pipeline(
                item=item,
                allowed_tables=allowed_tables,
                run_id=run_id,
                dataset_name=dataset_name,
                run_name=run_name,
                parent_trace_id=parent_trace_id,
            )

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
                context.error = agent_response.error or "agent_did_not_produce_sql"

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

        except TimeoutError:
            context.timed_out = True
            context.error = "agent_call_timed_out"
            logger.warning("evaluation_runner.timeout", item_id=item.id)
        except Exception as exc:
            context.agent_crashed = True
            context.error = str(exc)
            logger.error(
                "evaluation_runner.item_pipeline_error",
                item_id=item.id,
                error=str(exc),
                exc_info=True,
            )

        context.total_latency_ms = time.monotonic() * 1000 - start_ms

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
