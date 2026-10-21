"""
evaluation_runner.py — Orchestrates a full dataset evaluation run.

This is the central coordinator of the evaluation platform. It:
  1. Loads the dataset from the DatasetProvider
  2. Resolves the connection's dialect and schema once
  3. Creates a Langfuse dataset run trace
  4. Evaluates each item (bounded concurrency via asyncio.Semaphore)
  5. Aggregates results into a DatasetRun
  6. Writes the summary back to the TraceReporter

Every item always produces a SampleRecord and is reported, whatever fails
inside its pipeline — one bad item never aborts the run.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime

import structlog

from src.domain.entities.dataset_run import DatasetRun
from src.domain.entities.evaluation_context import EvaluationContext
from src.domain.entities.evaluation_result import EvaluationResult
from src.domain.entities.query_result import QueryResult
from src.domain.evaluators.sql_comparison_utils import transpile_sql
from src.domain.services.deadline import run_with_deadline
from src.domain.services.evaluation_engine import EvaluationEngine
from src.domain.services.metrics_aggregator import MetricsAggregator, SampleRecord
from src.ports.agent_client import AgentClient
from src.ports.connection_metadata_provider import ConnectionMetadataProvider
from src.ports.dataset_provider import DatasetProvider
from src.ports.query_executor import QueryExecutor
from src.ports.trace_reporter import TraceReporter

logger = structlog.get_logger(__name__)

ProgressCallback = Callable[[int, int], None]
ItemCallback = Callable[[SampleRecord], None]

# Upper bound for reporting one sample to Langfuse (link retries included)
_REPORT_TIMEOUT_SECONDS = 180.0


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
        evaluation_timeout: float = 600.0,
        metadata_provider: ConnectionMetadataProvider | None = None,
        agent_dialect: str = "trino",
        numeric_tolerance: int = 6,
    ) -> None:
        self._dataset_provider = dataset_provider
        self._agent_client = agent_client
        self._query_executor = query_executor
        self._reporter = reporter
        self._engine = evaluation_engine
        self._aggregator = metrics_aggregator
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._agent_timeout = agent_timeout
        self._evaluation_timeout = evaluation_timeout
        self._metadata_provider = metadata_provider
        self._agent_dialect = agent_dialect
        self._numeric_tolerance = numeric_tolerance

    async def run(
        self,
        dataset_name: str,
        allowed_tables: list[str],
        run_name: str | None = None,
        question_ids: list[str] | None = None,
        limit: int | None = None,
        offset: int | None = None,
        agent_kwargs: dict | None = None,
        progress_callback: ProgressCallback | None = None,
        item_callback: ItemCallback | None = None,
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
            agent_kwargs: Extra agent arguments; connection_id also selects
                the connection that expected/generated SQL run against.
            progress_callback: Called with (completed, total) after each item.
            item_callback: Called with each item's SampleRecord once it is done.

        Returns:
            Completed DatasetRun with all metrics.
        """
        run_id = str(uuid.uuid4())
        run_name = run_name or f"eval-run-{run_id[:8]}"
        started_at = datetime.now(tz=UTC)
        agent_kwargs = agent_kwargs or {}
        connection_id = agent_kwargs.get("connection_id")

        logger.info(
            "evaluation_runner.start",
            dataset_name=dataset_name,
            run_id=run_id,
            run_name=run_name,
            allowed_tables=allowed_tables,
            connection_id=connection_id,
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

        # ── 2. Connection metadata (dialect + schema), resolved once ──────────
        dialect, schema_map = await self._resolve_connection_metadata(connection_id)

        # ── 3. Create top-level trace ─────────────────────────────────────────
        trace_id = await self._reporter.create_dataset_run_trace(
            dataset_name=dataset_name,
            run_id=run_id,
            metadata={
                "run_name": run_name,
                "allowed_tables": allowed_tables,
                "item_count": len(items),
                "connection_id": connection_id,
                "connection_dialect": dialect,
            },
        )

        # ── 4. Evaluate all items (bounded concurrency) ───────────────────────
        completed = 0

        async def evaluate_and_track(item) -> SampleRecord:
            nonlocal completed
            record = await self._evaluate_item(
                item=item,
                allowed_tables=allowed_tables,
                run_id=run_id,
                run_name=run_name,
                parent_trace_id=trace_id,
                agent_kwargs=agent_kwargs,
                dialect=dialect,
                schema_map=schema_map,
            )
            completed += 1
            for callback, args in (
                (item_callback, (record,)),
                (progress_callback, (completed, len(items))),
            ):
                if callback is None:
                    continue
                try:
                    callback(*args)
                except Exception as exc:
                    logger.warning("evaluation_runner.progress_callback_failed", error=str(exc))
            return record

        records: list[SampleRecord] = await asyncio.gather(
            *(evaluate_and_track(item) for item in items)
        )

        # ── 5. Aggregate metrics ──────────────────────────────────────────────
        dataset_run = self._aggregator.aggregate(
            dataset_name=dataset_name,
            run_id=run_id,
            run_name=run_name,
            records=list(records),
            started_at=started_at,
            langfuse_trace_id=trace_id,
        )

        # ── 6. Write summary back ──────────────────────────────────────────────
        await self._reporter.report_run_summary(run=dataset_run, trace_id=trace_id)
        await self._reporter.flush()

        logger.info(
            "evaluation_runner.complete",
            run_id=run_id,
            total=dataset_run.total_cases,
            passed=dataset_run.passed,
            contains=dataset_run.accuracy.contains_accuracy,
            composite=dataset_run.accuracy.composite_score,
            duration_s=dataset_run.duration_seconds,
        )
        return dataset_run

    async def _resolve_connection_metadata(
        self, connection_id: int | None
    ) -> tuple[str | None, dict[str, dict[str, str]]]:
        if self._metadata_provider is None:
            return None, {}
        dialect = await self._metadata_provider.get_dialect(connection_id)
        schema_map = await self._metadata_provider.get_table_schema_map(connection_id)
        logger.info(
            "evaluation_runner.connection_metadata",
            connection_id=connection_id,
            dialect=dialect,
            tables=len(schema_map),
        )
        return dialect, schema_map

    async def _evaluate_item(
        self,
        item,
        allowed_tables: list[str],
        run_id: str,
        run_name: str,
        parent_trace_id: str,
        agent_kwargs: dict,
        dialect: str | None = None,
        schema_map: dict[str, dict[str, str]] | None = None,
    ) -> SampleRecord:
        """
        Evaluate a single dataset item under the concurrency semaphore.

        Never raises: whatever fails, the item is scored with what is known
        and reported, so it always appears in the Langfuse dataset run.
        """
        async with self._semaphore:
            start_ms = time.monotonic() * 1000
            context = EvaluationContext(
                dataset_item=item,
                run_id=run_id,
                query=item.query,
                expected_sql=item.expected_sql,
                allowed_tables=allowed_tables,
                connection_id=agent_kwargs.get("connection_id"),
                dialect=dialect,
                agent_dialect=self._agent_dialect,
                schema_map=schema_map or {},
            )
            self._adapt_expected_sql(context)
            context.metadata["evaluation_config"] = {
                "numeric_tolerance": self._numeric_tolerance,
                "connection_dialect": dialect,
                "agent_dialect": self._agent_dialect,
            }

            results: list[EvaluationResult]
            try:
                await self._run_agent(context, agent_kwargs)
                results = await run_with_deadline(
                    self._execute_and_evaluate(context),
                    self._evaluation_timeout,
                    label=f"evaluation of {item.id}",
                )
            except TimeoutError:
                context.timed_out = True
                context.error = context.error or (
                    f"evaluation_timed_out_after_{int(self._evaluation_timeout)}s"
                )
                logger.warning("evaluation_runner.evaluation_timeout", item_id=item.id)
                results = self._error_results(context.error)
            except Exception as exc:
                context.error = context.error or f"evaluation_pipeline_error: {exc}"
                logger.error(
                    "evaluation_runner.item_failed",
                    item_id=item.id,
                    error=str(exc),
                    exc_info=True,
                )
                results = self._error_results(context.error)

            context.total_latency_ms = time.monotonic() * 1000 - start_ms
            try:
                await run_with_deadline(
                    self._reporter.report_sample(
                        context=context,
                        results=results,
                        parent_trace_id=parent_trace_id,
                        dataset_item_id=item.id,
                        run_name=run_name,
                    ),
                    _REPORT_TIMEOUT_SECONDS,
                    label=f"reporting of {item.id}",
                )
            except Exception as exc:
                logger.error(
                    "evaluation_runner.report_sample_failed",
                    item_id=item.id,
                    error=str(exc),
                )
            return SampleRecord(context=context, results=results)

    @staticmethod
    def _adapt_expected_sql(context: EvaluationContext) -> None:
        """
        Benchmark gold SQL is written for its own engine (BIRD: SQLite);
        transpile it to the connection's dialect before it is executed.
        """
        source_dialect = (context.dataset_item.metadata or {}).get("sql_dialect")
        if not source_dialect or not context.dialect or source_dialect == context.dialect:
            return
        transpiled = transpile_sql(
            context.expected_sql, read=source_dialect, write=context.dialect
        )
        if transpiled:
            context.metadata["original_expected_sql"] = context.expected_sql
            context.expected_sql = transpiled

    def _error_results(self, error: str) -> list[EvaluationResult]:
        return [EvaluationResult.from_error(name, error) for name in self._engine.evaluator_names]

    async def _run_agent(self, context: EvaluationContext, agent_kwargs: dict) -> None:
        """Call the agent; failures are recorded on the context, never raised."""
        agent_start = time.monotonic() * 1000
        try:
            # The agent streams keep-alive pings, so the HTTP client's read
            # timeout never fires on a stuck call: this deadline is the real limit
            agent_response = await run_with_deadline(
                self._agent_client.run(
                    query=context.query,
                    allowed_tables=context.allowed_tables,
                    **agent_kwargs,
                ),
                self._agent_timeout,
                label=f"agent call for {context.dataset_item.id}",
            )
        except TimeoutError:
            context.timed_out = True
            context.agent_call_timed_out = True
            context.error = "agent_call_timed_out"
            logger.warning("evaluation_runner.agent_timeout", item_id=context.dataset_item.id)
            return
        except Exception as exc:
            context.agent_crashed = True
            context.error = str(exc) or type(exc).__name__
            logger.error(
                "evaluation_runner.agent_crashed",
                item_id=context.dataset_item.id,
                error=context.error,
            )
            return
        finally:
            context.agent_latency_ms = time.monotonic() * 1000 - agent_start

        context.agent_response = agent_response
        context.generated_sql = agent_response.sql_query

        if not agent_response.succeeded:
            if agent_response.sql_query is not None:
                # Agent produced SQL but didn't complete normally (e.g. HITL escalation
                # after a Trino error). The SQL still exists and will be evaluated.
                context.error = "sql_execution_failure"
            else:
                context.error = agent_response.error or "agent_did_not_produce_sql"

        for key in (
            "refiner_iteration_count",
            "token_usage",
            "model_version",
            "agent_version",
            "prompt_version",
            "feature_flags",
        ):
            val = agent_response.metadata.get(key)
            if val is not None:
                context.metadata[key] = val

    async def _execute_and_evaluate(self, context: EvaluationContext) -> list[EvaluationResult]:
        """Execute expected and generated SQL, then run every evaluator."""
        coros = []
        if context.expected_sql:
            coros.append(self._execute_expected(context))
        if context.generated_sql:
            coros.append(self._execute_generated(context))
        if coros:
            await asyncio.gather(*coros)
        await self._count_truncated_results(context)

        if context.generated_result and context.generated_result.success:
            context.metadata["total_execution_time_ms"] = context.generated_result.execution_time_ms
        if context.expected_result is not None and not context.expected_result.success:
            context.metadata["expected_sql_error"] = context.expected_result.error

        return await self._engine.run_all(context)

    async def _execute_expected(self, context: EvaluationContext) -> None:
        context.expected_result = await self._safe_execute(
            context.expected_sql, context.connection_id
        )

    async def _execute_generated(self, context: EvaluationContext) -> None:
        """
        Run the agent's SQL on the connection. The agent writes SQL for its own
        engine (Trino); when that fails on a connection with another dialect,
        retry once with the SQL transpiled to the connection's dialect.
        """
        sql = context.generated_sql or ""
        result = await self._safe_execute(sql, context.connection_id)
        context.executed_generated_sql = sql
        context.generated_result = result

        if result.success or not context.dialect or context.dialect == context.agent_dialect:
            return
        transpiled = transpile_sql(sql, read=context.agent_dialect, write=context.dialect)
        if not transpiled or transpiled.strip() == sql.strip():
            return

        retry = await self._safe_execute(transpiled, context.connection_id)
        context.metadata["generated_sql_raw_error"] = result.error
        if retry.success:
            context.executed_generated_sql = transpiled
            context.generated_result = retry
            context.metadata["generated_sql_transpiled"] = True
            logger.info(
                "evaluation_runner.generated_sql_transpiled",
                item_id=context.dataset_item.id,
                dialect=context.dialect,
            )

    async def _count_truncated_results(self, context: EvaluationContext) -> None:
        """
        When both results hit the executor's row cap their rows can't be
        compared, but their full row counts can: different counts settle
        the comparison as a mismatch.
        """
        expected, generated = context.expected_result, context.generated_result
        if not (expected and generated and expected.truncated and generated.truncated):
            return
        generated_sql = context.executed_generated_sql or context.generated_sql or ""
        exp_total, gen_total = await asyncio.gather(
            self._count_rows(context.expected_sql, context.connection_id),
            self._count_rows(generated_sql, context.connection_id),
        )
        expected.total_row_count = exp_total
        generated.total_row_count = gen_total

    async def _count_rows(self, sql: str, connection_id: int | None) -> int | None:
        inner = sql.strip().rstrip(";")
        result = await self._safe_execute(
            f"SELECT COUNT(*) FROM ({inner}) AS __eval_count", connection_id
        )
        if not result.success or not result.rows:
            return None
        try:
            return int(float(str(result.rows[0][0]).replace(",", "")))
        except (TypeError, ValueError):
            return None

    async def _safe_execute(self, sql: str, connection_id: int | None) -> QueryResult:
        try:
            return await self._query_executor.execute(sql, connection_id=connection_id)
        except Exception as exc:
            logger.error("evaluation_runner.sql_execution_error", error=str(exc))
            return QueryResult.failure(f"execution error: {exc}")
