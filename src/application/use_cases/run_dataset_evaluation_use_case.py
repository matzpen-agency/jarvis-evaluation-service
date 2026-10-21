"""
run_dataset_evaluation_use_case.py — Orchestrates a dataset evaluation request.

Responsibilities:
  1. Resolve allowed_tables (production tables from backend API + additional_tables)
  2. Delegate to EvaluationRunner
  3. Map DatasetRun → RunDatasetResponse DTO
"""

from __future__ import annotations


import structlog

from src.application.dto.run_dataset_request import RunDatasetRequest
from src.application.dto.run_dataset_response import (
    AccuracyStatsDTO,
    FailureAnalysisDTO,
    FailureCategoryDTO,
    LatencyStatsDTO,
    RunDatasetResponse,
)
from src.domain.entities.dataset_run import DatasetRun
from src.domain.services.evaluation_runner import EvaluationRunner
from src.ports.connection_metadata_provider import ConnectionMetadataProvider

logger = structlog.get_logger(__name__)


class RunDatasetEvaluationUseCase:
    """
    Application-layer use case for running a dataset evaluation.

    This class is the entry point for the evaluation workflow.
    It should contain no infrastructure imports — only domain
    and application concerns.
    """

    def __init__(
        self,
        runner: EvaluationRunner,
        backend_table_resolver: BackendTableResolver,
    ) -> None:
        self._runner = runner
        self._resolver = backend_table_resolver

    async def execute(self, request: RunDatasetRequest) -> RunDatasetResponse:
        """
        Run an evaluation and return the response DTO.

        Raises:
            DatasetNotFoundError: If the Langfuse dataset doesn't exist.
        """
        logger.info(
            "use_case.run_dataset_evaluation.start",
            dataset_name=request.dataset_name,
            additional_tables=request.additional_tables,
        )

        # ── Resolve allowed tables ─────────────────────────────────────────────
        production_tables = await self._resolver.get_production_tables()
        allowed_tables = list(
            {*production_tables, *request.additional_tables}
        )

        logger.info(
            "use_case.tables_resolved",
            production_count=len(production_tables),
            additional_count=len(request.additional_tables),
            total=len(allowed_tables),
        )

        # ── Run evaluation ─────────────────────────────────────────────────────
        dataset_run = await self._runner.run(
            dataset_name=request.dataset_name,
            allowed_tables=allowed_tables,
            question_ids=request.question_ids,
            limit=request.limit,
            offset=request.offset,
        )

        return self._to_response(dataset_run)

    @staticmethod
    def _to_response(run: DatasetRun) -> RunDatasetResponse:
        """Map DatasetRun domain entity to RunDatasetResponse DTO."""
        from src.application.dto.run_dataset_response import (
            PerformanceStatsDTO,
            RunDatasetCaseResultDTO,
        )
        return RunDatasetResponse(
            dataset_name=run.dataset_name,
            run_id=run.run_id,
            total_cases=run.total_cases,
            passed=run.passed,
            failed=run.failed,
            failure_rate=run.failure_rate,
            langfuse_trace_id=run.langfuse_trace_id,
            duration_seconds=run.duration_seconds,
            latency=LatencyStatsDTO(
                p50=run.latency.p50,
                p95=run.latency.p95,
                p99=run.latency.p99,
                average=run.latency.average,
                minimum=run.latency.minimum,
                maximum=run.latency.maximum,
                total_samples=run.latency.total_samples,
            ),
            accuracy=AccuracyStatsDTO(
                execution_accuracy=run.accuracy.execution_accuracy,
                contains_accuracy=run.accuracy.contains_accuracy,
                sql_exact_match=run.accuracy.sql_exact_match,
                time_shift_score=run.accuracy.time_shift_score,
                component_match=run.accuracy.component_match,
                schema_hallucination=run.accuracy.schema_hallucination,
                dialect_error=run.accuracy.dialect_error,
                composite_score=run.accuracy.composite_score,
                evaluated_counts=run.accuracy.evaluated_counts,
            ),
            failure_analysis=FailureAnalysisDTO(
                total_failures=run.failure_analysis.total_failures,
                failure_rate=run.failure_analysis.failure_rate,
                agent_crash_count=run.failure_analysis.agent_crash_count,
                agent_crash_rate=run.failure_analysis.agent_crash_rate,
                sql_execution_failure_count=run.failure_analysis.sql_execution_failure_count,
                sql_execution_failure_rate=run.failure_analysis.sql_execution_failure_rate,
                trino_failure_count=run.failure_analysis.trino_failure_count,
                trino_failure_rate=run.failure_analysis.trino_failure_rate,
                timeout_count=run.failure_analysis.timeout_count,
                timeout_rate=run.failure_analysis.timeout_rate,
                validation_failure_count=run.failure_analysis.validation_failure_count,
                validation_failure_rate=run.failure_analysis.validation_failure_rate,
                categories=[
                    FailureCategoryDTO(
                        category=c.category,
                        count=c.count,
                        rate=c.rate,
                    )
                    for c in run.failure_analysis.categories
                ],
            ),
            performance=PerformanceStatsDTO(
                average_total_execution_time_ms=run.performance.average_total_execution_time_ms,
                average_time_to_first_row_ms=run.performance.average_time_to_first_row_ms,
                total_token_usage=run.performance.total_token_usage,
                average_token_usage=run.performance.average_token_usage,
                average_refiner_iterations=run.iteration_stats.average_iterations,
            ),
            cases=[
                RunDatasetCaseResultDTO(
                    question_id=c.question_id,
                    generated_sql=c.generated_sql,
                    expected_sql=c.expected_sql,
                    succeeded=c.succeeded,
                    error=c.error,
                    scores=c.scores,
                    not_applicable=c.not_applicable,
                )
                for c in run.cases
            ],
        )


class BackendTableResolver(ConnectionMetadataProvider):
    """
    Resolves the list of production tables by calling the backend REST API.

    Falls back to an empty list if the backend is unavailable. Knows table
    names only (no columns) and ignores connection_id; Jeen metadata is the
    source of connection schemas.
    """

    def __init__(self, backend_url: str, timeout: float = 30.0) -> None:
        self._url = f"{backend_url}/api/agent/tables"
        self._timeout = timeout

    async def get_dialect(self, connection_id: int | None) -> str | None:
        return None

    async def get_table_schema_map(
        self, connection_id: int | None = None
    ) -> dict[str, dict[str, str]]:
        """
        Column metadata isn't available from the backend (connection schemas
        live in Jeen metadata), so schema-dependent evaluators report
        "not applicable" on this path.
        """
        return {}

    async def get_production_tables(self) -> list[str]:
        """
        Fetch production-status table names from the backend.

        Returns empty list on any error to allow evaluation to proceed
        with only the additional_tables provided in the request.
        """
        import httpx

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.get(
                    self._url,
                    params={"status": "production"},
                )
                response.raise_for_status()
                tables = response.json()
                # Handle both list[str] and list[{name: str}] responses
                if tables and isinstance(tables[0], dict):
                    return [t.get("name", "") for t in tables if t.get("name")]
                return [t for t in tables if isinstance(t, str)]
        except Exception as exc:
            logger.warning(
                "backend_table_resolver.failed",
                error=str(exc),
                fallback="empty_list",
            )
            return []
