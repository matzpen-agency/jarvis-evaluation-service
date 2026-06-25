"""
container.py — Dependency injection container.

Wires all abstractions (ports) to their concrete implementations (infrastructure).
FastAPI's Depends() system is used throughout — no service locator pattern.

The API layer never imports infrastructure classes directly.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated, Any

from fastapi import Depends

from src.application.use_cases.run_dataset_evaluation_use_case import (
    BackendTableResolver,
    RunDatasetEvaluationUseCase,
)
from src.config.settings import Settings
from src.domain.evaluators.contains_evaluator import ContainsEvaluator
from src.domain.evaluators.execution_accuracy_evaluator import (
    ExecutionAccuracyEvaluator,
)
from src.domain.evaluators.sql_exact_match_evaluator import SqlExactMatchEvaluator
from src.domain.evaluators.time_shift_evaluator import TimeShiftEvaluator
from src.domain.metrics.composite_score_calculator import CompositeWeights
from src.domain.services.evaluation_engine import EvaluationEngine
from src.domain.services.evaluation_runner import EvaluationRunner
from src.domain.services.metrics_aggregator import MetricsAggregator
from src.infrastructure.langfuse.langfuse_client_factory import create_langfuse_client
from src.infrastructure.langfuse.langfuse_dataset_provider import (
    LangfuseDatasetProvider,
)
from src.infrastructure.langfuse.langfuse_reporter import LangfuseReporter
from src.infrastructure.langfuse.text_to_sql_dataset_item_parser import (
    TextToSqlDatasetItemParser,
)
from src.infrastructure.text_to_sql_agent.text_to_sql_agent_client import (
    TextToSqlAgentClient,
)
from src.infrastructure.trino.backend_query_executor import BackendQueryExecutor
from src.ports.query_executor import QueryExecutor


@lru_cache
def get_settings() -> Settings:
    """Return the singleton Settings instance."""
    return Settings()


# ── Infrastructure factories ───────────────────────────────────────────────────


def get_langfuse_client(
    settings: Annotated[Settings, Depends(get_settings)],
):
    """Create and return the Langfuse SDK client (None if unconfigured)."""
    return create_langfuse_client(settings)


def get_query_executor(
    settings: Annotated[Settings, Depends(get_settings)],
) -> QueryExecutor:
    """Create and return the Backend query executor."""
    return BackendQueryExecutor(
        backend_url=settings.BACKEND_URL,
        query_endpoint=settings.BACKEND_QUERY_ENDPOINT,
        timeout=settings.BACKEND_TIMEOUT,
        enabled=settings.TRINO_ENABLED,
    )


# ── Domain service factories ───────────────────────────────────────────────────


def get_evaluation_engine(
    settings: Annotated[Settings, Depends(get_settings)],
    query_executor: Annotated[QueryExecutor, Depends(get_query_executor)],
) -> EvaluationEngine:
    """
    Build the EvaluationEngine with all registered evaluator plugins.

    To add a new evaluator: instantiate it here and append to the list.
    No other code changes are required.
    """
    evaluators = [
        ExecutionAccuracyEvaluator(
            numeric_tolerance=settings.NUMERIC_COMPARISON_TOLERANCE
        ),
        ContainsEvaluator(
            numeric_tolerance=settings.NUMERIC_COMPARISON_TOLERANCE
        ),
        SqlExactMatchEvaluator(),
        TimeShiftEvaluator(
            query_executor=query_executor,
            offsets_days=settings.TIME_SHIFT_OFFSETS_DAYS,
            numeric_tolerance=settings.NUMERIC_COMPARISON_TOLERANCE,
        ),
    ]
    return EvaluationEngine(evaluators)


def get_metrics_aggregator(
    settings: Annotated[Settings, Depends(get_settings)],
) -> MetricsAggregator:
    """Build the MetricsAggregator with configured composite weights."""
    weights = CompositeWeights(
        execution_accuracy=settings.COMPOSITE_WEIGHT_EXECUTION_ACCURACY,
        contains_accuracy=settings.COMPOSITE_WEIGHT_CONTAINS_ACCURACY,
        sql_exact_match=settings.COMPOSITE_WEIGHT_SQL_EXACT_MATCH,
        time_shift=settings.COMPOSITE_WEIGHT_TIME_SHIFT,
    )
    return MetricsAggregator(weights=weights)


# ── Use case factory ───────────────────────────────────────────────────────────


def get_run_dataset_use_case(
    settings: Annotated[Settings, Depends(get_settings)],
    langfuse_client: Annotated[Any, Depends(get_langfuse_client)],
    query_executor: Annotated[QueryExecutor, Depends(get_query_executor)],
    evaluation_engine: Annotated[EvaluationEngine, Depends(get_evaluation_engine)],
    metrics_aggregator: Annotated[MetricsAggregator, Depends(get_metrics_aggregator)],
) -> RunDatasetEvaluationUseCase:
    """Wire all dependencies and return the fully configured use case."""

    parser = TextToSqlDatasetItemParser()
    dataset_provider = LangfuseDatasetProvider(client=langfuse_client, parser=parser)
    agent_client = TextToSqlAgentClient(settings=settings)
    reporter = LangfuseReporter(client=langfuse_client)
    resolver = BackendTableResolver(
        backend_url=settings.BACKEND_URL,
        timeout=settings.BACKEND_TIMEOUT,
    )

    runner = EvaluationRunner(
        dataset_provider=dataset_provider,
        agent_client=agent_client,
        query_executor=query_executor,
        reporter=reporter,
        evaluation_engine=evaluation_engine,
        metrics_aggregator=metrics_aggregator,
        max_concurrency=settings.MAX_CONCURRENT_EVALUATIONS,
    )

    return RunDatasetEvaluationUseCase(
        runner=runner,
        backend_table_resolver=resolver,
    )
