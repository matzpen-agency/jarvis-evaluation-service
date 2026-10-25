"""
container.py — Dependency injection container.

Wires all abstractions (ports) to their concrete implementations (infrastructure).
FastAPI's Depends() system is used throughout — no service locator pattern.

The API layer never imports infrastructure classes directly.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Annotated, Any

import langfuse as lf_sdk
import trino
import trino.auth
from fastapi import Depends

from src.application.use_cases.run_dataset_evaluation_use_case import (
    BackendTableResolver,
    RunDatasetEvaluationUseCase,
)
from src.application.use_cases.run_mcp_dataset_evaluation_use_case import (
    RunMcpDatasetEvaluationUseCase,
)
from src.config.settings import Settings
from src.domain.evaluators.component_match_evaluator import ComponentMatchEvaluator
from src.domain.evaluators.contains_evaluator import ContainsEvaluator
from src.domain.evaluators.dialect_validity_evaluator import DialectValidityEvaluator
from src.domain.evaluators.execution_accuracy_evaluator import (
    ExecutionAccuracyEvaluator,
)
from src.domain.evaluators.schema_validity_evaluator import (
    SchemaValidityEvaluator,
)
from src.domain.evaluators.sql_exact_match_evaluator import SqlExactMatchEvaluator
from src.domain.evaluators.time_shift_evaluator import TimeShiftEvaluator
from src.domain.metrics.composite_score_calculator import CompositeWeights
from src.domain.services.evaluation_engine import EvaluationEngine
from src.domain.services.evaluation_runner import EvaluationRunner
from src.domain.services.metrics_aggregator import MetricsAggregator
from src.infrastructure.bird.bird_dataset_loader import BirdDatasetLoader
from src.infrastructure.jeen.jeen_metadata_client import (
    JeenConnectionMetadataProvider,
    JeenMetadataClient,
    JeenQueryExecutor,
)
from src.infrastructure.langfuse.langfuse_client_factory import create_langfuse_client
from src.infrastructure.langfuse.langfuse_dataset_provider import (
    LangfuseDatasetProvider,
)
from src.infrastructure.langfuse.langfuse_reporter import LangfuseReporter
from src.infrastructure.langfuse.text_to_sql_dataset_item_parser import (
    TextToSqlDatasetItemParser,
)
from src.infrastructure.llm.llm_order_judge import LlmOrderJudge
from src.infrastructure.text_to_sql_agent.mcp_agent_client import McpAgentClient
from src.infrastructure.text_to_sql_agent.text_to_sql_agent_client import (
    TextToSqlAgentClient,
)
from src.infrastructure.trino.trino_query_executor import TrinoQueryExecutor
from src.ports.connection_metadata_provider import ConnectionMetadataProvider
from src.ports.order_judge import OrderJudge
from src.ports.query_executor import QueryExecutor


@lru_cache
def get_settings() -> Settings:
    """Return the singleton Settings instance."""
    return Settings()


# ── Infrastructure factories ───────────────────────────────────────────────────


@lru_cache
def _get_cached_langfuse_client():
    """Create and return the Langfuse SDK client (None if unconfigured) (Cached)."""
    return create_langfuse_client(get_settings())


def get_langfuse_client() -> lf_sdk.Langfuse | None:
    """Dependency provider for Langfuse SDK client."""
    return _get_cached_langfuse_client()


@lru_cache
def _get_bird_loader() -> BirdDatasetLoader:
    settings = get_settings()
    return BirdDatasetLoader(
        path=_project_path(settings.BIRD_DATASET_PATH),
        include_evidence=settings.BIRD_INCLUDE_EVIDENCE,
    )


def _project_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    return os.path.join(project_root, path)


@lru_cache
def _get_order_judge() -> OrderJudge | None:
    """LLM order judge shared across requests (it caches decisions), or None
    when disabled or no LLM key is configured."""
    settings = get_settings()
    api_key = settings.llm_api_key
    if not settings.ORDER_JUDGE_ENABLED or not api_key:
        return None
    return LlmOrderJudge(
        base_url=settings.LLM_BASE_URL,
        api_key=api_key,
        model=settings.LLM_MODEL,
        timeout=settings.LLM_TIMEOUT,
        cache_path=_project_path(settings.ORDER_JUDGE_CACHE_PATH),
    )


@lru_cache
def _get_cached_jeen_client() -> JeenMetadataClient | None:
    """Jeen metadata client shared across requests (it caches schema), or None."""
    settings = get_settings()
    api_key = settings.JEEN_METADATA_MCP_KEY.get_secret_value()
    if not api_key or not settings.JEEN_METADATA_MCP_URL:
        return None
    return JeenMetadataClient(
        url=settings.JEEN_METADATA_MCP_URL,
        api_key=api_key,
        timeout=settings.JEEN_METADATA_TIMEOUT,
        max_rows=settings.JEEN_RUN_QUERY_MAX_ROWS,
        max_concurrency=settings.JEEN_MAX_CONCURRENT_QUERIES,
    )


def get_query_executor(
    settings: Annotated[Settings, Depends(get_settings)],
) -> QueryExecutor:
    """
    Executor for expected/generated SQL: Jeen run_query on the item's
    connection when Jeen is configured (direct Trino when no connection_id),
    otherwise direct Trino.
    """
    trino_executor = _get_trino_query_executor(settings)
    jeen_client = _get_cached_jeen_client()
    if jeen_client is None:
        return trino_executor
    return JeenQueryExecutor(jeen_client, fallback=trino_executor)


def _get_trino_query_executor(settings: Settings) -> QueryExecutor:
    """Create and return a direct Trino query executor."""

    def _connection_factory():
        verify_val = settings.TRINO_VERIFY
        if isinstance(verify_val, str):
            if verify_val.lower() in ("false", "0", "no", "off"):
                verify_val = False
            elif verify_val.lower() in ("true", "1", "yes", "on"):
                verify_val = True

        kwargs: dict = {
            "host": settings.TRINO_HOST,
            "port": settings.TRINO_PORT,
            "user": settings.TRINO_USER,
            "catalog": settings.TRINO_CATALOG,
            "schema": settings.TRINO_SCHEMA,
            "http_scheme": settings.TRINO_HTTP_SCHEME,
            "request_timeout": settings.TRINO_REQUEST_TIMEOUT,
            "verify": verify_val,
        }
        trino_password = settings.TRINO_PASSWORD.get_secret_value()
        if trino_password:
            kwargs["auth"] = trino.auth.BasicAuthentication(
                settings.TRINO_USER, trino_password
            )
        if settings.TRINO_CERT_PATH and settings.TRINO_KEY_PATH:
            kwargs["auth"] = trino.auth.CertificateAuthentication(settings.TRINO_CERT_PATH, settings.TRINO_KEY_PATH)
        return trino.dbapi.connect(**kwargs)

    return TrinoQueryExecutor(
        connection_factory=_connection_factory,
        enabled=settings.TRINO_ENABLED,
    )


def get_backend_table_resolver(
    settings: Annotated[Settings, Depends(get_settings)],
) -> BackendTableResolver:
    """Create and return the backend table resolver."""
    return BackendTableResolver(
        backend_url=settings.BACKEND_URL,
        timeout=settings.BACKEND_TIMEOUT,
    )


def get_connection_metadata_provider(
    resolver: Annotated[BackendTableResolver, Depends(get_backend_table_resolver)],
) -> ConnectionMetadataProvider:
    """Connection dialect/schema from Jeen when configured, else the backend tables."""
    jeen_client = _get_cached_jeen_client()
    if jeen_client is None:
        return resolver
    return JeenConnectionMetadataProvider(jeen_client)


# ── Domain service factories ───────────────────────────────────────────────────


def get_evaluation_engine(
    settings: Annotated[Settings, Depends(get_settings)],
    query_executor: Annotated[QueryExecutor, Depends(get_query_executor)],
    table_resolver: Annotated[
        ConnectionMetadataProvider, Depends(get_connection_metadata_provider)
    ],
) -> EvaluationEngine:
    """
    Build the EvaluationEngine with all registered evaluator plugins.

    To add a new evaluator: instantiate it here and append to the list.
    No other code changes are required.
    """
    evaluators = [
        ExecutionAccuracyEvaluator(
            numeric_tolerance=settings.NUMERIC_COMPARISON_TOLERANCE,
            order_judge=_get_order_judge(),
        ),
        ContainsEvaluator(
            numeric_tolerance=settings.NUMERIC_COMPARISON_TOLERANCE,
            order_judge=_get_order_judge(),
        ),
        SqlExactMatchEvaluator(),
        TimeShiftEvaluator(
            query_executor=query_executor,
            offsets_days=settings.TIME_SHIFT_OFFSETS_DAYS,
            numeric_tolerance=settings.NUMERIC_COMPARISON_TOLERANCE,
            table_resolver=table_resolver,
            shift_timeout_seconds=settings.TIME_SHIFT_TIMEOUT,
            order_judge=_get_order_judge(),
        ),
        ComponentMatchEvaluator(),
        SchemaValidityEvaluator(table_resolver=table_resolver),
        DialectValidityEvaluator(),
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
        component_match=settings.COMPOSITE_WEIGHT_COMPONENT_MATCH,
        schema_validity=settings.COMPOSITE_WEIGHT_SCHEMA_VALIDITY,
        dialect_validity=settings.COMPOSITE_WEIGHT_DIALECT_VALIDITY,
    )
    return MetricsAggregator(weights=weights)


# ── Use case factory ───────────────────────────────────────────────────────────


def get_run_dataset_use_case(
    settings: Annotated[Settings, Depends(get_settings)],
    langfuse_client: Annotated[Any, Depends(get_langfuse_client)],
    query_executor: Annotated[QueryExecutor, Depends(get_query_executor)],
    evaluation_engine: Annotated[EvaluationEngine, Depends(get_evaluation_engine)],
    metrics_aggregator: Annotated[MetricsAggregator, Depends(get_metrics_aggregator)],
    resolver: Annotated[BackendTableResolver, Depends(get_backend_table_resolver)],
    metadata_provider: Annotated[
        ConnectionMetadataProvider, Depends(get_connection_metadata_provider)
    ],
) -> RunDatasetEvaluationUseCase:
    """Wire all dependencies and return the fully configured use case."""

    parser = TextToSqlDatasetItemParser()
    dataset_provider = LangfuseDatasetProvider(
        client=langfuse_client,
        parser=parser,
        bird_loader=_get_bird_loader(),
    )
    agent_client = TextToSqlAgentClient(settings=settings)
    reporter = LangfuseReporter(client=langfuse_client)

    runner = EvaluationRunner(
        dataset_provider=dataset_provider,
        agent_client=agent_client,
        query_executor=query_executor,
        reporter=reporter,
        evaluation_engine=evaluation_engine,
        metrics_aggregator=metrics_aggregator,
        max_concurrency=settings.MAX_CONCURRENT_EVALUATIONS,
        agent_timeout=settings.AGENT_TIMEOUT,
        evaluation_timeout=settings.EVALUATION_ITEM_TIMEOUT,
        metadata_provider=metadata_provider,
        agent_dialect=settings.AGENT_SQL_DIALECT,
        numeric_tolerance=settings.NUMERIC_COMPARISON_TOLERANCE,
    )

    return RunDatasetEvaluationUseCase(
        runner=runner,
        backend_table_resolver=resolver,
    )


def get_run_mcp_dataset_use_case(
    settings: Annotated[Settings, Depends(get_settings)],
    langfuse_client: Annotated[Any, Depends(get_langfuse_client)],
    query_executor: Annotated[QueryExecutor, Depends(get_query_executor)],
    evaluation_engine: Annotated[EvaluationEngine, Depends(get_evaluation_engine)],
    metrics_aggregator: Annotated[MetricsAggregator, Depends(get_metrics_aggregator)],
    metadata_provider: Annotated[
        ConnectionMetadataProvider, Depends(get_connection_metadata_provider)
    ],
) -> RunMcpDatasetEvaluationUseCase:
    """Wire MCP agent dependencies and return the configured use case."""

    parser = TextToSqlDatasetItemParser()
    dataset_provider = LangfuseDatasetProvider(
        client=langfuse_client,
        parser=parser,
        bird_loader=_get_bird_loader(),
    )
    agent_client = McpAgentClient(settings=settings)
    reporter = LangfuseReporter(client=langfuse_client)

    runner = EvaluationRunner(
        dataset_provider=dataset_provider,
        agent_client=agent_client,
        query_executor=query_executor,
        reporter=reporter,
        evaluation_engine=evaluation_engine,
        metrics_aggregator=metrics_aggregator,
        max_concurrency=settings.MAX_CONCURRENT_EVALUATIONS,
        agent_timeout=settings.AGENT_TIMEOUT,
        evaluation_timeout=settings.EVALUATION_ITEM_TIMEOUT,
        metadata_provider=metadata_provider,
        agent_dialect=settings.AGENT_SQL_DIALECT,
        numeric_tolerance=settings.NUMERIC_COMPARISON_TOLERANCE,
    )

    return RunMcpDatasetEvaluationUseCase(runner=runner, metadata_provider=metadata_provider)
